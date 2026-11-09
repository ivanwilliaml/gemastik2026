"""Author topic modeling notebook untuk data ratusan ribu baris; eksekusi hanya di Kaggle (dua dataset)."""
from pathlib import Path
from textwrap import dedent
import ast, base64, json, tempfile, hashlib
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'a_tanpa_label/01_topic_modeling/01_topic_modeling_besar.ipynb'
UNIVERSAL = ROOT / 'preprocessing_universal/stepwise.py'
CELLS = []


def md(s): CELLS.append(nb.v4.new_markdown_cell(dedent(s).strip()))
def code(s): CELLS.append(nb.v4.new_code_cell(dedent(s).strip()))
def part(title, note, *sources):
    md('## ' + title + '\n\n' + note)
    for source in sources: code(source)


md(r'''
# Topic modeling · data besar (ratusan ribu baris)
**Versi skala besar dari `01_topic_modeling_review`. Target: Run All di Kaggle T4 jauh di bawah 5 jam.**

| Masalah di data besar | Solusi |
|---|---|
| Banyak teks identik | Model jalan di **teks unik** `U`, dengan `count` sebagai bobot. Ukuran topik dihitung per baris asli |
| Grid model × k × seed mahal | Pemilihan dilakukan di **sampel `N_FIT` teks unik** |
| Fit ulang di semua data mahal | Model dari sampel diterapkan ke semua teks: `transform` untuk LDA/NMF, `predict` untuk KMeans, dan centroid e5 untuk HDBSCAN. Kecocokan sampel dengan hasil penuh dicek di bagian 09 |
| Stemming lambat | Hanya kata yang lolos `MIN_DF`, 4 proses `forkserver` |

**Model** sama dengan versi review: acak, TF-IDF + KMeans, LDA, NMF, e5 + KMeans, dan e5 + UMAP + HDBSCAN (gaya BERTopic). Backbone e5 di-freeze.

**Aturan pemilihan tanpa label** juga sama: topik raksasa dan outlier berlebihan dilarang, lalu kandidat diurutkan dengan rata-rata peringkat TQ (coherence × diversity) dan stabilitas antar seed. Di versi review, stabilitas adalah kriteria yang paling searah dengan audit (Spearman +0,60), sedangkan TQ +0,19.

**Data:** `news` = 91 ribu judul berita, kategori sebagai audit. `gojek` = 225 ribu ulasan aplikasi, tanpa label topik.
''')

part('00 · Konfigurasi', 'Parameter papermill. `audit` diisi `None` kalau labelnya tidak ada. `desc` adalah kolom angka opsional untuk membaca topik, misalnya rating.', r'''
DATASET = "news"
SEED, TOP = 42, 10
N_ROWS, N_FIT = None, 30000
K_GRID = [8, 10, 15, 20, 25, 30]
SEEDS = 3
MAX_SHARE, MAX_OUTLIER = .40, .30
MIN_DF, DF_MAX, MIN_TOPIC_SHARE, MERGE_JACCARD, GIANT_GENERIC = 5, .25, .005, .30, .25
MAX_LEN, BATCH = 128, 256
BUDGET_MINUTES = 290
PRESETS = {"news": dict(file="indonesian-news-title.csv", text="title", audit="category", desc=None),
           "gojek": dict(file="GojekAppReviewV4.0.0-V4.9.3_Cleaned.csv", text="content", audit=None, desc="score")}
''', r'''
from pathlib import Path
P = PRESETS[DATASET]; FILE, TEXT_COL, AUDIT_COL, DESC_COL = P["file"], P["text"], P["audit"], P["desc"]
OUT = Path(f"/kaggle/working/hasil_topik_besar_{DATASET}")
print("dataset:", DATASET, "| teks:", TEXT_COL, "| audit:", AUDIT_COL or "tidak ada")
''')

part('01 · Lingkungan Kaggle & pencatat waktu', 'Waktu tiap tahap dicatat, dan notebook dibuat gagal kalau total melewati `BUDGET_MINUTES`.', r'''
import sys, os, subprocess
assert Path("/kaggle/input").exists(), "Jalankan notebook ini di Kaggle."
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "umap-learn>=0.5,<0.6", "gensim>=4.4,<5", "ftfy>=6.3,<7",
                "emoji>=2.14,<3", "langid==1.1.6", "Sastrawi==1.0.1"], check=True)
os.environ.update(TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_PROGRESS_BARS="1", USE_TF="0", USE_FLAX="0")
''', r'''
import json, time, random, re, html, unicodedata, itertools, warnings
from collections import Counter
from contextlib import contextmanager
import numpy as np, pandas as pd, torch, emoji, ftfy
from IPython.display import display
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
warnings.filterwarnings("ignore")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
OUT.mkdir(exist_ok=True); INPUT = Path("/kaggle/input"); RUN_START = time.perf_counter(); TIMES = []
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "tidak ada")

@contextmanager
def stage(name, n=None):
    start = time.perf_counter(); yield
    TIMES.append({"tahap": name, "detik": time.perf_counter() - start, "n": n})
    print(f"[waktu] {name}: {TIMES[-1]['detik']:.0f} detik | total {(time.perf_counter() - RUN_START) / 60:.1f} menit")
def show(title, table):
    print("\n" + title); display((table.to_frame() if isinstance(table, pd.Series) else table).round(3))
def distribution(values):
    n = values.value_counts(dropna=False); return pd.DataFrame({"n": n, "%": n / (len(values) or np.nan) * 100})
pd.set_option("display.max_columns", None, "display.max_colwidth", 100, "display.width", 200)
''')

part('02 · Muat data & dedupe', 'Kualitas diaudit di semua baris. Model jalan di tabel `U` berisi teks unik. Label audit per teks unik diambil dari suara terbanyak, lalu disembunyikan sampai bagian 10.', r'''
with stage("muat_data"):
    path = next(INPUT.rglob(FILE)); raw = pd.read_csv(path, low_memory=False)
text = raw[TEXT_COL]
quality = pd.DataFrame({"teks_null": text.isna(), "teks_non_string": text.notna() & ~text.map(lambda x: isinstance(x, str)),
    "teks_kosong": text.fillna("").astype(str).str.strip().eq("")})
show("Kualitas input", pd.DataFrame({"n": quality.sum(), "%": quality.mean() * 100}))
data = raw.loc[~quality.any(axis=1)].reset_index(names="row_id")
if N_ROWS and len(data) > N_ROWS: data = data.sample(N_ROWS, random_state=SEED).reset_index(drop=True)
data["key"] = data[TEXT_COL].astype(str).str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
agg = {"text": (TEXT_COL, "first"), "count": ("key", "size")}
if DESC_COL: agg["desc"] = (DESC_COL, "mean")
if AUDIT_COL: agg["_audit"] = (AUDIT_COL, lambda s: s.astype(str).mode().iloc[0])
U = data.groupby("key", sort=False).agg(**agg).reset_index(); W = U["count"].to_numpy()
show("Ukuran setelah dedupe", pd.Series({"baris": len(data), "teks_unik": len(U), "hemat_%": 100 * (1 - len(U) / len(data)), "teks_terbanyak_n": W.max()}))
show("Teks paling sering · kandidat topik palsu", U.nlargest(TOP, "count")[["text", "count"]])
''')

part('03 · EDA wajib topic modeling · angka', 'Pertanyaannya sama dengan versi review: panjang dokumen, kosakata setelah pruning, kata di mana-mana, kolokasi, noise, dan bahasa. Semua dihitung berbobot baris.', r'''
with stage("eda", len(U)):
    t = U.text.astype(str); words = t.str.split().str.len().to_numpy()
    show("3.1 Panjang dokumen (berbobot baris)", pd.Series({"kata_median": np.median(np.repeat(words, W)), "<5_kata_%": 100 * W[words < 5].sum() / W.sum(),
        "<10_kata_%": 100 * W[words < 10].sum() / W.sum(), ">50_kata_%": 100 * W[words > 50].sum() / W.sum()}))
    seqs = t.str.casefold().str.findall(r"[^\W\d_]+")
    counts = pd.Series(Counter(w for s in seqs for w in s), dtype="int64").sort_values(ascending=False)
    docfreq = pd.Series(Counter(w for s in seqs for w in set(s)), dtype="int64").reindex(counts.index) / len(seqs)
    kept = docfreq[(docfreq * len(seqs) >= MIN_DF) & (docfreq <= DF_MAX)]
    show("3.2 Kosakata & pruning · teks unik", pd.Series({"tipe_unik": len(counts), "hapax_%_tipe": 100 * counts.eq(1).mean(),
        "kosakata_setelah_pruning": len(kept), "token_tersisa_%": 100 * counts.reindex(kept.index).sum() / counts.sum(), f"kata_df>{DF_MAX:.0%}": int(docfreq.gt(DF_MAX).sum())}))
    show("Top words", pd.DataFrame({"n": counts.head(15), "%_dokumen": 100 * docfreq.head(15)}))
    pairs = Counter(" ".join(s[i:i + 2]) for s in seqs for i in range(len(s) - 1)); total = counts.sum()
    colloc = pd.DataFrame([{"bigram": b, "n": n, "PMI": np.log(n * total / (counts[b.split()[0]] * counts[b.split()[1]]))} for b, n in pairs.items() if n >= 30])
    show("3.3 Kolokasi · kandidat frasa", colloc.sort_values("PMI", ascending=False).head(TOP) if len(colloc) else colloc)
    NOISE = {"url": r"https?://|www\.", "html": r"</?[a-zA-Z][^>]*>", "mention": r"@\w+", "hashtag": r"#\w+", "angka": r"\d",
             "huruf_ulang": r"([^\W\d_])\1{2,}", "non_ASCII": r"[^\x00-\x7f]", "slang_contoh": r"\b(?:gk|gak|ga|bgt|yg|udh|klo|jg|tdk)\b"}
    flags = pd.DataFrame({n: t.str.contains(p, flags=re.I) for n, p in NOISE.items()})
    show("3.4 Noise · % baris", (flags.mul(W, axis=0).sum() / W.sum() * 100).sort_values(ascending=False))
''')

md('## 04 · Preprocessing dari universal\n\nFungsi berikut diambil dari `preprocessing_universal/stepwise.py`. BoW agresif dipakai LDA, NMF, dan TF-IDF, sedangkan e5 memakai teks utuh. Stemming dikerjakan sebelum GPU dipakai, supaya proses paralelnya aman.')
u = UNIVERSAL.read_text(encoding='utf-8'); lines = u.splitlines(); tree = ast.parse(u)
code(u.split('# %% resources\n', 1)[1].split('# %% validate', 1)[0])
functions = {node.name: '\n'.join(lines[node.lineno - 1:node.end_lineno]) for node in tree.body if isinstance(node, ast.FunctionDef)}
for names in [['valid_text', 'repair_encoding', 'normalize_unicode', 'remove_html', 'decode_html_entities', 'remove_invisible', 'normalize_whitespace'],
              ['replace_urls', 'replace_emails', 'replace_phones', 'replace_mentions', 'segment_hashtags', 'lowercase'],
              ['normalize_slang', 'normalize_elongation', 'normalize_laughter', 'normalize_reduplication'],
              ['remove_emoji', 'remove_emoticons', 'remove_numbers', 'remove_punctuation', 'reduce_repeated_words']]:
    code('\n\n'.join(functions[name] for name in names))
code(r'''
def stem_series(series):
    """Stem hanya kata yang lolos MIN_DF, 4 proses forkserver."""
    import multiprocessing as mp
    Path("stem_worker.py").write_text("from Sastrawi.Stemmer.StemmerFactory import StemmerFactory\n_s = None\n"
        "def stem(w):\n    global _s\n    _s = _s or StemmerFactory().create_stemmer()\n    return _s.stem(w)\n")
    if "." not in sys.path: sys.path.insert(0, ".")
    import stem_worker
    vocab = pd.Series(Counter(w for s in series.str.split() for w in set(s)))
    frequent = [w for w in vocab[vocab >= MIN_DF].index if w not in PROTECTED | NEGATIONS]
    with mp.get_context("forkserver").Pool(4) as pool:
        stems = dict(zip(frequent, pool.map(stem_worker.stem, frequent, chunksize=256)))
    print(f"stemming {len(frequent)} kata dari {len(vocab)} kata unik")
    return series.map(lambda x: " ".join(stems.get(w, w) for w in x.split()))
def chain(text, steps):
    for step in steps: text = step(text)
    return text
def apply(series, steps):
    cache = {s: chain(s, steps) for s in series.unique()}; return series.map(cache)
SAFE = [valid_text, repair_encoding, normalize_unicode, lambda s: remove_html(s) if "<" in s else s, decode_html_entities, remove_invisible, normalize_whitespace]
NORMAL = [segment_hashtags, lowercase, normalize_slang, normalize_elongation, normalize_laughter, normalize_reduplication]
CLEAN = [remove_emoji, remove_emoticons, remove_numbers, remove_punctuation, reduce_repeated_words, normalize_whitespace]
STOP = STOPWORDS | set("""akan ada adalah agar aja aku apa atau bahwa banyak baru bila bisa buat bukan cuma dah dalam dapat demi dgn
    dong gitu harus hal hanya ini itu jadi jika juga kalau kami kamu karena kata ke kenapa kok kini lagi lah lain lalu lebih
    loh maka masih mau mereka meski mungkin nah namun nanti nya oleh pada para pernah pun saat saja sama sampai sangat
    sebagai sedang sejak sekarang semua seperti setelah sih sudah supaya tapi telah tentang terhadap tersebut tetapi
    untuk usai yaitu yang ya url user email phone wkwk haha tak tidak gak ga nggak enggak belum jangan bukan""".split())
with stage("preprocessing", len(U)):
    U["safe"] = apply(U.text, SAFE); U["embed"] = apply(U.safe, [replace_urls, replace_emails, replace_phones, replace_mentions, normalize_whitespace])
    U["normal"] = apply(U.embed, NORMAL); bow = apply(U.normal, CLEAN).str.split()
    df_bow = pd.Series(Counter(w for s in bow for w in set(s))) / len(bow); DOMAIN_STOP = set(df_bow[df_bow > DF_MAX].index)
    U["bow"] = bow.map(lambda s: " ".join(w for w in s if w not in STOP | DOMAIN_STOP and len(w) > 2))
with stage("stemming", len(U)):
    U["bow_stem"] = stem_series(U.bow)
show(f"Stopword domain · df > {DF_MAX:.0%}", df_bow[df_bow > DF_MAX].sort_values(ascending=False).rename("df"))
show("Audit preprocessing", pd.DataFrame({v: {"kosong_%_baris": 100 * W[U[v].str.strip().eq("").to_numpy()].sum() / W.sum(),
    "kata_median": U[v].str.split().str.len().median(), "kosakata": len(set(" ".join(U[v]).split()))} for v in ["safe", "normal", "bow", "bow_stem"]}).T)
rng = np.random.default_rng(SEED); S = np.sort(rng.choice(len(U), min(N_FIT, len(U)), replace=False))
''')

part('05 · Fungsi evaluasi tanpa label', 'Fungsinya sama dengan versi review. Bedanya, ukuran topik dihitung berbobot `count`, dan semua fungsi menerima `idx`, yaitu baris `U` yang dinilai.', r'''
REF = {}
def reference(view):
    if view not in REF:
        vectorizer = CountVectorizer(binary=True, min_df=MIN_DF, token_pattern=r"\S+").fit(U[view])
        matrix = vectorizer.transform(U[view]).tocsc(); REF[view] = (matrix, vectorizer.vocabulary_, np.asarray(matrix.sum(0)).ravel())
    return REF[view]
def npmi(words, view):
    matrix, vocab, docfreq = reference(view); n = matrix.shape[0]; ids = [vocab[w] for w in words if w in vocab]; scores = []
    for a, b in itertools.combinations(ids, 2):
        both = matrix[:, a].multiply(matrix[:, b]).sum() / n
        scores.append(-1. if both == 0 else np.log(both / (docfreq[a] * docfreq[b] / n ** 2)) / -np.log(both))
    return float(np.mean(scores)) if scores else np.nan
def ctfidf(labels, view, idx, n=25):
    texts = U[view].to_numpy()[idx]; topics = [c for c in np.unique(labels) if c >= 0]
    joined = [" ".join(texts[labels == c]) for c in topics]
    vectorizer = CountVectorizer(token_pattern=r"\S+").fit(joined); tf = vectorizer.transform(joined).toarray().astype(float)
    weight = tf * np.log1p(tf.sum() / len(topics) / np.maximum(tf.sum(0), 1)); names = vectorizer.get_feature_names_out()
    return {c: list(names[np.argsort(-weight[i])[:n]]) for i, c in enumerate(topics)}
def match_stability(runs):
    scores = []
    for a, b in itertools.combinations(runs, 2):
        A, B = list(a.values()), list(b.values())
        sim = np.array([[len(set(x[:10]) & set(y[:10])) / len(set(x[:10]) | set(y[:10])) for y in B] for x in A])
        r, c = linear_sum_assignment(-sim); scores.append(sim[r, c].sum() / max(len(A), len(B)))
    return float(np.mean(scores))
def audit_scores(labels, idx):
    truth = U["_audit"].to_numpy()[idx]; w = W[idx]
    table = pd.crosstab(labels, truth, values=w, aggfunc="sum").fillna(0).to_numpy(); r, c = linear_sum_assignment(-table)
    return {"audit_NMI": normalized_mutual_info_score(np.repeat(truth, w), np.repeat(labels, w)), "audit_purity": table.max(1).sum() / w.sum(),
            "audit_acc_hungarian": table[r, c].sum() / w.sum()}
rows, STORE = [], {}
def evaluate(name, topics, labels, idx, view, **extra):
    labels = np.asarray(labels); valid = labels >= 0; w = W[idx]
    share = pd.Series(w[valid]).groupby(labels[valid]).sum() / w.sum()
    per_topic = {c: npmi(ws[:TOP], view) for c, ws in topics.items()}; top25 = [x for ws in topics.values() for x in ws[:25]]
    coherence = np.nanmean(list(per_topic.values())); diversity = len(set(top25)) / max(len(top25), 1)
    row = {"model": name, **extra, "n": len(idx), "k": len(topics), "outlier_%": 100 * w[~valid].sum() / w.sum(),
           "max_share_%": 100 * share.max() if len(share) else np.nan, "NPMI": coherence, "diversity": diversity, "TQ": (coherence + 1) / 2 * diversity}
    if AUDIT_COL: row.update(audit_scores(labels, idx))
    rows.append(row); STORE[name] = {"topics": topics, "labels": labels, "idx": idx, "view": view}; return row
''')

part('06 · Representasi & ablasi di sampel', 'e5 dihitung sekali untuk semua teks unik, karena dibutuhkan model e5, penugasan outlier, dan analisis. Ablasi preprocessing memakai NMF di sampel `S`.', r'''
def locate(name, hf_id):
    found = list(INPUT.glob(f"**/models/{name}/config.json")); return str(found[0].parent) if found else hf_id
def embed(model_id, texts):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_id); model = AutoModel.from_pretrained(model_id).cuda().half().eval()
    order = np.argsort([len(x) for x in texts]); parts = []
    with torch.inference_mode():
        for i in range(0, len(texts), BATCH):
            enc = tokenizer([texts[j] for j in order[i:i + BATCH]], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda")
            hidden = model(**enc).last_hidden_state; mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            parts.append(((hidden * mask).sum(1) / mask.sum(1)).float().cpu().numpy())
    vectors = np.zeros((len(texts), parts[0].shape[1]), "float32"); vectors[order] = np.concatenate(parts)
    del model; torch.cuda.empty_cache(); return normalize(vectors)
with stage("embed_e5_penuh", len(U)):
    E5 = embed(locate("e5", "intfloat/multilingual-e5-base"), ("query: " + U.embed).tolist())
    E5C = normalize(E5 - E5.mean(0))   # dipusatkan: kemiripan antar topik tidak lagi semuanya ~0,99
def factor_topics(components, names, n=25): return {i: list(names[np.argsort(-row)[:n]]) for i, row in enumerate(components)}
def vectorizers(view):
    tfidf = TfidfVectorizer(min_df=MIN_DF, max_df=.5, token_pattern=r"\S+", sublinear_tf=True, max_features=50000).fit(U[view].iloc[S])
    count = CountVectorizer(min_df=MIN_DF, max_df=.5, token_pattern=r"\S+", max_features=50000).fit(U[view].iloc[S])
    return tfidf, count
K_REF = K_GRID[len(K_GRID) // 2]; ablation = []
with stage("ablasi", len(S)):
    for view in ["normal", "bow", "bow_stem"]:
        tfidf, _ = vectorizers(view); X = tfidf.transform(U[view].iloc[S])
        model = NMF(K_REF, init="nndsvda", max_iter=300, random_state=SEED); Wm = model.fit_transform(X)
        ablation.append({"view": view, **evaluate(f"ablasi_{view}", factor_topics(model.components_, tfidf.get_feature_names_out()),
                                                    np.where(Wm.max(1) > 0, Wm.argmax(1), -1), S, view, algo="ablasi")})
rows = [r for r in rows if r.get("algo") != "ablasi"]
ablation = pd.DataFrame(ablation).drop(columns=["model", "algo"]); show(f"6.1 Ablasi · NMF k={K_REF}, sampel {len(S)}", ablation)
BOW_VIEW = ablation.sort_values("TQ").iloc[-1]["view"]; print("Versi BoW terpilih:", BOW_VIEW)
''')

part('07 · Grid di sampel', 'Lima model × `K_GRID` × 3 seed di sampel `S`. HDBSCAN memvariasikan `min_cluster_size`. Pembanding acak dibuat dari kata acak.', r'''
TFIDF, COUNT = vectorizers(BOW_VIEW)
X_t, X_c = TFIDF.transform(U[BOW_VIEW].iloc[S]), COUNT.transform(U[BOW_VIEW].iloc[S])
names_t, names_c = TFIDF.get_feature_names_out(), COUNT.get_feature_names_out(); MODELS = {}
def fit_once(algo, k, seed):
    if algo == "tfidf_kmeans":
        m = KMeans(k, n_init=1, random_state=seed).fit(X_t); return ctfidf(m.labels_, BOW_VIEW, S), m.labels_, m
    if algo == "e5_kmeans":
        m = KMeans(k, n_init=1, random_state=seed).fit(E5[S]); return ctfidf(m.labels_, BOW_VIEW, S), m.labels_, m
    if algo == "lda":
        m = LatentDirichletAllocation(k, learning_method="online", batch_size=2048, max_iter=10, random_state=seed); theta = m.fit_transform(X_c)
        return factor_topics(m.components_, names_c), theta.argmax(1), m
    m = NMF(k, init="nndsvda" if seed == SEED else "random", max_iter=300, random_state=seed); Wm = m.fit_transform(X_t)
    return factor_topics(m.components_, names_t), np.where(Wm.max(1) > 0, Wm.argmax(1), -1), m
with stage("grid_sampel", len(S)):
    for algo in ["tfidf_kmeans", "lda", "nmf", "e5_kmeans"]:
        for k in K_GRID:
            fits = [fit_once(algo, k, SEED + s) for s in range(SEEDS)]
            evaluate(f"{algo}_k{k}", fits[0][0], fits[0][1], S, BOW_VIEW, algo=algo, stability=match_stability([f[0] for f in fits]))
            MODELS[f"{algo}_k{k}"] = fits[0][2]
''', r'''
import umap
with stage("bertopic_sampel", len(S)):
    reducers = [umap.UMAP(n_neighbors=15, n_components=5, min_dist=0., metric="cosine", random_state=SEED + s, low_memory=True).fit(E5[S]) for s in range(SEEDS)]
    for size in [len(S) // 400, len(S) // 200, len(S) // 100]:
        fits = [HDBSCAN(min_cluster_size=size, min_samples=10).fit_predict(r.embedding_) for r in reducers]
        topics = [ctfidf(f, BOW_VIEW, S) for f in fits]
        evaluate(f"bertopic_mcs{size}", topics[0], fits[0], S, BOW_VIEW, algo="bertopic", stability=match_stability(topics), min_cluster_size=size)
        MODELS[f"bertopic_mcs{size}"] = fits[0]
rand = np.random.default_rng(SEED)
evaluate("acak_k10", {i: list(rand.choice(names_t, 25, replace=False)) for i in range(10)}, rand.integers(0, 10, len(S)), S, BOW_VIEW, algo="acak", stability=0.)
board = pd.DataFrame(rows)
show("Grid · tanpa audit", board.drop(columns=[c for c in board if c.startswith("audit_")]).sort_values("TQ", ascending=False).head(12))
''')

part('08 · Pemilihan tanpa label & C_V', 'Larangan topik raksasa dan outlier berlebihan, lalu rata-rata peringkat TQ dan stabilitas. C_V (gensim) dihitung di sampel untuk 5 kandidat teratas, sebagai laporan saja.', r'''
pool = board[(board.algo != "acak") & (board["max_share_%"] <= 100 * MAX_SHARE) & (board["outlier_%"] <= 100 * MAX_OUTLIER)].copy()
pool = pool if len(pool) else board[board.algo != "acak"].copy()
pool["skor_gabungan"] = pool[["TQ", "stability"]].rank(pct=True).mean(axis=1)
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
with stage("c_v", len(S)):
    tokens = U[BOW_VIEW].iloc[S].str.split().tolist(); dictionary = Dictionary(tokens)
    def c_v(name):
        topics = [[w for w in ws[:TOP] if w in dictionary.token2id] for ws in STORE[name]["topics"].values()]
        return CoherenceModel(topics=[t for t in topics if len(t) >= 2], texts=tokens, dictionary=dictionary, coherence="c_v", processes=1).get_coherence()
    top5 = pool.sort_values("skor_gabungan", ascending=False).head(5).copy(); top5["C_V"] = [c_v(m) for m in top5.model]
show("5 kandidat teratas", top5[["model", "algo", "k", "TQ", "NPMI", "diversity", "stability", "C_V", "max_share_%", "outlier_%", "skor_gabungan"]])
FINAL_BASE = top5.iloc[0].model; ALGO = top5.iloc[0].algo; print("TERPILIH:", FINAL_BASE)
''')

part('09 · Terapkan ke semua teks unik & post-processing', '''Model dari sampel diterapkan ke semua teks: `transform` untuk LDA/NMF, `predict` untuk KMeans, dan centroid e5 untuk HDBSCAN. **Cek transfer:** di baris sampel, topik hasil penerapan dibandingkan dengan topik saat fit. Lalu post-processing seperti versi review:
- outlier ke topik terdekat
- gabung topik kembar (Jaccard kata kunci ≥ `MERGE_JACCARD`)
- gabung topik di bawah `MIN_TOPIC_SHARE`
- label ulang dengan c-TF-IDF dari semua anggota''', r'''
ALL = np.arange(len(U)); sample_labels = STORE[FINAL_BASE]["labels"]
def nearest_topic(labels_known, idx_known, idx_target):
    keep = [c for c in np.unique(labels_known) if c >= 0]
    centers = normalize(np.vstack([E5[idx_known][labels_known == c].mean(0) for c in keep]))
    return np.array(keep)[(E5[idx_target] @ centers.T).argmax(1)]
with stage("terapkan_penuh", len(U)):
    model = MODELS[FINAL_BASE]; text_all = U[BOW_VIEW]
    if ALGO == "nmf": Wm = model.transform(TFIDF.transform(text_all)); labels = np.where(Wm.max(1) > 0, Wm.argmax(1), -1)
    elif ALGO == "lda": labels = model.transform(COUNT.transform(text_all)).argmax(1)
    elif ALGO == "tfidf_kmeans": labels = model.predict(TFIDF.transform(text_all))
    elif ALGO == "e5_kmeans": labels = model.predict(E5)
    else:
        labels = np.full(len(U), -1); labels[S] = sample_labels
        rest = np.setdiff1d(ALL, S); labels[rest] = nearest_topic(sample_labels, S, rest)
transfer = adjusted_rand_score(sample_labels, labels[S])
print(f"Transfer sampel → penuh: ARI {transfer:.3f} di {len(S)} teks sampel")
''', r'''
with stage("post_processing", len(U)):
    steps = []
    if (labels < 0).any():
        bad = np.where(labels < 0)[0]; good = np.where(labels >= 0)[0]
        labels = labels.copy(); labels[bad] = nearest_topic(labels[good], good, bad); steps.append(f"{len(bad)} outlier")
    topics_now = ctfidf(labels, BOW_VIEW, ALL); merged = []
    for a, b in itertools.combinations(sorted(topics_now), 2):
        A, B = set(topics_now[a][:10]), set(topics_now[b][:10])
        if len(A & B) / len(A | B) >= MERGE_JACCARD: merged.append((a, b))
    for a, b in merged:
        small, large = sorted([a, b], key=lambda c: W[labels == c].sum()); labels[labels == small] = large
    if merged: steps.append(f"gabung {len(merged)} pasangan kembar")
    share = pd.Series(W).groupby(labels).sum() / W.sum(); tiny = share[share < MIN_TOPIC_SHARE].index
    if len(tiny):
        moving = np.where(np.isin(labels, tiny))[0]; staying = np.where(~np.isin(labels, tiny))[0]
        labels[moving] = nearest_topic(labels[staying], staying, moving); steps.append(f"gabung {len(tiny)} topik kecil")
    labels = pd.Series(labels).map({c: i for i, c in enumerate(sorted(np.unique(labels)))}).to_numpy()
    final_topics = ctfidf(labels, BOW_VIEW, ALL)
    evaluate("FINAL_penuh", final_topics, labels, ALL, BOW_VIEW, algo="final")
board = pd.DataFrame(rows)
show("Sebelum dan sesudah · tanpa audit", board[board.model.isin([FINAL_BASE, "FINAL_penuh"])].drop(columns=[c for c in board if c.startswith("audit_")]))
print("Langkah:", steps or "tidak ada")
''')

part('10 · Evaluasi & kelemahan', 'Kolom audit dibuka di sini kalau ada, tanpa mengubah keputusan di atas. Kemiripan antar topik dihitung di ruang e5 yang sudah dipusatkan. Topik raksasa yang kata kuncinya generik ditandai sebagai **topik sampah**.', r'''
audit_cols = [c for c in board if c.startswith("audit_")]
show("10.1 Semua kandidat", board.sort_values(audit_cols[0] if audit_cols else "TQ", ascending=False).head(20))
if AUDIT_COL:
    grid = board[board.algo.isin(["tfidf_kmeans", "lda", "nmf", "e5_kmeans", "bertopic"])]
    final_nmi = board.loc[board.model.eq("FINAL_penuh"), "audit_NMI"].iloc[0]
    print(f"FINAL_penuh: NMI {final_nmi:.3f} | terbaik di grid sampel: {grid.sort_values('audit_NMI').iloc[-1].model} NMI {grid.audit_NMI.max():.3f}")
    trust = pd.DataFrame({m: spearmanr(grid[m], grid.audit_NMI, nan_policy="omit") for m in ["TQ", "NPMI", "diversity", "stability", "max_share_%", "k"]}, index=["spearman", "p"]).T
    show("10.2 Kriteria tanpa label vs NMI audit", trust.sort_values("spearman", ascending=False))
    show("NMI terbaik per model", grid.groupby("algo").audit_NMI.max().sort_values(ascending=False))
''', r'''
matrix, vocab, docfreq = reference(BOW_VIEW); generic = {w for w, i in vocab.items() if docfreq[i] / matrix.shape[0] > .03}
order = sorted(final_topics); centers = normalize(np.vstack([E5C[labels == c].mean(0) for c in order]))
cosine = (E5C * centers[labels]).sum(1)
cards = []
for c in order:
    m = labels == c; w = W[m]; ws = final_topics[c]
    row = {"topik": c, "label": " / ".join(ws[:3]), "baris": int(w.sum()), "%_baris": 100 * w.sum() / W.sum(), "teks_unik": int(m.sum()),
           "kata_kunci": ", ".join(ws[:TOP]), "NPMI": npmi(ws[:TOP], BOW_VIEW), "kata_generik_%": 100 * np.mean([x in generic for x in ws[:TOP]]),
           "porsi_teks_terbanyak_%": 100 * w.max() / w.sum(), "kemiripan_ke_pusat": cosine[m].mean(),
           "contoh": " | ".join(U.text[m].iloc[np.argsort(-cosine[m])[:2]].str.slice(0, 60))}
    if DESC_COL: row[f"rata_{DESC_COL}"] = np.average(U.desc[m], weights=w)
    if AUDIT_COL:
        audit = pd.Series(w).groupby(U["_audit"].to_numpy()[m]).sum().sort_values(ascending=False)
        row.update(audit_mayoritas=audit.index[0], audit_purity=audit.iloc[0] / w.sum())
    cards.append(row)
cards = pd.DataFrame(cards)
show("10.3 Kartu topik · urut dari NPMI terendah", cards.sort_values("NPMI"))
flags = pd.DataFrame({"topik": cards.topik, "label": cards.label, "topik_sampah": (cards["%_baris"] > 100 * GIANT_GENERIC) & (cards["kata_generik_%"] >= 30),
    "NPMI_negatif": cards.NPMI < 0, "didominasi_satu_teks": cards["porsi_teks_terbanyak_%"] > 30, "kecil": cards["%_baris"] < 1})
show("10.4 Bendera kelemahan", flags[flags.drop(columns=["topik", "label"]).any(axis=1)])
''', r'''
sim = centers @ centers.T; np.fill_diagonal(sim, -1)
close = sorted([(a, b, sim[a, b]) for a in range(len(order)) for b in range(a + 1, len(order))], key=lambda x: -x[2])[:6]
show("10.5 Pasangan topik paling mirip · e5 dipusatkan", pd.DataFrame(close, columns=["topik_a", "topik_b", "cosine"]).assign(
    label_a=lambda f: f.topik_a.map(cards.set_index("topik").label), label_b=lambda f: f.topik_b.map(cards.set_index("topik").label)))
length = pd.cut(U[BOW_VIEW].str.split().str.len(), [-1, 0, 2, 5, 12, np.inf], labels=["kosong", "1-2", "3-5", "6-12", ">12"])
slices = pd.DataFrame({"baris_%": pd.Series(W).groupby(length.to_numpy(), observed=True).sum() / W.sum() * 100,
                       "kemiripan_ke_pusat": pd.Series(cosine).groupby(length.to_numpy(), observed=True).mean()})
if AUDIT_COL:
    truth = U["_audit"].to_numpy()
    slices["NMI_slice"] = [normalized_mutual_info_score(np.repeat(truth[length.to_numpy() == v], W[length.to_numpy() == v]),
                                                        np.repeat(labels[length.to_numpy() == v], W[length.to_numpy() == v])) for v in slices.index]
show("10.6 Kelemahan menurut panjang dokumen", slices)
show("10.7 Paling jauh dari pusat topik", U.assign(topik=labels, kemiripan=cosine).nsmallest(6, "kemiripan")[["text", "count", "topik", "kemiripan"]])
''')

part('11 · Keluaran & waktu', '`topics.csv` berisi topik dan label untuk setiap baris asli.', r'''
label_text = cards.set_index("topik").label
data["topik"] = data.key.map(dict(zip(U.key, labels))); assert data.topik.notna().all()
data[["row_id", TEXT_COL, "topik"]].assign(label=data.topik.map(label_text)).to_csv(OUT / "topics.csv", index=False)
board.to_csv(OUT / "kandidat.csv", index=False); cards.to_csv(OUT / "kartu_topik.csv", index=False)
times = pd.DataFrame(TIMES); times["menit"] = times.detik / 60; show("Waktu per tahap", times)
total = (time.perf_counter() - RUN_START) / 60
import importlib.metadata as metadata
manifest = {"dataset": DATASET, "rows": len(data), "unique": len(U), "n_fit": len(S), "bow_view": BOW_VIEW, "base": FINAL_BASE,
    "k_final": len(final_topics), "transfer_ARI": transfer, "post": steps, "minutes": total, "times": TIMES,
    "final": {k: v for k, v in board.loc[board.model.eq("FINAL_penuh")].iloc[0].items() if isinstance(v, (int, float, np.floating))},
    "gpu": torch.cuda.get_device_name(0), "versions": {p: metadata.version(p) for p in ["torch", "transformers", "scikit-learn", "umap-learn", "gensim"]}}
(OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
print(f"Total {total:.1f} menit | batas {BUDGET_MINUTES} menit"); assert total < BUDGET_MINUTES
''')

md(r'''
## Batas interpretasi
- Pilihan dibuat di sampel. Bagian 09 mengukur seberapa sama hasil sampel dan hasil penerapan penuh.
- Kategori berita bukan topik. NMI audit hanya sinyal kasar: topik koheren yang lintas kategori, misalnya COVID, justru menurunkan NMI.
- Stopword dan stopword domain bergantung data, jadi periksa tabel di bagian 04.
''')

for cell in CELLS:
    if cell.cell_type == 'code': ast.parse(cell.source)
next(c for c in CELLS if c.cell_type == 'code').metadata['tags'] = ['parameters']
notebook = nb.v4.new_notebook(cells=CELLS, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}, "pilot": {"task": "topic_besar", "preprocessing_source_sha256": hashlib.sha256(u.encode()).hexdigest()}})
nb.validate(notebook); nb.write(notebook, DEST)
kernel = Path(tempfile.gettempdir()) / 'gemastik_topic_besar_kernel'; kernel.mkdir(exist_ok=True)
payload = base64.b64encode(nb.writes(notebook).encode()).decode()
runner = f'''import base64, pathlib, os, traceback
import papermill as pm
os.environ['PYTHONHASHSEED'] = '42'
source = pathlib.Path('/kaggle/working/_source.ipynb'); source.write_bytes(base64.b64decode({payload!r}))
for dataset, name in [('news', '01_topic_modeling_besar.ipynb'), ('gojek', '01_topic_modeling_besar_gojek.ipynb')]:
    try:
        pm.execute_notebook(str(source), '/kaggle/working/' + name, parameters={{'DATASET': dataset}},
                            kernel_name='python3', cwd='/kaggle/working', progress_bar=False, log_output=True, execution_timeout=3000)
    except Exception:
        traceback.print_exc()
source.unlink(missing_ok=True)
'''
nb.write(nb.v4.new_notebook(cells=[nb.v4.new_code_cell(runner)], metadata=notebook.metadata), kernel / 'topicbesar.ipynb')
meta = {"id": "ivanwllm/gemastik-topic-besar", "title": "Gemastik Topic Besar", "code_file": "topicbesar.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4", "enable_internet": True,
        "dataset_sources": ["ibamibrahim/indonesian-news-title", "ucupsedaya/gojek-app-reviews-bahasa-indonesia"],
        "kernel_sources": ["ivanwllm/gemastik-nlp-00-setup-offline"], "competition_sources": [], "model_sources": []}
(kernel / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
print('Wrote', DEST, '| cells', len(CELLS), '| code lines', sum(len(c.source.splitlines()) for c in CELLS if c.cell_type == 'code'))
