"""Author clustering notebook untuk data ratusan ribu baris; eksekusi hanya di Kaggle."""
from pathlib import Path
from textwrap import dedent
import ast, base64, json, tempfile, hashlib
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'a_tanpa_label/03_clustering/03_clustering_besar.ipynb'
UNIVERSAL = ROOT / 'preprocessing_universal/stepwise.py'
CELLS = []


def md(s): CELLS.append(nb.v4.new_markdown_cell(dedent(s).strip()))
def code(s): CELLS.append(nb.v4.new_code_cell(dedent(s).strip()))
def part(title, note, *sources):
    md('## ' + title + '\n\n' + note)
    for source in sources: code(source)


md(r'''
# Clustering · data besar (ratusan ribu baris)
**Versi skala besar dari `03_clustering_review`. Target: Run All di Kaggle T4 di bawah 2 jam.**

Data contoh: [ulasan aplikasi Gojek](https://www.kaggle.com/datasets/ucupsedaya/gojek-app-reviews-bahasa-indonesia), 225 ribu baris bahasa Indonesia informal, tanpa label topik.

**Yang diubah dibanding versi 5 ribu baris supaya muat waktu**

| Masalah di data besar | Solusi di notebook ini |
|---|---|
| Banyak teks identik ("mantap", "bagus") | Semua model jalan di **teks unik**, lalu hasil dipetakan kembali ke setiap baris, dengan jumlah kemunculan sebagai bobot |
| Grid k × representasi × bootstrap mahal | Pemilihan dilakukan di **sampel `N_FIT` teks unik**. Fit final sekali di semua teks unik |
| Agglomerative dan konsensus butuh memori n² | Hanya dijalankan di sub-sampel `N_PAIRWISE` |
| Embedding pretrained paling lama | Hanya di teks unik, diurutkan menurut panjang. IndoBERT hanya di sampel, karena cuma pembanding |
| Stemming, Word2Vec, LDA lambat | Stemming dicache per kata unik, Word2Vec memakai 4 worker dan 5 epoch, LDA mode online di sampel |

**Aturan pemilihan** sama dengan `03_clustering_review`: larangan cluster raksasa, lalu rata-rata peringkat silhouette di ruang e5 dan stabilitas. Semua pilihan dibuat tanpa label. Sampel yang dipakai memilih dicek ulang: apakah cluster dari sampel sama dengan cluster dari fit penuh (bagian 09). Waktu tiap tahap dicatat di bagian 11.
''')

part('00 · Konfigurasi', 'Parameter papermill. Untuk data lain, tambah preset. `desc` adalah kolom angka opsional yang hanya dipakai untuk mendeskripsikan cluster, misalnya rating. Kolom ini tidak dipakai memilih.', r'''
DATASET = "gojek"
SEED, TOP = 42, 12
N_ROWS = None                 # None = semua baris
N_FIT, N_PAIRWISE = 30000, 10000
K_GRID = [6, 8, 10, 15, 20, 25]
BOOTSTRAP, SUBSAMPLE = 3, .8
MAX_SHARE = .40
MIN_DF, DF_MAX, MIN_CLUSTER_SHARE = 5, .25, .005
MAX_LEN, BATCH = 128, 256
BUDGET_MINUTES = 110
PRESETS = {"gojek": dict(file="GojekAppReviewV4.0.0-V4.9.3_Cleaned.csv", text="content", desc="score"),
           "mbg": dict(file="comment_sentiment.csv", text="Comment", desc="Likes")}
''', r'''
from pathlib import Path
FILE, TEXT_COL, DESC_COL = PRESETS[DATASET]["file"], PRESETS[DATASET]["text"], PRESETS[DATASET]["desc"]
OUT = Path(f"/kaggle/working/hasil_clustering_besar_{DATASET}")
print("dataset:", DATASET, "| teks:", TEXT_COL, "| deskripsi:", DESC_COL)
''')

part('01 · Lingkungan Kaggle & pencatat waktu', '`stage` mencatat durasi setiap tahap. Ringkasannya ada di bagian 11, dan bisa dipakai memperkirakan waktu kalau data bertambah.', r'''
import sys, os, subprocess
assert Path("/kaggle/input").exists(), "Jalankan notebook ini di Kaggle."
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "umap-learn>=0.5,<0.6", "gensim>=4.4,<5", "ftfy>=6.3,<7",
                "emoji>=2.14,<3", "datasketch>=1.6,<3", "langid==1.1.6", "Sastrawi==1.0.1"], check=True)
os.environ.update(TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_PROGRESS_BARS="1", USE_TF="0", USE_FLAX="0")
''', r'''
import json, time, random, re, html, unicodedata, itertools, hashlib, warnings
from collections import Counter
from contextlib import contextmanager
import numpy as np, pandas as pd, torch, emoji, ftfy
from IPython.display import display
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, HDBSCAN
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
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

part('02 · Muat data & dedupe', 'Audit kualitas dilakukan pada semua baris. Setelah itu baris dikelompokkan menurut teks yang sama (casefold dan spasi dirapikan). Semua tahap berikutnya bekerja di tabel `U` berisi teks unik, dengan `count` sebagai jumlah barisnya.', r'''
with stage("muat_data"):
    path = next(INPUT.rglob(FILE)); raw = pd.read_csv(path, low_memory=False)
assert TEXT_COL in raw, f"Kolom tersedia: {raw.columns.tolist()}"
text = raw[TEXT_COL]
quality = pd.DataFrame({"teks_null": text.isna(), "teks_non_string": text.notna() & ~text.map(lambda x: isinstance(x, str)),
    "teks_kosong": text.fillna("").astype(str).str.strip().eq(""),
    "teks_pseudo_null": text.fillna("").astype(str).str.strip().str.casefold().isin(["nan", "null", "none", "n/a", "-"])})
show("Skema", pd.DataFrame({"tipe": raw.dtypes.astype(str), "missing_%": raw.isna().mean() * 100, "unik": raw.nunique()}))
show("Kualitas input · flag dapat overlap", pd.DataFrame({"n": quality.sum(), "%": quality.mean() * 100}))
''', r'''
data = raw.loc[~quality.any(axis=1)].reset_index(names="row_id")
if N_ROWS and len(data) > N_ROWS: data = data.sample(N_ROWS, random_state=SEED).reset_index(drop=True)
data["key"] = data[TEXT_COL].astype(str).str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
agg = {"text": (TEXT_COL, "first"), "count": ("key", "size")}
if DESC_COL: agg["desc"] = (DESC_COL, "mean")
U = data.groupby("key", sort=False).agg(**agg).reset_index()
show("Ukuran setelah dedupe", pd.Series({"baris_valid": len(data), "teks_unik": len(U), "hemat_%": 100 * (1 - len(U) / len(data)),
    "baris_di_teks_berulang_%": 100 * U["count"][U["count"] > 1].sum() / len(data), "teks_terbanyak_n": U["count"].max()}))
show("Teks paling sering · ini yang akan menggelembungkan cluster", U.nlargest(TOP, "count")[["text", "count"]])
''')

part('03 · EDA wajib clustering · angka', 'Sama seperti versi 5 ribu baris. Statistik murah dihitung pada semua baris. Near-duplicate dan deteksi bahasa memakai sampel, dan ukuran sampelnya tertulis di judul tabel.', r'''
with stage("eda", len(data)):
    t = data[TEXT_COL].astype(str); words = t.str.split().str.len()
    show("3.1 Panjang & teks pendek · semua baris", pd.DataFrame({"karakter": t.str.len(), "kata": words}).describe(percentiles=[.05, .25, .5, .75, .95, .99]).T)
    show("Porsi pendek", pd.Series({"<3_kata_%": 100 * words.lt(3).mean(), "<5_kata_%": 100 * words.lt(5).mean(), ">100_kata_%": 100 * words.gt(100).mean()}))
    seqs = U.text.str.casefold().str.findall(r"[^\W\d_]+")
    counts = pd.Series(Counter(w for s in seqs for w in s), dtype="int64").sort_values(ascending=False)
    docfreq = pd.Series(Counter(w for s in seqs for w in set(s)), dtype="int64").reindex(counts.index) / len(seqs)
    show("3.2 Kosakata · teks unik", pd.Series({"token": counts.sum(), "tipe_unik": len(counts), "TTR_%": 100 * len(counts) / counts.sum(),
        "hapax_%_tipe": 100 * counts.eq(1).mean(), "tipe_df<MIN_DF_%": 100 * (docfreq * len(seqs)).lt(MIN_DF).mean(),
        "kata_df>DF_MAX": int(docfreq.gt(DF_MAX).sum()), "teks_tanpa_kata": int(seqs.str.len().eq(0).sum())}))
    show("Top words · % teks unik", pd.DataFrame({"n": counts.head(TOP), "%_dokumen": 100 * docfreq.head(TOP)}))
    show("Top bigram", pd.Series(Counter(" ".join(s[i:i + 2]) for s in seqs for i in range(len(s) - 1)), dtype="int64").nlargest(TOP))
''', r'''
NOISE = {"url": r"https?://|www\.", "html": r"</?[a-zA-Z][^>]*>", "mention": r"@\w+", "hashtag": r"#\w+",
    "emoji": r"[\U0001F300-\U0001FAFF☀-➿]", "angka": r"\d", "huruf_ulang": r"([^\W\d_])\1{2,}",
    "tanda_baca_ulang": r"[!?.,]{2,}", "kapital_semua": r"\b[A-Z]{3,}\b", "non_ASCII": r"[^\x00-\x7f]",
    "encoding_rusak": r"�|Ã.|Â.", "slang_contoh": r"\b(?:gk|gak|ga|bgt|yg|utk|udh|klo|jg|tdk)\b", "tawa": r"\b(?:wk){2,}\w*|\b(?:ha){2,}\b"}
with stage("eda_noise", len(U)):
    flags = pd.DataFrame({n: U.text.str.contains(p, flags=0 if n == "kapital_semua" else re.I) for n, p in NOISE.items()})
    show("3.3 Variasi noise · % baris (berbobot count)", pd.DataFrame({"%": flags.mul(U["count"], axis=0).sum() / len(data) * 100}).sort_values("%", ascending=False))
''', r'''
from datasketch import MinHash, MinHashLSH
from langid.langid import LanguageIdentifier, model as language_model
with stage("eda_sampel", 20000):
    probe = U.key.sample(min(20000, len(U)), random_state=SEED).tolist()
    index, grams_of, linked = MinHashLSH(threshold=.9, num_perm=64), {}, set()
    for i, k in enumerate(probe):
        if len(k) < 20: continue
        grams = {k[j:j + 5] for j in range(len(k) - 4)}; sig = MinHash(num_perm=64, seed=SEED); sig.update_batch([g.encode() for g in sorted(grams)])
        for o in index.query(sig):
            if len(grams & grams_of[o]) / len(grams | grams_of[o]) >= .9: linked |= {o, str(i)}
        index.insert(str(i), sig); grams_of[str(i)] = grams
    print(f"3.4 Near-duplicate di sampel 20.000 teks unik: {len(linked)} teks ({100 * len(linked) / len(probe):.1f}%)")
    identifier = LanguageIdentifier.from_modelstring(language_model, norm_probs=True)
    language = U.text.sample(min(1000, len(U)), random_state=SEED).map(lambda s: identifier.classify(s[:2000]))
    show("3.5 Estimasi bahasa · sampel 1.000, yakin >= 0.8", distribution(language.map(lambda p: p[0] if p[1] >= .8 else "tidak_pasti")).head(8))
if DESC_COL: show(f"Kolom deskripsi {DESC_COL} · hanya untuk membaca cluster", distribution(data[DESC_COL]).head(10))
decisions = pd.DataFrame([
    {"temuan": "teks unik / baris", "angka": f"{len(U) / len(data):.1%}", "keputusan": "model di teks unik, bobot = count"},
    {"temuan": "baris < 3 kata", "angka": f"{words.lt(3).mean():.1%}", "keputusan": "teks sangat pendek, embedding diuji; cluster pendek dicek di 10"},
    {"temuan": f"kata df > {DF_MAX:.0%}", "angka": str(int(docfreq.gt(DF_MAX).sum())), "keputusan": "stopword domain untuk BoW"},
    {"temuan": "noise tertinggi", "angka": f"{flags.mul(U['count'], axis=0).sum().idxmax()}", "keputusan": "safe cleaning + placeholder; normalisasi untuk BoW"},
    {"temuan": "hapax % tipe", "angka": f"{counts.eq(1).mean():.1%}", "keputusan": f"min_df={MIN_DF}; stemming diuji"}])
show("3.6 Temuan → keputusan", decisions)
''')

md('## 04 · Preprocessing dari universal\n\nFungsi berikut diambil dari `preprocessing_universal/stepwise.py`, dengan dua jenis input seperti versi 5 ribu baris. Semua langkah memakai cache per teks unik, dan stemming memakai cache per kata unik.')
u = UNIVERSAL.read_text(encoding='utf-8'); lines = u.splitlines(); tree = ast.parse(u)
code(u.split('# %% resources\n', 1)[1].split('# %% validate', 1)[0])
functions = {node.name: '\n'.join(lines[node.lineno - 1:node.end_lineno]) for node in tree.body if isinstance(node, ast.FunctionDef)}
for names in [['valid_text', 'repair_encoding', 'normalize_unicode', 'remove_html', 'decode_html_entities', 'remove_invisible', 'normalize_whitespace'],
              ['replace_urls', 'replace_emails', 'replace_phones', 'replace_mentions', 'segment_hashtags', 'lowercase'],
              ['normalize_slang', 'normalize_elongation', 'normalize_laughter', 'normalize_reduplication'],
              ['remove_emoji', 'remove_emoticons', 'remove_numbers', 'remove_punctuation', 'reduce_repeated_words']]:
    code('\n\n'.join(functions[name] for name in names))
start = [n.lineno for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_stemmer'][0] - 2
end = [n.end_lineno for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'stem_text'][0]
code('from functools import lru_cache\n' + '\n'.join(lines[start:end]))
md('**4.1 Versi teks.** Versinya sama dengan versi 5 ribu baris: `embed` untuk transformer, lalu `bow` dan `bow_stem` untuk TF-IDF, LDA, NMF, dan Word2Vec. `bow_stem` hanya dibuat kalau ablasi 6.1 memilihnya. Untuk ablasi, stemming dihitung di sampel dulu.')
code(r'''
def stem_series(series):
    """Stem hanya kata yang lolos MIN_DF, 4 proses paralel. Kata langka tetap, toh dibuang TF-IDF."""
    # forkserver: proses anak tidak mewarisi thread CUDA/torch, yang bisa membuat fork macet.
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
    return series.map(lambda t: " ".join(stems.get(w, w) for w in t.split()))
def chain(text, steps):
    for step in steps: text = step(text)
    return text
def apply(series, steps):
    cache = {s: chain(s, steps) for s in series.unique()}; return series.map(cache)
SAFE = [valid_text, repair_encoding, normalize_unicode, lambda s: remove_html(s) if "<" in s else s,
        decode_html_entities, remove_invisible, normalize_whitespace]
PLACEHOLDER = [replace_urls, replace_emails, replace_phones, replace_mentions, normalize_whitespace]
NORMAL = [segment_hashtags, lowercase, normalize_slang, normalize_elongation, normalize_laughter, normalize_reduplication]
CLEAN = [remove_emoji, remove_emoticons, remove_numbers, remove_punctuation, reduce_repeated_words, normalize_whitespace]
STOP = STOPWORDS | set("""akan ada adalah agar aja aku apa atau bahwa banyak baru bila bisa buat bukan cuma dah dalam dapat demi dgn
    dong gitu harus hal hanya ini itu jadi jika juga kalau kami kamu karena kata ke kenapa kok kini lagi lah lain lalu lebih
    loh maka masih mau mereka meski mungkin nah namun nanti nya oleh pada para pernah pun saat saja sama sampai sangat
    sebagai sedang sejak sekarang semua seperti setelah sih sudah supaya tapi telah tentang terhadap tersebut tetapi
    untuk usai yaitu yang ya url user email phone wkwk haha""".split())
with stage("preprocessing", len(U)):
    U["safe"] = apply(U.text, SAFE); U["embed"] = apply(U.safe, PLACEHOLDER); U["normal"] = apply(U.embed, NORMAL)
    bow = apply(U.normal, CLEAN).str.split()
    df_bow = pd.Series(Counter(w for s in bow for w in set(s))) / len(bow)
    DOMAIN_STOP = set(df_bow[df_bow > DF_MAX].index)
    U["bow"] = bow.map(lambda s: " ".join(w for w in s if w not in STOP | DOMAIN_STOP and len(w) > 1))
show(f"Stopword domain · df > {DF_MAX:.0%}", df_bow[df_bow > DF_MAX].sort_values(ascending=False).rename("df"))
rng = np.random.default_rng(SEED); S = np.sort(rng.choice(len(U), min(N_FIT, len(U)), replace=False))
with stage("stemming_sampel", len(S)):
    U["bow_stem"] = None; U.loc[S, "bow_stem"] = stem_series(U.bow.iloc[S]).to_numpy()
show("Audit preprocessing · teks unik", pd.DataFrame({v: {"kosong_%": 100 * U[v].iloc[S].astype(str).str.strip().eq("").mean(),
    "kata_median": U[v].iloc[S].astype(str).str.split().str.len().median()} for v in ["safe", "embed", "normal", "bow", "bow_stem"]}).T)
show("Before/after", U.iloc[S][["text", "embed", "bow", "bow_stem"]].sample(5, random_state=SEED))
''')

part('05 · Fungsi evaluasi tanpa label', 'Kriterianya sama dengan versi 5 ribu baris: stabilitas ARI, NPMI, diversity, dan silhouette. Bedanya, ukuran cluster dihitung berbobot `count`, supaya cluster berisi ribuan ulasan "mantap" yang identik terlihat besar, sebagaimana kenyataannya. NPMI memakai korpus `bow` dari semua teks unik.', r'''
with stage("referensi_npmi", len(U)):
    reference = CountVectorizer(binary=True, min_df=MIN_DF, token_pattern=r"\S+").fit(U.bow)
    DOC = reference.transform(U.bow).tocsc(); VOCAB_REF = reference.vocabulary_; N_DOC = DOC.shape[0]
    DF_REF = np.asarray(DOC.sum(0)).ravel()

def keywords(labels, idx, n=10):
    texts = U.bow.to_numpy()[idx]; clusters = [c for c in np.unique(labels) if c >= 0]
    joined = [" ".join(texts[labels == c]) for c in clusters]
    vectorizer = CountVectorizer(token_pattern=r"\S+").fit(joined); tf = vectorizer.transform(joined).toarray().astype(float)
    weight = tf * np.log1p(tf.sum() / len(clusters) / np.maximum(tf.sum(0), 1)); names = vectorizer.get_feature_names_out()
    return {c: list(names[np.argsort(-weight[i])[:n]]) for i, c in enumerate(clusters)}

def npmi(words):
    ids = [VOCAB_REF[w] for w in words if w in VOCAB_REF]; scores = []
    for a, b in itertools.combinations(ids, 2):
        both = DOC[:, a].multiply(DOC[:, b]).sum() / N_DOC
        scores.append(-1. if both == 0 else np.log(both / (DF_REF[a] * DF_REF[b] / N_DOC ** 2)) / -np.log(both))
    return float(np.mean(scores)) if scores else np.nan

def stability(X, k, return_runs=False):
    r, runs = np.random.default_rng(SEED), []
    for b in range(BOOTSTRAP):
        sub = r.choice(X.shape[0], int(SUBSAMPLE * X.shape[0]), replace=False)
        runs.append(KMeans(k, n_init=1, random_state=SEED + b).fit(X[sub]).predict(X))
    score = float(np.mean([adjusted_rand_score(a, b) for a, b in itertools.combinations(runs, 2)]))
    return (score, runs) if return_runs else score

rows = []
def evaluate(name, labels, idx, X=None, **extra):
    labels = np.asarray(labels); valid = labels >= 0; weight = U["count"].to_numpy()[idx]
    share = pd.Series(weight[valid]).groupby(labels[valid]).sum() / weight.sum()
    keys = keywords(labels, idx); top = [w for ws in keys.values() for w in ws]
    row = {"model": name, **extra, "n": len(idx), "k": len(share), "noise_%": 100 * (~valid).mean(), "max_share_%": 100 * share.max(),
           "min_share_%": 100 * share.min(), "NPMI": np.nanmean([npmi(ws) for ws in keys.values()]), "diversity": len(set(top)) / max(len(top), 1)}
    pick = np.random.default_rng(SEED).choice(np.where(valid)[0], min(3000, int(valid.sum())), replace=False)
    if X is not None and len(np.unique(labels[pick])) > 1:
        row["silhouette_own"] = silhouette_score(X[pick], labels[pick], metric="cosine")
        if "E5_S" in globals() and np.isin(idx, S).all():
            row["silhouette_e5"] = silhouette_score(E5_S[np.searchsorted(S, idx[pick])], labels[pick], metric="cosine")
        if not hasattr(X, "tocsr"): row["DB_own"] = davies_bouldin_score(X[pick], labels[pick])
    rows.append(row); return row
print("NPMI referensi:", len(VOCAB_REF), "kata dari", N_DOC, "teks unik")
''')

part('06 · Representasi', 'TF-IDF, LSA, dan Word2Vec dihitung untuk semua teks unik, karena murah. IndoBERT dan e5 dihitung di sampel `S` dulu. Representasi transformer yang terpilih baru di-embed penuh di bagian 09.', r'''
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

TRANSFORMERS = {"indobert_mean": (lambda: locate("indobert", "indobenchmark/indobert-base-p1"), lambda f: f.embed.str.lower()),
                "e5": (lambda: locate("e5", "intfloat/multilingual-e5-base"), lambda f: "query: " + f.embed)}
def tfidf_fit(texts):
    return TfidfVectorizer(min_df=MIN_DF, max_df=.5, ngram_range=(1, 2), max_features=50000, sublinear_tf=True).fit(texts)
''')
md('**6.1 Ablasi preprocessing BoW di sampel `S`.** e5 di sampel dihitung lebih dulu, karena dipakai aturan pemilihan. Versi dipilih dengan aturan yang sama seperti bagian 07.')
code(r'''
with stage("embed_sampel_e5", len(S)):
    E5_S = embed(TRANSFORMERS["e5"][0](), TRANSFORMERS["e5"][1](U.iloc[S]).tolist())
K_REF = K_GRID[len(K_GRID) // 2]; ablation = []
with stage("ablasi_bow", len(S)):
    for view in ["normal", "bow", "bow_stem"]:
        texts = U[view].iloc[S].astype(str); X = tfidf_fit(texts).transform(texts)
        labels = KMeans(K_REF, n_init=3, random_state=SEED).fit_predict(X)
        ablation.append({"view": view, "vocab": X.shape[1], "stability_ARI": stability(X, K_REF), **evaluate(f"ablasi_{view}", labels, S, X, algo="ablasi")})
rows = [r for r in rows if r.get("algo") != "ablasi"]
ablation = pd.DataFrame(ablation).drop(columns=["model", "algo"])
show(f"Ablasi preprocessing · TF-IDF KMeans k={K_REF}, sampel {len(S)}", ablation)
pool = ablation[ablation["max_share_%"] <= 100 * MAX_SHARE].copy()
pool = pool if len(pool) else ablation.copy()
pool["skor_gabungan"] = pool[["silhouette_e5", "stability_ARI"]].rank(pct=True).mean(axis=1)
BOW_VIEW = pool.sort_values(["skor_gabungan", "NPMI"]).iloc[-1]["view"]
if BOW_VIEW == "bow_stem":
    with stage("stemming_penuh", len(U)): U["bow_stem"] = stem_series(U.bow)
print("Versi BoW terpilih:", BOW_VIEW)
''')
md('**6.2 Bangun representasi.** `FULL` berisi matriks semua teks unik, dan `SAMPLE` berisi baris sampel.')
code(r'''
from gensim.models import Word2Vec
FULL, SAMPLE, rep_info = {}, {}, []
with stage("tfidf_lsa", len(U)):
    FULL["tfidf"] = tfidf_fit(U[BOW_VIEW]).transform(U[BOW_VIEW])
    FULL["lsa"] = normalize(TruncatedSVD(100, random_state=SEED).fit_transform(FULL["tfidf"]))
with stage("word2vec", len(U)):
    tokens = U[BOW_VIEW].str.split().tolist()
    w2v = Word2Vec(tokens, vector_size=100, window=5, min_count=MIN_DF, sg=1, epochs=5, workers=4, seed=SEED)
    FULL["word2vec"] = normalize(np.array([np.mean([w2v.wv[w] for w in s if w in w2v.wv], 0) if any(w in w2v.wv for w in s)
                                          else np.zeros(100) for s in tokens], dtype="float32"))
for name, X in FULL.items():
    SAMPLE[name] = X[S]; rep_info.append({"rep": name, "dibuat_untuk": "semua teks unik", "dimensi": X.shape[1],
        "vektor_nol_%": 100 * np.mean(np.asarray(abs(X).sum(1)).ravel() == 0)})
for name, (model_path, view) in TRANSFORMERS.items():
    if name == "e5":
        SAMPLE[name] = E5_S
    else:
        with stage(f"embed_sampel_{name}", len(S)): SAMPLE[name] = embed(model_path(), view(U.iloc[S]).tolist())
    rep_info.append({"rep": name, "dibuat_untuk": f"sampel {len(S)}", "dimensi": SAMPLE[name].shape[1], "vektor_nol_%": 0.})
show("Representasi", pd.DataFrame(rep_info))
''')

part('07 · Grid KMeans di sampel', 'Lima representasi × `K_GRID`, termasuk stabilitas bootstrap. Semuanya dikerjakan di sampel `S`, jadi waktunya tidak bergantung pada ukuran data.\n\n' + '**Aturan pemilihan tanpa label.** Cluster raksasa dilarang (`max_share_% <= MAX_SHARE`). Sisanya diurutkan dengan rata-rata peringkat **silhouette di ruang e5** dan **stabilitas**, lalu NPMI dipakai sebagai pemecah seri. Aturan ini berasal dari audit data berita di `03_clustering_review`: Spearman terhadap NMI adalah +0,61 untuk silhouette e5, +0,32 untuk stabilitas, +0,28 untuk NPMI, dan −0,78 untuk ukuran cluster terbesar. Aturan lama (stabilitas, lalu NPMI) memilih Word2Vec dengan NMI 0,116, sedangkan aturan ini memilih e5 dengan NMI 0,197. Karena dirancang dari satu dataset berlabel, aturan ini adalah prior, bukan jaminan untuk data lain.', r'''
with stage("grid_kmeans", len(S)):
    for name, X in SAMPLE.items():
        for k in K_GRID:
            labels = KMeans(k, n_init=3, random_state=SEED).fit_predict(X)
            evaluate(f"{name}_kmeans", labels, S, X, rep=name, algo="kmeans", stability_ARI=stability(X, k))
grid = pd.DataFrame(rows)
show("Grid KMeans · sampel", grid.sort_values("NPMI", ascending=False).head(15))
pool = grid[grid["max_share_%"] <= 100 * MAX_SHARE].copy()
pool = pool if len(pool) else grid.copy()
pool["skor_gabungan"] = pool[["silhouette_e5", "stability_ARI"]].rank(pct=True).mean(axis=1)
base = pool.sort_values(["skor_gabungan", "NPMI"]).iloc[-1]
REP_STAR, K_STAR = base.rep, int(base.k)
show("5 kandidat teratas menurut aturan", pool.sort_values("skor_gabungan", ascending=False)[["rep", "k", "silhouette_e5", "stability_ARI", "NPMI", "max_share_%", "skor_gabungan"]].head(5))
print(f"Lolos larangan cluster raksasa: {len(pool)} dari {len(grid)} | TERPILIH: {REP_STAR} k={K_STAR}")
show("NPMI & stabilitas terbaik per representasi", grid.groupby("rep")[["NPMI", "stability_ARI"]].max())
''')

part('08 · Algoritma lain di sampel', 'LDA, NMF, dan UMAP + HDBSCAN di sampel `S`. Agglomerative dan konsensus co-association butuh memori n², jadi dijalankan di sub-sampel `N_PAIRWISE`. Pembanding acak memakai ukuran cluster yang sama.', r'''
X_s = SAMPLE[REP_STAR]
with stage("algoritma_sampel", len(S)):
    sample_labels = KMeans(K_STAR, n_init=10, random_state=SEED).fit_predict(X_s)
    evaluate("kmeans_sampel_n10", sample_labels, S, X_s, rep=REP_STAR, algo="kmeans_n10")
    evaluate("acak_ukuran_sama", np.random.default_rng(SEED).permutation(sample_labels), S, X_s, rep=REP_STAR, algo="acak")
    counts_s = CountVectorizer(min_df=MIN_DF, max_df=.5, token_pattern=r"\S+").fit_transform(U[BOW_VIEW].iloc[S])
    lda = LatentDirichletAllocation(K_STAR, learning_method="online", batch_size=2048, max_iter=10, random_state=SEED).fit_transform(counts_s)
    evaluate("lda_argmax", lda.argmax(1), S, SAMPLE["tfidf"], rep="count", algo="lda")
    nmf = NMF(K_STAR, init="nndsvda", max_iter=300, random_state=SEED).fit_transform(SAMPLE["tfidf"])
    evaluate("nmf_argmax", nmf.argmax(1), S, SAMPLE["tfidf"], rep="tfidf", algo="nmf")
''', r'''
import umap
with stage("umap_hdbscan_sampel", len(S)):
    reducer = umap.UMAP(n_neighbors=30, n_components=10, min_dist=0., metric="cosine", random_state=SEED, low_memory=True)
    U_s = reducer.fit_transform(SAMPLE["e5"])
    hdb = HDBSCAN(min_cluster_size=max(30, len(S) // 200), min_samples=10).fit_predict(U_s)
    evaluate("e5_umap_hdbscan", hdb, S, SAMPLE["e5"], rep="e5_umap", algo="hdbscan")
P = np.sort(np.random.default_rng(SEED + 1).choice(len(S), min(N_PAIRWISE, len(S)), replace=False))
with stage("pairwise_subsampel", len(P)):
    dense_p = SAMPLE["lsa"][P] if hasattr(X_s, "tocsr") else X_s[P]
    evaluate("agglo_ward_subsampel", AgglomerativeClustering(K_STAR, linkage="ward").fit_predict(dense_p), S[P], dense_p, rep=REP_STAR, algo="agglomerative")
    runs = [KMeans(K_STAR, n_init=1, random_state=SEED + s).fit_predict(dense_p) for s in range(10)]
    onehot = np.hstack([np.eye(K_STAR, dtype="float32")[r] for r in runs]); coassoc = onehot @ onehot.T / len(runs); del onehot
    consensus = AgglomerativeClustering(K_STAR, metric="precomputed", linkage="average").fit_predict(1 - coassoc); del coassoc
    evaluate("konsensus_subsampel", consensus, S[P], dense_p, rep=REP_STAR, algo="konsensus")
    evaluate("kmeans_subsampel_pembanding", sample_labels[P], S[P], dense_p, rep=REP_STAR, algo="pembanding")
board = pd.DataFrame(rows)
show("Perbandingan algoritma · sampel", board[~board.algo.eq("kmeans")])
''')

part('09 · Fit final di semua teks unik', 'Representasi terpilih dihitung untuk semua teks unik, lalu KMeans k terpilih di-fit sekali. MiniBatchKMeans dipakai kalau teks unik lebih dari 300 ribu. **Cek transfer:** cluster dari fit penuh dibandingkan dengan cluster dari sampel, pada baris sampel. ARI tinggi berarti sampel mewakili data.', r'''
if REP_STAR in FULL: X_full = FULL[REP_STAR]
else:
    with stage(f"embed_penuh_{REP_STAR}", len(U)):
        model_path, view = TRANSFORMERS[REP_STAR]; X_full = embed(model_path(), view(U).tolist())
with stage("kmeans_penuh", len(U)):
    Model = MiniBatchKMeans if len(U) > 300000 else KMeans
    final_model = Model(K_STAR, n_init=3, random_state=SEED, **({"batch_size": 8192} if Model is MiniBatchKMeans else {})).fit(X_full)
    labels_full = final_model.labels_
print(f"Transfer sampel → penuh: ARI {adjusted_rand_score(sample_labels, labels_full[S]):.3f} (pada {len(S)} teks sampel)")
''', r'''
def center_cosine(X, labels, order):
    """Cosine tiap baris ke pusat clusternya. Aman untuk matriks sparse besar, tanpa toarray."""
    Xn = normalize(X); out = np.zeros(X.shape[0]); centers = []
    for c in order:
        m = labels == c; center = np.asarray(Xn[m].mean(0)).ravel(); center /= np.linalg.norm(center) or 1
        centers.append(center); out[m] = np.asarray(Xn[m] @ center).ravel()
    return out, np.vstack(centers)

with stage("post_processing", len(U)):
    weight = U["count"].to_numpy(); share = pd.Series(weight).groupby(labels_full).sum() / weight.sum()
    tiny = set(share[share < MIN_CLUSTER_SHARE].index)
    if tiny:
        keep = [c for c in share.index if c not in tiny]
        _, centers = center_cosine(X_full, labels_full, keep)
        mask = np.isin(labels_full, list(tiny)); labels_full = labels_full.copy()
        labels_full[mask] = np.array(keep)[np.asarray(normalize(X_full[mask]) @ centers.T).argmax(1)]
zero = np.asarray(abs(X_full).sum(1)).ravel() == 0
if zero.any():  # teks tanpa kata yang dikenal: diarahkan lewat e5, bukan ditumpuk ke satu cluster
    with stage("vektor_nol_lewat_e5", int(zero.sum())):
        in_sample = ~zero[S]; known = np.unique(labels_full[S][in_sample])
        e5_centers = normalize(np.vstack([E5_S[in_sample & (labels_full[S] == c)].mean(0) for c in known]))
        e5_zero = embed(TRANSFORMERS["e5"][0](), TRANSFORMERS["e5"][1](U.iloc[np.where(zero)[0]]).tolist())
        labels_full = labels_full.copy(); labels_full[zero] = known[(e5_zero @ e5_centers.T).argmax(1)]
print("Teks vektor nol diarahkan lewat e5:", int(zero.sum()), "teks unik,", int(U["count"].to_numpy()[zero].sum()), "baris")
ALL = np.arange(len(U))
evaluate("FINAL_penuh", labels_full, ALL, X_full, rep=REP_STAR, algo="final")
print("Cluster kecil digabung:", len(tiny), "| teks identik otomatis satu cluster karena model jalan di teks unik")
''')

part('10 · Evaluasi & kelemahan', 'Data ini tidak punya label topik, jadi kelemahan dibaca dari stabilitas per cluster, kata kunci, jarak ke pusat, porsi teks pendek, dan dominasi teks berulang. `desc` (rating) dipakai sebagai deskripsi, bukan kebenaran.', r'''
labels = labels_full
keys = keywords(labels, ALL); order = sorted(keys)
cosine, _ = center_cosine(X_full, labels, order)
_, boot = stability(X_s, len(order), return_runs=True)
sample_final = labels[S]
def cluster_stability(c):
    member = sample_final == c
    return np.mean([max((member & (r == j)).sum() / max((member | (r == j)).sum(), 1) for j in np.unique(r)) for r in boot]) if member.any() else np.nan
short = U.bow.str.split().str.len().le(2).to_numpy()
cards = []
for c in order:
    m = labels == c; w = U["count"].to_numpy()[m]
    row = {"cluster": c, "baris": int(w.sum()), "%_baris": 100 * w.sum() / len(data), "teks_unik": int(m.sum()),
           "porsi_teks_terbanyak_%": 100 * w.max() / w.sum(), "teks_<=2_kata_%": 100 * (w * short[m]).sum() / w.sum(),
           "kata_kunci": ", ".join(keys[c][:8]), "NPMI": npmi(keys[c]), "stabilitas_jaccard": cluster_stability(c),
           "kemiripan_ke_pusat": cosine[m].mean(), "contoh": " | ".join(U.text[m].iloc[np.argsort(-cosine[m])[:2]].str.slice(0, 60))}
    if DESC_COL: row[f"rata_{DESC_COL}"] = np.average(U.desc[m], weights=w)
    cards.append(row)
cards = pd.DataFrame(cards)
show("10.1 Kartu cluster · urut dari paling tidak stabil", cards.sort_values("stabilitas_jaccard"))
''', r'''
board = pd.DataFrame(rows)
show("10.2 Semua kandidat · sampel kecuali FINAL", board.drop(columns=[c for c in ["DB_own"] if c in board]).sort_values("NPMI", ascending=False))
flags_c = pd.DataFrame({"cluster": cards.cluster,
    "didominasi_satu_teks": cards["porsi_teks_terbanyak_%"] > 30, "mayoritas_teks_pendek": cards["teks_<=2_kata_%"] > 50,
    "tidak_stabil": cards.stabilitas_jaccard < .6, "raksasa": cards["%_baris"] > 100 * MAX_SHARE, "NPMI_rendah": cards.NPMI < board.NPMI.median()})
show("10.3 Bendera kelemahan per cluster", flags_c[flags_c.drop(columns="cluster").any(axis=1)])
length = pd.cut(U.bow.str.split().str.len(), [-1, 0, 2, 5, 15, np.inf], labels=["kosong", "1-2", "3-5", "6-15", ">15"])
show("10.4 Kemiripan ke pusat menurut panjang · rendah = cluster kurang yakin",
     pd.DataFrame({"teks_unik": length.value_counts(sort=False), "baris": pd.Series(U["count"].to_numpy()).groupby(length.to_numpy(), observed=True).sum(),
                   "kemiripan_ke_pusat": pd.Series(cosine).groupby(length.to_numpy(), observed=True).mean()}))
show("10.5 Paling jauh dari pusat · kandidat outlier atau salah tempat", U.assign(cluster=labels, kemiripan=cosine).nsmallest(8, "kemiripan")[["text", "count", "cluster", "kemiripan"]])
''')

part('11 · Keluaran & waktu', 'Label setiap baris asli disimpan di `clusters.csv`. Tabel waktu menunjukkan tahap mana yang ikut membesar kalau data bertambah, yaitu tahap yang memakai semua baris (`n` = semua), dan mana yang tetap karena hanya memakai sampel.', r'''
data["cluster"] = data.key.map(dict(zip(U.key, labels)))
assert data.cluster.notna().all()
data[["row_id", TEXT_COL, "cluster"]].assign(kata_kunci=data.cluster.map({c: ", ".join(keys[c][:5]) for c in order})).to_csv(OUT / "clusters.csv", index=False)
board.to_csv(OUT / "kandidat.csv", index=False); cards.to_csv(OUT / "kartu_cluster.csv", index=False)
times = pd.DataFrame(TIMES); times["menit"] = times.detik / 60
times["skala"] = np.where(times.n.fillna(0) >= len(U), "ikut_ukuran_data", "tetap_(sampel)")
show("Waktu per tahap", times)
total = (time.perf_counter() - RUN_START) / 60
grow = times.loc[times.skala.eq("ikut_ukuran_data"), "menit"].sum()
print(f"Total {total:.1f} menit | tahap yang ikut ukuran data {grow:.1f} menit | perkiraan kalau teks unik 2x: {total + grow:.1f} menit | batas {BUDGET_MINUTES} menit")
import importlib.metadata as metadata
manifest = {"dataset": DATASET, "rows": len(data), "unique": len(U), "n_fit": len(S), "bow_view": BOW_VIEW, "rep": REP_STAR, "k": K_STAR,
    "transfer_ARI": adjusted_rand_score(sample_labels, labels_full[S]), "minutes": total, "times": TIMES, "gpu": torch.cuda.get_device_name(0),
    "versions": {p: metadata.version(p) for p in ["torch", "transformers", "scikit-learn", "umap-learn", "gensim", "pandas"]}}
(OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
assert total < BUDGET_MINUTES, "Melebihi anggaran waktu"
''')

md(r'''
## Batas interpretasi
- Pilihan dibuat di sampel. Bagian 09 mengukur seberapa sama hasil sampel dan hasil penuh, tetapi bukan jaminan untuk semua dataset.
- Tanpa label topik, kualitas dinilai lewat kriteria tidak langsung. Di versi 5 ribu baris dengan audit, terlihat kriteria mana yang searah dengan kebenaran. Pakai tabel itu untuk memutuskan seberapa percaya pada NPMI dan stabilitas di sini.
- Rating (`desc`) hanya untuk membaca cluster, misalnya cluster keluhan cenderung rating rendah. Rating bukan label topik.
''')

for cell in CELLS:
    if cell.cell_type == 'code': ast.parse(cell.source)
next(c for c in CELLS if c.cell_type == 'code').metadata['tags'] = ['parameters']
notebook = nb.v4.new_notebook(cells=CELLS, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}, "pilot": {"task": "clustering_besar", "preprocessing_source_sha256": hashlib.sha256(u.encode()).hexdigest()}})
nb.validate(notebook); nb.write(notebook, DEST)
kernel = Path(tempfile.gettempdir()) / 'gemastik_clustering_besar_kernel'; kernel.mkdir(exist_ok=True)
payload = base64.b64encode(nb.writes(notebook).encode()).decode()
runner = f'''import base64, pathlib, os
import papermill as pm
os.environ['PYTHONHASHSEED'] = '42'
source = pathlib.Path('/kaggle/working/_source.ipynb'); source.write_bytes(base64.b64decode({payload!r}))
try:
    pm.execute_notebook(str(source), '/kaggle/working/03_clustering_besar.ipynb', kernel_name='python3', cwd='/kaggle/working', progress_bar=False, log_output=True)
finally:
    source.unlink(missing_ok=True)
'''
nb.write(nb.v4.new_notebook(cells=[nb.v4.new_code_cell(runner)], metadata=notebook.metadata), kernel / 'besar.ipynb')
meta = {"id": "ivanwllm/gemastik-clustering-besar", "title": "Gemastik Clustering Besar", "code_file": "besar.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4", "enable_internet": True,
        "dataset_sources": ["ucupsedaya/gojek-app-reviews-bahasa-indonesia", "dzikriraihan/indonesian-youtube-comments-mbg-program"],
        "kernel_sources": ["ivanwllm/gemastik-nlp-00-setup-offline"], "competition_sources": [], "model_sources": []}
(kernel / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
print('Wrote', DEST, '| cells', len(CELLS), '| code lines', sum(len(c.source.splitlines()) for c in CELLS if c.cell_type == 'code'))
