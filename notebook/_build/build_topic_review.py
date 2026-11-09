"""Author topic modeling review notebook; eksekusi hanya di Kaggle (papermill, dua dataset)."""
from pathlib import Path
from textwrap import dedent
import ast, base64, json, tempfile, hashlib
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'a_tanpa_label/01_topic_modeling/01_topic_modeling_review.ipynb'
UNIVERSAL = ROOT / 'preprocessing_universal/stepwise.py'
CELLS = []


def md(s): CELLS.append(nb.v4.new_markdown_cell(dedent(s).strip()))
def code(s): CELLS.append(nb.v4.new_code_cell(dedent(s).strip()))
def part(title, note, *sources):
    md('## ' + title + '\n\n' + note)
    for source in sources: code(source)


md(r'''
# Topic modeling · versi review
**Pilot satu kasus: EDA angka → preprocessing yang diuji ablasi → 5 model topik + baseline acak → post-processing → evaluasi kelemahan.**

Run All di **Kaggle GPU**. Semua angka berasal dari data asli, tanpa grafik otomatis.

| Model | Representasi | Kata kunci topik dari | Cara kerja |
|---|---|---|---|
| Acak | kata acak dari kosakata | — | batas bawah coherence |
| TF-IDF + KMeans | BoW agresif | c-TF-IDF | baseline klasik: satu dokumen satu topik |
| LDA | hitungan kata | distribusi kata topik | probabilistik, dokumen bercampur topik |
| NMF | TF-IDF | matriks H | faktorisasi, biasanya lebih tajam dari LDA untuk teks pendek |
| e5 + KMeans | embedding e5 frozen | c-TF-IDF | semantik, satu dokumen satu topik |
| e5 + UMAP + HDBSCAN (gaya BERTopic) | embedding e5 frozen | c-TF-IDF | jumlah topik otomatis, ada outlier |

**Kriteria tanpa label** (tidak memakai kategori):
- **NPMI coherence:** apakah kata kunci topik muncul bersama di dokumen.
- **Diversity:** porsi kata unik di top-25 semua topik, yang menunjukkan apakah topik saling berbeda.
- **Topic quality** (TQ) = NPMI ternormalisasi × diversity, mengikuti Dieng dkk. 2020.
- **Stabilitas:** kemiripan kata kunci antar 3 seed.

Ditambah larangan topik raksasa. C_V coherence (gensim) dihitung untuk kandidat teratas.

**Data:** `DATASET="news"` memakai judul berita, dengan kategori sebagai audit yang baru dibuka di bagian 10. `DATASET="mbg"` memakai komentar YouTube MBG tanpa label, yaitu kondisi seperti lomba.
''')

part('00 · Konfigurasi', 'Parameter papermill. Untuk data lain, tambah preset: nama file, kolom teks, dan kolom audit (`None` kalau tidak ada).', r'''
DATASET = "news"
SEED, N_ROWS, TOP = 42, 5000, 10
K_GRID = [5, 8, 10, 12, 15, 20]
SEEDS = 3                              # stabilitas: kata kunci dibandingkan antar 3 seed
MAX_SHARE, MAX_OUTLIER = .40, .30      # larangan topik raksasa dan outlier HDBSCAN berlebihan
MIN_DF, DF_MAX, MIN_TOPIC_SHARE, MERGE_JACCARD = 3, .25, .01, .30
MAX_LEN, BATCH = 128, 64
PRESETS = {"news": dict(file="indonesian-news-title.csv", text="title", audit="category"),
           "mbg": dict(file="comment_sentiment.csv", text="Comment", audit=None)}
''', r'''
from pathlib import Path
FILE, TEXT_COL, AUDIT_COL = PRESETS[DATASET]["file"], PRESETS[DATASET]["text"], PRESETS[DATASET]["audit"]
OUT = Path(f"/kaggle/working/hasil_topik_{DATASET}")
print("dataset:", DATASET, "| teks:", TEXT_COL, "| audit:", AUDIT_COL or "tidak ada")
''')

part('01 · Lingkungan Kaggle', 'Paket tambahan dipasang di Kaggle. Model e5 diambil dari paket offline kalau tersedia, dan dari Hugging Face kalau tidak. Backbone e5 di-freeze dan hanya dipakai untuk feature extraction.', r'''
import sys, os, subprocess
assert Path("/kaggle/input").exists(), "Jalankan notebook ini di Kaggle."
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "umap-learn>=0.5,<0.6", "gensim>=4.4,<5", "ftfy>=6.3,<7",
                "emoji>=2.14,<3", "datasketch>=1.6,<3", "langid==1.1.6", "Sastrawi==1.0.1"], check=True)
os.environ.update(TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_PROGRESS_BARS="1", USE_TF="0", USE_FLAX="0")
''', r'''
import json, time, random, re, html, unicodedata, itertools, hashlib, warnings
from collections import Counter
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
OUT.mkdir(exist_ok=True); INPUT = Path("/kaggle/input"); RUN_START = time.perf_counter()
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "tidak ada")

def show(title, table):
    print("\n" + title); display((table.to_frame() if isinstance(table, pd.Series) else table).round(3))
def distribution(values):
    n = values.value_counts(dropna=False); return pd.DataFrame({"n": n, "%": n / (len(values) or np.nan) * 100})
pd.set_option("display.max_columns", None, "display.max_colwidth", 110, "display.width", 200)
''')

part('02 · Muat data mentah & audit sebelum drop', 'Missing dan teks invalid diaudit sebelum dibuang. Kolom audit langsung dipisah ke `_audit`, dan baru dipakai di bagian 10.', r'''
path = next(INPUT.rglob(FILE)); raw = pd.read_csv(path)
assert TEXT_COL in raw, f"Kolom tersedia: {raw.columns.tolist()}"
show("Skema sebelum pembersihan", pd.DataFrame({"tipe": raw.dtypes.astype(str), "missing": raw.isna().sum(),
     "missing_%": raw.isna().mean() * 100, "unik_non_null": raw.nunique(dropna=True)}))
text = raw[TEXT_COL]
quality = pd.DataFrame({"teks_null": text.isna(), "teks_non_string": text.notna() & ~text.map(lambda x: isinstance(x, str)),
    "teks_kosong": text.fillna("").astype(str).str.strip().eq(""),
    "teks_pseudo_null": text.fillna("").astype(str).str.strip().str.casefold().isin(["nan", "null", "none", "n/a", "-"])})
if AUDIT_COL: quality["audit_null"] = raw[AUDIT_COL].isna()
show("Kualitas input · flag dapat overlap", pd.DataFrame({"n": quality.sum(), "%": quality.mean() * 100}))
data = raw.loc[~quality.any(axis=1)].reset_index(names="row_id")
if N_ROWS and len(data) > N_ROWS: data = data.sample(N_ROWS, random_state=SEED).reset_index(drop=True)
data["_audit"] = data[AUDIT_COL].astype(str) if AUDIT_COL else None
print("File:", path.name, "| baris mentah:", len(raw), "| invalid:", int(quality.any(axis=1).sum()), "| baris pilot:", len(data))
''')

part('03 · EDA wajib topic modeling · angka', 'Pertanyaan EDA untuk topic modeling: apakah dokumennya cukup panjang (LDA butuh banyak kata per dokumen), apakah kosakatanya cukup padat setelah pruning, kata apa yang ada di mana-mana (akan muncul di semua topik), frasa dua kata apa yang sering muncul (kandidat bigram), noise apa yang perlu dibuang, dan apakah teks identik atau template akan membentuk topik palsu. Setiap angka menghasilkan keputusan di 3.6.', r'''
t = data[TEXT_COL].astype(str)
key = t.str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
words = t.str.split().str.len()
show("3.1 Ukuran, duplikat, panjang dokumen", pd.Series({"dokumen": len(t), "duplikat_normal": key.duplicated().sum(),
    "duplikat_normal_%": 100 * key.duplicated().mean(), "grup_duplikat": key.value_counts().gt(1).sum(),
    "dok_<5_kata_%": 100 * words.lt(5).mean(), "dok_<10_kata_%": 100 * words.lt(10).mean(), "dok_>100_kata_%": 100 * words.gt(100).mean()}))
show("Panjang kata · LDA lemah di dokumen sangat pendek", words.describe(percentiles=[.05, .25, .5, .75, .95, .99]))
''', r'''
seqs = t.str.casefold().str.findall(r"[^\W\d_]+")
counts = pd.Series(Counter(w for s in seqs for w in s), dtype="int64").sort_values(ascending=False)
docfreq = pd.Series(Counter(w for s in seqs for w in set(s)), dtype="int64").reindex(counts.index) / len(seqs)
kept = docfreq[(docfreq * len(seqs) >= MIN_DF) & (docfreq <= DF_MAX)]
show("3.2 Kosakata & pruning", pd.Series({"token": counts.sum(), "tipe_unik": len(counts), "hapax_%_tipe": 100 * counts.eq(1).mean(),
    "hapax_%_token": 100 * counts.eq(1).sum() / counts.sum(), f"tipe_df>={MIN_DF}": int((docfreq * len(seqs) >= MIN_DF).sum()),
    f"kata_df>{DF_MAX:.0%}": int(docfreq.gt(DF_MAX).sum()), "kosakata_setelah_pruning": len(kept),
    "token_tersisa_%": 100 * counts.reindex(kept.index).sum() / counts.sum(),
    "dok_kosong_setelah_pruning_%": 100 * seqs.map(lambda s: not any(w in kept.index for w in s)).mean()}))
show("Top words · kandidat stopword umum/domain", pd.DataFrame({"n": counts.head(TOP + 5), "%_dokumen": 100 * docfreq.head(TOP + 5)}))
''', r'''
pair_counts = Counter(" ".join(s[i:i + 2]) for s in seqs for i in range(len(s) - 1))
total_tokens = counts.sum()
collocation = pd.DataFrame([{"bigram": b, "n": n, "PMI": np.log(n * total_tokens / (counts[b.split()[0]] * counts[b.split()[1]]))}
                            for b, n in pair_counts.items() if n >= 10])
show("3.3 Kolokasi · bigram sering dan PMI tinggi, kandidat frasa untuk topik", collocation.sort_values("PMI", ascending=False).head(TOP) if len(collocation) else collocation)
NOISE = {"url": r"https?://|www\.", "html": r"</?[a-zA-Z][^>]*>", "html_entity": r"&(?:#\d+|[a-zA-Z]+);", "mention": r"@\w+",
    "hashtag": r"#\w+", "emoji": r"[\U0001F300-\U0001FAFF☀-➿]", "angka": r"\d", "huruf_ulang": r"([^\W\d_])\1{2,}",
    "kapital_semua": r"\b[A-Z]{3,}\b", "non_ASCII": r"[^\x00-\x7f]", "slang_contoh": r"\b(?:gk|gak|ga|bgt|yg|utk|udh|klo|jg|tdk)\b",
    "tawa": r"\b(?:wk){2,}\w*|\b(?:ha){2,}\b"}
flags = pd.DataFrame({n: t.str.contains(p, flags=0 if n == "kapital_semua" else re.I) for n, p in NOISE.items()})
show("3.4 Variasi noise · % dokumen", pd.DataFrame({"n": flags.sum(), "%": flags.mean() * 100}).sort_values("n", ascending=False))
''', r'''
from langid.langid import LanguageIdentifier, model as language_model
identifier = LanguageIdentifier.from_modelstring(language_model, norm_probs=True)
language = t.sample(min(500, len(t)), random_state=SEED).map(lambda s: identifier.classify(s[:2000]))
show("3.5 Estimasi bahasa · sampel 500, yakin >= 0.8", distribution(language.map(lambda p: p[0] if p[1] >= .8 else "tidak_pasti")).head(8))
prefix = seqs.map(lambda s: " ".join(s[:3])); top_prefix = prefix[prefix.str.count(" ").ge(2)].value_counts().head(5)
show("Template · awalan 3 kata tersering", top_prefix.to_frame("n").assign(**{"%": 100 * top_prefix / len(t)}))
if AUDIT_COL: show("Kategori audit · hanya untuk penilaian", distribution(data["_audit"]))
decisions = pd.DataFrame([
    {"temuan": "dok < 10 kata", "angka": f"{words.lt(10).mean():.1%}", "keputusan": "LDA kemungkinan lemah; model satu-topik-per-dokumen (KMeans, HDBSCAN) ikut diuji" if words.lt(10).mean() > .5 else "LDA layak"},
    {"temuan": f"kata df > {DF_MAX:.0%}", "angka": str(int(docfreq.gt(DF_MAX).sum())), "keputusan": "dibuang sebagai stopword domain di BoW"},
    {"temuan": "token tersisa setelah pruning", "angka": f"{counts.reindex(kept.index).sum() / counts.sum():.1%}", "keputusan": "stopword dan kata langka dibuang; stemming diuji di 6.1"},
    {"temuan": "kolokasi PMI tinggi", "angka": str(int((collocation.PMI > 5).sum()) if len(collocation) else 0), "keputusan": "bigram (gensim Phrases) diuji di 6.1"},
    {"temuan": "duplikat normal", "angka": f"{key.duplicated().mean():.1%}", "keputusan": "duplikat bisa membentuk topik palsu; dicek di 10.4"},
    {"temuan": "noise tertinggi", "angka": f"{flags.mean().idxmax()} {flags.mean().max():.1%}", "keputusan": "safe cleaning + normalisasi untuk BoW"}])
show("3.6 Temuan → keputusan", decisions)
''')

md('## 04 · Preprocessing dari universal\n\nFungsi berikut diambil langsung dari `preprocessing_universal/stepwise.py`. Topic model berbasis kata (LDA, NMF, KMeans TF-IDF) butuh BoW agresif: kata umum, angka, dan tanda baca dibuang, karena kata seperti itu akan muncul di semua topik. Model embedding butuh kalimat utuh.')
u = UNIVERSAL.read_text(encoding='utf-8'); lines = u.splitlines(); tree = ast.parse(u)
code(u.split('# %% resources\n', 1)[1].split('# %% validate', 1)[0])
functions = {node.name: '\n'.join(lines[node.lineno - 1:node.end_lineno]) for node in tree.body if isinstance(node, ast.FunctionDef)}
for names in [['valid_text', 'repair_encoding', 'normalize_unicode', 'remove_html', 'decode_html_entities', 'remove_invisible', 'normalize_whitespace'],
              ['replace_urls', 'replace_emails', 'replace_phones', 'replace_mentions', 'segment_hashtags', 'lowercase'],
              ['normalize_slang', 'normalize_elongation', 'normalize_laughter', 'normalize_reduplication'],
              ['remove_emoji', 'remove_emoticons', 'remove_numbers', 'remove_punctuation', 'reduce_repeated_words']]:
    code('\n\n'.join(functions[name] for name in names))
md('**4.1 Versi teks.** `embed` dipakai e5. `normal` = lowercase dan normalisasi kata, tanpa pembuangan. `bow` = normal tanpa emoji, angka, tanda baca, stopword umum, dan stopword domain. `bow_stem` = bow + stemming Sastrawi. `bow_bigram` = bow dengan frasa dua kata dari gensim Phrases, misalnya `virus_corona`. Ketiga versi BoW diuji di 6.1.')
code(r'''
def stem_series(series):
    """Stem hanya kata yang lolos MIN_DF, 4 proses forkserver (aman walau CUDA sudah aktif)."""
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
SAFE = [valid_text, repair_encoding, normalize_unicode, lambda s: remove_html(s) if "<" in s else s,
        decode_html_entities, remove_invisible, normalize_whitespace]
PLACEHOLDER = [replace_urls, replace_emails, replace_phones, replace_mentions, normalize_whitespace]
NORMAL = [segment_hashtags, lowercase, normalize_slang, normalize_elongation, normalize_laughter, normalize_reduplication]
CLEAN = [remove_emoji, remove_emoticons, remove_numbers, remove_punctuation, reduce_repeated_words, normalize_whitespace]
STOP = STOPWORDS | set("""akan ada adalah agar aja aku apa atau bahwa banyak baru bila bisa buat bukan cuma dah dalam dapat demi dgn
    dong gitu harus hal hanya ini itu jadi jika juga kalau kami kamu karena kata ke kenapa kok kini lagi lah lain lalu lebih
    loh maka masih mau mereka meski mungkin nah namun nanti nya oleh pada para pernah pun saat saja sama sampai sangat
    sebagai sedang sejak sekarang semua seperti setelah sih sudah supaya tapi telah tentang terhadap tersebut tetapi
    untuk usai yaitu yang ya url user email phone wkwk haha tak tidak gak ga nggak enggak belum jangan bukan""".split())
data["safe"] = apply(data[TEXT_COL], SAFE); data["embed"] = apply(data.safe, PLACEHOLDER); data["normal"] = apply(data.embed, NORMAL)
bow = apply(data.normal, CLEAN).str.split()
df_bow = pd.Series(Counter(w for s in bow for w in set(s))) / len(bow)
DOMAIN_STOP = set(df_bow[df_bow > DF_MAX].index)
show(f"Stopword domain · df > {DF_MAX:.0%}", df_bow[df_bow > DF_MAX].sort_values(ascending=False).rename("df"))
data["bow"] = bow.map(lambda s: " ".join(w for w in s if w not in STOP | DOMAIN_STOP and len(w) > 2))
data["bow_stem"] = stem_series(data.bow)
from gensim.models.phrases import Phrases
phrases = Phrases(data.bow.str.split().tolist(), min_count=5, threshold=10, delimiter="_")
data["bow_bigram"] = [" ".join(phrases[s]) for s in data.bow.str.split()]
show("Bigram yang terbentuk", pd.Series(Counter(w for s in data.bow_bigram.str.split() for w in s if "_" in w)).nlargest(TOP))
''')
code(r'''
views = ["safe", "embed", "normal", "bow", "bow_stem", "bow_bigram"]
show("Audit preprocessing", pd.DataFrame({v: {"kosong_%": 100 * data[v].str.strip().eq("").mean(), "kata_median": data[v].str.split().str.len().median(),
    "kosakata": len(set(" ".join(data[v]).split()))} for v in views}).T)
show("Before/after", data[[TEXT_COL, "bow", "bow_stem", "bow_bigram"]].sample(4, random_state=SEED))
''')

part('05 · Fungsi evaluasi tanpa label', '''Kriteria di sini **tidak memakai label**. Semua model menghasilkan dua hal yang sama: daftar kata kunci per topik dan topik dominan setiap dokumen. Dengan begitu, model yang cara kerjanya berbeda tetap bisa dibandingkan.
- **NPMI** dihitung dengan co-occurrence dokumen, pada korpus versi teks yang dipakai model itu sendiri.
- **Stabilitas:** model di-fit dengan 3 seed. Topik antar seed dipasangkan dengan Hungarian matching, lalu Jaccard top-10 kata dirata-rata.''', r'''
REF = {}
def reference(view):
    if view not in REF:
        vectorizer = CountVectorizer(binary=True, min_df=MIN_DF, token_pattern=r"\S+").fit(data[view])
        matrix = vectorizer.transform(data[view]).tocsc()
        REF[view] = (matrix, vectorizer.vocabulary_, np.asarray(matrix.sum(0)).ravel())
    return REF[view]

def npmi(words, view):
    matrix, vocab, docfreq = reference(view); n = matrix.shape[0]
    ids = [vocab[w] for w in words if w in vocab]; scores = []
    for a, b in itertools.combinations(ids, 2):
        both = matrix[:, a].multiply(matrix[:, b]).sum() / n
        scores.append(-1. if both == 0 else np.log(both / (docfreq[a] * docfreq[b] / n ** 2)) / -np.log(both))
    return float(np.mean(scores)) if scores else np.nan

def ctfidf(labels, view, n=25):
    texts = data[view].to_numpy(); topics = [c for c in np.unique(labels) if c >= 0]
    joined = [" ".join(texts[labels == c]) for c in topics]
    vectorizer = CountVectorizer(token_pattern=r"\S+").fit(joined); tf = vectorizer.transform(joined).toarray().astype(float)
    weight = tf * np.log1p(tf.sum() / len(topics) / np.maximum(tf.sum(0), 1)); names = vectorizer.get_feature_names_out()
    return {c: list(names[np.argsort(-weight[i])[:n]]) for i, c in enumerate(topics)}

def match_stability(runs):
    """runs: list of {topik: [kata]} dari seed berbeda. Rata-rata Jaccard top-10 setelah Hungarian matching."""
    scores = []
    for a, b in itertools.combinations(runs, 2):
        A, B = list(a.values()), list(b.values())
        sim = np.array([[len(set(x[:10]) & set(y[:10])) / len(set(x[:10]) | set(y[:10])) for y in B] for x in A])
        r, c = linear_sum_assignment(-sim); scores.append(sim[r, c].sum() / max(len(A), len(B)))
    return float(np.mean(scores))
''', r'''
def audit_scores(labels):
    truth = data["_audit"].to_numpy(); valid = labels >= 0
    table = pd.crosstab(labels, truth).to_numpy(); r, c = linear_sum_assignment(-table)
    return {"audit_NMI": normalized_mutual_info_score(truth, labels), "audit_ARI": adjusted_rand_score(truth, labels),
            "audit_NMI_tanpa_outlier": normalized_mutual_info_score(truth[valid], labels[valid]),
            "audit_acc_hungarian": table[r, c].sum() / len(labels), "audit_purity": table.max(1).sum() / len(labels)}

rows, STORE = [], {}
def evaluate(name, topics, labels, view, **extra):
    """topics: {topik: [kata]}; labels: topik dominan tiap dokumen (-1 = outlier)."""
    labels = np.asarray(labels); valid = labels >= 0
    share = pd.Series(labels[valid]).value_counts(normalize=True) * valid.mean()
    per_topic = {c: npmi(ws[:TOP], view) for c, ws in topics.items()}
    top25 = [w for ws in topics.values() for w in ws[:25]]
    coherence = np.nanmean(list(per_topic.values())); diversity = len(set(top25)) / max(len(top25), 1)
    row = {"model": name, "view": view, **extra, "k": len(topics), "outlier_%": 100 * (~valid).mean(),
           "max_share_%": 100 * share.max() if len(share) else np.nan, "NPMI": coherence, "diversity": diversity,
           "TQ": (coherence + 1) / 2 * diversity}
    if AUDIT_COL: row.update(audit_scores(labels))
    rows.append(row); STORE[name] = {"topics": topics, "labels": labels, "view": view, "per_topic_npmi": per_topic}
    return row
print("Fungsi evaluasi siap.")
''')

part('06 · Representasi & ablasi preprocessing', 'e5 dihitung sekali (frozen, mean pooling, awalan `query: `). Ablasi preprocessing memakai NMF, karena paling cepat dan paling sensitif terhadap pilihan kata. Versi BoW dipilih dari TQ, bukan dari audit.', r'''
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

start = time.perf_counter()
E5 = embed(locate("e5", "intfloat/multilingual-e5-base"), ("query: " + data.embed).tolist())
print(f"e5: {E5.shape} dalam {time.perf_counter() - start:.0f} detik")
def tfidf(view): v = TfidfVectorizer(min_df=MIN_DF, max_df=.5, token_pattern=r"\S+", sublinear_tf=True); return v, v.fit_transform(data[view])
def counts_of(view): v = CountVectorizer(min_df=MIN_DF, max_df=.5, token_pattern=r"\S+"); return v, v.fit_transform(data[view])
def factor_topics(components, names, n=25): return {i: list(names[np.argsort(-row)[:n]]) for i, row in enumerate(components)}
''', r'''
K_REF = K_GRID[len(K_GRID) // 2]; ablation = []
for view in ["normal", "bow", "bow_stem", "bow_bigram"]:
    vectorizer, X = tfidf(view)
    model = NMF(K_REF, init="nndsvda", max_iter=400, random_state=SEED); W = model.fit_transform(X)
    labels = np.where(W.max(1) > 0, W.argmax(1), -1)
    ablation.append({"view": view, "vocab": X.shape[1], **evaluate(f"ablasi_{view}", factor_topics(model.components_, vectorizer.get_feature_names_out()), labels, view, algo="ablasi")})
rows = [r for r in rows if r.get("algo") != "ablasi"]
ablation = pd.DataFrame(ablation).drop(columns=["model", "algo"])
show(f"6.1 Ablasi preprocessing · NMF k={K_REF}", ablation)
BOW_VIEW = ablation.sort_values("TQ").iloc[-1]["view"]
print("Versi BoW terpilih (TQ tertinggi):", BOW_VIEW)
''')

part('07 · Model: grid k untuk setiap model', 'Setiap model dicoba pada semua `K_GRID` dengan 3 seed. Baris yang dicatat adalah seed pertama, dan stabilitas dihitung dari ketiga seed. HDBSCAN tidak memakai k, jadi yang divariasikan adalah `min_cluster_size`.', r'''
tfidf_vec, X_tfidf = tfidf(BOW_VIEW); count_vec, X_count = counts_of(BOW_VIEW)
names_tfidf, names_count = tfidf_vec.get_feature_names_out(), count_vec.get_feature_names_out()
THETA = {}

def fit_once(algo, k, seed):
    if algo == "tfidf_kmeans":
        labels = KMeans(k, n_init=1, random_state=seed).fit_predict(X_tfidf); return ctfidf(labels, BOW_VIEW), labels, None
    if algo == "e5_kmeans":
        labels = KMeans(k, n_init=1, random_state=seed).fit_predict(E5); return ctfidf(labels, BOW_VIEW), labels, None
    if algo == "lda":
        model = LatentDirichletAllocation(k, learning_method="batch", max_iter=25, random_state=seed); theta = model.fit_transform(X_count)
        return factor_topics(model.components_, names_count), theta.argmax(1), theta
    model = NMF(k, init="nndsvda" if seed == SEED else "random", max_iter=400, random_state=seed); W = model.fit_transform(X_tfidf)
    theta = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    return factor_topics(model.components_, names_tfidf), np.where(W.max(1) > 0, W.argmax(1), -1), theta

start = time.perf_counter()
for algo in ["tfidf_kmeans", "lda", "nmf", "e5_kmeans"]:
    for k in K_GRID:
        fits = [fit_once(algo, k, SEED + s) for s in range(SEEDS)]
        evaluate(f"{algo}_k{k}", fits[0][0], fits[0][1], BOW_VIEW, algo=algo, stability=match_stability([f[0] for f in fits]))
        THETA[f"{algo}_k{k}"] = fits[0][2]
print(f"grid selesai dalam {time.perf_counter() - start:.0f} detik")
''', r'''
import umap
reductions = [umap.UMAP(n_neighbors=15, n_components=5, min_dist=0., metric="cosine", random_state=SEED + s).fit_transform(E5) for s in range(SEEDS)]
for size in [max(10, len(data) // 200), max(15, len(data) // 100), max(25, len(data) // 50)]:
    fits = []
    for U in reductions:
        labels = HDBSCAN(min_cluster_size=size, min_samples=5).fit_predict(U); fits.append((ctfidf(labels, BOW_VIEW), labels))
    evaluate(f"bertopic_mcs{size}", fits[0][0], fits[0][1], BOW_VIEW, algo="bertopic", stability=match_stability([f[0] for f in fits]), min_cluster_size=size)
rng = np.random.default_rng(SEED); vocab_pool = np.array(names_tfidf)
random_topics = {i: list(rng.choice(vocab_pool, 25, replace=False)) for i in range(10)}
evaluate("acak_k10", random_topics, rng.integers(0, 10, len(data)), BOW_VIEW, algo="acak", stability=0.)
board = pd.DataFrame(rows)
show("Grid · tanpa kolom audit", board.drop(columns=[c for c in board if c.startswith("audit_")]).sort_values("TQ", ascending=False).head(15))
''')

part('08 · Pemilihan tanpa label & C_V', '''**Aturan:** topik raksasa (`max_share_% > MAX_SHARE`) dan outlier berlebihan (`outlier_% > MAX_OUTLIER`) dilarang. Kandidat yang tersisa diurutkan dengan rata-rata peringkat **TQ** dan **stabilitas**. Untuk 5 kandidat teratas dihitung juga **C_V** (gensim, sliding window), yaitu coherence yang paling sering dilaporkan di makalah. C_V hanya dilaporkan, tidak dipakai memilih.''', r'''
pool = board[(board.algo != "acak") & (board["max_share_%"] <= 100 * MAX_SHARE) & (board["outlier_%"] <= 100 * MAX_OUTLIER)].copy()
pool = pool if len(pool) else board[board.algo != "acak"].copy()
pool["skor_gabungan"] = pool[["TQ", "stability"]].rank(pct=True).mean(axis=1)
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
tokens = data[BOW_VIEW].str.split().tolist(); dictionary = Dictionary(tokens)
def c_v(name):
    topics = [[w for w in ws[:TOP] if w in dictionary.token2id] for ws in STORE[name]["topics"].values()]
    return CoherenceModel(topics=[t for t in topics if len(t) >= 2], texts=tokens, dictionary=dictionary, coherence="c_v", processes=1).get_coherence()
top5 = pool.sort_values("skor_gabungan", ascending=False).head(5).copy()
top5["C_V"] = [c_v(m) for m in top5.model]
show("5 kandidat teratas menurut aturan", top5[["model", "algo", "k", "TQ", "NPMI", "diversity", "stability", "C_V", "max_share_%", "outlier_%", "skor_gabungan"]])
FINAL_BASE = top5.iloc[0].model
show("Terbaik per model · menurut skor gabungan", pool.sort_values("skor_gabungan", ascending=False).groupby("algo").head(1)[["model", "k", "TQ", "stability", "skor_gabungan"]])
print("TERPILIH:", FINAL_BASE)
''')

part('09 · Post-processing', '''Diterapkan pada model terpilih, dan setiap langkah dinilai ulang.
- **Outlier ke topik terdekat:** kalau ada outlier (HDBSCAN), dokumen outlier diarahkan ke topik dengan centroid e5 terdekat, supaya semua dokumen punya topik.
- **Gabung topik kembar:** pasangan topik dengan Jaccard top-10 kata ≥ `MERGE_JACCARD` digabung, karena topik yang hampir sama membingungkan saat presentasi.
- **Gabung topik kecil:** topik di bawah `MIN_TOPIC_SHARE` digabung ke topik dengan centroid e5 terdekat.
- **Label topik:** kata kunci dihitung ulang dengan c-TF-IDF dari dokumen anggotanya, lalu 3 kata teratas dijadikan label.''', r'''
def to_nearest(labels, members):
    keep = [c for c in np.unique(labels) if c >= 0 and c not in members]
    centers = normalize(np.vstack([E5[labels == c].mean(0) for c in keep]))
    moved = labels.copy(); mask = np.isin(labels, list(members)) | (labels < 0)
    if mask.any(): moved[mask] = np.array(keep)[(E5[mask] @ centers.T).argmax(1)]
    return moved

labels = STORE[FINAL_BASE]["labels"].copy(); topics = STORE[FINAL_BASE]["topics"]; steps = []
if (labels < 0).any():
    labels = to_nearest(labels, set()); evaluate("post_outlier", ctfidf(labels, BOW_VIEW), labels, BOW_VIEW, algo="post"); steps.append("outlier")
merged_pairs = []
for a, b in itertools.combinations(sorted(topics), 2):
    A, B = set(topics[a][:10]), set(topics[b][:10])
    if len(A & B) / len(A | B) >= MERGE_JACCARD: merged_pairs.append((a, b))
for a, b in merged_pairs:
    small, large = sorted([a, b], key=lambda c: (labels == c).sum())
    labels[labels == small] = large
if merged_pairs:
    evaluate("post_gabung_kembar", ctfidf(labels, BOW_VIEW), labels, BOW_VIEW, algo="post"); steps.append(f"gabung {len(merged_pairs)} pasangan")
share = pd.Series(labels).value_counts(normalize=True)
if (share < MIN_TOPIC_SHARE).any():
    labels = to_nearest(labels, set(share[share < MIN_TOPIC_SHARE].index)); steps.append(f"gabung {int((share < MIN_TOPIC_SHARE).sum())} topik kecil")
labels = pd.Series(labels).map({c: i for i, c in enumerate(sorted(np.unique(labels)))}).to_numpy()
final_topics = ctfidf(labels, BOW_VIEW)
evaluate("FINAL", final_topics, labels, BOW_VIEW, algo="final")
board = pd.DataFrame(rows)
show("Post-processing · tanpa audit", board[board.model.isin([FINAL_BASE, "post_outlier", "post_gabung_kembar", "FINAL"])].drop(columns=[c for c in board if c.startswith("audit_")]))
print("Langkah:", steps or "tidak ada yang perlu", "| pasangan kembar:", merged_pairs)
''')

part('10 · Evaluasi & kelemahan', 'Kolom audit dibuka mulai di sini, tanpa mengubah keputusan di atas. Bagian ini menjawab tiga pertanyaan: seberapa jauh pilihan tanpa label dari yang terbaik, kriteria tanpa label mana yang searah dengan kebenaran, dan topik mana yang lemah.', r'''
audit_cols = [c for c in board if c.startswith("audit_")]
cols = ["model", "algo", "k", "TQ", "NPMI", "diversity", "stability", "max_share_%", "outlier_%"] + audit_cols
show("10.1 Semua kandidat", board[[c for c in cols if c in board]].sort_values(audit_cols[0] if audit_cols else "TQ", ascending=False))
if AUDIT_COL:
    ranked = board[~board.algo.isin(["acak", "post"])].sort_values("audit_NMI", ascending=False).reset_index(drop=True)
    final_nmi = board.loc[board.model.eq("FINAL"), "audit_NMI"].iloc[0]
    print(f"FINAL: NMI {final_nmi:.3f}, peringkat {int((ranked.audit_NMI > final_nmi).sum()) + 1} dari {len(ranked)} | terbaik dengan label: "
          f"{ranked.model[0]} NMI {ranked.audit_NMI[0]:.3f} | acak {board.loc[board.model.eq('acak_k10'), 'audit_NMI'].iloc[0]:.3f}")
    grid = board[board.algo.isin(["tfidf_kmeans", "lda", "nmf", "e5_kmeans", "bertopic"])]
    trust = pd.DataFrame({m: spearmanr(grid[m], grid.audit_NMI, nan_policy="omit") for m in ["TQ", "NPMI", "diversity", "stability", "max_share_%", "k"]},
                         index=["spearman", "p"]).T
    show("10.2 Kriteria tanpa label vs NMI audit · positif = bisa dipercaya", trust.sort_values("spearman", ascending=False))
    show("NMI terbaik per model", grid.groupby("algo").audit_NMI.max().sort_values(ascending=False))
''', r'''
vectors = normalize(np.vstack([E5[labels == c].mean(0) for c in sorted(final_topics)]))
cosine = (E5 * vectors[labels]).sum(1)
docfreq_ref = reference(BOW_VIEW)
generic = {w for w, d in zip(*[list(docfreq_ref[1].keys()), docfreq_ref[2][list(docfreq_ref[1].values())] / len(data)]) if d > .05}
dup_key = data.normal.str.replace(r"\s+", " ", regex=True).str.strip()
cards = []
for c, ws in sorted(final_topics.items()):
    m = labels == c
    row = {"topik": c, "label": " / ".join(ws[:3]), "n": int(m.sum()), "%": 100 * m.mean(), "kata_kunci": ", ".join(ws[:TOP]),
           "NPMI": npmi(ws[:TOP], BOW_VIEW), "kata_generik_%": 100 * np.mean([w in generic for w in ws[:TOP]]),
           "porsi_duplikat_%": 100 * dup_key[m].duplicated(keep=False).mean(), "kemiripan_ke_pusat": cosine[m].mean(),
           "contoh": " | ".join(data.loc[m].assign(s=cosine[m]).nlargest(2, "s")[TEXT_COL].str.slice(0, 70))}
    if AUDIT_COL:
        audit = pd.Series(data["_audit"].to_numpy()[m]).value_counts(normalize=True)
        row.update(audit_mayoritas=audit.index[0], audit_purity=audit.iloc[0])
    cards.append(row)
cards = pd.DataFrame(cards)
show("10.3 Kartu topik · urut dari NPMI terendah", cards.sort_values("NPMI"))
''', r'''
flags = pd.DataFrame({"topik": cards.topik, "label": cards.label, "NPMI_negatif": cards.NPMI < 0,
    "kata_generik_tinggi": cards["kata_generik_%"] >= 40, "didominasi_duplikat": cards["porsi_duplikat_%"] >= 30,
    "raksasa": cards["%"] > 100 * MAX_SHARE, "kecil": cards["%"] < 2})
show("10.4 Bendera kelemahan per topik", flags[flags.drop(columns=["topik", "label"]).any(axis=1)])
sim = vectors @ vectors.T; np.fill_diagonal(sim, -1); pairs = np.dstack(np.unravel_index(np.argsort(-sim, axis=None), sim.shape))[0]
close = [(int(a), int(b), sim[a, b]) for a, b in pairs if a < b][:5]
show("10.5 Pasangan topik paling mirip (centroid e5) · kandidat digabung", pd.DataFrame(close, columns=["topik_a", "topik_b", "cosine"]).assign(
    label_a=lambda f: f.topik_a.map(cards.set_index("topik").label), label_b=lambda f: f.topik_b.map(cards.set_index("topik").label)))
length = pd.cut(data[BOW_VIEW].str.split().str.len(), [-1, 0, 3, 6, 12, np.inf], labels=["kosong", "1-3", "4-6", "7-12", ">12"])
slice_rows = []
for value, ids in pd.Series(np.arange(len(data))).groupby(length.to_numpy(), observed=True):
    ids = ids.to_numpy(); row = {"panjang_bow": value, "n": len(ids), "kemiripan_ke_pusat": cosine[ids].mean()}
    if AUDIT_COL: row["NMI_slice"] = normalized_mutual_info_score(data["_audit"].to_numpy()[ids], labels[ids])
    slice_rows.append(row)
show("10.6 Kelemahan menurut panjang dokumen", pd.DataFrame(slice_rows))
base_theta = THETA.get(FINAL_BASE)
if base_theta is not None:
    print(f"10.7 Dokumen ambigu (proporsi topik dominan < 0.4): {100 * (base_theta.max(1) < .4).mean():.1f}%")
show("10.8 Paling jauh dari pusat topik · kandidat salah topik", data[[TEXT_COL]].assign(topik=labels, kemiripan=cosine).nsmallest(6, "kemiripan"))
''')

part('11 · Keluaran', 'File `topics.csv` berisi topik dan label untuk setiap dokumen. `kartu_topik.csv` berisi ringkasan topik untuk presentasi.', r'''
data[["row_id", TEXT_COL]].assign(topik=labels, label=[cards.set_index("topik").label[c] for c in labels], kemiripan=cosine).to_csv(OUT / "topics.csv", index=False)
board.to_csv(OUT / "kandidat.csv", index=False); cards.to_csv(OUT / "kartu_topik.csv", index=False); ablation.to_csv(OUT / "ablasi_preprocessing.csv", index=False)
import importlib.metadata as metadata
manifest = {"dataset": DATASET, "rows": len(data), "bow_view": BOW_VIEW, "base": FINAL_BASE, "k_final": len(final_topics), "post": steps,
    "final": {k: v for k, v in board.loc[board.model.eq("FINAL")].iloc[0].items() if isinstance(v, (int, float, np.floating))},
    "seconds": time.perf_counter() - RUN_START, "gpu": torch.cuda.get_device_name(0),
    "versions": {p: metadata.version(p) for p in ["torch", "transformers", "scikit-learn", "umap-learn", "gensim", "pandas"]}}
(OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
print(json.dumps(manifest, indent=2, default=float))
''')

md(r'''
## Sumber & batas interpretasi
- Preprocessing diambil dari `preprocessing_universal/stepwise.py`. Stopword tambahan dan stopword domain bergantung pada data, jadi periksa tabel 4.1.
- NPMI mengikuti Röder dkk. 2015, dan C_V memakai implementasi gensim. Topic quality = coherence × diversity mengikuti Dieng dkk. 2020 (ETM), dengan NPMI dinormalisasi ke 0–1.
- c-TF-IDF dan pipeline e5 + UMAP + HDBSCAN mengikuti ide [BERTopic](https://maartengr.github.io/BERTopic/), diimplementasikan langsung tanpa library BERTopic.
- Kategori berita bukan topik "benar", karena satu kategori bisa berisi beberapa topik. NMI audit hanya sinyal kasar.
- Satu sampel 5.000 dokumen. Stabilitas mengukur variasi seed, bukan variasi sampel data.
''')

for cell in CELLS:
    if cell.cell_type == 'code': ast.parse(cell.source)
next(c for c in CELLS if c.cell_type == 'code').metadata['tags'] = ['parameters']
notebook = nb.v4.new_notebook(cells=CELLS, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}, "pilot": {"task": "topic_modeling", "preprocessing_source_sha256": hashlib.sha256(u.encode()).hexdigest()}})
nb.validate(notebook); DEST.parent.mkdir(parents=True, exist_ok=True); nb.write(notebook, DEST)
kernel = Path(tempfile.gettempdir()) / 'gemastik_topic_review_kernel'; kernel.mkdir(exist_ok=True)
payload = base64.b64encode(nb.writes(notebook).encode()).decode()
runner = f'''import base64, pathlib, os, traceback
import papermill as pm
os.environ['PYTHONHASHSEED'] = '42'
source = pathlib.Path('/kaggle/working/_source.ipynb'); source.write_bytes(base64.b64decode({payload!r}))
for dataset, name in [('news', '01_topic_modeling_review.ipynb'), ('mbg', '01_topic_modeling_review_mbg.ipynb')]:
    try:
        pm.execute_notebook(str(source), '/kaggle/working/' + name, parameters={{'DATASET': dataset}},
                            kernel_name='python3', cwd='/kaggle/working', progress_bar=False, log_output=True, execution_timeout=2400)
    except Exception:
        traceback.print_exc()
source.unlink(missing_ok=True)
'''
nb.write(nb.v4.new_notebook(cells=[nb.v4.new_code_cell(runner)], metadata=notebook.metadata), kernel / 'topic.ipynb')
meta = {"id": "ivanwllm/gemastik-topic-review", "title": "Gemastik Topic Review", "code_file": "topic.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4", "enable_internet": True,
        "dataset_sources": ["ibamibrahim/indonesian-news-title", "dzikriraihan/indonesian-youtube-comments-mbg-program"],
        "kernel_sources": ["ivanwllm/gemastik-nlp-00-setup-offline"], "competition_sources": [], "model_sources": []}
(kernel / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
print('Wrote', DEST, '| cells', len(CELLS), '| code lines', sum(len(c.source.splitlines()) for c in CELLS if c.cell_type == 'code'))
