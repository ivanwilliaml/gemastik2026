"""Author weak supervision notebook untuk data ratusan ribu baris; eksekusi hanya di Kaggle."""
from pathlib import Path
from textwrap import dedent
import ast, base64, json, tempfile, hashlib
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'a_tanpa_label/04_weak_supervision/04_weak_supervision_besar.ipynb'
UNIVERSAL = ROOT / 'preprocessing_universal/stepwise.py'
CELLS = []


def md(s): CELLS.append(nb.v4.new_markdown_cell(dedent(s).strip()))
def code(s): CELLS.append(nb.v4.new_code_cell(dedent(s).strip()))
def part(title, note, *sources):
    md('## ' + title + '\n\n' + note)
    for source in sources: code(source)


md(r'''
# Weak supervision · data besar tanpa label
**Untuk kasus: ratusan ribu teks, label tidak ada atau sangat sedikit, tetapi tetap harus mengeluarkan kelas untuk setiap baris.**

Alurnya:

1. **Labeling function (LF):** beberapa sumber label lemah. Setiap LF boleh abstain kalau tidak yakin.
2. **Label model:** gabungan semua LF. Bobot setiap LF diperkirakan dari kesepakatannya dengan LF lain, tanpa label.
3. **End model:** classifier yang dilatih dari label lemah, sehingga bisa memberi kelas pada baris yang tidak tersentuh LF dan menggeneralisasi lebih jauh.

| Jenis | Sumber | Murah? |
|---|---|---|
| LF kata kunci | leksikon positif/negatif + penanganan negasi ("tidak bagus" → negatif) | ya, semua baris |
| LF pertanyaan | tanda tanya dan kata tanya, tanpa kata sentimen → netral | ya |
| LF emoji | emoji positif/negatif | ya |
| LF e5 prototipe | kemiripan embedding e5 frozen ke deskripsi label | ya, sekitar 1 menit untuk 135 ribu teks |
| LF zero-shot NLI | mDeBERTa-v3 MNLI-XNLI, hanya di sampel `N_NLI` | mahal, jadi hanya sampel |
| End model | mayoritas, LF kata kunci, TF-IDF+LR, e5+LR, **IndoBERT fine-tune dari label lemah**, ensemble | |

**Data contoh:** [ulasan Gojek](https://www.kaggle.com/datasets/ucupsedaya/gojek-app-reviews-bahasa-indonesia), 225 ribu baris. Rating bintang disembunyikan sebagai **audit**: 1–2 = negatif, 3 = netral, 4–5 = positif. Rating adalah sinyal manusia asli, tapi tetap berisik, misalnya bintang 5 yang isinya keluhan.

`N_DEV = 0` berarti tanpa label sama sekali. `N_DEV > 0` mensimulasikan panitia yang memberi sedikit label: label itu dipakai untuk menimbang LF dan memilih model, dan barisnya dikeluarkan dari audit.
''')

part('00 · Konfigurasi', 'Parameter papermill. Untuk kasus MBG, ganti preset, deskripsi label, dan kata kunci domain. Label boleh lebih dari tiga, dan setiap label cukup diberi deskripsi serta kata kunci.', r'''
DATASET = "gojek"
SEED, TOP = 42, 10
N_DEV = 0                          # 0 = tanpa label; misal 500 = panitia memberi 500 label
N_NLI, N_BERT = 20000, 30000       # sampel untuk zero-shot NLI dan untuk fine-tune IndoBERT
E5_MARGIN, NLI_MIN, CONF_MIN = .02, .50, .60
BERT_EPOCHS, BERT_LEN, BATCH = 2, 64, 256
BUDGET_MINUTES = 290
LABELS = ["negatif", "netral", "positif"]
DESCRIPTIONS = {"negatif": ["keluhan, kecewa, aplikasi buruk dan bermasalah", "pelayanan jelek, lambat, mahal, mengecewakan"],
                "netral": ["pertanyaan atau informasi tanpa penilaian", "bertanya cara memakai fitur atau meminta informasi"],
                "positif": ["pujian, puas, aplikasi bagus dan membantu", "pelayanan baik, cepat, ramah, memuaskan"]}
HYPOTHESES = {"negatif": "Teks ini berisi keluhan atau penilaian negatif.",
              "netral": "Teks ini netral, berisi pertanyaan atau informasi tanpa penilaian.",
              "positif": "Teks ini berisi pujian atau penilaian positif."}
PRESETS = {"gojek": dict(file="GojekAppReviewV4.0.0-V4.9.3_Cleaned.csv", text="content", audit="score",
                         audit_map={1: "negatif", 2: "negatif", 3: "netral", 4: "positif", 5: "positif"},
                         pos_extra="driver ramah cepat sampai tepat waktu", neg_extra="cancel batal saldo hilang topup gagal ongkir"),
           "mbg": dict(file="comment_sentiment.csv", text="Comment", audit="Sentiment",
                       audit_map={"negative": "negatif", "neutral": "netral", "positive": "positif"},
                       pos_extra="bergizi bermanfaat dukung mendukung setuju lanjutkan", neg_extra="korupsi koruptor keracunan basi tolak hentikan pemborosan")}
''', r'''
from pathlib import Path
P = PRESETS[DATASET]; FILE, TEXT_COL, AUDIT_COL = P["file"], P["text"], P["audit"]
OUT = Path(f"/kaggle/working/hasil_weak_{DATASET}_dev{N_DEV}")
K = len(LABELS); LID = {name: i for i, name in enumerate(LABELS)}
print("dataset:", DATASET, "| label:", LABELS, "| N_DEV:", N_DEV)
''')

part('01 · Lingkungan Kaggle & pencatat waktu', 'Model diambil dari paket offline kalau ada, dan dari Hugging Face kalau tidak. e5 dan mDeBERTa hanya dipakai inferensi (frozen). IndoBERT di-fine-tune dari label lemah.', r'''
import sys, os, subprocess
assert Path("/kaggle/input").exists(), "Jalankan notebook ini di Kaggle."
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ftfy>=6.3,<7", "emoji>=2.14,<3", "langid==1.1.6", "sentencepiece>=0.2,<1"], check=True)
os.environ.update(TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_PROGRESS_BARS="1", USE_TF="0", USE_FLAX="0")
''', r'''
import json, time, random, re, html, unicodedata, warnings
from collections import Counter
from contextlib import contextmanager
import numpy as np, pandas as pd, torch, emoji, ftfy
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
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
def locate(name, hf_id):
    found = list(INPUT.glob(f"**/models/{name}/config.json")); return str(found[0].parent) if found else hf_id
pd.set_option("display.max_columns", None, "display.max_colwidth", 100, "display.width", 200)
''')

part('02 · Muat data, dedupe, dan pisahkan audit', 'Semua model jalan di **teks unik** `U`, dengan `count` sebagai bobot, jadi skor yang dilaporkan setara skor per baris. Label audit dihitung per teks unik dengan suara terbanyak, lalu disembunyikan di `U["_audit"]`. Kalau `N_DEV > 0`, sejumlah teks unik diambil sebagai dev berlabel.', r'''
with stage("muat_data"):
    path = next(INPUT.rglob(FILE)); raw = pd.read_csv(path, low_memory=False)
text = raw[TEXT_COL]
quality = pd.DataFrame({"teks_null": text.isna(), "teks_non_string": text.notna() & ~text.map(lambda x: isinstance(x, str)),
    "teks_kosong": text.fillna("").astype(str).str.strip().eq("")})
show("Kualitas input", pd.DataFrame({"n": quality.sum(), "%": quality.mean() * 100}))
data = raw.loc[~quality.any(axis=1)].reset_index(names="row_id")
data["key"] = data[TEXT_COL].astype(str).str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
data["_audit"] = data[AUDIT_COL].map(P["audit_map"]) if AUDIT_COL in data else None
U = data.groupby("key", sort=False).agg(text=(TEXT_COL, "first"), count=("key", "size"),
    _audit=("_audit", lambda s: s.mode().iloc[0] if s.notna().any() else None)).reset_index()
W = U["count"].to_numpy()
rng = np.random.default_rng(SEED)
DEV = np.sort(rng.choice(len(U), N_DEV, replace=False)) if N_DEV else np.array([], dtype=int)
is_dev = np.isin(np.arange(len(U)), DEV)
show("Ukuran", pd.Series({"baris": len(data), "teks_unik": len(U), "baris_di_teks_berulang_%": 100 * W[W > 1].sum() / len(data),
    "dev_berlabel": len(DEV), "teks_terbanyak": U.text[W.argmax()], "n_terbanyak": W.max()}))
''')

part('03 · EDA wajib weak supervision · angka', 'Yang paling menentukan di weak supervision: seberapa pendek teksnya (kata kunci jarang kena), seberapa sering ada negasi (kata kunci bisa terbalik), emoji (sumber LF gratis), bahasa campuran (leksikon Indonesia meleset), dan teks berulang (satu kesalahan LF terulang ribuan kali). Distribusi label audit **tidak** ditampilkan di sini, supaya tidak memengaruhi rancangan LF.', r'''
with stage("eda", len(U)):
    t = U.text.astype(str); words = t.str.split().str.len()
    show("3.1 Panjang (berbobot baris)", pd.Series({"kata_median": np.median(np.repeat(words, W)), "<=2_kata_%": 100 * W[words <= 2].sum() / W.sum(),
        "<=5_kata_%": 100 * W[words <= 5].sum() / W.sum(), ">30_kata_%": 100 * W[words > 30].sum() / W.sum()}))
    flags = pd.DataFrame({"negasi": t.str.contains(r"\b(?:tidak|tak|gak|ga|nggak|enggak|bukan|belum|kurang|jangan)\b", case=False),
        "tanda_tanya": t.str.contains(r"\?"), "emoji": t.map(lambda s: emoji.emoji_count(s) > 0),
        "kata_inggris": t.str.contains(r"\b(?:good|nice|best|bad|the|is|very|thanks|please)\b", case=False),
        "slang": t.str.contains(r"\b(?:gk|gak|ga|bgt|yg|udh|klo|jg|tdk|mantul|mantab)\b", case=False),
        "huruf_ulang": t.str.contains(r"([^\W\d_])\1{2,}"), "angka": t.str.contains(r"\d")})
    show("3.2 Kondisi teks · % baris", (flags.mul(W, axis=0).sum() / W.sum() * 100).sort_values(ascending=False))
    seqs = t.str.casefold().str.findall(r"[^\W\d_]+")
    counts = pd.Series(Counter(w for s, c in zip(seqs, W) for w in s), dtype="int64").sort_values(ascending=False)
    show("3.3 Kata teratas (berbobot baris) · bahan leksikon", counts.head(25))
    show("Teks paling sering · satu kesalahan LF di sini terulang ribuan kali", U.nlargest(TOP, "count")[["text", "count"]])
''')

md('## 04 · Preprocessing dari universal\n\nLF kata kunci butuh teks yang dinormalisasi, supaya "gak bagus" dan "ga bgs" terbaca sama. Model e5, NLI, dan IndoBERT memakai teks yang hanya dibersihkan secara teknis.')
u = UNIVERSAL.read_text(encoding='utf-8'); lines = u.splitlines(); tree = ast.parse(u)
code(u.split('# %% resources\n', 1)[1].split('# %% validate', 1)[0])
functions = {node.name: '\n'.join(lines[node.lineno - 1:node.end_lineno]) for node in tree.body if isinstance(node, ast.FunctionDef)}
for names in [['valid_text', 'repair_encoding', 'normalize_unicode', 'remove_html', 'decode_html_entities', 'remove_invisible', 'normalize_whitespace'],
              ['replace_urls', 'replace_emails', 'replace_phones', 'replace_mentions', 'lowercase', 'normalize_slang', 'normalize_elongation',
               'normalize_laughter', 'remove_punctuation']]:
    code('\n\n'.join(functions[name] for name in names))
code(r'''
def chain(text, steps):
    for step in steps: text = step(text)
    return text
def apply(series, steps):
    cache = {s: chain(s, steps) for s in series.unique()}; return series.map(cache)
SAFE = [valid_text, repair_encoding, normalize_unicode, lambda s: remove_html(s) if "<" in s else s, decode_html_entities, remove_invisible, normalize_whitespace]
with stage("preprocessing", len(U)):
    U["safe"] = apply(U.text, SAFE)
    U["embed"] = apply(U.safe, [replace_urls, replace_emails, replace_phones, replace_mentions, normalize_whitespace])
    U["normal"] = apply(U.embed, [lowercase, normalize_slang, normalize_elongation, normalize_laughter, remove_punctuation, normalize_whitespace])
show("Before/after", U[["text", "normal"]].sample(5, random_state=SEED))
''')

part('05 · Labeling function', 'Setiap LF menghasilkan satu kolom di matriks `L`: indeks label, atau −1 kalau abstain. Leksikon ditulis eksplisit di sini supaya mudah diperiksa dan diganti untuk domain lain.', r'''
POS = set("""bagus baik mantap mantab mantul keren hebat puas memuaskan membantu terbantu cepat mudah gampang praktis ramah sopan
    nyaman aman murah hemat terbaik top oke ok good nice best great love suka senang sukses makasih terimakasih thanks lancar
    responsif profesional amanah enak sip jos joss recommended rekomen luar_biasa""".split()) | set(P["pos_extra"].split())
NEG = set("""jelek buruk parah payah kecewa mengecewakan lambat lama lemot lelet error eror bug ngebug crash gagal susah sulit ribet
    mahal rugi tipu penipu penipuan bohong kesal kesel marah benci sampah hancur rusak gangguan masalah komplain aneh menyesal
    nyesel bad worst lambat""".split()) | set(P["neg_extra"].split())
NEGATORS = {"tidak", "tak", "bukan", "belum", "kurang", "jangan", "enggak"}
QUESTION = {"bagaimana", "kenapa", "mengapa", "kapan", "apakah", "dimana", "gimana", "berapa", "bisakah", "apa"}
POS_EMOJI, NEG_EMOJI = set("😍👍🙏❤😊🥰👌💯✨😁😄🤩💪🥳"), set("😡😠😤👎😭😢💔🤬😒😞😩🙄")

def keyword_score(text):
    tokens = text.split(); score = 0
    for i, w in enumerate(tokens):
        polarity = 1 if w in POS else -1 if w in NEG else 0
        if polarity and NEGATORS & set(tokens[max(0, i - 2):i]): polarity = -polarity
        score += polarity
    return score

with stage("lf_murah", len(U)):
    score = U.normal.map(keyword_score).to_numpy()
    lf_keyword = np.where(score > 0, LID["positif"], np.where(score < 0, LID["negatif"], -1))
    asks = U.text.str.contains(r"\?") | U.normal.map(lambda s: bool(QUESTION & set(s.split()[:2])))
    lf_question = np.where(asks.to_numpy() & (score == 0), LID["netral"], -1) if "netral" in LID else np.full(len(U), -1)
    emo = U.text.map(lambda s: sum(c in POS_EMOJI for c in s) - sum(c in NEG_EMOJI for c in s)).to_numpy()
    lf_emoji = np.where(emo > 0, LID["positif"], np.where(emo < 0, LID["negatif"], -1))
''', r'''
def embed(model_id, texts, max_len=128, batch=BATCH):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_id); model = AutoModel.from_pretrained(model_id).cuda().half().eval()
    order = np.argsort([len(x) for x in texts]); parts = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            enc = tokenizer([texts[j] for j in order[i:i + batch]], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to("cuda")
            hidden = model(**enc).last_hidden_state; mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            parts.append(((hidden * mask).sum(1) / mask.sum(1)).float().cpu().numpy())
    vectors = np.zeros((len(texts), parts[0].shape[1]), "float32"); vectors[order] = np.concatenate(parts)
    del model; torch.cuda.empty_cache(); return normalize(vectors)

with stage("lf_e5", len(U)):
    E5_ID = locate("e5", "intfloat/multilingual-e5-base")
    E5 = embed(E5_ID, ("query: " + U.embed).tolist())
    desc = [("query: " + d, name) for name in LABELS for d in DESCRIPTIONS[name]]
    desc_vec = embed(E5_ID, [d for d, _ in desc], batch=16)
    protos = normalize(np.vstack([desc_vec[[i for i, (_, n) in enumerate(desc) if n == name]].mean(0) for name in LABELS]))
    sim = E5 @ protos.T; ordered = np.sort(sim, 1)
    lf_e5 = np.where(ordered[:, -1] - ordered[:, -2] >= E5_MARGIN, sim.argmax(1), -1)
''', r'''
with stage("lf_nli_sampel", N_NLI):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    nli_id = locate("mdeberta_nli", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    tok = AutoTokenizer.from_pretrained(nli_id); nli = AutoModelForSequenceClassification.from_pretrained(nli_id).cuda().eval()
    label2id = {k.lower(): v for k, v in nli.config.label2id.items()}; ent, con = label2id["entailment"], label2id["contradiction"]
    S_NLI = np.sort(rng.choice(np.where(~is_dev)[0], min(N_NLI, int((~is_dev).sum())), replace=False))
    premises = U.embed.to_numpy()[S_NLI]; order = np.argsort([len(p) for p in premises]); nli_prob = np.zeros((len(S_NLI), K), "float32")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(order), 128):
            idx = order[i:i + 128]
            for j, name in enumerate(LABELS):
                enc = tok(list(premises[idx]), [HYPOTHESES[name]] * len(idx), truncation=True, max_length=128, padding=True, return_tensors="pt").to("cuda")
                logits = nli(**enc).logits.float()
                nli_prob[idx, j] = torch.softmax(logits[:, [ent, con]], 1)[:, 0].cpu().numpy()
    del nli; torch.cuda.empty_cache()
    nli_prob = nli_prob / nli_prob.sum(1, keepdims=True)
    lf_nli = np.full(len(U), -1); lf_nli[S_NLI] = np.where(nli_prob.max(1) >= NLI_MIN, nli_prob.argmax(1), -1)
''', r'''
L = np.column_stack([lf_keyword, lf_question, lf_emoji, lf_e5, lf_nli]); LF_NAMES = ["kata_kunci", "pertanyaan", "emoji", "e5_prototipe", "nli_zero_shot"]
covered = L >= 0
def lf_table(L, rows=slice(None)):
    out = []
    for j, name in enumerate(LF_NAMES):
        c = covered[rows, j]; w = W[rows]; others = np.delete(covered[rows], j, 1).any(1)
        Lr = L[rows]; rest = np.delete(Lr, j, 1)
        conflict = c & ((rest >= 0) & (rest != Lr[:, [j]])).any(1)
        dist = pd.Series(L[rows][c, j]).map(dict(enumerate(LABELS))).value_counts(normalize=True)
        out.append({"LF": name, "coverage_%_baris": 100 * w[c].sum() / w.sum(), "overlap_%": 100 * (c & others).mean(),
                    "conflict_%": 100 * conflict.mean(), **{f"porsi_{k}": dist.get(k, 0) for k in LABELS}})
    return pd.DataFrame(out)
show("5.1 Analisis LF di sampel NLI · coverage, overlap, conflict (tanpa label)", lf_table(L, S_NLI))
print(f"Baris tanpa satu LF pun di luar NLI: {100 * W[~covered[:, :4].any(1)].sum() / W.sum():.1f}%")
''')

part('06 · Label model', '''Dua cara menggabungkan LF.
- **Majority vote:** suara terbanyak. Kalau seri, baris itu abstain.
- **Weighted vote:** akurasi setiap LF diperkirakan dari kesepakatannya dengan suara LF lain (leave-one-out) di baris yang tersentuh minimal dua LF, lalu dijadikan bobot log-odds. Kalau ada dev berlabel, akurasi diambil langsung dari dev.

Hasilnya berupa probabilitas kelas. Baris dengan keyakinan ≥ `CONF_MIN` menjadi data latih end model.''', r'''
def vote(L, weights):
    score = np.zeros((len(L), K))
    for j, w in enumerate(weights):
        hit = L[:, j] >= 0; score[np.where(hit)[0], L[hit, j]] += w
    prob = np.exp(score - score.max(1, keepdims=True)); prob[score.sum(1) == 0] = 1 / K
    return prob / prob.sum(1, keepdims=True), score.sum(1) > 0

mv_prob, mv_any = vote(L, np.ones(L.shape[1]))
mv = np.where(mv_any & (np.sort(mv_prob, 1)[:, -1] > np.sort(mv_prob, 1)[:, -2]), mv_prob.argmax(1), -1)
acc = []
for j in range(L.shape[1]):
    others, _ = vote(np.delete(L, j, 1), np.ones(L.shape[1] - 1))
    rows = covered[:, j] & np.delete(covered, j, 1).any(1)
    if is_dev.any() and (covered[:, j] & is_dev).sum() >= 20:
        truth = U["_audit"].map(LID).to_numpy(); rows_dev = covered[:, j] & is_dev
        acc.append(float((L[rows_dev, j] == truth[rows_dev]).mean()))
    else:
        acc.append(float((L[rows, j] == others[rows].argmax(1)).mean()) if rows.any() else 1 / K)
acc = np.clip(np.array(acc), 1 / K + 1e-3, .99)
weights = np.log(acc * (K - 1) / (1 - acc))
lm_prob, lm_any = vote(L, weights); lm = np.where(lm_any, lm_prob.argmax(1), -1)
show("6.1 Bobot LF", pd.DataFrame({"LF": LF_NAMES, "akurasi_perkiraan": acc, "bobot": weights, "sumber": "dev" if is_dev.any() else "kesepakatan antar LF"}))
train_mask = (lm >= 0) & (lm_prob.max(1) >= CONF_MIN) & ~is_dev
show("6.2 Data latih dari label lemah", pd.Series({"teks_unik_berlabel_lemah": int(train_mask.sum()), "coverage_%_baris": 100 * W[lm >= 0].sum() / W.sum(),
    **{f"porsi_{k}": (lm[train_mask] == i).mean() for i, k in enumerate(LABELS)}}))
''')

part('07 · End model', 'Lima pendekatan dilatih dari label lemah yang sama, ditambah ensemble rata-rata tanpa bobot. Kelas diberi bobot seimbang, dan baris berlabel lemah diberi bobot sebesar keyakinan label model. IndoBERT dilatih di sampel `N_BERT` yang seimbang per kelas.', r'''
y_weak, conf = lm[train_mask], lm_prob[train_mask].max(1); PROB = {}
majority = np.bincount(y_weak, minlength=K).argmax()
PROB["01_mayoritas"] = np.eye(K)[np.full(len(U), majority)]
PROB["02_kata_kunci_saja"] = np.eye(K)[np.where(lf_keyword >= 0, lf_keyword, majority)]
with stage("tfidf_lr", len(U)):
    tfidf = TfidfVectorizer(min_df=3, ngram_range=(1, 2), max_features=200000, sublinear_tf=True).fit(U.normal)
    X = tfidf.transform(U.normal)
    PROB["03_tfidf_lr"] = LogisticRegression(C=2, max_iter=2000, class_weight="balanced").fit(X[train_mask], y_weak, sample_weight=conf).predict_proba(X)
with stage("e5_lr", len(U)):
    PROB["04_e5_lr"] = LogisticRegression(C=2, max_iter=2000, class_weight="balanced").fit(E5[train_mask], y_weak, sample_weight=conf).predict_proba(E5)
''', r'''
with stage("indobert_finetune", N_BERT):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    bert_id = locate("indobert", "indobenchmark/indobert-base-p1")
    tok = AutoTokenizer.from_pretrained(bert_id)
    model = AutoModelForSequenceClassification.from_pretrained(bert_id, num_labels=K, ignore_mismatched_sizes=True).cuda()
    pool = np.where(train_mask)[0]; per_class = N_BERT // K
    pick = np.concatenate([rng.permutation(pool[lm[pool] == c])[:per_class] for c in range(K)]); rng.shuffle(pick)
    texts, target = U.embed.str.lower().to_numpy(), torch.tensor(lm)
    class_w = torch.tensor(len(pick) / (K * np.maximum(np.bincount(lm[pick], minlength=K), 1)), dtype=torch.float).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=.01); bs = 64
    steps = BERT_EPOCHS * int(np.ceil(len(pick) / bs)); scheduler = get_linear_schedule_with_warmup(optimizer, int(.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda"); model.train()
    for epoch in range(BERT_EPOCHS):
        for i in range(0, len(pick), bs):
            idx = pick[i:i + bs]; enc = tok(list(texts[idx]), truncation=True, max_length=BERT_LEN, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), target[idx].cuda(), weight=class_w)
            optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); scheduler.step()
    model.eval(); order = np.argsort([len(x) for x in texts]); bert_prob = np.zeros((len(U), K), "float32")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]; enc = tok(list(texts[idx]), truncation=True, max_length=BERT_LEN, padding=True, return_tensors="pt").to("cuda")
            bert_prob[idx] = torch.softmax(model(**enc).logits.float(), 1).cpu().numpy()
    del model; torch.cuda.empty_cache()
PROB["05_indobert_weak"] = bert_prob
PROB["06_ensemble_rata"] = (PROB["03_tfidf_lr"] + PROB["04_e5_lr"] + PROB["05_indobert_weak"]) / 3
print(f"IndoBERT dilatih di {len(pick)} teks berlabel lemah, {BERT_EPOCHS} epoch")
''')

part('08 · Pemilihan & post-processing', '''**Tanpa dev:** ensemble rata-rata dipakai sebagai default, karena tiga representasi yang berbeda membuat kesalahan masing-masing saling meredam. Tabel kesepakatan antar model hanya ditampilkan, tidak dipakai memilih.

**Dengan dev:** model dengan macro-F1 dev tertinggi dipilih. Lalu pengali kelas dicoba secara cross-fit di dev, dan dipakai hanya kalau naik minimal 0,005.

**Konsistensi duplikat** terjadi otomatis, karena prediksi dibuat per teks unik.''', r'''
names = list(PROB)
agree = pd.DataFrame([[np.average(PROB[a].argmax(1) == PROB[b].argmax(1), weights=W) for b in names] for a in names], index=names, columns=names)
show("8.1 Kesepakatan antar model · % baris (tanpa label)", 100 * agree)
show("Distribusi prediksi · % baris", pd.DataFrame({n: pd.Series(np.bincount(PROB[n].argmax(1), weights=W, minlength=K) / W.sum(), index=LABELS) for n in names}).T * 100)
def macro(y, p, w=None): return f1_score(y, p, labels=range(K), average="macro", zero_division=0, sample_weight=w)
multipliers = np.ones(K)
if is_dev.any():
    y_dev = U["_audit"].map(LID).to_numpy()[DEV]
    dev_scores = pd.Series({n: macro(y_dev, PROB[n][DEV].argmax(1), W[DEV]) for n in names}).sort_values(ascending=False)
    show("8.2 Macro-F1 di dev berlabel", dev_scores)
    FINAL = dev_scores.index[0]
    def fit_mult(p, y, w):
        m = np.ones(K)
        for _ in range(2):
            for c in range(K):
                m[c] = max([.5, .75, 1, 1.25, 1.5, 2], key=lambda v: macro(y, (p * np.where(np.arange(K) == c, v, m)).argmax(1), w))
        return m
    p_dev, cross = PROB[FINAL][DEV], np.zeros(len(DEV), int)
    for a, b in StratifiedKFold(2, shuffle=True, random_state=SEED).split(p_dev, y_dev):
        cross[b] = (p_dev[b] * fit_mult(p_dev[a], y_dev[a], W[DEV][a])).argmax(1)
    gain = macro(y_dev, cross, W[DEV]) - macro(y_dev, p_dev.argmax(1), W[DEV])
    if gain >= .005: multipliers = fit_mult(p_dev, y_dev, W[DEV])
    print(f"Pengali kelas cross-fit: kenaikan {gain:+.4f} -> {'dipakai ' + str(multipliers) if gain >= .005 else 'tidak dipakai'}")
else:
    FINAL = "06_ensemble_rata"
final_prob = PROB[FINAL] * multipliers; final_prob /= final_prob.sum(1, keepdims=True); final = final_prob.argmax(1)
print("FINAL:", FINAL)
''')

part('09 · Evaluasi & kelemahan (audit)', 'Label audit dibuka di sini. Baris dev dikeluarkan dari audit. Semua skor berbobot jumlah baris, jadi setara skor per baris asli. Rating sebagai kebenaran juga berisik: tabel 9.6 menunjukkan contohnya.', r'''
truth = U["_audit"].map(LID).to_numpy(); evaluable = ~is_dev & ~pd.isna(truth)
y, w = truth[evaluable].astype(int), W[evaluable]
show("9.0 Distribusi label audit · % baris", pd.Series(np.bincount(y, weights=w, minlength=K) / w.sum() * 100, index=LABELS))
rows = []
for j, name in enumerate(LF_NAMES):
    c = covered[evaluable, j]
    rows.append({"sumber": "LF " + name, "coverage_%": 100 * w[c].sum() / w.sum(), "akurasi_saat_aktif": np.average(L[evaluable][c, j] == y[c], weights=w[c]) if c.any() else np.nan,
                 "macro_f1_saat_aktif": macro(y[c], L[evaluable][c, j], w[c]) if c.any() else np.nan})
for name, labels in [("label model: majority", mv), ("label model: weighted", lm)]:
    c = labels[evaluable] >= 0
    rows.append({"sumber": name, "coverage_%": 100 * w[c].sum() / w.sum(), "akurasi_saat_aktif": np.average(labels[evaluable][c] == y[c], weights=w[c]),
                 "macro_f1_saat_aktif": macro(y[c], labels[evaluable][c], w[c])})
show("9.1 LF dan label model · hanya di baris yang mereka beri label", pd.DataFrame(rows))
models = pd.DataFrame([{"model": n, "macro_f1": macro(y, PROB[n][evaluable].argmax(1), w), "accuracy": np.average(PROB[n][evaluable].argmax(1) == y, weights=w),
    **{f"f1_{k}": f1_score(y, PROB[n][evaluable].argmax(1), labels=[i], average="macro", zero_division=0, sample_weight=w) for i, k in enumerate(LABELS)}} for n in names])
models.loc[len(models)] = {"model": f"FINAL ({FINAL})", "macro_f1": macro(y, final[evaluable], w), "accuracy": np.average(final[evaluable] == y, weights=w),
    **{f"f1_{k}": f1_score(y, final[evaluable], labels=[i], average="macro", zero_division=0, sample_weight=w) for i, k in enumerate(LABELS)}}
show("9.2 End model · semua baris non-dev", models)
nli_rows = np.intersect1d(S_NLI, np.where(evaluable)[0]); yn = truth[nli_rows].astype(int)
print(f"Zero-shot NLI langsung di sampel {len(nli_rows)}: macro-F1 {macro(yn, nli_prob[np.searchsorted(S_NLI, nli_rows)].argmax(1), W[nli_rows]):.3f}"
      f" | FINAL di sampel yang sama: {macro(yn, final[nli_rows], W[nli_rows]):.3f}")
''', r'''
cm = pd.DataFrame(confusion_matrix(y, final[evaluable], labels=range(K), sample_weight=w), index=LABELS, columns=LABELS)
show("9.3 Confusion FINAL · % per kelas audit (baris=asli, kolom=prediksi)", cm.div(cm.sum(1), axis=0) * 100)
frame = pd.DataFrame({"kata": U.text.str.split().str.len().to_numpy()[evaluable], "y": y, "p": final[evaluable], "w": w})
for name, flag in flags.items(): frame[name] = flag.to_numpy()[evaluable]
frame["tanpa_LF_murah"] = ~covered[evaluable, :4].any(1)
frame["panjang"] = pd.cut(frame.kata, [0, 2, 5, 15, 40, np.inf], labels=["1-2", "3-5", "6-15", "16-40", ">40"])
slices = []
for column in ["panjang", "negasi", "tanda_tanya", "emoji", "kata_inggris", "slang", "tanpa_LF_murah"]:
    for value, part_ in frame.groupby(column, observed=True):
        slices.append({"slice": column, "nilai": str(value), "baris_%": 100 * part_.w.sum() / w.sum(), "macro_f1": macro(part_.y, part_.p, part_.w),
                       "error_%": 100 * np.average(part_.y != part_.p, weights=part_.w)})
show("9.4 Di mana model lemah", pd.DataFrame(slices))
''', r'''
analysis = U.loc[evaluable, ["text", "count"]].assign(audit=[LABELS[i] for i in y], prediksi=[LABELS[i] for i in final[evaluable]],
                                                       yakin=final_prob[evaluable].max(1), skor_kata_kunci=score[evaluable])
wrong = analysis[analysis.audit != analysis.prediksi]
show("9.5 Salah dengan keyakinan tertinggi · sering rating yang tidak cocok dengan isinya", wrong.nlargest(8, "yakin"))
show("9.6 Salah yang paling sering muncul (berbobot baris)", wrong.nlargest(8, "count"))
show("9.7 Netral menurut audit, ditebak apa?", analysis[analysis.audit.eq("netral")].prediksi.value_counts(normalize=True) * 100)
weakest = models.iloc[-1][[f"f1_{k}" for k in LABELS]].astype(float).idxmin()
print(f"Kelas terlemah FINAL: {weakest} | macro-F1 FINAL {models.iloc[-1].macro_f1:.3f} | terbaik di tabel 9.2: {models.sort_values('macro_f1').iloc[-1].model}")
''')

part('10 · Keluaran & waktu', '`prediksi.csv` berisi label untuk setiap baris asli. Waktu tiap tahap dicatat, dan notebook dibuat gagal kalau total melewati `BUDGET_MINUTES`.', r'''
label_of = dict(zip(U.key, [LABELS[i] for i in final])); conf_of = dict(zip(U.key, final_prob.max(1)))
data[["row_id", TEXT_COL]].assign(prediksi=data.key.map(label_of), yakin=data.key.map(conf_of)).to_csv(OUT / "prediksi.csv", index=False)
models.to_csv(OUT / "end_model.csv", index=False); pd.DataFrame(slices).to_csv(OUT / "slices.csv", index=False)
times = pd.DataFrame(TIMES); times["menit"] = times.detik / 60; show("Waktu per tahap", times)
total = (time.perf_counter() - RUN_START) / 60
import importlib.metadata as metadata
manifest = {"dataset": DATASET, "n_dev": N_DEV, "rows": len(data), "unique": len(U), "final": FINAL, "multipliers": multipliers.tolist(),
    "lf_weights": dict(zip(LF_NAMES, weights.tolist())), "audit": models.iloc[-1].to_dict(), "minutes": total, "times": TIMES,
    "gpu": torch.cuda.get_device_name(0), "versions": {p: metadata.version(p) for p in ["torch", "transformers", "scikit-learn", "pandas"]}}
(OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
print(f"Total {total:.1f} menit | batas {BUDGET_MINUTES} menit")
assert total < BUDGET_MINUTES
''')

md(r'''
## Batas interpretasi
- Leksikon, deskripsi label, dan hipotesis NLI ditulis manual untuk ulasan aplikasi. Untuk MBG, ganti `pos_extra`, `neg_extra`, dan deskripsinya, lalu periksa tabel 5.1 sebelum mempercayai hasil.
- Bobot label model tanpa dev diperkirakan dari kesepakatan antar LF. LF yang salah dengan cara yang sama, misalnya e5 dan NLI yang sama-sama terkecoh sarkasme, akan saling menguatkan.
- Rating bintang bukan label sentimen sempurna, jadi skor audit adalah batas bawah kasar.
- Satu seed. IndoBERT hanya dilatih 2 epoch di sampel. Hasilnya tidak membuktikan model mana yang selalu unggul.
''')

for cell in CELLS:
    if cell.cell_type == 'code': ast.parse(cell.source)
next(c for c in CELLS if c.cell_type == 'code').metadata['tags'] = ['parameters']
notebook = nb.v4.new_notebook(cells=CELLS, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}, "pilot": {"task": "weak_supervision_besar", "preprocessing_source_sha256": hashlib.sha256(u.encode()).hexdigest()}})
nb.validate(notebook); DEST.parent.mkdir(parents=True, exist_ok=True); nb.write(notebook, DEST)
kernel = Path(tempfile.gettempdir()) / 'gemastik_weak_besar_kernel'; kernel.mkdir(exist_ok=True)
payload = base64.b64encode(nb.writes(notebook).encode()).decode()
runner = f'''import base64, pathlib, os, traceback
import papermill as pm
os.environ['PYTHONHASHSEED'] = '42'
source = pathlib.Path('/kaggle/working/_source.ipynb'); source.write_bytes(base64.b64decode({payload!r}))
for n_dev, name in [(0, '04_weak_supervision_besar.ipynb'), (500, '04_weak_supervision_besar_dev500.ipynb')]:
    try:
        pm.execute_notebook(str(source), '/kaggle/working/' + name, parameters={{'N_DEV': n_dev}},
                            kernel_name='python3', cwd='/kaggle/working', progress_bar=False, log_output=True, execution_timeout=3000)
    except Exception:
        traceback.print_exc()
source.unlink(missing_ok=True)
'''
nb.write(nb.v4.new_notebook(cells=[nb.v4.new_code_cell(runner)], metadata=notebook.metadata), kernel / 'weak.ipynb')
meta = {"id": "ivanwllm/gemastik-weak-besar", "title": "Gemastik Weak Besar", "code_file": "weak.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaT4", "enable_internet": True,
        "dataset_sources": ["ucupsedaya/gojek-app-reviews-bahasa-indonesia", "dzikriraihan/indonesian-youtube-comments-mbg-program"],
        "kernel_sources": ["ivanwllm/gemastik-nlp-00-setup-offline"], "competition_sources": [], "model_sources": []}
(kernel / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
print('Wrote', DEST, '| cells', len(CELLS), '| code lines', sum(len(c.source.splitlines()) for c in CELLS if c.cell_type == 'code'))
