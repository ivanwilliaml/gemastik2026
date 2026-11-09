"""Build ONE multiclass pilot; author locally, execute only on Kaggle."""
from pathlib import Path
from textwrap import dedent
import ast,base64,json,tempfile,hashlib
import nbformat as nb
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'b_klasifikasi/10_multi_kelas/10_multi_kelas_review.ipynb'
UNIVERSAL=ROOT/'preprocessing_universal/stepwise.py'
CELLS=[]
def md(s):CELLS.append(nb.v4.new_markdown_cell(dedent(s).strip()))
def code(s):CELLS.append(nb.v4.new_code_cell(dedent(s).strip()))
def part(title,note,*sources):
 md('## '+title+'\n\n'+note)
 for source in sources:code(source)
md('''
# Multiclass · versi review
**Pilot satu kasus: EDA angka → preprocessing sesuai model → 5 pendekatan → evaluasi kelemahan.**

Run All pada **Kaggle GPU**, internet aktif. Semua angka EDA berasal dari data asli; tidak ada grafik otomatis atau hasil yang ditulis manual. Untuk dataset baru, ubah konfigurasi dan sumber input.

| Pendekatan | Representasi | Tujuan |
|---|---|---|
| Mayoritas | Tanpa fitur | Batas bawah |
| Logistic Regression | Word TF-IDF | Baseline teks; dua versi preprocessing diuji |
| Linear SVM | Word + char TF-IDF | Pola kata dan variasi ejaan |
| Logistic Regression | Mean Word2Vec, dilatih di train | Embedding statis |
| IndoBERTweet | Tokenizer + contextual BERT | Fine-tuning model bahasa |

**Data contoh:** [komentar YouTube MBG](https://www.kaggle.com/datasets/dzikriraihan/indonesian-youtube-comments-mbg-program). Label contoh diperlakukan sebagai pseudo-label mengikuti notebook awal. Confidence Score adalah keyakinan pemberi label, bukan bukti label benar. Skor terhadap label ini tidak membuktikan akurasi sentimen manusia.

**Validasi:** grup duplikat dibentuk sebelum split; train/validation/holdout sekitar 60/20/20. Semua model memakai baris yang sama. Validation untuk memilih varian/model dan epoch. Holdout hanya dievaluasi setelah keputusan dibekukan; skor validation adalah skor pengembangan, bukan estimasi final bebas bias.
''')
part('00 · Konfigurasi','Fokus pilot 5.000 baris; N_ROWS=None untuk seluruh CSV. SOURCE_COL/GROUP_COL boleh None. GROUP_COL harus merepresentasikan unit yang tidak boleh bocor, misalnya video/thread/pengguna.', '''
from pathlib import Path
SEED, N_ROWS, EDA_N, TOP = 42, 5000, 5000, 15
DATA_FILE = None  # None: cari comment_sentiment.csv di input Kaggle
TEXT_COL, LABEL_COL, CONF_COL = "Comment", "Sentiment", "Confidence Score"
SOURCE_COL, GROUP_COL = None, "Commenter ID"
MODEL_ID = "indolem/indobertweet-base-uncased"
EPOCHS, MAX_LEN, BATCH_SIZE = 3, 128, 16
NEAR_DUP, NEAR_THRESHOLD = True, .9
CLASS_WEIGHT, MIN_POST_GAIN = "balanced", .005
OUT = Path("/kaggle/working/hasil_review")
''')
part('01 · Lingkungan Kaggle','Paket tambahan dipasang pada Kaggle. Tidak menginstal atau melatih model pada komputer lokal. Model pretrained memakai bundle Kaggle jika tersedia, atau Hugging Face bila tidak tersedia.', '''
import sys,subprocess,os
assert Path("/kaggle/input").exists(), "Jalankan notebook ini di Kaggle."
subprocess.run([sys.executable,"-m","pip","install","-q",
    "gensim>=4.4,<5","ftfy>=6.3,<7","emoji>=2.14,<3","datasketch>=1.6,<3","langid==1.1.6"],check=True)
os.environ.update(TOKENIZERS_PARALLELISM="false",HF_HUB_DISABLE_PROGRESS_BARS="1",USE_TF="0",USE_FLAX="0")
''','''
import json,time,random,re,html,unicodedata,gc,hashlib
from collections import Counter
import numpy as np
import pandas as pd
import torch,emoji,ftfy
from IPython.display import display
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(2)
assert torch.cuda.is_available(), "Aktifkan GPU Kaggle untuk menjalankan kelima pendekatan."
OUT.mkdir(exist_ok=True)
print("GPU:",torch.cuda.get_device_name(0))
''','''
def show(title,table):
    print("\\n"+title)
    display((table.to_frame() if isinstance(table,pd.Series) else table).round(3))

def distribution(values):
    n=values.value_counts(dropna=False)
    return pd.DataFrame({"n":n,"%":n/(len(values) or np.nan)*100})

def macro(y,p):return f1_score(y,p,labels=np.arange(C),average="macro",zero_division=0)
pd.set_option("display.max_columns",None,"display.max_colwidth",130)
RUN_START=time.perf_counter()
''')
part('02 · Muat data mentah & audit sebelum drop','Audit ini dilakukan sebelum membuang missing/teks invalid. Nama kolom tidak ditebak diam-diam. Informasi sumber/grup dipakai bila disediakan; tidak dibuat sintetis.', '''
INPUT=Path("/kaggle/input")
path=Path(DATA_FILE) if DATA_FILE else next(INPUT.rglob("comment_sentiment.csv"))
raw=pd.read_csv(path)
assert {TEXT_COL,LABEL_COL}.issubset(raw), f"Kolom tersedia: {raw.columns.tolist()}"
show("Skema sebelum pembersihan",pd.DataFrame({"tipe":raw.dtypes.astype(str),"missing":raw.isna().sum(),
    "missing_%":raw.isna().mean()*100,"unik_non_null":raw.nunique(dropna=True)}))
print("File:",path,"| total baris:",len(raw),"| baris duplikat ekstra:",raw.duplicated().sum())
''','''
raw_text=raw[TEXT_COL].where(raw[TEXT_COL].map(lambda x:isinstance(x,str)),"").fillna("")
quality=pd.DataFrame({"teks_null":raw[TEXT_COL].isna(),"teks_non_string":raw[TEXT_COL].notna() & ~raw[TEXT_COL].map(lambda x:isinstance(x,str)),
    "teks_kosong":raw_text.str.strip().eq(""),"label_null":raw[LABEL_COL].isna(),
    "label_kosong":raw[LABEL_COL].astype("string").str.strip().eq("").fillna(False),
    "teks_pseudo_null":raw_text.str.strip().str.casefold().isin(["nan","null","none","n/a","-"])})
show("Kualitas input · flag dapat overlap",pd.DataFrame({"n":quality.sum(),"%":quality.mean()*100}))
valid=~quality.any(axis=1)
data=raw.loc[valid].copy().reset_index(names="row_id")
data[LABEL_COL]=data[LABEL_COL].astype(str).str.strip()
if N_ROWS and len(data)>N_ROWS:data=data.sample(N_ROWS,random_state=SEED).reset_index(drop=True)
print("Dikeluarkan karena invalid:",int((~valid).sum()),"| baris pilot:",len(data))
''')
md('''## 03 · Fungsi dari preprocessing universal
Fungsi berikut disalin langsung dari stepwise.py agar notebook tetap bisa dicopy sendiri. Pilihan sentimen: perbaikan teknis, placeholder sosial, normalisasi terkurasi, emoji tetap tersedia sebagai teks/karakter. Negasi, angka, dan istilah domain dipertahankan. Stemming, stopword removal, penghapusan emoji/angka, terjemahan daerah, dan typo otomatis tidak diaktifkan.

`minimal`: safe + casefold untuk TF-IDF/char. `normalized`: normalisasi terkurasi untuk eksperimen TF-IDF/Word2Vec. `bert`: safe + resep IndoBERTweet. Luasnya perubahan dibuktikan pada audit sebelum/sesudah, manfaatnya diuji pada validation.
''')
u=UNIVERSAL.read_text(encoding='utf-8');lines=u.splitlines();tree=ast.parse(u)
resources=u.split('# %% resources\n',1)[1].split('# %% validate',1)[0]
code(resources)
function_groups=[['map_words','valid_text','repair_encoding','normalize_unicode'],['decode_html_entities','remove_invisible','normalize_whitespace'],
 ['replace_urls','replace_emails','replace_phones','replace_mentions'],['segment_hashtags','lowercase','normalize_slang'],
 ['normalize_elongation','normalize_laughter','normalize_reduplication'],['normalize_punctuation_repeat','describe_emoji','indobertweet_view']]
functions={node.name:'\n'.join(lines[node.lineno-1:node.end_lineno]) for node in tree.body if isinstance(node,ast.FunctionDef)}
for names in function_groups:code('\n\n'.join(functions[name] for name in names))
code('''
def chain(text,steps):
    for step in steps:text=step(text)
    return text
safe_steps=[valid_text,repair_encoding,normalize_unicode,decode_html_entities,remove_invisible,normalize_whitespace]
normal_steps=[replace_emails,replace_urls,replace_phones,replace_mentions,segment_hashtags,lowercase,
    normalize_slang,normalize_elongation,normalize_laughter,normalize_reduplication,normalize_punctuation_repeat,describe_emoji,normalize_whitespace]
cache={s:chain(s,safe_steps) for s in data[TEXT_COL].unique()}
data["safe"]=data[TEXT_COL].map(cache)
print("Kosong setelah safe:",data.safe.eq("").sum())
data=data.loc[data.safe.ne("")].reset_index(drop=True)
data["minimal"]=data.safe.str.casefold()
cache={s:chain(s,normal_steps) for s in data.safe.unique()}
data["normalized"]=data.safe.map(cache)
data["bert"]=data.safe.map(indobertweet_view)
''')
part('04 · Grup duplikat sebelum split','Exact group memakai safe+casefold+spasi. Opsional near-duplicate: kandidat MinHash atas char 5-gram, diverifikasi Jaccard >= NEAR_THRESHOLD. Ini aproksimasi: kandidat yang terlewat tetap mungkin. Teks <20 karakter hanya exact; grup transitive. GROUP_COL yang diisi turut menyatukan grup.', '''
def duplicate_groups(frame):
    from datasketch import MinHash,MinHashLSH
    keys=frame.safe.str.casefold().str.replace(r"\\s+"," ",regex=True)
    parent=list(range(len(frame)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def merge(i,j):parent[root(i)]=root(j)
    first={}
    for i,key in enumerate(keys):
        if key in first:merge(i,first[key])
        else:first[key]=i
    if GROUP_COL:
        assert GROUP_COL in frame, f"Kolom grup tidak ada: {GROUP_COL}"
        for ids in frame.groupby(GROUP_COL).indices.values():
            for i in ids[1:]:merge(int(ids[0]),int(i))
    if NEAR_DUP:
        index=MinHashLSH(threshold=NEAR_THRESHOLD,num_perm=64)
        shingles={}
        for key,i in first.items():
            if len(key)<20:continue
            grams={key[j:j+5] for j in range(len(key)-4)}
            signature=MinHash(num_perm=64,seed=SEED)
            signature.update_batch([g.encode() for g in sorted(grams)])
            for candidate in index.query(signature):
                other=shingles[candidate]
                if len(grams&other)/len(grams|other)>=NEAR_THRESHOLD:merge(i,int(candidate))
            index.insert(str(i),signature);shingles[str(i)]=grams
    return np.array([root(i) for i in range(len(frame))])
''','''
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder().fit(data[LABEL_COL]);classes=encoder.classes_;C=len(classes)
assert C>=3, "Pilot ini untuk klasifikasi multiclass."
data["y"]=encoder.transform(data[LABEL_COL])
data["dup_group"]=duplicate_groups(data)
assert data.groupby("y").dup_group.nunique().min()>=5, "Perlu >=5 grup independen per kelas; ubah skema split untuk data kecil."
fold=np.full(len(data),-1)
for k,(_,idx) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(data,data.y,data.dup_group)):fold[idx]=k
train=data.loc[fold>=2].copy();val=data.loc[fold==1].copy();holdout=data.loc[fold==0].copy()
for a,b in [(train,val),(train,holdout),(val,holdout)]:assert set(a.dup_group).isdisjoint(b.dup_group)
for frame in [train,val,holdout]:assert set(frame.y)==set(range(C)), "Ada kelas hilang pada split; revisi skema sebelum training."
show("Pembagian baris",pd.DataFrame({name:frame[LABEL_COL].value_counts() for name,frame in [("train",train),("validation",val),("holdout",holdout)]}).fillna(0))
print("Grup independen:",data.dup_group.nunique(),"| overlap grup antar-split: 0")
''')
part('05 · EDA wajib multiclass · angka','EDA mendalam berikut hanya train. Persentase noise memakai semua baris train; vocabulary memakai sampel train maksimal EDA_N. Validation/holdout tidak menjadi sumber kamus, vocabulary, Word2Vec, atau normalizer yang belajar dari data.', '''
t=train[TEXT_COL].astype(str);words=t.str.split().str.len();chars=t.str.len()
key=train.safe.str.casefold().str.replace(r"\\s+"," ",regex=True)
show("Distribusi kelas train",distribution(train[LABEL_COL]))
show("Panjang karakter/kata",pd.DataFrame({"karakter":chars,"kata":words}).describe(percentiles=[.01,.05,.5,.9,.95,.99,.999]).T)
show("Duplikat train",pd.Series({"persis_ekstra":t.duplicated().sum(),"normal_ekstra":key.duplicated().sum(),
    "normal_ekstra_%":key.duplicated().mean()*100,"grup_dup":key.value_counts().gt(1).sum()}))
show("Panjang per kelas",words.groupby(train[LABEL_COL]).describe(percentiles=[.5,.95,.99]))
''','''
sample=t.sample(min(EDA_N,len(t)),random_state=SEED)
seqs=sample.str.casefold().str.findall(r"[^\\W\\d_]+")
counts=pd.Series(Counter(w for seq in seqs for w in seq),dtype="int64").sort_values(ascending=False)
docs=pd.Series(Counter(w for seq in seqs for w in set(seq)),dtype="int64")
show("Vocabulary train · sampel",pd.Series({"dokumen":len(seqs),"token":counts.sum(),"unik":len(counts),
    "TTR_%":100*len(counts)/(counts.sum() or np.nan),"hapax":counts.eq(1).sum(),"hapax_%_tipe":100*counts.eq(1).mean(),
    "hapax_%_token":100*counts.eq(1).sum()/(counts.sum() or np.nan),"langka_<=5_%_tipe":100*counts.le(5).mean()}))
show("Top words",counts.head(TOP).rename("n").to_frame().assign(pct_token=counts.head(TOP)/(counts.sum() or np.nan)*100,
    dokumen=docs.reindex(counts.head(TOP).index),pct_dokumen=docs.reindex(counts.head(TOP).index)/len(seqs)*100))
''','''
for n in [2,3]:
    bag=pd.Series(Counter(" ".join(seq[i:i+n]) for seq in seqs for i in range(len(seq)-n+1)),dtype="int64").sort_values(ascending=False)
    show(f"Top {n}-gram tanpa melompati stopword",bag.head(TOP).rename("n").to_frame().assign(pct=bag.head(TOP)/(bag.sum() or np.nan)*100))
show("Contoh hapax",counts[counts.eq(1)].head(TOP))
odd=counts.index.to_series().str.len().ge(20)|counts.index.to_series().str.count(r"([^\\W\\d_])\\1{2,}").gt(0)
show("Kandidat kata panjang/berulang · bukan pasti typo",counts[odd].head(TOP))
''')
code('''
NOISE={"url":r"https?://|www\\.","mention":r"@\\w+","hashtag":r"#\\w+","angka":r"\\d",
    "emoji/simbol":r"[\\U0001F300-\\U0001FAFF\\u2600-\\u27BF]","kapital":r"\\b[A-Z]{3,}\\b",
    "huruf_ulang":r"([^\\W\\d_])\\1{2,}","baca_ulang":r"[!?.,]{2,}","html":r"</?[a-zA-Z][^>]*>",
    "encoding":r"\\ufffd|Ã.|Â.","invisible":r"[\\u200b\\ufeff]","newline":r"[\\n\\r]",
    "email":r"\\b[\\w.+-]+@[\\w.-]+\\.[a-zA-Z]{2,}","non_ASCII":r"[^\\x00-\\x7f]",
    "html_entity":r"&(?:#\\d+|[a-zA-Z]+);","spasi_tepi":r"^\\s|\\s$","spasi_ulang":r" {2,}",
    "kontrol":r"[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]","huruf_angka":r"\\b(?=\\w*[a-zA-Z])(?=\\w*\\d)\\w+\\b",
    "slang_contoh":r"\\b(?:gk|gak|nggak|ga|bgt|yg|utk|udh|klo|jg)\\b",
    "negasi":r"\\b(?:tidak|tak|bukan|belum|jangan|tanpa|gak|nggak|ga|gk)\\b"}
flags=pd.DataFrame({name:t.str.count(pattern,flags=0 if name=="kapital" else re.I).gt(0) for name,pattern in NOISE.items()})
show("Noise train · % seluruh baris, flag overlap",pd.DataFrame({"n":flags.sum(),"%":flags.mean()*100}).sort_values("n",ascending=False))
show("Noise per kelas (%)",flags.groupby(train[LABEL_COL]).mean()*100)
if SOURCE_COL:
    assert SOURCE_COL in train
    show("Sumber",distribution(train[SOURCE_COL]));show("Label per sumber (%)",pd.crosstab(train[SOURCE_COL],train[LABEL_COL],normalize="index")*100)
''')
code('''
conflict=train.groupby(key)[LABEL_COL].nunique().loc[lambda s:s.gt(1)]
print("Grup teks dengan konflik label di train:",len(conflict))
show("Contoh konflik · perlu review konteks",train.loc[key.isin(conflict.index),[TEXT_COL,LABEL_COL]].head(TOP))
if CONF_COL in train:
    confidence=pd.to_numeric(train[CONF_COL],errors="coerce")
    show("Confidence pemberi label per kelas",confidence.groupby(train[LABEL_COL]).describe(percentiles=[.1,.5,.9]))
show("Contoh teks per kelas · train",train.groupby(LABEL_COL)[TEXT_COL].apply(lambda s:s.sample(min(3,len(s)),random_state=SEED).str.slice(0,180)).to_frame("teks"))
''')
code('''
from langid.langid import LanguageIdentifier,model as language_model
from threadpoolctl import threadpool_limits
language_sample=sample.sample(min(300,len(sample)),random_state=SEED)
identifier=LanguageIdentifier.from_modelstring(language_model,norm_probs=True)
with threadpool_limits(limits=1):
    language=language_sample.map(lambda s:identifier.classify(s[:3000]) if len(s.split())>=5 else ("tidak_pasti",np.nan))
lang=language.map(lambda p:p[0] if p[1]>=.8 else "tidak_pasti")
show("Estimasi bahasa · % sampel, satu bahasa dominan per dokumen",distribution(lang))
print("Bahasa serumpun/campur kode/teks pendek dapat salah; tidak_pasti bukan bahasa Indonesia otomatis.")
''')
part('06 · Audit preprocessing & batas token','Fungsi deterministik boleh diterapkan pada semua split; semua statistik pemilihan berikut hanya train/validation. Negasi/emoji/angka tetap tersedia, meskipun normalisasi dapat mengubah bentuknya. Tidak membuang baris valid berdasarkan distribusi label.', '''
audit=[]
for view in ["minimal","normalized","bert"]:
    audit.append({"versi":view,"berubah_%":train[view].ne(train.safe).mean()*100,"kosong":train[view].eq("").sum(),
        "unik":train[view].nunique(),"kata_p50":train[view].str.split().str.len().median(),"kata_p95":train[view].str.split().str.len().quantile(.95)})
show("Dampak preprocessing train",pd.DataFrame(audit).set_index("versi"))
show("Before/after train",train[[TEXT_COL,"minimal","normalized","bert"]].head(5))
assert all(frame[["minimal","normalized","bert"]].ne("").all().all() for frame in [train,val,holdout])
''','''
from transformers import AutoTokenizer
candidates=list(INPUT.glob("**/models/indobertweet/config.json"))
model_path=str(candidates[0].parent) if candidates else MODEL_ID
tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=bool(candidates))
token_lengths=train.bert.map(lambda s:len(tokenizer.encode(s,truncation=False)))
show("Panjang token model train",token_lengths.describe(percentiles=[.5,.9,.95,.99]))
show("Truncation train · % baris tiap kelas",pd.DataFrame({str(k):token_lengths.gt(k) for k in [64,128,256,512]}).groupby(train[LABEL_COL]).mean()*100)
print("Batas run:",MAX_LEN,"| terpotong:",f"{token_lengths.gt(MAX_LEN).mean():.2%}")
''')
part('07 · Pencatat evaluasi validation','Semua pendekatan memakai train/validation yang sama. Macro-F1 dipilih sebagai metrik kasus ini; accuracy/weighted-F1/balanced accuracy ikut dilaporkan. TF-IDF, Word2Vec, bobot kelas hanya belajar dari train. Holdout belum diprediksi.', '''
from sklearn.pipeline import make_pipeline,make_union
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import FunctionTransformer
from gensim.models import Word2Vec
models={};val_pred={};val_prob={};records=[]
y_train,y_val=train.y.to_numpy(),val.y.to_numpy()
def metrics(y,p):
    return {"macro_f1":macro(y,p),"weighted_f1":f1_score(y,p,average="weighted",zero_division=0),
        "accuracy":accuracy_score(y,p),"balanced_acc":balanced_accuracy_score(y,p)}

def register(name,estimator,view):
    start=time.perf_counter();estimator.fit(train[view],y_train)
    seconds=time.perf_counter()-start
    models[name]=(estimator,view);val_pred[name]=estimator.predict(val[view])
    if hasattr(estimator,"predict_proba"):val_prob[name]=estimator.predict_proba(val[view])
    records.append({"model":name,"view":view,"fit_seconds":seconds,**metrics(y_val,val_pred[name])})
    print(name,records[-1])
''')
part('08 · Model 1: mayoritas','Sanity baseline. Hanya memakai distribusi label train.', '''
register("01_mayoritas",DummyClassifier(strategy="most_frequent"),"minimal")
''')
part('09 · Model 2: TF-IDF word + Logistic Regression','Uji minimal vs normalized pada validation. Dua varian dihitung sebagai satu keluarga model. Pemilihan preprocessing pada validation membuat skor validation optimistis; holdout tetap terpisah.', '''
def word_tfidf():
    return TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=1.,max_features=60000,sublinear_tf=True)
def logistic():return LogisticRegression(C=2,max_iter=1000,class_weight=CLASS_WEIGHT,random_state=SEED)
variants={};ablation=[]
for view in ["minimal","normalized"]:
    estimator=make_pipeline(word_tfidf(),logistic())
    start=time.perf_counter();estimator.fit(train[view],y_train)
    variants[view]=estimator
    ablation.append({"view":view,"fit_seconds":time.perf_counter()-start,**metrics(y_val,estimator.predict(val[view]))})
show("Ablasi preprocessing · validation",pd.DataFrame(ablation))
best_view=max(ablation,key=lambda r:r["macro_f1"])["view"]
name="02_word_tfidf_lr";models[name]=(variants[best_view],best_view)
val_pred[name]=variants[best_view].predict(val[best_view]);val_prob[name]=variants[best_view].predict_proba(val[best_view])
records.append({"model":name,**next(r for r in ablation if r["view"]==best_view)})
''')
part('10 · Model 3: word+char TF-IDF + Linear SVM','Char n-gram membantu menangkap variasi ejaan dan fragmen kata; manfaat tetap dinilai dari validation. SVM menghasilkan margin, bukan probabilitas terkalibrasi.', '''
char_tfidf=TfidfVectorizer(analyzer="char",ngram_range=(3,5),min_df=2,max_features=80000,sublinear_tf=True)
svm=make_pipeline(make_union(word_tfidf(),char_tfidf),LinearSVC(C=1,class_weight=CLASS_WEIGHT,random_state=SEED))
register("03_word_char_svm",svm,"minimal")
''')
part('11 · Model 4: Word2Vec + Logistic Regression','Word2Vec dilatih hanya pada train, bukan mengambil embedding dari validation/holdout. Mean pooling menghilangkan urutan kata; embedding korpus kecil mungkin lemah. Coverage/OOV validation dan vektor nol dilaporkan.', '''
def w2v_tokens(text):return re.findall(r"[^\\W\\d_]+",text.casefold())
train_tokens=train.normalized.map(w2v_tokens)
w2v_start=time.perf_counter()
w2v=Word2Vec(sentences=train_tokens,vector_size=100,window=5,min_count=2,sg=1,epochs=20,workers=1,seed=SEED,
    hashfxn=lambda s:int(hashlib.md5(s.encode()).hexdigest()[:8],16))

def mean_w2v(texts):
    rows=[]
    for text in texts:
        vectors=[w2v.wv[w] for w in w2v_tokens(text) if w in w2v.wv]
        rows.append(np.mean(vectors,axis=0) if vectors else np.zeros(w2v.vector_size))
    return np.asarray(rows,dtype="float32")
register("04_word2vec_lr",make_pipeline(FunctionTransformer(mean_w2v),logistic()),"normalized")
records[-1]["fit_seconds"]=time.perf_counter()-w2v_start
''','''
val_tokens=val.normalized.map(w2v_tokens)
known=sum(w in w2v.wv for seq in val_tokens for w in seq);total=sum(map(len,val_tokens))
show("Coverage Word2Vec validation",pd.Series({"vocab_train":len(w2v.wv),"token_val":total,"OOV_%":100*(1-known/(total or np.nan)),
    "dokumen_vektor_nol_%":100*np.all(mean_w2v(val.normalized)==0,axis=1).mean()}))
word_vocab=models["02_word_tfidf_lr"][0][0].vocabulary_
show("OOV word TF-IDF validation",pd.Series({"vocab_train":len(word_vocab),
    "catatan_n":sum(len(seq) for seq in val_tokens),"token_di_luar_vocab_%":100*sum(w not in word_vocab for seq in val_tokens for w in seq)/(total or np.nan)}))
''')
part('12 · Model 5: fine-tuning IndoBERTweet','Trainer memilih checkpoint dari validation macro-F1. Loss berbobot kelas dihitung dari train. Dynamic padding + truncation MAX_LEN. Recipe tokenizer mengikuti model card; tidak menambahkan stemming/slang normalization pada BERT.', '''
from transformers import AutoModelForSequenceClassification,DataCollatorWithPadding,Trainer,TrainingArguments,set_seed
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
set_seed(SEED)
def encode(frame):
    dataset=Dataset.from_dict({"text":frame.bert.tolist(),"labels":frame.y.tolist()})
    return dataset.map(lambda batch:tokenizer(batch["text"],truncation=True,max_length=MAX_LEN),batched=True,remove_columns="text")
encoded_train,encoded_val=encode(train),encode(val)
weights=compute_class_weight("balanced",classes=np.arange(C),y=y_train) if CLASS_WEIGHT=="balanced" else np.ones(C)
class WeightedTrainer(Trainer):
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        labels=inputs.pop("labels");output=model(**inputs)
        loss=torch.nn.functional.cross_entropy(output.logits,labels,weight=torch.tensor(weights,device=output.logits.device,dtype=output.logits.dtype))
        return (loss,output) if return_outputs else loss
''','''
import inspect
argument_names=inspect.signature(TrainingArguments).parameters
args=dict(output_dir=str(OUT/"bert_checkpoints"),num_train_epochs=EPOCHS,learning_rate=2e-5,weight_decay=.01,
    per_device_train_batch_size=BATCH_SIZE,per_device_eval_batch_size=BATCH_SIZE*2,save_strategy="epoch",logging_strategy="epoch",
    load_best_model_at_end=True,metric_for_best_model="macro_f1",greater_is_better=True,save_total_limit=1,
    report_to="none",disable_tqdm=True,seed=SEED,fp16=True,dataloader_num_workers=0)
args["eval_strategy" if "eval_strategy" in argument_names else "evaluation_strategy"]="epoch"
model=AutoModelForSequenceClassification.from_pretrained(model_path,num_labels=C,
    id2label=dict(enumerate(classes)),label2id={v:i for i,v in enumerate(classes)},ignore_mismatched_sizes=True)
trainer=WeightedTrainer(model=model,args=TrainingArguments(**args),train_dataset=encoded_train,eval_dataset=encoded_val,
    data_collator=DataCollatorWithPadding(tokenizer),compute_metrics=lambda p:{"macro_f1":macro(p.label_ids,p.predictions.argmax(1))})
start=time.perf_counter();trainer.train();bert_seconds=time.perf_counter()-start
''','''
from scipy.special import softmax
logits=trainer.predict(encoded_val).predictions
val_prob["05_indobertweet"]=softmax(logits,axis=1)
val_pred["05_indobertweet"]=logits.argmax(1)
records.append({"model":"05_indobertweet","view":"bert","fit_seconds":bert_seconds,**metrics(y_val,logits.argmax(1))})
show("Riwayat training BERT",pd.DataFrame(trainer.state.log_history))
show("Perbandingan validation",pd.DataFrame(records).sort_values("macro_f1",ascending=False))
''')
part('13 · Pilih model & post-processing','Model dipilih dari validation macro-F1. Pengali kelas hanya dicoba bila pemenang punya predict_proba. Dampak pengali diuji cross-fit dua fold grup pada validation; angka ini tetap skor pengembangan karena model juga dipilih di validation. Holdout belum dibuka. SVM margin tidak diperlakukan sebagai probabilitas.', '''
leaderboard=pd.DataFrame(records).sort_values("macro_f1",ascending=False).reset_index(drop=True)
best_name=leaderboard.loc[0,"model"]
print("Pemenang validation:",best_name)
post_weights=np.ones(C);use_post=False

def fit_multipliers(prob,target):
    weights=np.ones(C)
    for _ in range(2):
        for k in range(C):
            best=macro(target,(prob*weights).argmax(1));chosen=weights[k]
            for value in [.5,.75,1.,1.25,1.5,2.]:
                candidate=weights.copy();candidate[k]=value
                score=macro(target,(prob*candidate).argmax(1))
                if score>best+1e-10:best,chosen=score,value
            weights[k]=chosen
    return weights
''','''
if best_name in val_prob:
    prob=val_prob[best_name];crossfit=np.empty(len(val),dtype=int)
    for tune,check in StratifiedGroupKFold(2,shuffle=True,random_state=SEED).split(val,y_val,val.dup_group):
        weights=fit_multipliers(prob[tune],y_val[tune])
        crossfit[check]=(prob[check]*weights).argmax(1)
    before,after=macro(y_val,prob.argmax(1)),macro(y_val,crossfit)
    use_post=bool(after>before+MIN_POST_GAIN)
    if use_post:post_weights=fit_multipliers(prob,y_val)
    show("Pengali kelas · validation cross-fit",pd.Series({"raw_macro_f1":before,"crossfit_macro_f1":after,"dipakai":use_post}))
else: print("Post-processing dilewati: pemenang SVM menghasilkan margin, bukan probabilitas.")
selection={"best_validation_model":best_name,"postprocessing":use_post,"multipliers":post_weights.tolist(),"classes":classes.tolist()}
(OUT/"selection.json").write_text(json.dumps(selection,indent=2))
print("Keputusan dibekukan sebelum holdout:",selection)
''')
part('14 · Evaluasi holdout sekali','Semua model kini dievaluasi pada holdout yang sama. Pemilihan model/pengali tidak diubah berdasarkan tabel holdout. Waktu inferensi dihitung per 1.000 teks, tidak termasuk fit. Skor terhadap pseudo-label bukan skor terhadap anotasi manusia.', '''
y_hold=holdout.y.to_numpy();hold_pred={};hold_prob={};hold_scores={};test_records=[]
for name,(estimator,view) in models.items():
    start=time.perf_counter();hold_pred[name]=estimator.predict(holdout[view])
    seconds=time.perf_counter()-start
    if hasattr(estimator,"predict_proba"):hold_prob[name]=estimator.predict_proba(holdout[view])
    elif hasattr(estimator,"decision_function"):hold_scores[name]=estimator.decision_function(holdout[view])
    test_records.append({"model":name,"infer_seconds_per1000":seconds/len(holdout)*1000,**metrics(y_hold,hold_pred[name])})
start=time.perf_counter();bert_logits=trainer.predict(encode(holdout)).predictions
hold_prob["05_indobertweet"]=softmax(bert_logits,axis=1);hold_pred["05_indobertweet"]=bert_logits.argmax(1)
test_records.append({"model":"05_indobertweet","infer_seconds_per1000":(time.perf_counter()-start)/len(holdout)*1000,**metrics(y_hold,bert_logits.argmax(1))})
assert len(hold_pred)==5 and all(len(pred)==len(holdout) for pred in hold_pred.values())
hold_table=pd.DataFrame(test_records).set_index("model")
show("Hasil holdout · urutan model validation",hold_table.reindex(leaderboard.model))
''','''
final_pred=(hold_prob[best_name]*post_weights).argmax(1) if use_post else hold_pred[best_name]
show("Pilihan final · holdout",pd.DataFrame([{"model":best_name,"postprocessing":use_post,**metrics(y_hold,final_pred)}]))
per_class=[]
for name,pred in {**hold_pred,"final_selected":final_pred}.items():
    report=classification_report(y_hold,pred,labels=np.arange(C),target_names=classes,output_dict=True,zero_division=0)
    per_class.extend({"model":name,"kelas":label,**report[label]} for label in classes)
per_class=pd.DataFrame(per_class)
show("Precision/recall/F1/support semua model",per_class)
cm=pd.DataFrame(confusion_matrix(y_hold,final_pred,labels=np.arange(C)),index=classes,columns=classes)
show("Confusion final · baris=aktual, kolom=prediksi",cm)
show("Confusion final · % per kelas aktual",cm.div(cm.sum(axis=1),axis=0)*100)
''')
part('15 · Di mana model lemah?','Slice menampilkan n, error rate, accuracy, dan macro-F1 untuk kelas aktual yang hadir dalam slice. F1 slice dengan komposisi kelas berbeda tidak langsung sebanding; lihat support/error rate. Confidence label rendah tidak otomatis berarti label salah.', '''
analysis=holdout.copy()
analysis["prediksi"]=encoder.inverse_transform(final_pred)
analysis["salah"]=final_pred!=y_hold
analysis["panjang"]=pd.cut(analysis.safe.str.split().str.len(),[-1,5,15,40,100,np.inf])
for name in ["negasi","slang_contoh","url","emoji/simbol"]:
    analysis[name]=analysis[TEXT_COL].astype(str).str.count(NOISE[name],flags=re.I).gt(0)
analysis["dipotong_token"]=analysis.bert.map(lambda s:len(tokenizer.encode(s,truncation=False))>MAX_LEN)
if CONF_COL in analysis:analysis["confidence_label"]=pd.cut(pd.to_numeric(analysis[CONF_COL],errors="coerce"),[-np.inf,.6,.8,.95,np.inf])
slice_cols=["panjang","negasi","slang_contoh","url","emoji/simbol","dipotong_token"]
if CONF_COL in analysis:slice_cols.append("confidence_label")
if SOURCE_COL:slice_cols.append(SOURCE_COL)
''','''
slice_rows=[]
for column in slice_cols:
    for value,frame in analysis.groupby(column,observed=True,dropna=False):
        target=frame.y.to_numpy();pred=encoder.transform(frame.prediksi)
        slice_rows.append({"slice":column,"nilai":str(value),"n":len(frame),"error_%":frame.salah.mean()*100,
            "accuracy":accuracy_score(target,pred),"macro_f1_kelas_hadir":f1_score(target,pred,labels=np.unique(target),average="macro",zero_division=0)})
slices=pd.DataFrame(slice_rows)
show("Performa menurut kondisi teks",slices)
if best_name in hold_prob:
    analysis["skor_model"]=hold_prob[best_name].max(axis=1);score_kind="probabilitas_raw_belum_terkalibrasi"
else:
    margins=np.sort(hold_scores[best_name],axis=1)
    analysis["skor_model"]=margins[:,-1]-margins[:,-2];score_kind="selisih_margin_SVM"
columns=[TEXT_COL,LABEL_COL,"prediksi","skor_model"]+([CONF_COL] if CONF_COL in analysis else [])
print("Jenis skor:",score_kind,"| jika pengali aktif, skor di sini tetap skor model sebelum pengali.")
show("Salah prediksi paling yakin · butuh review manusia",analysis[analysis.salah].sort_values("skor_model",ascending=False)[columns].head(TOP))
''','''
word_model=models["02_word_tfidf_lr"][0]
features=word_model[0].get_feature_names_out();coefficients=word_model[-1].coef_
show("Kata/ngram pendorong tiap kelas · asosiasi model linear",pd.DataFrame({label:features[np.argsort(coefficients[i])[-10:][::-1]] for i,label in enumerate(classes)}))
weakest=per_class[per_class.model.eq("final_selected")].sort_values("f1-score").iloc[0]
print("Kelas F1 terendah:",weakest.kelas,"| F1:",round(weakest["f1-score"],3),"| support:",int(weakest.support))
print("Slice error tertinggi dengan n>=20:")
display(slices[slices.n.ge(20)].sort_values("error_%",ascending=False).head(5))
''')
part('16 · Ringkasan keputusan & keluaran','File prediksi bernama holdout_predictions, bukan submission kompetisi. Tidak ada label holdout yang dipakai fitting. Untuk deployment/kompetisi, retrain setelah prosedur dipilih, lalu transform test tanpa fit ulang vocabulary pada test.', '''
decisions=pd.DataFrame([
    {"temuan":"duplikat/near-duplicate","bukti":f"{data.dup_group.nunique()} grup / {len(data)} baris","tindakan":"grup dibentuk sebelum split; overlap=0"},
    {"temuan":"preprocessing","bukti":best_view,"tindakan":"dipilih lewat ablasi validation; BERT memakai resep sendiri"},
    {"temuan":"panjang token","bukti":f"{token_lengths.gt(MAX_LEN).mean():.2%} train > {MAX_LEN}","tindakan":"batas eksplisit; slice truncation dilaporkan"},
    {"temuan":"model pilihan","bukti":best_name,"tindakan":"dipilih validation, holdout hanya laporan"},
    {"temuan":"post-processing","bukti":str(use_post),"tindakan":"pengali hanya jika cross-fit validation naik > MIN_POST_GAIN"},
    {"temuan":"kelas terlemah","bukti":str(weakest.kelas),"tindakan":"baca error, label ambigu, support, dan konteks sebelum eksperimen berikutnya"}])
show("Temuan → keputusan",decisions)
leaderboard.to_csv(OUT/"validation_models.csv",index=False)
hold_table.to_csv(OUT/"holdout_models.csv")
per_class.to_csv(OUT/"per_class.csv",index=False);slices.to_csv(OUT/"error_slices.csv",index=False)
analysis[["row_id",TEXT_COL,LABEL_COL,"prediksi","salah","skor_model"]].to_csv(OUT/"holdout_predictions.csv",index=False)
''','''
import importlib.metadata as metadata
manifest={"run":"multiclass_review","rows":len(data),"raw_rows":len(raw),"split_rows":{"train":len(train),"validation":len(val),"holdout":len(holdout)},
    "dataset_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"seed":SEED,"models_evaluated":len(hold_pred),
    "selection":selection,"final_holdout":metrics(y_hold,final_pred),"gpu":torch.cuda.get_device_name(0),
    "gpu_peak_GB":torch.cuda.max_memory_allocated()/1e9,"seconds":time.perf_counter()-RUN_START,
    "versions":{k:metadata.version(k) for k in ["torch","transformers","scikit-learn","gensim","pandas"]}}
assert manifest["gpu_peak_GB"]>0
(OUT/"run_manifest.json").write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
''')
md('''
## Sumber & batas interpretasi
- Fungsi preprocessing disalin dari notebook universal proyek ini; kamus contoh harus ditinjau untuk domain baru. Tidak ada jaminan normalisasi selalu menaikkan skor.
- [IndoBERTweet model card](https://huggingface.co/indolem/indobertweet-base-uncased): input lowercase, placeholder mention/URL, emoji ke teks.
- [scikit-learn: leakage](https://scikit-learn.org/stable/common_pitfalls.html), [group cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html): fit representasi hanya train, pisahkan grup terkait.
- [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html): embedding corpus kecil tidak setara pretrained embedding korpus besar.
- Holdout tunggal memberi satu estimasi. Setelah pola notebook disetujui, repeated/group CV dan beberapa seed dapat mengukur variasi hasil. Hasil pseudo-label perlu dibedakan dari evaluasi label manusia.
''')
# Authoring only: parse notebook source, never execute modeling locally.
for cell in CELLS:
    if cell.cell_type=='code':ast.parse(cell.source)
notebook=nb.v4.new_notebook(cells=CELLS,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"},
    "pilot":{"task":"multiclass","preprocessing_source_sha256":hashlib.sha256(u.encode()).hexdigest()}})
DEST.parent.mkdir(parents=True,exist_ok=True);nb.validate(notebook);nb.write(notebook,DEST)
# Remote runner saves the executed notebook as a downloadable Kaggle output.
kernel=Path(tempfile.gettempdir())/'gemastik_multiclass_review_kernel'
payload=base64.b64encode(nb.writes(notebook).encode()).decode()
runner=f'''import base64, pathlib, os
import papermill as pm
os.environ['PYTHONHASHSEED']='42'
source=pathlib.Path('/kaggle/working/_source.ipynb')
source.write_bytes(base64.b64decode({payload!r}))
try:
    pm.execute_notebook(str(source),'/kaggle/working/10_multi_kelas_review.ipynb',kernel_name='python3',cwd='/kaggle/working',progress_bar=False,log_output=True)
finally:
    source.unlink(missing_ok=True)
'''
nb.write(nb.v4.new_notebook(cells=[nb.v4.new_code_cell(runner)],metadata=notebook.metadata),kernel/'review.ipynb')
meta={"id":"ivanwllm/gemastik-multiclass-review","title":"Gemastik Multiclass Review","code_file":"review.ipynb","language":"python","kernel_type":"notebook","is_private":True,
    "enable_gpu":True,"machine_shape":"NvidiaTeslaT4","enable_internet":True,"dataset_sources":["dzikriraihan/indonesian-youtube-comments-mbg-program"],
    "kernel_sources":["ivanwllm/gemastik-nlp-00-setup-offline"],"competition_sources":[],"model_sources":[]}
(kernel/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
print('Wrote:',DEST,'| cells:',len(CELLS),'| code lines:',sum(len(c.source.splitlines()) for c in CELLS if c.cell_type=='code'))
