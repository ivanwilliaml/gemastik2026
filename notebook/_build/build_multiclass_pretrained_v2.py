"""Author multiclass V2. Notebook execution/training takes place only on Kaggle."""
from pathlib import Path
from textwrap import dedent
import ast,json,hashlib,copy,tempfile
import nbformat as nb
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'b_klasifikasi/10_multi_kelas/10_multi_kelas_v2_pretrained.ipynb'
V1=nb.read(ROOT/'b_klasifikasi/10_multi_kelas/10_multi_kelas_review.ipynb',4)
UNIVERSAL=ROOT/'preprocessing_universal/stepwise.py'
CELLS=[]
def md(s):CELLS.append(nb.v4.new_markdown_cell(dedent(s).strip()))
def code(s):CELLS.append(nb.v4.new_code_cell(dedent(s).strip()))
def section(title,note,*sources):
 md('## '+title+'\n\n'+note)
 for source in sources:code(source)
def reuse(indices):
 for i in indices:
  c=copy.deepcopy(V1.cells[i])
  if c.cell_type=='code':c.outputs=[];c.execution_count=None
  c.metadata={};CELLS.append(c)
md('''
# Multiclass v2 · lima pretrained encoder
**Eksperimen normalisasi per model + enam teknik post-processing.**

Kaggle GPU, internet aktif. Lima model yang diminta, masing-masing dua input: **native** vs **normalisasi slang+elongation terkurasi**. Total 10 fine-tuning; budget epoch/panjang token sama, tokenizer sesuai model. TF-IDF/Word2Vec tidak digunakan pada v2.

| Model | Model ID | Input native |
|---|---|---|
| IndoBERTweet Base | indolem/indobertweet-base-uncased | Lowercase, @USER/HTTPURL, emoji ke teks |
| IndoBERT Base | indobenchmark/indobert-base-p1 | Safe cleaning, lowercase |
| XLM-R Base | FacebookAI/xlm-roberta-base | Safe cleaning, pertahankan case/emoji/bahasa |
| IndoBERT IndoLEM | indolem/indobert-base-uncased | Safe cleaning, lowercase |
| NusaBERT Base | LazarusNLP/NusaBERT-base | Safe cleaning, lowercase |

**Normalisasi** hanya mengganti slang yang sudah dikurasi dan elongation yang cocok whitelist. Setelahnya tetap mengikuti recipe model masing-masing. Tidak menerjemahkan bahasa daerah, membuang negasi/angka, stemming, atau stopword removal. Manfaat diuji; varian normalized tidak otomatis dipilih.

**Data/split:** benchmark MBG yang sama dengan v1, dengan split v2 deterministik yang menyatukan pengguna, duplikat safe, dan collision hasil normalisasi sebelum split. Dataset ini sudah dieksplorasi di v1; holdout benchmark ini bukan test eksternal baru atau perbandingan berpasangan persis dengan v1. Semua keputusan v2 memakai validation; tabel holdout dibuka setelah keputusan dibekukan. Label contoh diperlakukan sebagai pseudo-label, bukan anotasi sentimen manusia.

**Post-processing:** threshold gate multiclass, bias kelas, probability ensemble, konsistensi duplikat, rule terverifikasi train, temperature calibration; tambahan ensemble+bias. Cross-fit validation mengurangi overfit parameter, tetapi tetap skor pengembangan karena pemilihan model/normalisasi juga memakai validation.
''')
section('00 · Konfigurasi','Sekitar 5.000 komentar; ubah DATA_FILE/kolom untuk data lain. Satu seed dan pencarian terbatas tidak membuktikan optimum global.', '''
from pathlib import Path
SEED,N_ROWS,EDA_N,TOP=42,5000,5000,15
DATA_FILE=None
TEXT_COL,LABEL_COL,CONF_COL="Comment","Sentiment","Confidence Score"
SOURCE_COL,GROUP_COL=None,"Commenter ID"
NEAR_DUP,NEAR_THRESHOLD=True,.9
EPOCHS,MAX_LEN,LEARNING_RATE=3,192,2e-5
WEIGHT_POWER=.5  # akar inverse-frequency dari train; lebih lunak daripada balanced penuh
POST_FOLDS,MIN_POST_GAIN,ENSEMBLE_CANDIDATES=3,.003,64
RULE_MIN_SUPPORT,RULE_MIN_PRECISION=8,.95
OUT=Path("/kaggle/working/hasil_v2")
''','''
MODEL_SPECS=[
 {"name":"01_indobertweet","id":"indolem/indobertweet-base-uncased","family":"tweet","batch":32,"accum":1,"revision":"32e28c05b47e33b6675d2670a1745c50a65e987a"},
 {"name":"02_indobert_p1","id":"indobenchmark/indobert-base-p1","family":"uncased","batch":32,"accum":1,"revision":"c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2"},
 {"name":"03_xlm_roberta","id":"FacebookAI/xlm-roberta-base","family":"cased","batch":16,"accum":2,"revision":"e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"},
 {"name":"04_indobert_indolem","id":"indolem/indobert-base-uncased","family":"uncased","batch":32,"accum":1,"revision":"7ccb3cd0f5b08ffbaa465aade22328e8600e23eb"},
 {"name":"05_nusabert","id":"LazarusNLP/NusaBERT-base","family":"uncased","batch":32,"accum":1,"revision":"e52beecf63a65daeae103c4fe8ad6edbf3e9743b"},
]
''')
section('01 · Lingkungan Kaggle','Dependensi dipasang di Kaggle. Model public diambil pada revision yang dipin; tidak ada token akses dalam notebook.', '''
import sys,os,subprocess
assert Path("/kaggle/input").exists(),"Jalankan di Kaggle GPU."
os.environ.update(USE_TF="0",USE_FLAX="0",TOKENIZERS_PARALLELISM="false",HF_HUB_DISABLE_PROGRESS_BARS="1",
    HF_HUB_ETAG_TIMEOUT="60",HF_HUB_DOWNLOAD_TIMEOUT="300")
subprocess.run([sys.executable,"-m","pip","install","-q","ftfy>=6.3,<7","emoji>=2.14,<3",
    "datasketch>=1.6,<3","langid==1.1.6","sentencepiece>=0.2,<1"],check=True)
''')
reuse([5,6])
# Replace v1 utility setup, preserving familiar small helpers.
CELLS[-2].source=CELLS[-2].source.replace('OUT.mkdir(exist_ok=True)','OUT.mkdir(exist_ok=True)')
section('02 · Data mentah & kualitas','Sebelum drop missing/invalid, audit seluruh CSV. Selanjutnya N_ROWS membatasi benchmark. Tidak ada data demo sintetis.')
reuse([8,9])
section('03 · Preprocessing universal · bagian yang sesuai','Fungsi berikut diambil langsung dari stepwise.py. Native dan normalized berbeda hanya pada dua langkah normalisasi kata; recipe model tidak berubah antara dua varian.')
u=UNIVERSAL.read_text(encoding='utf-8');lines=u.splitlines();tree=ast.parse(u)
resources=u.split('# %% resources\n',1)[1].split('# %% validate',1)[0]
wanted={'SLANG','PROTECTED','NEGATIONS','AMBIGUOUS','VOCAB','TOKEN','WORD','URL'}
resource_nodes=ast.parse(resources).body
code('\n'.join(ast.get_source_segment(resources,node) for node in resource_nodes if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in wanted for t in node.targets)))
functions={node.name:'\n'.join(lines[node.lineno-1:node.end_lineno]) for node in tree.body if isinstance(node,ast.FunctionDef)}
for names in [['map_words','valid_text','repair_encoding','normalize_unicode'],['decode_html_entities','remove_invisible','normalize_whitespace'],
              ['normalize_slang','normalize_elongation','indobertweet_view']]:
 code('\n\n'.join(functions[name] for name in names))
code('''
def safe_clean(value):
    for function in [valid_text,repair_encoding,normalize_unicode,decode_html_entities,remove_invisible,normalize_whitespace]:value=function(value)
    return value
cache={s:safe_clean(s) for s in data[TEXT_COL].unique()}
data["safe"]=data[TEXT_COL].map(cache)
print("Kosong setelah safe:",data.safe.eq("").sum())
data=data.loc[data.safe.ne("")].reset_index(drop=True)
cache={s:normalize_elongation(normalize_slang(s)) for s in data.safe.unique()}
data["normal"]=data.safe.map(cache)

def model_input(frame,spec,view):
    texts=frame["normal" if view=="normalized" else "safe"]
    if spec["family"]=="tweet":return texts.map(indobertweet_view)
    return texts.str.lower() if spec["family"]=="uncased" else texts
''')
section('04 · Split grup yang sama untuk semua eksperimen','Pengguna yang sama atau teks duplikat/near-duplicate tidak boleh melintasi split. Kandidat near-duplicate diverifikasi Jaccard; deteksi ini tetap aproksimasi.')
reuse([20,21])
# Prevent normalized variants of the same text from crossing any split.
CELLS[-2].source=CELLS[-2].source.replace('    if GROUP_COL:',
    '    normal_first={}\n    for i,key in enumerate(frame.normal.str.casefold().str.replace(r"\\s+"," ",regex=True)):\n        if key in normal_first:merge(i,normal_first[key])\n        else:normal_first[key]=i\n    if GROUP_COL:')
CELLS[-2].source=CELLS[-2].source.replace('    return np.array([root(i) for i in range(len(frame))])',
    '    components=np.array([root(i) for i in range(len(frame))])\n    return pd.Series(np.arange(len(frame))).groupby(components).transform("min").to_numpy()')
code('''
split=pd.concat([frame[["row_id","dup_group"]].assign(split=name) for name,frame in [("train",train),("validation",val),("holdout",holdout)]])
split.to_csv(OUT/"split.csv",index=False)
print("SHA split:",hashlib.sha256(split.sort_values("row_id").to_csv(index=False).encode()).hexdigest())
''')
section('05 · EDA train · angka saja','Missing/skema ada sebelum drop pada 02. Di sini: class distribution, duplikat, length/kuantil, hapax, top words/ngram, noise per kelas, konflik label, confidence pemberi label, contoh, dan estimasi bahasa. Bahasa estimasi tidak diperlakukan sebagai ground truth.')
reuse([23,24,25,26,27,28])
code('''
show("Audit normalisasi train",pd.DataFrame({"berubah_%":[train.normal.ne(train.safe).mean()*100],
    "kosong":[train.normal.eq("").sum()],"unik_safe":[train.safe.nunique()],"unik_normal":[train.normal.nunique()]}))
show("Contoh perubahan",train.loc[train.normal.ne(train.safe),["safe","normal"]].head(TOP))
REGIONAL_MARKERS=set("ora mboten kowe piye durung ngono saiki teu henteu abdi kumaha naon ngeunah indak rancak lamak bana".split())
show("Penanda regional · indikasi saja",pd.Series({"n_train":train.safe.str.casefold().str.findall(r"[^\\W\\d_]+").map(lambda s:len(set(s)&REGIONAL_MARKERS)>=2).sum()}))
''')
section('06 · Post-processing · fungsi keputusan','Threshold gate memilih probabilitas tertinggi di kelas yang lolos ambang; jika tidak ada yang lolos, kembali ke argmax. Ini berbeda dari sekadar p/threshold, yang ekuivalen dengan bias di log-probability. Bias kelas di sini ditambahkan ke log(p). Parameter dicari untuk macro-F1, bukan accuracy.', '''
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

def threshold_predict(prob,thresholds):
    eligible=prob>=thresholds
    pred=np.where(eligible,prob,-1).argmax(1)
    return np.where(eligible.any(axis=1),pred,prob.argmax(1))

def fit_threshold(prob,target):
    thresholds=np.zeros(C)
    for _ in range(2):
        for k in range(C):
            best=macro(target,threshold_predict(prob,thresholds))
            for value in [0,.2,.3,.4,.5,.6,.7,.8]:
                candidate=thresholds.copy();candidate[k]=value
                score=macro(target,threshold_predict(prob,candidate))
                if score>best+1e-10:best,thresholds=score,candidate
    return thresholds
''','''
def bias_prob(prob,bias):return softmax(np.log(np.clip(prob,1e-12,1))+bias,axis=1)

def fit_bias(prob,target):
    bias=np.zeros(C)
    for _ in range(2):
        for k in range(C):
            best=macro(target,bias_prob(prob,bias).argmax(1))
            for value in [-1.,-.6,-.3,0,.3,.6,1.]:
                candidate=bias.copy();candidate[k]=value
                score=macro(target,bias_prob(prob,candidate).argmax(1))
                if score>best+1e-10:best,bias=score,candidate
    return bias
''','''
def mix(stack,weights):return np.einsum("mnc,m->nc",stack,weights)

def fit_blend(stack,target):
    m=len(stack);rng=np.random.default_rng(SEED)
    candidates=np.vstack([np.eye(m),np.ones((1,m))/m,rng.dirichlet(np.ones(m),ENSEMBLE_CANDIDATES)])
    scores=[macro(target,mix(stack,w).argmax(1)) for w in candidates]
    return candidates[int(np.argmax(scores))]

def temperature_prob(prob,temperature):return softmax(np.log(np.clip(prob,1e-12,1))/temperature,axis=1)
def fit_temperature(prob,target):
    result=minimize_scalar(lambda x:log_loss(target,temperature_prob(prob,np.exp(x)),labels=np.arange(C)),
                           bounds=(np.log(.25),np.log(4.)),method="bounded")
    return float(np.exp(result.x))
''','''
def duplicate_prob(prob,keys):
    # Konsistensi berdasarkan exact safe+casefold+spasi, bukan ID pengguna.
    frame=pd.DataFrame(prob,columns=range(C))
    frame["key"]=np.asarray(keys)
    return frame.groupby("key",sort=False)[list(range(C))].transform("mean").to_numpy()

def probability_metrics(target,prob):
    clipped=np.clip(prob,1e-12,1);clipped=clipped/clipped.sum(axis=1,keepdims=True)
    confidence=clipped.max(1);correct=clipped.argmax(1)==target;ece=0.
    bins=np.minimum((confidence*10).astype(int),9)
    for k in range(10):
        mask=bins==k
        if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-confidence[mask].mean())
    return {"NLL":log_loss(target,clipped,labels=np.arange(C)),
        "Brier_sum":np.square(clipped-np.eye(C)[target]).sum(axis=1).mean(),"ECE_argmax_10bins":ece}
''')
md('''**Sanity check dilakukan di Kaggle sebelum training.** Temperature scalar harus menjaga argmax, setiap transformasi probabilitas harus tetap berjumlah satu. NLL/Brier/ECE mengukur probabilitas; threshold/rule hanya mengubah keputusan label sehingga metrik probabilitasnya tetap merujuk distribusi asal.''')
code('''
fixture=np.full((C*2,C),.2/(C-1));fixture[np.arange(C*2),np.arange(C*2)%C]=.8
fixture_y=np.arange(C*2)%C
assert np.array_equal(temperature_prob(fixture,2.).argmax(1),fixture.argmax(1))
assert np.allclose(bias_prob(fixture,np.zeros(C)),fixture)
assert np.array_equal(threshold_predict(fixture,np.zeros(C)),fixture.argmax(1))
assert np.allclose(mix(np.stack([fixture,fixture]),np.array([.4,.6])),fixture)
assert np.allclose(duplicate_prob(fixture,np.arange(len(fixture))),fixture)
for transformed in [bias_prob(fixture,fit_bias(fixture,fixture_y)),temperature_prob(fixture,fit_temperature(fixture,fixture_y))]:
    assert np.allclose(transformed.sum(axis=1),1)
print("Sanity check post-processing: PASS")
''')
section('07 · Rule-based correction yang dibatasi','Aturan template sentimen ini bukan kebenaran universal. Hanya aturan dengan support train >= RULE_MIN_SUPPORT dan agreement pseudo-label train >= RULE_MIN_PRECISION yang diaktifkan. Bukan menggunakan label validation/holdout untuk membuat kamus. Aturan bisa tidak aktif sama sekali; override hanya pada model ber-confidence rendah.', '''
RULES=[("positive",r"^(?:sangat )?(?:bagus|mantap|keren)(?: sekali| banget)?[.! ]*$"),
       ("positive",r"^(?:saya )?(?:setuju|mendukung)(?: program ini| mbg)?[.! ]*$"),
       ("negative",r"^(?:saya )?tidak setuju(?: program ini| mbg)?[.! ]*$"),
       ("negative",r"^(?:program ini |mbg )?(?:buruk|jelek|gagal)(?: sekali| banget)?[.! ]*$")]
rule_text=train.safe.str.casefold();active_rules=[];rule_audit=[]
for label,pattern in RULES:
    hits=rule_text.str.fullmatch(pattern,na=False);support=int(hits.sum())
    agreement=float(train.loc[hits,LABEL_COL].eq(label).mean()) if support else np.nan
    active=support>=RULE_MIN_SUPPORT and agreement>=RULE_MIN_PRECISION and label in classes
    if active:active_rules.append((int(encoder.transform([label])[0]),pattern))
    rule_audit.append({"label":label,"pattern":pattern,"n_train":support,"agreement_train":agreement,"aktif":bool(active)})
show("Audit aturan train",pd.DataFrame(rule_audit))
''','''
def rule_targets(texts):
    result=np.full(len(texts),-1)
    for i,text in enumerate(texts):
        matches={label for label,pattern in active_rules if re.fullmatch(pattern,text.casefold())}
        if len(matches)==1:result[i]=next(iter(matches))
    return result

def rule_predict(prob,targets,gate):
    pred=prob.argmax(1);mask=(targets>=0)&(prob.max(1)<gate)
    return np.where(mask,targets,pred)

def fit_rule_gate(prob,target,rules):
    gates=[0.,.4,.5,.6,.7,.8]
    return max(gates,key=lambda g:macro(target,rule_predict(prob,rules,g)))
''')
section('08 · Trainer bersama','Setiap pasangan model/input dimulai dari pretrained checkpoint yang sama, bukan meneruskan training varian sebelumnya. Bobot loss hanya dari train. Batch XLM-R lebih kecil dengan accumulation agar effective batch sebanding. Model terbaik per keluarga disimpan; optimizer checkpoint sementara dibersihkan untuk menghemat ruang.', '''
import shutil,inspect
from transformers import AutoTokenizer,AutoModelForSequenceClassification,Trainer,TrainingArguments,DataCollatorWithPadding,set_seed
from datasets import Dataset,disable_progress_bars
disable_progress_bars()
from sklearn.utils.class_weight import compute_class_weight
from scipy.special import softmax
weights=compute_class_weight("balanced",classes=np.arange(C),y=train.y.to_numpy())**WEIGHT_POWER
weights=weights/weights.mean()
class WeightedTrainer(Trainer):
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        target=inputs.pop("labels");result=model(**inputs)
        loss=torch.nn.functional.cross_entropy(result.logits.float(),target,weight=torch.tensor(weights,dtype=torch.float32,device=result.logits.device))
        return (loss,result) if return_outputs else loss

def encode(texts,tokenizer,labels=None):
    payload={"text":list(texts)}
    if labels is not None:payload["labels"]=list(labels)
    return Dataset.from_dict(payload).map(lambda b:tokenizer(b["text"],truncation=True,max_length=MAX_LEN),batched=True,remove_columns="text")
''','''
def make_trainer(spec,run,tokenizer,encoded_train,encoded_val):
    model=AutoModelForSequenceClassification.from_pretrained(spec["id"],revision=spec["revision"],trust_remote_code=False,
        num_labels=C,id2label=dict(enumerate(classes)),label2id={label:i for i,label in enumerate(classes)},ignore_mismatched_sizes=True)
    steps=int(np.ceil(len(train)/(spec["batch"]*max(1,torch.cuda.device_count())*spec["accum"]))*EPOCHS)
    args=TrainingArguments(output_dir=str(OUT/"checkpoints"/run),num_train_epochs=EPOCHS,learning_rate=LEARNING_RATE,
        per_device_train_batch_size=spec["batch"],per_device_eval_batch_size=spec["batch"],gradient_accumulation_steps=spec["accum"],
        warmup_steps=int(.1*steps),weight_decay=.01,optim="adamw_torch",fp16=True,
        eval_strategy="epoch",save_strategy="epoch",save_only_model=True,save_total_limit=1,logging_strategy="epoch",
        load_best_model_at_end=True,metric_for_best_model="macro_f1",greater_is_better=True,
        report_to="none",disable_tqdm=True,seed=SEED,data_seed=SEED,dataloader_num_workers=0)
    trainer=WeightedTrainer(model=model,args=args,train_dataset=encoded_train,eval_dataset=encoded_val,
        data_collator=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8),compute_metrics=lambda p:{"macro_f1":macro(p.label_ids,p.predictions.argmax(1))})
    trainer.model_accepts_loss_kwargs=False
    return trainer
''','''
def label_metrics(target,pred):
    return {"macro_f1":macro(target,pred),"weighted_f1":f1_score(target,pred,average="weighted",zero_division=0),
            "accuracy":accuracy_score(target,pred),"balanced_acc":balanced_accuracy_score(target,pred)}
records=[];predictions={};best_per_model={};token_audit=[]

def prepare_inputs(spec,view,tokenizer):
    a,b,c=[model_input(frame,spec,view) for frame in [train,val,holdout]]
    ids=tokenizer(a.tolist(),truncation=False)["input_ids"]
    lengths=pd.Series(list(map(len,ids)))
    token_audit.append({"model":spec["name"],"view":view,"token_p50":lengths.median(),"token_p95":lengths.quantile(.95),
        "token_p99":lengths.quantile(.99),"terpotong_%":lengths.gt(MAX_LEN).mean()*100,
        "UNK_%_termasuk_special":100*sum(seq.count(tokenizer.unk_token_id) for seq in ids)/(sum(map(len,ids)) or np.nan)})
    return encode(a,tokenizer,train.y),encode(b,tokenizer,val.y),encode(c,tokenizer)
''')
md('''**Prediksi holdout tanpa label:** logits holdout dihitung saat model masih di GPU agar tidak perlu memuat ulang sepuluh model. Holdout dataset untuk Trainer.predict **tidak membawa labels**; metriknya baru dihitung setelah seluruh model/normalisasi/post-processing dipilih dan disimpan. Tidak ada tuning memakai skor holdout.''')
code('''
def run_experiment(spec,view):
    run=spec["name"]+"__"+view;set_seed(SEED);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(spec["id"],revision=spec["revision"],trust_remote_code=False)
    a,b,c=prepare_inputs(spec,view,tokenizer)
    trainer=make_trainer(spec,run,tokenizer,a,b)
    start=time.perf_counter();trainer.train();seconds=time.perf_counter()-start
    vp=softmax(trainer.predict(b).predictions,axis=1)
    predict_start=time.perf_counter();hp=softmax(trainer.predict(c).predictions,axis=1)
    infer_seconds=(time.perf_counter()-predict_start)/len(holdout)*1000
    assert vp.shape==(len(val),C) and hp.shape==(len(holdout),C) and np.isfinite(vp).all() and np.isfinite(hp).all()
    assert np.allclose(vp.sum(axis=1),1,atol=1e-5) and np.allclose(hp.sum(axis=1),1,atol=1e-5)
    predictions[run]={"validation":vp,"holdout":hp}
    row={"model":spec["name"],"model_id":spec["id"],"view":view,"run":run,"fit_seconds":seconds,
        "parameters":trainer.model.num_parameters(),"infer_seconds_per1000":infer_seconds,"gpu_peak_GB":torch.cuda.max_memory_allocated()/1e9,
        **label_metrics(val.y.to_numpy(),vp.argmax(1)),**probability_metrics(val.y.to_numpy(),vp)}
    records.append(row)
    if spec["name"] not in best_per_model or row["macro_f1"]>best_per_model[spec["name"]]["macro_f1"]:
        best_per_model[spec["name"]]=row
        folder=OUT/"models"/spec["name"];trainer.save_model(str(folder));tokenizer.save_pretrained(str(folder))
        (folder/"input_policy.json").write_text(json.dumps({"view":view,"family":spec["family"],"model_id":spec["id"],"revision":spec["revision"]},indent=2))
    np.savez_compressed(OUT/(run+".npz"),validation=vp,holdout=hp)
    pd.DataFrame(trainer.state.log_history).to_csv(OUT/(run+"_training.csv"),index=False)
    (OUT/"progress.json").write_text(json.dumps(records,indent=2,default=float))
    print(json.dumps(row,default=float),flush=True)
    checkpoint=Path(trainer.args.output_dir).resolve()
    del trainer,tokenizer,a,b,c;gc.collect();torch.cuda.empty_cache()
    assert checkpoint.is_relative_to((OUT/"checkpoints").resolve())
    shutil.rmtree(checkpoint)
''')
section('09 · Sepuluh fine-tuning','Urutan: native dan normalized untuk masing-masing lima model. Hyperparameter dasar sama, head klasifikasi baru untuk setiap run. Normalisasi dipilih per keluarga melalui validation macro-F1; tidak memilih lewat holdout. Ini pencarian terbatas, bukan pembuktian model tertentu selalu lebih unggul.', '''
print("Jumlah GPU:",torch.cuda.device_count(),"| eksperimen:",len(MODEL_SPECS)*2)
for spec in MODEL_SPECS:
    for view in ["native","normalized"]:
        print("START",spec["name"],view,flush=True)
        run_experiment(spec,view)
assert len(records)==10 and len(best_per_model)==5
validation_table=pd.DataFrame(records).sort_values("macro_f1",ascending=False)
show("Validation · 10 eksperimen",validation_table.drop(columns=["model_id","parameters"]))
show("Tokenization train per model/input",pd.DataFrame(token_audit))
''')
section('10 · Pilih input per model & audit pengaruh normalisasi','Selisih normalized-native dihitung dari validation. Nilai negatif berarti normalisasi merugikan pada run ini. Pilihan input tiap model, model dasar post-processing, dan urutan model dibekukan sebelum holdout metrics.', '''
normalization_comparison=pd.DataFrame(records).pivot(index="model",columns="view",values="macro_f1")
normalization_comparison["delta_normalized_minus_native"]=normalization_comparison.normalized-normalization_comparison.native
show("Apakah normalisasi membantu? · validation",normalization_comparison)
winners=sorted(best_per_model.values(),key=lambda row:row["macro_f1"],reverse=True)
selected_runs=[row["run"] for row in winners]
best_run=selected_runs[0]
stack_val=np.stack([predictions[run]["validation"] for run in selected_runs])
stack_hold=np.stack([predictions[run]["holdout"] for run in selected_runs])
base_val=predictions[best_run]["validation"];base_hold=predictions[best_run]["holdout"]
y_val=val.y.to_numpy()
show("Pemenang per keluarga · validation",pd.DataFrame(winners)[["model","view","macro_f1","NLL","fit_seconds"]])
print("Model dasar post-processing:",best_run)
''')
section('11 · Cross-fit enam teknik + gabungan','Semua metode memakai fold grup validation yang sama. Threshold/bias/ensemble/rule gate/temperature dipelajari pada bagian tune, dinilai pada bagian check. Duplicate consistency hanya menggabungkan probabilitas exact-text di prediction batch, tidak mengambil label train atau ID pengguna. Temperatur scalar mengubah confidence, bukan argmax. Aturan bisa tetap nonaktif bila dukungan tidak cukup.', '''
val_keys=val.safe.str.casefold().str.replace(r"\\s+"," ",regex=True).to_numpy()
hold_keys=holdout.safe.str.casefold().str.replace(r"\\s+"," ",regex=True).to_numpy()
val_rules=rule_targets(val.safe.tolist());hold_rules=rule_targets(holdout.safe.tolist())
METHODS=["raw","threshold","class_bias","ensemble","duplicate_consistency","rule_correction","temperature","ensemble_bias"]
oof_label={name:np.empty(len(val),dtype=int) for name in METHODS}
oof_prob={name:np.empty_like(base_val) for name in METHODS}
fold_stats=[]
for tune,check in StratifiedGroupKFold(POST_FOLDS,shuffle=True,random_state=SEED).split(val,y_val,val.dup_group):
    p,target=base_val[tune],y_val[tune];q=base_val[check]
    threshold=fit_threshold(p,target);bias=fit_bias(p,target);blend=fit_blend(stack_val[:,tune],target)
    gate=fit_rule_gate(p,target,val_rules[tune]);temperature=fit_temperature(p,target)
    blended_tune=mix(stack_val[:,tune],blend);blended_check=mix(stack_val[:,check],blend)
    combined_bias=fit_bias(blended_tune,target)
    probs={"raw":q,"threshold":q,"class_bias":bias_prob(q,bias),"ensemble":blended_check,
        "duplicate_consistency":duplicate_prob(q,val_keys[check]),"rule_correction":q,
        "temperature":temperature_prob(q,temperature),"ensemble_bias":bias_prob(blended_check,combined_bias)}
    for name,prob in probs.items():
        pred=threshold_predict(prob,threshold) if name=="threshold" else rule_predict(prob,val_rules[check],gate) if name=="rule_correction" else prob.argmax(1)
        oof_prob[name][check]=prob;oof_label[name][check]=pred
    fold_stats.append({"tune_n":len(tune),"check_n":len(check),"temperature":temperature,"gate":gate,
        "threshold":threshold.tolist(),"bias":bias.tolist(),"ensemble_weights":blend.tolist()})
''','''
post_validation=pd.DataFrame([{"method":name,**label_metrics(y_val,oof_label[name]),**probability_metrics(y_val,oof_prob[name])} for name in METHODS])
raw_score=float(post_validation.loc[post_validation.method.eq("raw"),"macro_f1"].iloc[0])
post_validation["delta_macro_f1"]=post_validation.macro_f1-raw_score
show("Post-processing · cross-fit validation",post_validation.sort_values("macro_f1",ascending=False))
print("Duplicate groups validation:",pd.Series(val_keys).value_counts().gt(1).sum(),"| baris rule match:",(val_rules>=0).sum())
assert np.array_equal(oof_label["raw"],oof_label["temperature"])
print("Temperature argmax tetap sama: PASS; perhatikan NLL/Brier/ECE, bukan berharap F1 naik.")
chosen=post_validation.sort_values("macro_f1",ascending=False).iloc[0]
chosen_method=chosen.method if chosen.macro_f1>raw_score+MIN_POST_GAIN else "raw"
print("Metode final dari cross-fit validation:",chosen_method)
''')
section('12 · Bekukan parameter post-processing','Setelah metode dipilih dari cross-fit validation, parameter metode dipasang sekali pada seluruh validation. Tidak ada label holdout dipakai untuk pemilihan bobot, threshold, bias, rule gate, atau temperature.', '''
full_threshold=fit_threshold(base_val,y_val);full_bias=fit_bias(base_val,y_val)
full_blend=fit_blend(stack_val,y_val);full_temperature=fit_temperature(base_val,y_val)
full_gate=fit_rule_gate(base_val,y_val,val_rules)
full_combined_bias=fit_bias(mix(stack_val,full_blend),y_val)
selection={"best_run":best_run,"selected_runs":selected_runs,"method":chosen_method,
    "thresholds":full_threshold.tolist(),"class_bias":full_bias.tolist(),"ensemble_weights":full_blend.tolist(),
    "temperature":full_temperature,"rule_gate":full_gate,"ensemble_bias":full_combined_bias.tolist(),"classes":classes.tolist()}
(OUT/"selection.json").write_text(json.dumps(selection,indent=2,default=float))
print("KEPUTUSAN DIBEKUKAN",json.dumps(selection,indent=2,default=float))
''')
section('13 · Holdout · semua model dan post-processing','Sekarang label holdout dipakai hanya untuk laporan. Seluruh tabel mempertahankan pilihan validation; tidak beralih ke metode yang kebetulan terbaik di holdout. NLL/Brier/ECE merujuk probabilitas argmax, sehingga threshold/rule yang hanya mengubah label tidak otomatis memperbaiki probabilitas.', '''
y_hold=holdout.y.to_numpy()
hold_raw=pd.DataFrame([{"model":row["model"],"view":row["view"],"run":row["run"],
    **label_metrics(y_hold,predictions[row["run"]]["holdout"].argmax(1)),
    **probability_metrics(y_hold,predictions[row["run"]]["holdout"])} for row in winners])
show("Lima model · input dipilih validation",hold_raw.drop(columns="run"))
all_hold=pd.DataFrame([{"run":row["run"],**label_metrics(y_hold,predictions[row["run"]]["holdout"].argmax(1))} for row in records])
show("Seluruh 10 run · tidak dipakai memilih input",all_hold)
post_hold_prob={"raw":base_hold,"threshold":base_hold,"class_bias":bias_prob(base_hold,full_bias),
    "ensemble":mix(stack_hold,full_blend),"duplicate_consistency":duplicate_prob(base_hold,hold_keys),
    "rule_correction":base_hold,"temperature":temperature_prob(base_hold,full_temperature),
    "ensemble_bias":bias_prob(mix(stack_hold,full_blend),full_combined_bias)}
post_hold_label={name:threshold_predict(prob,full_threshold) if name=="threshold" else
    rule_predict(prob,hold_rules,full_gate) if name=="rule_correction" else prob.argmax(1) for name,prob in post_hold_prob.items()}
''','''
post_hold=pd.DataFrame([{"method":name,**label_metrics(y_hold,post_hold_label[name]),**probability_metrics(y_hold,post_hold_prob[name])} for name in METHODS])
raw_hold_score=macro(y_hold,post_hold_label["raw"])
post_hold["delta_macro_f1_vs_raw"]=post_hold.macro_f1-raw_hold_score
show("Eksperimen post-processing · holdout",post_hold)
print("Rule match holdout:",(hold_rules>=0).sum(),"| label berubah karena rule:",(post_hold_label["rule_correction"]!=post_hold_label["raw"]).sum())
print("Duplikat exact-key holdout:",pd.Series(hold_keys).value_counts().gt(1).sum(),"grup")
final_pred=post_hold_label[chosen_method];final_prob=post_hold_prob[chosen_method]
show("Pilihan final sesuai validation",post_hold.loc[post_hold.method.eq(chosen_method)])
assert np.array_equal(post_hold_label["raw"],post_hold_label["temperature"])
''')
section('14 · Precision, recall, F1 & confusion','Tabel seluruh model/raw dan metode post-processing menunjukkan kelas yang dikorbankan atau terbantu. Confusion: baris aktual, kolom prediksi. Tidak ada grafik otomatis.', '''
per_class=[]
reported={row["model"]:predictions[row["run"]]["holdout"].argmax(1) for row in winners}
reported.update({"post_"+name:pred for name,pred in post_hold_label.items()})
for name,pred in reported.items():
    result=classification_report(y_hold,pred,labels=np.arange(C),target_names=classes,output_dict=True,zero_division=0)
    per_class.extend({"model/metode":name,"kelas":label,**result[label]} for label in classes)
per_class=pd.DataFrame(per_class)
show("Per-class metrics",per_class)
cm=pd.DataFrame(confusion_matrix(y_hold,final_pred,labels=np.arange(C)),index=classes,columns=classes)
show("Confusion final jumlah",cm)
show("Confusion final % per kelas aktual",cm.div(cm.sum(axis=1),axis=0)*100)
''')
section('15 · Analisis kelemahan & indikasi multilingual','Penanda regional/campur kode adalah indikasi, bukan label bahasa terverifikasi. Support kecil tidak cukup membuktikan NusaBERT/XLM-R unggul untuk bahasa tertentu. F1 slice hanya memakai kelas aktual yang hadir; selalu lihat n/error rate.', '''
analysis=holdout.copy();analysis["prediksi"]=encoder.inverse_transform(final_pred);analysis["salah"]=final_pred!=y_hold
analysis["prob_label_dipilih"]=final_prob[np.arange(len(final_pred)),final_pred]
analysis["panjang"]=pd.cut(analysis.safe.str.split().str.len(),[-1,5,15,40,100,np.inf])
for name in ["negasi","slang_contoh","url","emoji/simbol"]:
    analysis[name]=analysis[TEXT_COL].astype(str).str.count(NOISE[name],flags=re.I).gt(0)
sets=analysis.safe.str.casefold().str.findall(r"[^\\W\\d_]+").map(set)
analysis["regional_indikator"]=sets.map(lambda s:len(s&REGIONAL_MARKERS)>=2)
en=set("the this that with would should have because very really".split());ind=set("yang tidak bukan belum dengan untuk karena sudah".split())
analysis["campur_indikator"]=sets.map(lambda s:len(s&en)>=2 and len(s&ind)>=2)
if CONF_COL in analysis:analysis["confidence_label"]=pd.cut(pd.to_numeric(analysis[CONF_COL],errors="coerce"),[-np.inf,.6,.8,.95,np.inf])
slice_columns=["panjang","negasi","slang_contoh","url","emoji/simbol","regional_indikator","campur_indikator"]
if CONF_COL in analysis:slice_columns.append("confidence_label")
''','''
slice_rows=[]
for name,pred in {**{row["model"]:predictions[row["run"]]["holdout"].argmax(1) for row in winners},"final_selected":final_pred}.items():
    for column in slice_columns:
        for value,ids in analysis.groupby(column,observed=True,dropna=False).indices.items():
            target=y_hold[ids];p=pred[ids]
            slice_rows.append({"model":name,"slice":column,"nilai":str(value),"n":len(ids),"error_%":100*(p!=target).mean(),
                "macro_f1_kelas_hadir":f1_score(target,p,labels=np.unique(target),average="macro",zero_division=0)})
slices=pd.DataFrame(slice_rows)
show("Kelemahan pilihan final",slices[slices.model.eq("final_selected")])
show("Model per indikasi bahasa",slices[slices["slice"].isin(["regional_indikator","campur_indikator"])])
cols=[TEXT_COL,LABEL_COL,"prediksi","prob_label_dipilih"]+([CONF_COL] if CONF_COL in analysis else [])
show("Salah dengan probabilitas label pilihan tertinggi · bukan bukti label sumber benar",analysis[analysis.salah].sort_values("prob_label_dipilih",ascending=False)[cols].head(TOP))
''','''
raw_preds=np.stack([predictions[run]["holdout"].argmax(1) for run in selected_runs])
analysis["jumlah_label_berbeda"]=np.apply_along_axis(lambda p:len(set(p)),0,raw_preds)
show("Disagreement lima model",distribution(analysis.jumlah_label_berbeda))
show("Contoh tidak sepakat",analysis[analysis.jumlah_label_berbeda.ge(3)][cols+["jumlah_label_berbeda"]].head(TOP))
worst=per_class[per_class["model/metode"].eq("post_"+chosen_method)].sort_values("f1-score").iloc[0]
print("Kelas terlemah:",worst.kelas,"| F1:",round(worst["f1-score"],3),"| support:",int(worst.support))
print("Perubahan post-processing pada holdout:",round(macro(y_hold,final_pred)-raw_hold_score,4),"macro-F1; pilihan tidak diganti berdasarkan hasil ini.")
''')
section('16 · Ekspor bukti run & penggunaan ulang','Model terbaik/input policy tiap keluarga disimpan di hasil_v2/models. Lima model itu masih hanya dilatih pada train. Untuk inference baru, terapkan safe cleaning + input_policy.json yang sama; ensemble mengikuti selected_runs/weights dan urutan classes pada selection.json. Duplicate consistency bergantung pada batch inference. Untuk produksi, refit setelah eksperimen disepakati.', '''
validation_table.to_csv(OUT/"validation_runs.csv",index=False)
normalization_comparison.to_csv(OUT/"normalization_comparison.csv")
pd.DataFrame(token_audit).to_csv(OUT/"tokenization.csv",index=False)
post_validation.to_csv(OUT/"post_validation.csv",index=False);post_hold.to_csv(OUT/"post_holdout.csv",index=False)
hold_raw.to_csv(OUT/"holdout_models.csv",index=False);all_hold.to_csv(OUT/"holdout_all_runs.csv",index=False)
per_class.to_csv(OUT/"per_class.csv",index=False);slices.to_csv(OUT/"error_slices.csv",index=False)
pd.DataFrame(rule_audit).to_csv(OUT/"rule_audit.csv",index=False)
analysis[["row_id",TEXT_COL,LABEL_COL,"prediksi","salah","prob_label_dipilih"]].to_csv(OUT/"holdout_predictions.csv",index=False)
''','''
import importlib.metadata as metadata
manifest={"experiment":"multiclass_pretrained_v2","seed":SEED,"raw_rows":len(raw),"rows":len(data),
    "split_rows":{"train":len(train),"validation":len(val),"holdout":len(holdout)},"split_sha256":hashlib.sha256(split.sort_values("row_id").to_csv(index=False).encode()).hexdigest(),
    "dataset_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"model_specs":MODEL_SPECS,"runs_completed":len(records),
    "epochs":EPOCHS,"max_len":MAX_LEN,"learning_rate":LEARNING_RATE,"weight_power":WEIGHT_POWER,
    "selection":selection,"final_holdout":label_metrics(y_hold,final_pred),"final_probability_metrics":probability_metrics(y_hold,final_prob),
    "gpu":torch.cuda.get_device_name(0),"gpu_count":torch.cuda.device_count(),"elapsed_seconds":time.perf_counter()-RUN_START,
    "versions":{p:metadata.version(p) for p in ["torch","transformers","pandas","scikit-learn","tokenizers"]}}
assert len(records)==10 and len(winners)==5 and all(row["gpu_peak_GB"]>0 for row in records)
(OUT/"run_manifest.json").write_text(json.dumps(manifest,indent=2,default=float))
print(json.dumps(manifest,indent=2,default=float))
''')
md('''
## Sumber & interpretasi
- [IndoBERTweet](https://huggingface.co/indolem/indobertweet-base-uncased), [IndoBERT p1](https://huggingface.co/indobenchmark/indobert-base-p1), [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base), [IndoBERT IndoLEM](https://huggingface.co/indolem/indobert-base-uncased), [NusaBERT](https://huggingface.co/LazarusNLP/NusaBERT-base). Revision dipin di konfigurasi.
- [Model card IndoBERTweet](https://huggingface.co/indolem/indobertweet-base-uncased) menetapkan lowercase, @USER/HTTPURL, dan emoji ke teks. Perlakuan konservatif pada model lain adalah pilihan eksperimen; bukan klaim normalisasi wajib dari model card.
- [Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599): temperature scaling. Temperature positif tunggal menjaga argmax multiclass; manfaatnya dievaluasi lewat NLL/Brier/ECE.
- [scikit-learn leakage](https://scikit-learn.org/stable/common_pitfalls.html): seluruh keputusan/tuning dipisahkan dari laporan holdout. Cross-fit post-processing bukan jaminan validation bebas selection bias.
- Brier_sum = rata-rata jumlah squared error seluruh kelas; ECE argmax memakai 10 bin confidence. Rule/threshold yang tidak mengubah distribusi tidak mengubah NLL/Brier/ECE_argmax.
- Skor benchmark pseudo-label dan subset indikasi bahasa tidak membuktikan akurasi pada label manusia atau semua bahasa daerah. Repeated group CV/multiple seed dan test eksternal beranotasi manusia diperlukan untuk klaim generalisasi lebih kuat.
''')
# Last cell mirrors Kaggle's already-executed notebook into a downloadable artifact.
code('''
import nbformat
from IPython import get_ipython
executed=Path("/kaggle/working/__notebook__.ipynb")
if executed.exists():
    finished=nbformat.read(executed,as_version=4)
    finished.metadata.pop("papermill",None)
    for cell in finished.cells:cell.metadata.pop("papermill",None)
    finished.cells[-1].execution_count=get_ipython().execution_count
    nbformat.write(finished,"/kaggle/working/10_multi_kelas_v2_pretrained.ipynb")
''')
for cell in CELLS:
 if cell.cell_type=='code':ast.parse(cell.source)
notebook=nb.v4.new_notebook(cells=CELLS,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"},
    "experiment":{"version":2,"preprocessing_source_sha256":hashlib.sha256(u.encode()).hexdigest()}})
nb.validate(notebook);nb.write(notebook,DEST)
kernel=Path(tempfile.gettempdir())/'gemastik_multiclass_pretrained_v2_kernel';kernel.mkdir(exist_ok=True)
nb.write(notebook,kernel/'v2.ipynb')
meta={"id":"ivanwllm/gemastik-multiclass-pretrained-v2","title":"Gemastik Multiclass Pretrained V2","code_file":"v2.ipynb","language":"python","kernel_type":"notebook",
    "is_private":True,"enable_gpu":True,"machine_shape":"NvidiaTeslaT4","enable_internet":True,
    "dataset_sources":["dzikriraihan/indonesian-youtube-comments-mbg-program"],"kernel_sources":[],"competition_sources":[],"model_sources":[]}
(kernel/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
(DEST.parent/'kernel-metadata.json').write_text(json.dumps({**meta,'code_file':DEST.name},indent=2))
print('Wrote',DEST,'| code cells',sum(c.cell_type=='code' for c in CELLS),'| lines',sum(len(c.source.splitlines()) for c in CELLS if c.cell_type=='code'))
