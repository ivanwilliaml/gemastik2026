"""Author four task-specific notebooks; model execution takes place on Kaggle."""
from pathlib import Path
from textwrap import dedent
import ast, json, copy, tempfile, base64
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
books = {}
cells = []

def md(s): cells.append(nb.v4.new_markdown_cell(dedent(s).strip()))
def code(s): cells.append(nb.v4.new_code_cell(dedent(s).strip()))
def section(title, note, *sources):
    md('## '+title+'\n\n'+note)
    for source in sources: code(source)

SETUP = r'''
import os,sys,subprocess,time,json,re,html,unicodedata,random,gc
from pathlib import Path
from collections import Counter
assert Path('/kaggle/input').exists(),'Jalankan di Kaggle.'
os.environ.update(USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1')
subprocess.run([sys.executable,'-m','pip','install','-q','ftfy>=6.3,<7','rank-bm25>=0.2,<1','sentencepiece>=0.2,<1','langid==1.1.6'],check=True)
import numpy as np,pandas as pd,torch,ftfy
from IPython.display import display
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import f1_score,classification_report,confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import Ridge,LogisticRegression
from transformers import AutoTokenizer,AutoModel,AutoModelForSequenceClassification
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(2)
assert torch.cuda.is_available(),'Aktifkan GPU Kaggle.'
OUT=Path('/kaggle/working')/RESULT_NAME;OUT.mkdir(exist_ok=True)
pd.set_option('display.max_columns',None,'display.max_colwidth',120)
START=time.perf_counter()
'''
HELPERS = r'''
def show(title,table):
    print('\n'+title)
    display((table.to_frame() if isinstance(table,pd.Series) else table).round(4))

def distribution(values):
    n=values.value_counts(dropna=False)
    return pd.DataFrame({'n':n,'%':100*n/(len(values) or np.nan)})

def clean(value):
    if not isinstance(value,str): return ''
    repaired=ftfy.fix_text(value,unescape_html=False,uncurl_quotes=False,fix_latin_ligatures=False,
        fix_character_width=False,remove_control_chars=False,normalization=None)
    text=unicodedata.normalize('NFC',html.unescape(repaired))
    text=text.translate(dict.fromkeys(map(ord,'\ufeff\u200b\u2060\x00'),None))
    text=re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]',' ',text)
    return re.sub(r'\s+',' ',text).strip()

def key(value): return clean(value).casefold()
def cap(frame,n): return frame.sample(min(n,len(frame)),random_state=SEED).reset_index(drop=True) if n else frame.reset_index(drop=True)
def locate(folder,repo):
    matches=list(Path('/kaggle/input').glob('**/models/'+folder+'/config.json'))
    return str(matches[0].parent) if matches else repo
NOISE={'url':r'https?://|www\.', 'mention':r'@\w+','hashtag':r'#\w+','angka':r'\d',
 'emoji/simbol':r'[\U0001F300-\U0001FAFF\u2600-\u27BF]','huruf_ulang':r'([^\W\d_])\1{2,}',
 'baca_ulang':r'[!?.,]{2,}','html':r'</?[a-zA-Z][^>]*>','encoding':r'\ufffd|Ã.|Â.',
 'non_ascii':r'[^\x00-\x7f]','spasi_ulang':r' {2,}','invisible':r'[\u200b\ufeff]',
 'KAPITAL':r'(?-i:\b[A-Z]{3,}\b)','huruf_angka':r'\b(?=\w*[a-z])(?=\w*\d)\w+\b',
 'slang_contoh':r'\b(?:gk|gak|nggak|ga|bgt|yg|utk|udh|klo|jg)\b',
 'negasi':r'\b(?:tidak|tak|bukan|belum|jangan|tanpa|gak|nggak|ga|not|never)\b'}
'''
EDA = r'''
def text_eda(values,title):
    valid=values.where(values.map(lambda v:isinstance(v,str)),'').fillna('')
    show(title+' kualitas',pd.Series({'n':len(values),'null':values.isna().sum(),'kosong/spasi':valid.str.strip().eq('').sum(),
        'unik_terisi':valid[valid.str.strip().ne('')].nunique(),'duplikat_ekstra_terisi':valid[valid.str.strip().ne('')].duplicated().sum()}))
    show(title+' length',pd.DataFrame({'karakter':valid.str.len(),'kata':valid.str.split().str.len()}).describe(percentiles=[.01,.5,.9,.95,.99]).T)
    sample=valid[valid.str.strip().ne('')].sample(min(EDA_N,valid.str.strip().ne('').sum()),random_state=SEED)
    tokens=sample.str.casefold().str.findall(r'[^\W\d_]+')
    count=pd.Series(Counter(w for seq in tokens for w in seq),dtype='int64').sort_values(ascending=False)
    show(title+' vocabulary sampel',pd.Series({'n_sample':len(sample),'token':count.sum(),'unik':len(count),
        'hapax':count.eq(1).sum(),'hapax_%_tipe':100*count.eq(1).mean(),'hapax_%_token':100*count.eq(1).sum()/(count.sum() or np.nan)}))
    show(title+' top words',count.head(TOP).rename('n').to_frame().assign(pct=count.head(TOP)/(count.sum() or np.nan)*100))
    for size in [2,3]:
        grams=pd.Series(Counter(' '.join(seq[i:i+size]) for seq in tokens for i in range(len(seq)-size+1)),dtype='int64').sort_values(ascending=False)
        show(title+f' top {size}-gram',grams.head(TOP))
    flags=pd.DataFrame({name:valid.str.count(regex,flags=re.I).gt(0) for name,regex in NOISE.items()})
    show(title+' noise · % semua baris, dapat overlap',pd.DataFrame({'n':flags.sum(),'%':flags.mean()*100}))
    terms=count.index.to_series().astype('string')
    weird=terms.str.len().ge(20)|terms.str.count(r'([^\W\d_])\1{2,}').gt(0)
    show(title+' kandidat kata aneh, bukan pasti typo',count[weird].head(TOP))
    from langid.langid import LanguageIdentifier,model
    from threadpoolctl import threadpool_limits
    detector=LanguageIdentifier.from_modelstring(model,norm_probs=True)
    lang_sample=sample.sample(min(150,len(sample)),random_state=SEED)
    with threadpool_limits(limits=1):
        lang=lang_sample.map(lambda s:detector.classify(s[:3000]) if len(s.split())>=5 else ('tidak_pasti',np.nan))
    show(title+' estimasi bahasa · % sampel, bukan gold',distribution(lang.map(lambda p:p[0] if p[1]>=.8 else 'tidak_pasti')))
'''
PAIRS = r'''
def prepare_pairs(raw,a,b,target):
    show('Missing sebelum drop',pd.DataFrame({'n':raw.isna().sum(),'%':raw.isna().mean()*100}))
    frame=raw.copy();frame['a']=frame[a].map(clean);frame['b']=frame[b].map(clean)
    frame['y']=pd.to_numeric(frame[target],errors='coerce')
    ok=frame.a.ne('')&frame.b.ne('')&frame.y.notna()&np.isfinite(frame.y)
    print('Dibuang invalid:',int((~ok).sum()))
    return frame.loc[ok].reset_index(drop=True)

def endpoints(frame): return set(frame.a.map(key))|set(frame.b.map(key))
def purge(frame,blocked):
    keep=~frame.a.map(key).isin(blocked)&~frame.b.map(key).isin(blocked)
    print('Pasangan dikeluarkan karena teks lintas split:',int((~keep).sum()))
    return frame.loc[keep].reset_index(drop=True)

def pair_groups(frame):
    parent={}
    def root(x):
        parent.setdefault(x,x)
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for a,b in zip(frame.a.map(key),frame.b.map(key)):parent[root(a)]=root(b)
    return pd.factorize([root(x) for x in frame.a.map(key)])[0]

def pair_features(frame):
    rows=[]
    for a,b in zip(frame.a,frame.b):
        x,y=set(a.casefold().split()),set(b.casefold().split());union=x|y
        rows.append([len(x&y)/max(1,len(union)),len(x-y)/max(1,len(x)),len(y-x)/max(1,len(y)),len(a)/max(1,len(b)),
            int(bool(re.search(NOISE['negasi'],a,re.I)))!=int(bool(re.search(NOISE['negasi'],b,re.I))),
            len(set(re.findall(r'\d+',a))^set(re.findall(r'\d+',b)))])
    return np.asarray(rows,dtype='float32')
'''
EMBED = r'''
_encoder=None;_encoder_tok=None
@torch.inference_mode()
def embed(texts,prefix='query: '):
    global _encoder,_encoder_tok
    if _encoder is None:
        path=locate('e5','intfloat/multilingual-e5-base')
        _encoder_tok=AutoTokenizer.from_pretrained(path)
        _encoder=AutoModel.from_pretrained(path).cuda().eval()
    texts=list(texts);output=[]
    for start in range(0,len(texts),32):
        batch=_encoder_tok([prefix+t for t in texts[start:start+32]],padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda')
        with torch.autocast('cuda',dtype=torch.float16):hidden=_encoder(**batch).last_hidden_state
        mask=batch.attention_mask.unsqueeze(-1)
        pooled=(hidden.float()*mask).sum(1)/mask.sum(1).clamp(min=1)
        output.append(torch.nn.functional.normalize(pooled,dim=1).cpu().numpy())
    return np.concatenate(output)

def release_encoder():
    global _encoder,_encoder_tok
    _encoder=None;_encoder_tok=None;gc.collect();torch.cuda.empty_cache()
'''
TRAINER = r'''
def fine_tune_pairs(train,val,test,regression=False,label_names=('entailment','neutral','contradiction')):
    from datasets import Dataset,Value,disable_progress_bars
    from transformers import Trainer,TrainingArguments,DataCollatorWithPadding,set_seed
    from scipy.stats import spearmanr
    disable_progress_bars();set_seed(SEED);release_encoder()
    path=locate('indobert','indobenchmark/indobert-base-p1');tok=AutoTokenizer.from_pretrained(path)
    lengths=pd.Series([len(seq) for seq in tok(train.a.tolist(),train.b.tolist(),truncation=False)['input_ids']])
    show('Panjang token pasangan train',lengths.describe(percentiles=[.5,.9,.95,.99]))
    print('Train melewati PAIR_MAX_LEN (%):',lengths.gt(PAIR_MAX_LEN).mean()*100)
    def encode(frame,labels=True):
        payload={'a':frame.a.tolist(),'b':frame.b.tolist()}
        if labels:payload['labels']=frame.y.astype(float if regression else int).tolist()
        ds=Dataset.from_dict(payload)
        if labels and regression:ds=ds.cast_column('labels',Value('float32'))
        return ds.map(lambda v:tok(v['a'],v['b'],truncation=True,max_length=PAIR_MAX_LEN),batched=True,remove_columns=['a','b'])
    model=AutoModelForSequenceClassification.from_pretrained(path,num_labels=1 if regression else 3,
        problem_type='regression' if regression else 'single_label_classification',ignore_mismatched_sizes=True)
    if not regression:
        model.config.id2label=dict(enumerate(label_names));model.config.label2id={v:i for i,v in enumerate(label_names)}
    def metric(p):
        return {'selection':float(spearmanr(p.label_ids,p.predictions.ravel()).statistic)} if regression else {'selection':f1_score(p.label_ids,p.predictions.argmax(1),average='macro',zero_division=0)}
    args=TrainingArguments(output_dir=str(OUT/'checkpoints'),num_train_epochs=EPOCHS,learning_rate=2e-5,
        per_device_train_batch_size=16,per_device_eval_batch_size=32,fp16=True,eval_strategy='epoch',save_strategy='epoch',
        save_only_model=True,save_total_limit=1,load_best_model_at_end=True,metric_for_best_model='selection',greater_is_better=True,
        report_to='none',logging_strategy='epoch',disable_tqdm=True,seed=SEED)
    trainer=Trainer(model=model,args=args,train_dataset=encode(train),eval_dataset=encode(val),data_collator=DataCollatorWithPadding(tok),compute_metrics=metric)
    trainer.train();a=trainer.predict(encode(val,False)).predictions;b=trainer.predict(encode(test,False)).predictions
    trainer.save_model(str(OUT/'model_indobert'));tok.save_pretrained(str(OUT/'model_indobert'))
    del trainer,model;gc.collect();torch.cuda.empty_cache()
    return (a.ravel(),b.ravel()) if regression else (softmax(a,axis=1),softmax(b,axis=1))
'''
PAIR_INFER = r'''
@torch.inference_mode()
def pair_logits(repo,frame,folder=None,max_len=256):
    path=locate(folder,repo) if folder else repo
    tok=AutoTokenizer.from_pretrained(path);model=AutoModelForSequenceClassification.from_pretrained(path).cuda().eval()
    mapping={str(k).lower():int(v) for k,v in model.config.label2id.items()};output=[]
    for start in range(0,len(frame),32):
        batch=frame.iloc[start:start+32]
        enc=tok(batch.a.tolist(),batch.b.tolist(),padding=True,truncation=True,max_length=max_len,return_tensors='pt').to('cuda')
        with torch.autocast('cuda',dtype=torch.float16):logits=model(**enc).logits
        output.append(logits.float().cpu().numpy())
    del model;gc.collect();torch.cuda.empty_cache()
    return np.concatenate(output),mapping
'''

def base(title,result,config,pairs=False,training=False):
    global cells
    cells=[];md(title)
    section('00 · Konfigurasi','Ubah sumber/kolom/ukuran sampel di sini. Tidak ada grafik otomatis.',"SEED,TOP=42,15\nRESULT_NAME="+repr(result)+'\n'+config)
    section('01 · Lingkungan & preprocessing teknis','Safe cleaning: encoding, NFC, HTML entity, invisible, whitespace. Pertahankan angka, negasi, case, entitas dan bahasa. Tanpa stemming/stopword removal pada encoder.',SETUP,HELPERS,EDA)
    if pairs:code(PAIRS)
    code(EMBED)
    if training:code(TRAINER)
    code(PAIR_INFER)

def finish(relative,slug,datasets=()):
    md('''## Interpretasi & sumber
Semua angka dihitung saat run; pemilihan validation tidak diubah setelah holdout. E5 memakai query: di kedua sisi STS, query:/passage: pada retrieval. [E5](https://huggingface.co/intfloat/multilingual-e5-base), [IndoBERT](https://huggingface.co/indobenchmark/indobert-base-p1), [IndoNLI](https://github.com/ir-nlp-csui/indonli).''')
    code("import importlib.metadata as metadata\nmanifest={'task':RESULT_NAME,'seconds':time.perf_counter()-START,'gpu':torch.cuda.get_device_name(0),'seed':SEED,'versions':{p:metadata.version(p) for p in ['torch','transformers','scikit-learn','pandas']}}\n(OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2))\nprint(manifest)")
    for c in cells:
        if c.cell_type=='code':ast.parse(c.source)
    notebook=nb.v4.new_notebook(cells=copy.deepcopy(cells),metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
    dest=ROOT/relative;nb.validate(notebook);nb.write(notebook,dest)
    books[slug]={'relative':relative,'notebook':notebook,'datasets':list(datasets)}
    print('Wrote',dest,'| code cells',sum(c.cell_type=='code' for c in cells))

base('''# Semantic similarity · review
Skor kontinu 0–5, lima pendekatan. Data [STSB terjemahan Indonesia](https://huggingface.co/datasets/LazarusNLP/stsb_mt_id): validation resmi dipakai sebagai pool pengembangan karena tidak ada train resmi; test resmi dipertahankan. Terjemahan mesin dapat mengubah makna. Pearson/Spearman utama, MAE/RMSE untuk skala; parafrase hanyalah turunan dengan batas konfigurasi.''',
 'hasil_similarity_review', 'N_POOL,N_TEST,EDA_N=1500,1379,3000\nEPOCHS,PAIR_MAX_LEN=3,256\nPARAPHRASE_MIN=4.0',True,True)
section('02 · Data, grouping & overlap','Prioritas test resmi, lalu split komponen teks dari pool pengembangan. Semua teks identik di kedua sisi menjadi satu komponen; train tidak boleh berbagi endpoint dengan validation/test.',r'''
from datasets import load_dataset
data=load_dataset('LazarusNLP/stsb_mt_id')
pool=prepare_pairs(cap(data['validation'].to_pandas(),N_POOL),'text_1','text_2','correlation')
test=prepare_pairs(cap(data['test'].to_pandas(),N_TEST),'text_1','text_2','correlation')
assert pool.y.between(0,5).all() and test.y.between(0,5).all()
pool=purge(pool,endpoints(test));groups=pair_groups(pool)
a,b=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=SEED).split(pool,groups=groups))
train,val=pool.iloc[a].copy(),pool.iloc[b].copy()
assert endpoints(train).isdisjoint(endpoints(val)) and endpoints(train).isdisjoint(endpoints(test)) and endpoints(val).isdisjoint(endpoints(test))
assert min(len(train),len(val),len(test))>=30
print('Train/validation/test:',len(train),len(val),len(test))
''')
section('03 · EDA wajib pasangan & skor','Train saja. Symmetry relevan untuk similarity, berbeda dengan NLI yang berarah.',r'''
text_eda(train.a,'Sisi A');text_eda(train.b,'Sisi B')
show('Distribusi skor',train.y.describe(percentiles=[.1,.25,.5,.75,.9]))
show('Distribusi bin skor',distribution(pd.cut(train.y,[-.01,1,2,3,4,5])))
show('Fitur pasangan',pd.DataFrame(pair_features(train),columns=['Jaccard','A_only','B_only','rasio_panjang','negasi_berbeda','angka_berbeda']).describe().T)
print('Pasangan identik:',train.a.map(key).eq(train.b.map(key)).sum(),'| duplicate pair:',train.duplicated(['a','b']).sum())
if 'data' in train:show('Sumber train',distribution(train['data']))
''')
section('04 · Lima pendekatan','Semua parameter representasi/target dipelajari dari train. Model 1=mean, 2=TF-IDF cosine berkalibrasi linear train, 3=Ridge fitur simetris, 4=E5 cosine berkalibrasi train, 5=fine-tuned IndoBERT pair regressor.',r'''
from scipy.stats import pearsonr,spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
def metrics(y,p):
    return {'pearson':float(pearsonr(y,p).statistic) if np.std(p)>1e-10 else np.nan,
        'spearman':float(spearmanr(y,p).statistic) if np.std(p)>1e-10 else np.nan,
        'MAE':float(np.abs(y-p).mean()),'RMSE':float(np.sqrt(np.mean((y-p)**2)))}
pred_val={};pred_test={}
pred_val['01_mean']=np.full(len(val),train.y.mean());pred_test['01_mean']=np.full(len(test),train.y.mean())
vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=60000,sublinear_tf=True).fit(pd.concat([train.a,train.b]))
def cosine(frame):return np.asarray(vectorizer.transform(frame.a).multiply(vectorizer.transform(frame.b)).sum(1)).ravel()
fit=LinearRegression().fit(cosine(train)[:,None],train.y)
for frame,target in [(val,pred_val),(test,pred_test)]:target['02_tfidf']=fit.predict(cosine(frame)[:,None])
def symmetric(frame):
    reverse=frame.assign(a=frame.b,b=frame.a)
    return (pair_features(frame)+pair_features(reverse))/2
fit=Ridge(alpha=1).fit(symmetric(train),train.y)
pred_val['03_ridge']=fit.predict(symmetric(val));pred_test['03_ridge']=fit.predict(symmetric(test))
''',r'''
cosines=[]
for frame in [train,val,test]:cosines.append(np.sum(embed(frame.a)*embed(frame.b),axis=1))
fit=LinearRegression().fit(cosines[0][:,None],train.y)
pred_val['04_e5']=fit.predict(cosines[1][:,None]);pred_test['04_e5']=fit.predict(cosines[2][:,None])
# Pair augmentation hanya train. Prediction symmetry enforcement diuji terpisah.
aug=pd.concat([train,train.assign(a=train.b,b=train.a)],ignore_index=True)
both_test=pd.concat([test,test.assign(a=test.b,b=test.a)],ignore_index=True)
both_val=pd.concat([val,val.assign(a=val.b,b=val.a)],ignore_index=True)
vp,hp=fine_tune_pairs(aug,val,pd.concat([both_val,both_test],ignore_index=True),regression=True)
pred_val['05_indobert']=(hp[:len(val)]+hp[len(val):2*len(val)])/2
offset=2*len(val);pred_test['05_indobert']=(hp[offset:offset+len(test)]+hp[offset+len(test):])/2
''')
section('05 · Post-processing & pemilihan','Clipping 0–5; linear blend TF-IDF/E5/IndoBERT; isotonic calibration diuji cross-fit grup validation dan hanya dipakai jika RMSE membaik tanpa penurunan Spearman berarti. Holdout belum dinilai.',r'''
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
for name in list(pred_val):
    pred_val[name]=np.clip(pred_val[name],0,5);pred_test[name]=np.clip(pred_test[name],0,5)
pred_val['06_blend']=sum(pred_val[k] for k in ['02_tfidf','04_e5','05_indobert'])/3
pred_test['06_blend']=sum(pred_test[k] for k in ['02_tfidf','04_e5','05_indobert'])/3
board=pd.DataFrame([{'model':name,**metrics(val.y.to_numpy(),p)} for name,p in pred_val.items()]).sort_values('spearman',ascending=False)
show('Validation',board);chosen=board.iloc[0]['model'];cross=np.zeros(len(val))
cal_groups=pair_groups(val);assert len(np.unique(cal_groups))>=3,'Validation perlu >=3 komponen teks untuk cross-fit; tambah data/revisi split.'
for a,b in GroupKFold(3).split(val,groups=cal_groups):
    cross[b]=IsotonicRegression(out_of_bounds='clip').fit(pred_val[chosen][a],val.y.iloc[a]).predict(pred_val[chosen][b])
before=metrics(val.y.to_numpy(),pred_val[chosen]);after=metrics(val.y.to_numpy(),cross)
use_iso=after['RMSE']<before['RMSE'] and after['spearman']>=before['spearman']-.01
cal=IsotonicRegression(out_of_bounds='clip').fit(pred_val[chosen],val.y)
final=cal.predict(pred_test[chosen]) if use_iso else pred_test[chosen]
cuts=np.linspace(0,5,101);target=val.y.to_numpy()>=PARAPHRASE_MIN
cut=max(cuts,key=lambda c:f1_score(target,pred_val[chosen]>=c,zero_division=0))
selection={'model':chosen,'isotonic':bool(use_iso),'paraphrase_cut':float(cut),'gold_paraphrase_min':PARAPHRASE_MIN}
(OUT/'selection.json').write_text(json.dumps(selection,indent=2));print(selection)
''')
section('06 · Holdout & kelemahan','Sumber, overlap tinggi tetapi skor rendah, angka/negasi berbeda, error absolut dan bias rentang skor. Threshold parafrase dipilih dari validation pada skor RAW model pilihan, bukan pada skala isotonic.',r'''
table=pd.DataFrame([{'model':name,**metrics(test.y.to_numpy(),p)} for name,p in pred_test.items()]+[{'model':'final_selected',**metrics(test.y.to_numpy(),final)}])
show('Holdout',table);table.to_csv(OUT/'models.csv',index=False)
audit=test.assign(pred=final,error=np.abs(test.y-final),paraphrase_pred=pred_test[chosen]>=cut)
print('F1 parafrase:',f1_score(test.y>=PARAPHRASE_MIN,audit.paraphrase_pred,zero_division=0))
audit['overlap']=pair_features(test)[:,0];audit['bin_score']=pd.cut(audit.y,[-.01,1,2,3,4,5])
show('Bias menurut skor',audit.groupby('bin_score',observed=True)[['y','pred','error']].agg(['size','mean']))
if 'data' in audit:
    rows=[{'sumber':name,'n':len(g),**metrics(g.y.to_numpy(),g.pred.to_numpy())} for name,g in audit.groupby('data') if len(g)>=3]
    show('Per sumber',pd.DataFrame(rows))
show('High overlap, low gold',audit[(audit.overlap>.6)&(audit.y<2)][['a','b','y','pred']].head(TOP))
show('Error terbesar',audit.nlargest(TOP,'error')[['a','b','y','pred','error']])
audit.to_csv(OUT/'predictions.csv',index=False)
''')
finish('e_pasangan_teks/20_kemiripan/20_kemiripan_review.ipynb','similarity')

base('''# NLI · tiga kelas · review
Premise → hypothesis: entailment, neutral, contradiction. [IndoNLI](https://github.com/ir-nlp-csui/indonli) beranotasi manusia, dengan test_lay dan test_expert terpisah. Tidak mereduksi neutral menjadi contradiction. Lima pendekatan, ditambah audit hypothesis-only untuk mendeteksi shortcut.''',
 'hasil_nli3_review','N_TRAIN,N_VALID,N_TEST_EACH,EDA_N=6000,1500,1000,3000\nEPOCHS,PAIR_MAX_LEN=3,256',True,True)
section('02 · IndoNLI & label mapping eksplisit','Baca JSONL langsung dari repository resmi, tidak menjalankan dataset loading script. Test expert/lay dipertahankan terpisah dalam audit. Endpoint overlap dipurge dari train lalu validation. Pair order tidak ditukar.',r'''
CLASSES=['entailment','neutral','contradiction'];mapping={'e':0,'n':1,'c':2}
def load_nli(name,n):
    raw=pd.read_json('https://raw.githubusercontent.com/ir-nlp-csui/indonli/main/data/indonli/'+name+'.jsonl',lines=True)
    assert raw.label.isin(mapping).all(),'Label tidak dikenali; cek sumber.'
    raw['target']=raw.label.map(mapping)
    return prepare_pairs(cap(raw,n),'premise','hypothesis','target')
test=pd.concat([load_nli('test_lay',N_TEST_EACH).assign(source='lay'),load_nli('test_expert',N_TEST_EACH).assign(source='expert')],ignore_index=True)
val=purge(load_nli('val',N_VALID),endpoints(test))
train=purge(load_nli('train',N_TRAIN),endpoints(test)|endpoints(val))
for frame in [train,val,test]:frame['y']=frame.y.astype(int);assert set(frame.y)=={0,1,2}
assert endpoints(train).isdisjoint(endpoints(val)|endpoints(test)) and endpoints(val).isdisjoint(endpoints(test))
print('Train/validation/test:',len(train),len(val),len(test))
''')
section('03 · EDA wajib NLI','Negasi, angka, arah pasangan, hypothesis-only artifacts, label konflik dan truncation perlu diperhatikan. Tidak melakukan swap augmentation: entailment tidak simetris.',r'''
text_eda(train.a,'Premise');text_eda(train.b,'Hypothesis')
show('Kelas train',distribution(train.y.map(dict(enumerate(CLASSES)))))
features=pair_features(train)
show('Fitur per kelas',pd.DataFrame(features,columns=['overlap','premise_only','hyp_only','length_ratio','negation_diff','number_diff']).groupby(train.y).mean())
print('Konflik pair-label:',train.groupby(['a','b']).y.nunique().gt(1).sum())
show('Contoh tiap kelas',train.groupby('y')[['a','b']].head(3))
''')
section('04 · Lima model','Mayoritas, pair TF-IDF + LR, E5 pair features + LR, mDeBERTa NLI pretrained, IndoBERT fine-tune tiga kelas. Label model NLI pretrained harus dipetakan lewat config, tidak ditebak dari urutan logits.',r'''
from scipy.sparse import hstack
from sklearn.dummy import DummyClassifier
vp={};hp={};y=train.y.to_numpy()
dummy=DummyClassifier(strategy='most_frequent').fit(np.zeros((len(train),1)),y)
vp['01_majority']=dummy.predict_proba(np.zeros((len(val),1)));hp['01_majority']=dummy.predict_proba(np.zeros((len(test),1)))
tf=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=40000,sublinear_tf=True).fit(pd.concat([train.a,train.b]))
def lexical(frame):
    a,b=tf.transform(frame.a),tf.transform(frame.b)
    return hstack([a,b,abs(a-b),a.multiply(b)],format='csr')
lr=LogisticRegression(max_iter=1200,class_weight='balanced',random_state=SEED).fit(lexical(train),y)
vp['02_pair_tfidf_lr']=lr.predict_proba(lexical(val));hp['02_pair_tfidf_lr']=lr.predict_proba(lexical(test))
hyp=LogisticRegression(max_iter=1200,class_weight='balanced',random_state=SEED).fit(tf.transform(train.b),y)
print('Hypothesis-only validation macro-F1:',f1_score(val.y,hyp.predict(tf.transform(val.b)),average='macro'))
''',r'''
def dense_features(frame):
    a,b=embed(frame.a),embed(frame.b)
    return np.hstack([a,b,np.abs(a-b),a*b,pair_features(frame)])
model=LogisticRegression(max_iter=1200,class_weight='balanced',random_state=SEED).fit(dense_features(train),y)
vp['03_e5_lr']=model.predict_proba(dense_features(val));hp['03_e5_lr']=model.predict_proba(dense_features(test))
release_encoder()
for frame,target in [(val,vp),(test,hp)]:
    logits,label_map=pair_logits('MoritzLaurer/mDeBERTa-v3-base-mnli-xnli',frame,'mdeberta_nli')
    assert set(CLASSES).issubset(label_map),'Model label mapping tidak eksplisit.'
    target['04_nli_pretrained']=softmax(logits[:,[label_map[k] for k in CLASSES]],axis=1)
vp['05_indobert'],hp['05_indobert']=fine_tune_pairs(train,val,test)
''')
section('05 · Validation & post-processing','Ensemble rata-rata dan bias kelas. Bias diuji cross-fit komponen teks validation. Parameter final hanya di-fit validation, bukan test. Prediction test dihitung tanpa label oleh Trainer.',r'''
from sklearn.model_selection import StratifiedGroupKFold
def macro(y,p):return f1_score(y,p,labels=[0,1,2],average='macro',zero_division=0)
vp['ensemble']=sum(vp.values())/len(vp);hp['ensemble']=sum(hp.values())/len(hp)
board=pd.Series({name:macro(val.y,p.argmax(1)) for name,p in vp.items()}).sort_values(ascending=False)
show('Validation macro-F1',board);chosen=board.index[0]
def fit_bias(p,y):
    bias=np.zeros(3)
    for _ in range(2):
        for k in range(3):
            best=macro(y,(np.log(np.clip(p,1e-10,1))+bias).argmax(1))
            for value in [-.6,-.3,0,.3,.6]:
                candidate=bias.copy();candidate[k]=value
                score=macro(y,(np.log(np.clip(p,1e-10,1))+candidate).argmax(1))
                if score>best+1e-10:best,bias=score,candidate
    return bias
cross=np.zeros(len(val),int)
cal_groups=pair_groups(val);assert len(np.unique(cal_groups))>=3,'Validation perlu >=3 grup untuk cross-fit.'
for a,b in StratifiedGroupKFold(3,shuffle=True,random_state=SEED).split(val,val.y,cal_groups):
    bias=fit_bias(vp[chosen][a],val.y.to_numpy()[a]);cross[b]=(np.log(np.clip(vp[chosen][b],1e-10,1))+bias).argmax(1)
use_bias=macro(val.y,cross)>board.iloc[0]+.003
bias=fit_bias(vp[chosen],val.y.to_numpy()) if use_bias else np.zeros(3)
(OUT/'selection.json').write_text(json.dumps({'model':chosen,'bias':bias.tolist(),'classes':CLASSES},indent=2))
''')
section('06 · Test lay/expert & kelemahan','Tidak menyimpulkan kelas neutral sebagai tidak entailment saja. Confusion dan error slices memperlihatkan high-overlap contradiction, perubahan angka/negasi, serta pergeseran lay→expert.',r'''
final=(np.log(np.clip(hp[chosen],1e-10,1))+bias).argmax(1)
rows=[{'model':name,'macro_f1':macro(test.y,p.argmax(1))} for name,p in hp.items()]+[{'model':'final_selected','macro_f1':macro(test.y,final)}]
show('Test keseluruhan',pd.DataFrame(rows));pd.DataFrame(rows).to_csv(OUT/'models.csv',index=False)
show('Per kelas',pd.DataFrame(classification_report(test.y,final,labels=[0,1,2],target_names=CLASSES,output_dict=True,zero_division=0)).T)
show('Confusion · aktual baris',pd.DataFrame(confusion_matrix(test.y,final,labels=[0,1,2]),index=CLASSES,columns=CLASSES))
audit=test.assign(pred=final,error=final!=test.y,overlap=pair_features(test)[:,0])
show('Lay vs expert',pd.DataFrame([{'source':name,'n':len(g),'macro_f1':macro(g.y,g.pred)} for name,g in audit.groupby('source')]))
for i,label in enumerate(CLASSES):show(label+' errors',audit[(audit.y==i)&audit.error][['a','b','y','pred','source']].head(5))
show('High overlap contradiction',audit[(audit.y==2)&(audit.overlap>.6)][['a','b','pred']].head(TOP))
audit.to_csv(OUT/'predictions.csv',index=False)
''')
finish('e_pasangan_teks/21_nli/21_nli_review.ipynb','nli3')

base('''# Text matching / retrieval · review
Lima pendekatan: TF-IDF, BM25, E5 chunk retrieval, RRF hybrid, hybrid+cross-encoder rerank. Ditambah kalibrasi skor untuk keputusan match/tidak match.

Default [IndoSum](https://www.kaggle.com/datasets/linkgish/indosum): kalimat ringkasan menjadi kueri proxy, artikel asal menjadi satu dokumen relevan. Ini benchmark turunan yang memihak overlap kata, bukan evaluasi kueri manusia atau relevance judgment lengkap. Untuk data nyata, isi dua CSV pada konfigurasi. Corpus diketahui saat indexing; label kueri test tidak dipakai memilih metode.''',
 'hasil_retrieval_review',
 'N_DOCS,N_QUERIES,EDA_N=2000,400,3000\nRERANK_TOP,CHUNK_WORDS,MAX_CHUNKS=20,120,6\nCORPUS_CSV,QUERIES_CSV=None,None',False,False)
section('02 · Corpus & kueri','CSV opsional: corpus kolom doc_id,text; queries kolom query,relevant_ids (JSON list doc_id). Multi-relevant didukung metrik. Default membuang artikel identik, bukan membuang isi volume tanpa pelaporan.',r'''
if CORPUS_CSV and QUERIES_CSV:
    corpus=pd.read_csv(CORPUS_CSV);queries=pd.read_csv(QUERIES_CSV)
    assert {'doc_id','text'}.issubset(corpus) and {'query','relevant_ids'}.issubset(queries)
    corpus['text']=corpus.text.map(clean);assert corpus.doc_id.astype(str).is_unique
    lookup={str(x):i for i,x in enumerate(corpus.doc_id)}
    gold=[{lookup[str(x)] for x in json.loads(v)} for v in queries.relevant_ids]
    queries['query']=queries['query'].map(clean)
else:
    path=next(Path('/kaggle/input').rglob('test.01.jsonl'));raw=pd.read_json(path,lines=True)
    corpus=raw.copy();corpus['text']=corpus.paragraphs.map(lambda p:' '.join(' '.join(s) for para in p for s in para)).map(clean)
    corpus['query']=corpus.summary.map(lambda seq:' '.join(seq[0]) if seq else '').map(clean)
    print('Artikel duplicate ekstra:',corpus.text.map(key).duplicated().sum())
    corpus=cap(corpus.loc[corpus.text.ne('')&corpus['query'].str.split().str.len().ge(5)].drop_duplicates('text'),N_DOCS)
    corpus['doc_id']=np.arange(len(corpus));queries=cap(corpus[['doc_id','query']],N_QUERIES)
    gold=[{int(x)} for x in queries.doc_id]
assert all(gold) and queries['query'].ne('').all() and corpus.text.ne('').all()
query_groups=queries['query'].map(key)
v,t=next(GroupShuffleSplit(n_splits=1,test_size=.6,random_state=SEED).split(queries,groups=query_groups))
assert set(query_groups.iloc[v]).isdisjoint(query_groups.iloc[t])
print('Corpus:',len(corpus),'| validation/test queries:',len(v),len(t))
''')
section('03 · EDA retrieval wajib','Ukuran index, panjang dokumen/kueri, vocabulary/noise, gold per kueri, duplikat kueri, chunk coverage dan lexical shortcut.',r'''
text_eda(corpus.text,'Dokumen');text_eda(queries['query'],'Kueri')
show('Jumlah gold per kueri',distribution(pd.Series(list(map(len,gold)))))
print('Kueri duplicate:',queries['query'].map(key).duplicated().sum())
overlap=[len(set(q.casefold().split())&set(corpus.text.iloc[min(g)].casefold().split()))/max(1,len(set(q.casefold().split()))) for q,g in zip(queries['query'],gold)]
show('Porsi kata kueri muncul pada salah satu gold · shortcut audit',pd.Series(overlap).describe())
''')
section('04 · Metrik ranking & sanity check','Recall@k memakai jumlah relevant ID, bukan hit rate saja. NDCG memakai binary relevance dengan IDCG menurut jumlah gold. Kandidat yang tidak berisi gold tidak dapat diperbaiki oleh reranker.',r'''
def query_metrics(row,truth):
    hits=np.array([int(int(x) in truth) for x in row]);positions=np.flatnonzero(hits[:10])
    result={'MRR@10':1/(positions[0]+1) if len(positions) else 0.}
    for k in [1,5,10]:result[f'Recall@{k}']=hits[:k].sum()/len(truth)
    gain=(hits[:10]/np.log2(np.arange(min(10,len(row)))+2)).sum()
    ideal=(1/np.log2(np.arange(min(10,len(truth)))+2)).sum()
    result['NDCG@10']=gain/ideal
    return result
def ranking_metrics(rank,ids):
    return pd.DataFrame([query_metrics(rank[i],gold[i]) for i in ids]).mean().to_dict()
assert query_metrics(np.array([1,0,2,3,4]),{1,2})['Recall@1']==.5
assert query_metrics(np.array([1,0,2,3,4]),{1,2})['MRR@10']==1
def rank_scores(scores,n=50):return np.argsort(-scores,axis=1,kind='stable')[:,:min(n,scores.shape[1])]
''')
section('05 · TF-IDF, BM25 & chunk E5','Lexical index memakai semua dokumen kandidat, bukan label kueri. Encoder menjaga case/angka. Maksimal MAX_CHUNKS per dokumen; tail yang tidak dibaca dilaporkan. Similarity kueri-dokumen adalah max similarity chunk.',r'''
tokenize=lambda s:re.findall(r'[^\W_]+',s.casefold())
tf=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=100000).fit(corpus.text)
tf_scores=(tf.transform(queries['query'])@tf.transform(corpus.text).T).toarray()
from rank_bm25 import BM25Okapi
bm=BM25Okapi(corpus.text.map(tokenize).tolist())
bm_scores=np.stack([bm.get_scores(tokenize(q)) for q in queries['query']])
ranks={'01_tfidf':rank_scores(tf_scores),'02_bm25':rank_scores(bm_scores)}
chunks=[];owners=[]
for i,text in enumerate(corpus.text):
    words=text.split()
    for start in list(range(0,len(words),CHUNK_WORDS))[:MAX_CHUNKS]:chunks.append(' '.join(words[start:start+CHUNK_WORDS]));owners.append(i)
owners=np.asarray(owners);chunk_matrix=embed(chunks,prefix='passage: ');query_matrix=embed(queries['query'],prefix='query: ')
chunk_scores=query_matrix@chunk_matrix.T
dense_scores=np.full((len(queries),len(corpus)),-np.inf,dtype='float32')
for i,row in enumerate(chunk_scores):np.maximum.at(dense_scores[i],owners,row)
ranks['03_e5_chunks']=rank_scores(dense_scores)
release_encoder()
print('Chunks:',len(chunks),'| dokumen dengan tail tidak dibaca (%):',corpus.text.str.split().str.len().gt(CHUNK_WORDS*MAX_CHUNKS).mean()*100)
''')
section('06 · Hybrid dipilih validation & rerank','RRF alpha dipilih dari validation NDCG@10. Rerank hanya top-k kandidat hybrid, memakai chunk kandidat yang paling dekat menurut E5. Test labels belum dipakai.',r'''
def rrf(alpha):
    score=np.zeros((len(queries),len(corpus)))
    for weight,rank in [(1-alpha,ranks['02_bm25']),(alpha,ranks['03_e5_chunks'])]:
        for i,row in enumerate(rank):score[i,row]+=weight/(60+np.arange(len(row))+1)
    return rank_scores(score)
alphas=[0,.25,.5,.75,1]
alpha=max(alphas,key=lambda a:ranking_metrics(rrf(a),v)['NDCG@10'])
ranks['04_hybrid']=rrf(alpha)
pair_rows=[];candidate_ids=[]
for i,row in enumerate(ranks['04_hybrid']):
    for doc in row[:RERANK_TOP]:
        positions=np.flatnonzero(owners==doc);best=positions[np.argmax(chunk_scores[i,positions])]
        pair_rows.append({'a':queries['query'].iloc[i],'b':chunks[best]});candidate_ids.append((i,int(doc)))
logits,_=pair_logits('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',pd.DataFrame(pair_rows),'reranker',max_len=384)
assert logits.shape[1]==1,'Reranker output bukan satu relevance logit.'
raw_scores=logits[:,0].reshape(len(queries),-1)
reranked=[]
for i,row in enumerate(ranks['04_hybrid']):
    head=row[:RERANK_TOP];order=np.argsort(-raw_scores[i],kind='stable')
    reranked.append(np.r_[head[order],row[RERANK_TOP:]])
ranks['05_hybrid_rerank']=np.asarray(reranked)
dev=pd.DataFrame([{'method':name,**ranking_metrics(rank,v)} for name,rank in ranks.items()]).sort_values('NDCG@10',ascending=False)
show('Validation ranking',dev);chosen=dev.iloc[0]['method']
(OUT/'selection.json').write_text(json.dumps({'method':chosen,'alpha':alpha,'rerank_top':RERANK_TOP},indent=2))
''')
section('07 · Binary text matching · skor reranker','Kalibrasi satu logit reranker dengan LR, group cross-fit per kueri pada validation. Threshold dari OOF validation. Negatif adalah kandidat non-gold; pada benchmark relevansi tak lengkap, sebagian bisa sebenarnya relevan. Metrik kandidat bukan metrik seluruh corpus.',r'''
from sklearn.model_selection import GroupKFold
candidate_target=np.array([int(doc in gold[q]) for q,doc in candidate_ids])
candidate_query=np.array([q for q,doc in candidate_ids])
flat=raw_scores.ravel()[:,None];dev_mask=np.isin(candidate_query,v);test_mask=np.isin(candidate_query,t)
assert len(np.unique(candidate_target[dev_mask]))==2,'Validation candidates perlu positif dan negatif.'
oof=np.zeros(dev_mask.sum());dev_x,dev_y=flat[dev_mask],candidate_target[dev_mask]
for a,b in GroupKFold(3).split(dev_x,groups=candidate_query[dev_mask]):
    cal=LogisticRegression().fit(dev_x[a],dev_y[a]);oof[b]=cal.predict_proba(dev_x[b])[:,1]
cut=max(np.linspace(.05,.95,37),key=lambda c:f1_score(dev_y,oof>=c,zero_division=0))
cal=LogisticRegression().fit(dev_x,dev_y);match_prob=cal.predict_proba(flat[test_mask])[:,1]
show('Matching test candidates',pd.DataFrame(classification_report(candidate_target[test_mask],match_prob>=cut,output_dict=True,zero_division=0)).T)
(OUT/'matching_policy.json').write_text(json.dumps({'threshold':float(cut),'coef':cal.coef_.tolist(),'intercept':cal.intercept_.tolist()}))
''')
section('08 · Holdout ranking & kelemahan','Laporan per-kueri, category jika ada, answer-missing-in-candidates, dan contoh gagal. Pilihan validation dipertahankan walaupun test metode lain lebih tinggi.',r'''
board=pd.DataFrame([{'method':name,**ranking_metrics(rank,t)} for name,rank in ranks.items()])
show('Test ranking',board);board.to_csv(OUT/'models.csv',index=False)
ceiling=np.mean([bool(set(ranks['04_hybrid'][i,:RERANK_TOP])&gold[i]) for i in t])
print('Hit ceiling kandidat rerank:',ceiling,'| metode final:',chosen)
audit=pd.DataFrame([{'query_id':i,'query':queries['query'].iloc[i],'top_doc':int(ranks[chosen][i,0]),
    'gold':sorted(gold[i]),**query_metrics(ranks[chosen][i],gold[i])} for i in t])
show('Kueri gagal',audit.nsmallest(10,'MRR@10'))
audit.to_csv(OUT/'query_results.csv',index=False)
print('Keterbatasan: default gold proxy satu artikel asal, bukan relevance judgment lengkap; ulangi pada kueri manusia sebelum klaim performa pencarian nyata.')
''')
finish('e_pasangan_teks/22_retrieval/22_retrieval_review.ipynb','retrieval',['linkgish/indosum'])

base('''# Tren topik/sentimen per waktu · review
Volume asli, proporsi topik, sentimen model dan pergeseran distribusi. Empat pendekatan topik: NMF, LDA, TF-IDF KMeans, E5 KMeans. Tiga pendekatan sentimen: leksikon negasi-aware, classifier pretrained, NLI zero-shot.

Model topik fit periode awal, dipilih pada periode tengah, dan diaudit pada periode akhir. Seluruh label topik berasal dari model beku yang sama agar ID topik dapat dibandingkan. Sentimen tanpa gold hanyalah estimasi, agreement antar model bukan accuracy. Perubahan waktu tidak membuktikan sebab-akibat.''',
 'hasil_temporal_review',
 "TEXT_COL,TIME_COL='comment','published_at'\nN_ROWS,N_FIT,EDA_N,N_SENTIMENT=None,5000,3000,1500\nFREQ,TIMEZONE,MIN_PERIOD='W','Asia/Jakarta',30\nK_GRID=[4,6,8]\nDOMAIN_STOP=set()",False,False)
section('02 · Waktu, volume dan kualitas','Missing diaudit sebelum drop. Duplikat dipertahankan untuk volume; fitting topik memakai unique text periode awal. Jika N_ROWS dibatasi, volume yang dilaporkan adalah volume sampel, bukan seluruh sumber.',r'''
path=next(Path('/kaggle/input').rglob('komentar-mbg.csv'));raw=pd.read_csv(path)
assert {TEXT_COL,TIME_COL}.issubset(raw),raw.columns.tolist()
show('Missing sumber',pd.DataFrame({'n':raw.isna().sum(),'%':raw.isna().mean()*100}))
raw['text']=raw[TEXT_COL].map(clean);raw['time']=pd.to_datetime(raw[TIME_COL],errors='coerce',format='mixed',utc=True)
print('Invalid date:',raw.time.isna().sum(),'| teks kosong:',raw.text.eq('').sum())
data=cap(raw.loc[raw.time.notna()&raw.text.ne('')].copy(),N_ROWS).sort_values('time').reset_index(drop=True)
data['period']=data.time.dt.tz_convert(TIMEZONE).dt.tz_localize(None).dt.to_period(FREQ).dt.start_time
periods=sorted(data.period.unique());assert len(periods)>=8,'Perlu >=8 periode; sesuaikan FREQ/data.'
cut1,cut2=periods[int(.6*len(periods))],periods[int(.8*len(periods))]
early=data.period<cut1;middle=(data.period>=cut1)&(data.period<cut2);late=data.period>=cut2
fit=cap(data.loc[early].drop_duplicates('text'),N_FIT)
assert min(early.sum(),middle.sum(),late.sum())>=30
show('Volume per periode',data.groupby('period').agg(n=('text','size'),unik=('text',lambda s:s.map(key).nunique())))
print('Waktu:',data.time.min(),data.time.max(),'| training sebelum:',cut1,'| final sejak:',cut2)
''')
section('03 · EDA temporal wajib','Audit volume/unik, panjang/noise, distribusi hari/jam, kelengkapan periode, dan kosakata awal vs akhir. Tidak menghapus periode kecil diam-diam.',r'''
text_eda(data.text,'Teks seluruh rentang')
show('Hari lokal',distribution(data.time.dt.tz_convert(TIMEZONE).dt.dayofweek))
show('Jam lokal',distribution(data.time.dt.tz_convert(TIMEZONE).dt.hour))
volume=data.groupby('period').size().sort_index()
grid=pd.period_range(volume.index.min(),volume.index.max(),freq=FREQ).start_time
show('Periode + dukungan',volume.rename('n').to_frame().assign(cukup=volume>=MIN_PERIOD))
print('Missing bins (kalender):',len(set(grid)-set(volume.index)))
print('Duplikat ekstra seluruh rentang:',data.text.map(key).duplicated().sum())
''')
section('04 · Representasi topic vs sentiment','Sentimen/embedding memakai safe text. BoW memakai normalisasi slang terkurasi + stopword fungsi, tanpa membuang negasi/angka. DOMAIN_STOP kosong secara default; tinjau kata domain sebelum menambahkan.',r'''
SLANG={'yg':'yang','utk':'untuk','gak':'tidak','ga':'tidak','nggak':'tidak','tdk':'tidak','blm':'belum','bgt':'banget','udh':'sudah','klo':'kalau'}
STOP=set('yang dan di ke dari untuk pada dengan ini itu atau juga saya kami kita mereka dia oleh sebagai dalam para serta adalah sebuah'.split())|DOMAIN_STOP
STOP-=set('tidak tak bukan belum jangan tanpa kurang'.split())
def bow(text):
    text=re.sub(r'https?://\S+|@\w+',' ',text.casefold())
    words=[SLANG.get(w,w) for w in re.findall(r'[^\W_]+',text)]
    return ' '.join(w for w in words if w not in STOP and len(w)>1)
data['bow']=data.text.map(bow);fit['bow']=fit.text.map(bow)
cv=CountVectorizer(min_df=3,max_df=.95,max_features=20000)
counts_fit=cv.fit_transform(fit.bow);counts_all=cv.transform(data.bow)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf=TfidfTransformer(sublinear_tf=True).fit(counts_fit)
x_fit=tfidf.transform(counts_fit);x_all=tfidf.transform(counts_all);vocab=cv.get_feature_names_out()
show('Before/after',data[['text','bow']].head(5))
print('OOV/empty BoW per waktu:',pd.Series(np.asarray(counts_all.sum(1)).ravel()==0).groupby(data.period).mean().to_dict())
''')
section('05 · Empat pendekatan topik, pilih hanya validation','Coherence dihitung pada dokumen validation dan top words train dengan vocabulary yang sama. Hindari membuat matriks vocabulary²: co-occurence hanya untuk kata topik. Label kategori tidak diperlukan. TQ adalah proxy, bukan kebenaran topik.',r'''
from sklearn.decomposition import NMF,LatentDirichletAllocation
from sklearn.cluster import MiniBatchKMeans
def top_terms(labels):
    result={}
    for topic in sorted(set(labels)-{-1}):
        mass=np.asarray(counts_fit[labels==topic].sum(0)).ravel()
        result[int(topic)]=np.argsort(-mass)[:10]
    return result
def coherence(topics,presence):
    scores=[];n=presence.shape[0]
    for words in topics.values():
        x=(presence[:,words]>0).astype('int64');freq=np.asarray(x.sum(0)).ravel()/n;joint=(x.T@x).toarray()/n
        for i in range(len(words)):
            for j in range(i):
                p=joint[i,j]
                scores.append(-1. if p==0 else 1. if p>=1 else np.log(p/max(1e-12,freq[i]*freq[j]))/-np.log(p))
    return float(np.mean(scores)) if scores else np.nan
models={};candidate=[]
def evaluate(name,model,fit_labels,all_labels):
    terms=top_terms(fit_labels);flat=[int(w) for topic in terms.values() for w in topic]
    npmi=coherence(terms,counts_all[middle.to_numpy()]);diversity=len(set(flat))/max(1,len(flat))
    share=pd.Series(all_labels[middle.to_numpy()]).value_counts(normalize=True)
    candidate.append({'name':name,'n_topics_fit':len(terms),'NPMI_val':npmi,'diversity':diversity,'TQ':(npmi+1)/2*diversity,'max_share_val':share.max()})
    models[name]={'model':model,'labels':all_labels,'terms':terms}
''',r'''
for k in K_GRID:
    nmf=NMF(n_components=k,init='nndsvda',random_state=SEED,max_iter=300).fit(x_fit)
    pred=nmf.transform(x_all).argmax(1);pred[np.asarray(counts_all.sum(1)).ravel()==0]=-1
    evaluate(f'nmf_k{k}',nmf,nmf.transform(x_fit).argmax(1),pred)
    lda=LatentDirichletAllocation(n_components=k,learning_method='online',max_iter=8,random_state=SEED,n_jobs=2).fit(counts_fit)
    pred=lda.transform(counts_all).argmax(1);pred[np.asarray(counts_all.sum(1)).ravel()==0]=-1
    evaluate(f'lda_k{k}',lda,lda.transform(counts_fit).argmax(1),pred)
    km=MiniBatchKMeans(n_clusters=k,random_state=SEED,n_init=5,batch_size=512).fit(x_fit)
    pred=km.predict(x_all);pred[np.asarray(counts_all.sum(1)).ravel()==0]=-1
    evaluate(f'tfidf_kmeans_k{k}',km,km.labels_,pred)
fit_e=embed(fit.text);all_e=embed(data.text);release_encoder()
for k in K_GRID:
    km=MiniBatchKMeans(n_clusters=k,random_state=SEED,n_init=5,batch_size=512).fit(fit_e)
    evaluate(f'e5_kmeans_k{k}',km,km.labels_,km.predict(all_e))
board=pd.DataFrame(candidate);pool=board[board.max_share_val.le(.8)&board.TQ.notna()&board.n_topics_fit.ge(2)]
assert len(pool),'Semua topik dominan/invalid; tinjau preprocessing dan grid.'
chosen=pool.sort_values('TQ',ascending=False).iloc[0]['name']
show('Model topik validation',board.sort_values('TQ',ascending=False));print('Dipilih:',chosen)
(OUT/'selection.json').write_text(json.dumps({'topic_model':chosen,'selection_end':str(cut2)},indent=2))
''')
section('06 · Audit periode akhir & tren topik','Tidak refit topik pada seluruh rentang setelah evaluasi; ID topik tetap stabil. Persentase termasuk topik -1=unknown. Rolling bersifat trailing, bukan centered smoothing yang melihat masa depan.',r'''
selected=models[chosen];data['topic']=selected['labels']
show('Top words train',pd.DataFrame({str(i):vocab[terms] for i,terms in selected['terms'].items()}))
print('NPMI validation:',coherence(selected['terms'],counts_all[middle.to_numpy()]),'| akhir:',coherence(selected['terms'],counts_all[late.to_numpy()]))
shares=pd.crosstab(data.period,data.topic,normalize='index')
enough=data.groupby('period').size().ge(MIN_PERIOD)
show('Proporsi topik mentah (%)',shares*100)
smooth=shares.rolling(3,min_periods=1).mean().where(enough,axis=0)
show('Proporsi trailing 3 periode · bin kurang data=NaN',smooth*100)
from scipy.spatial.distance import jensenshannon
shift=pd.Series({shares.index[i]:jensenshannon(shares.iloc[i-1],shares.iloc[i],base=2) for i in range(1,len(shares))})
show('Perubahan komposisi terbesar · bukan kausalitas',shift.nlargest(TOP))
for topic in sorted(data.topic.unique()):show(f'Contoh topik {topic}',data.loc[data.topic.eq(topic),['time','text']].head(3))
shares.to_csv(OUT/'topic_share.csv');board.to_csv(OUT/'topic_models.csv',index=False)
''')
section('07 · Tiga estimasi sentimen','Sampel acak di setiap periode agar coverage waktu jelas. Leksikon hanya baseline sederhana. Model classifier dan NLI dapat salah domain/sarkasme; softmax bukan bukti label benar. Tidak mengisi baris yang belum dinilai menjadi netral.',r'''
quota=max(1,N_SENTIMENT//data.period.nunique())
sentiment=pd.concat([g.sample(min(quota,len(g)),random_state=SEED) for _,g in data.groupby('period')]).copy()
canonical=['negative','neutral','positive']
POS=set('bagus mantap baik setuju dukung sukses hebat enak'.split());NEG=set('buruk jelek gagal korupsi keracunan bohong rugi parah tolak'.split());NEGATE=set('tidak tak bukan belum jangan tanpa kurang ga gak nggak'.split())
def lexical_sent(text):
    words=re.findall(r'[^\W\d_]+',text.casefold());score=0
    for i,w in enumerate(words):
        s=int(w in POS)-int(w in NEG)
        score+=-s if set(words[max(0,i-3):i])&NEGATE else s
    return 2 if score>0 else 0 if score<0 else 1
pred={'lexicon':np.array([lexical_sent(t) for t in sentiment.text])}
repo='w11wo/indonesian-roberta-base-sentiment-classifier'
tok=AutoTokenizer.from_pretrained(repo);model=AutoModelForSequenceClassification.from_pretrained(repo).cuda().eval()
label_map={k.lower():v for k,v in model.config.label2id.items()};assert set(canonical).issubset(label_map)
prob=[]
with torch.inference_mode():
    for start in range(0,len(sentiment),32):
        enc=tok(sentiment.text.iloc[start:start+32].tolist(),padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda')
        with torch.autocast('cuda',dtype=torch.float16):logits=model(**enc).logits
        prob.append(softmax(logits.float().cpu().numpy()[:,[label_map[k] for k in canonical]],axis=1))
prob=np.concatenate(prob);pred['sentiment_pretrained']=prob.argmax(1)
del model;gc.collect();torch.cuda.empty_cache()
''',r'''
hypotheses=['Teks ini menyampaikan keluhan, kritik, atau penolakan.','Teks ini hanya menyampaikan informasi netral.','Teks ini menyampaikan pujian, kepuasan, atau dukungan.']
pairs=pd.DataFrame([{'a':text,'b':hyp} for text in sentiment.text for hyp in hypotheses])
logits,mapping=pair_logits('MoritzLaurer/mDeBERTa-v3-base-mnli-xnli',pairs,'mdeberta_nli')
assert 'entailment' in mapping
nli_prob=softmax(logits[:,mapping['entailment']].reshape(len(sentiment),3),axis=1)
pred['nli_zero_shot']=nli_prob.argmax(1)
for name,value in pred.items():sentiment[name]=value
sentiment['polarity_pretrained']=prob[:,2]-prob[:,0]
sentiment['polarity_nli']=nli_prob[:,2]-nli_prob[:,0]
show('Coverage sentimen · n sumber vs n dinilai',data.groupby('period').size().rename('n_sumber').to_frame().join(sentiment.groupby('period').size().rename('n_dinilai')))
show('Agreement, bukan accuracy',pd.DataFrame({a:{b:np.mean(pred[a]==pred[b]) for b in pred} for a in pred}))
''')
section('08 · Tren, ketidakpastian sampel & kelemahan','Interval mean adalah sampling SE untuk skor model, bukan interval akurasi sentimen. Komentar duplikat/serial dependence dapat membuatnya terlalu sempit; n_unique disertakan. Periode kecil tetap terlihat dengan flag.',r'''
rows=[]
for period,g in sentiment.groupby('period'):
    n=len(g);mean=g.polarity_pretrained.mean();se=g.polarity_pretrained.std()/np.sqrt(n)
    rows.append({'period':period,'n_dinilai':n,'n_unique':g.text.map(key).nunique(),'n_sumber':int((data.period==period).sum()),
        'negatif_%':g.sentiment_pretrained.eq(0).mean()*100,'netral_%':g.sentiment_pretrained.eq(1).mean()*100,
        'positif_%':g.sentiment_pretrained.eq(2).mean()*100,'mean_polarity':mean,'sampling_lo':mean-1.96*se,'sampling_hi':mean+1.96*se,
        'agreement_nli':g.sentiment_pretrained.eq(g.nli_zero_shot).mean(),'cukup_sampel':n>=MIN_PERIOD})
trend=pd.DataFrame(rows).set_index('period').sort_index()
trend['trailing_3']=trend.mean_polarity.rolling(3,min_periods=1).mean()
show('Tren sentimen & dukungan',trend)
show('Perubahan skor terbesar',trend.mean_polarity.diff().abs().nlargest(5))
show('Contoh model tidak sepakat',sentiment.loc[sentiment.sentiment_pretrained.ne(sentiment.nli_zero_shot),['time','text','sentiment_pretrained','nli_zero_shot']].head(TOP))
trend.to_csv(OUT/'sentiment_trend.csv');sentiment.to_csv(OUT/'sentiment_sample.csv',index=False)
print('Tidak mengklaim model paling akurat tanpa gold label; gunakan contoh tidak sepakat untuk audit manusia.')
''')
finish('a_tanpa_label/02_tren_topik_waktu/02_tren_topik_waktu_review.ipynb','temporal',['neevvv/indonesia-mbg-sentiment-analysis'])

# Local sources/configs, one remote runner to avoid simultaneous GPU quota issues.
runner_dir=ROOT/'_build/pair_time_kaggle';runner_dir.mkdir(exist_ok=True)
payloads={name:base64.b64encode(nb.writes(item['notebook']).encode()).decode() for name,item in books.items()}
sources=sorted({source for item in books.values() for source in item['datasets']})
runner='''import base64,pathlib,json,traceback,os
import papermill as pm
os.environ['PYTHONHASHSEED']='42'
TASKS=PAYLOADS
report={}
for name,payload in TASKS.items():
    source=pathlib.Path('/kaggle/working')/('_source_'+name+'.ipynb')
    source.write_bytes(base64.b64decode(payload))
    try:
        pm.execute_notebook(str(source),'/kaggle/working/'+name+'_review.ipynb',kernel_name='python3',cwd='/kaggle/working',progress_bar=False,log_output=True)
        report[name]={'status':'complete'}
    except Exception as error:
        traceback.print_exc();report[name]={'status':'error','error':str(error)[-2000:]}
    finally:
        source.unlink(missing_ok=True)
        pathlib.Path('/kaggle/working/suite_status.json').write_text(json.dumps(report,indent=2))
if any(v['status']=='error' for v in report.values()):raise RuntimeError('Sebagian notebook gagal; lihat suite_status.json dan output masing-masing.')
'''.replace('PAYLOADS',repr(payloads))
runner_nb=nb.v4.new_notebook(cells=[nb.v4.new_code_cell(runner)],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
nb.write(runner_nb,runner_dir/'runner.ipynb')
meta={'id':'ivanwllm/gemastik-pair-time-review','title':'Gemastik Pair Time Review','code_file':'runner.ipynb','language':'python','kernel_type':'notebook',
      'is_private':True,'enable_gpu':True,'machine_shape':'NvidiaTeslaT4','enable_internet':True,'dataset_sources':sources,
      'kernel_sources':['ivanwllm/gemastik-nlp-00-setup-offline'],'competition_sources':[],'model_sources':[]}
(runner_dir/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
(runner_dir/'suite_files.json').write_text(json.dumps({k:v['relative'] for k,v in books.items()},indent=2),encoding='utf-8')
