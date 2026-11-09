"""Author a self-contained multilabel pilot; training and inference run on Kaggle."""
from pathlib import Path
from textwrap import dedent
import ast,copy,json,base64
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'b_klasifikasi/12_multi_label/12_multi_label_review.ipynb'
tree=ast.parse((ROOT/'_build/build_pair_time_review.py').read_text(encoding='utf-8'))
shared={node.targets[0].id:ast.literal_eval(node.value) for node in tree.body
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
        and node.targets[0].id in {'SETUP','HELPERS','EDA','EMBED'}}
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(dedent(s).strip()))
def code(s):cells.append(nb.v4.new_code_cell(dedent(s).strip()))
def section(title,note,*sources):
    md('## '+title+'\n\n'+note)
    for source in sources:code(source)

md('''# Multilabel · review
Satu teks dapat memiliki beberapa label, atau tidak memiliki label aktif. **Sigmoid independen**, bukan softmax/argmax. Missing label adalah unknown, bukan otomatis 0.

Lima pendekatan: semua-negatif, word TF-IDF + binary relevance LR, word+char TF-IDF + LR, E5 + LR, dan IndoBERTweet fine-tuning masked BCE. Post-processing: threshold 0,5 vs global vs per-label, probability ensemble, serta pemeriksaan hierarki HS.

Data latihan: [Indonesian abusive and hate speech Twitter](https://www.kaggle.com/datasets/ilhamfp31/indonesian-abusive-and-hate-speech-twitter-text). Tweet memuat kata kasar; kata tersebut dipertahankan sebagai informasi task, bukan disensor. Ini klasifikasi ujaran kebencian, bukan otomatis analisis sentimen MBG. Ganti kolom/label untuk domain lain.

Grup exact duplicate dibentuk sebelum split dan distratifikasi multilabel pada level grup. Semua model memakai split yang sama. Threshold dipilih lewat cross-fit validation, holdout baru dinilai setelah keputusan dibekukan. Tetap ada selection bias validation; repeated group split diperlukan untuk klaim peningkatan kecil.''')
section('00 · Konfigurasi','LABEL_COLS dapat diganti. N_ROWS=None memakai seluruh sumber. Nilai label harus 0/1 atau missing; angka lain dianggap kesalahan skema.',r'''
SEED,TOP,EDA_N=42,15,3000
RESULT_NAME='hasil_multilabel_review'
DATA_FILE=None
TEXT_COL='Tweet'
LABEL_COLS=['HS','Abusive','HS_Individual','HS_Group','HS_Religion','HS_Race','HS_Physical','HS_Gender','HS_Other','HS_Weak','HS_Moderate','HS_Strong']
N_ROWS,EPOCHS,MAX_LEN=10000,3,192
THRESHOLD_MIN_POS,THRESHOLD_MIN_NEG,MIN_GAIN=5,5,.003
GRID=[.01,.03,.05,.08,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.7,.8,.9,.95]
''')
section('01 · Lingkungan','Training/inference hanya di Kaggle GPU. Helper EDA memakai hitungan/tabel, tanpa grafik.',
        shared['SETUP'].replace("'langid==1.1.6'","'langid==1.1.6','iterative-stratification>=0.1,<1'"),shared['HELPERS'],shared['EDA'],shared['EMBED'])
section('02 · Data mentah, partial labels & preprocessing','Byte escape emoji dipulihkan hanya jika UTF-8 valid. Simpan raw text. USER/URL adalah placeholder dataset; input IndoBERTweet disesuaikan ke @USER/HTTPURL. Tidak melakukan stemming, membuang negasi, emoji, atau kata kasar.',r'''
if DATA_FILE:path=Path(DATA_FILE)
else:
    paths=[p for p in Path('/kaggle/input').rglob('data.csv') if 'indonesian-abusive-and-hate-speech-twitter-text' in str(p)]
    assert len(paths)==1,f'Isi DATA_FILE secara eksplisit; kandidat={paths}'
    path=paths[0]
raw=pd.read_csv(path,encoding='latin-1')
assert {TEXT_COL,*LABEL_COLS}.issubset(raw.columns),raw.columns.tolist()
show('Missing sumber sebelum drop',pd.DataFrame({'n':raw.isna().sum(),'%':raw.isna().mean()*100}))
show('Tipe sumber',raw.dtypes.astype(str))
raw['row_id']=np.arange(len(raw));raw['text']=raw[TEXT_COL].map(clean)
raw[LABEL_COLS]=raw[LABEL_COLS].apply(pd.to_numeric,errors='coerce')
assert raw[LABEL_COLS].stack().isin([0,1]).all(),'Label nonmissing bukan 0/1.'
print('Teks kosong dibuang:',raw.text.eq('').sum(),'| seluruh label unknown:',raw[LABEL_COLS].isna().all(axis=1).sum())
data=cap(raw.loc[raw.text.ne('')&raw[LABEL_COLS].notna().any(axis=1)].copy(),N_ROWS)
''',r'''
import emoji
def decode_bytes(match):
    try:return bytes.fromhex(match[0].replace('\\x','')).decode('utf-8')
    except UnicodeDecodeError:return match[0]
def safe_tweet(text):return clean(re.sub(r'(?:\\x[0-9a-fA-F]{2})+',decode_bytes,text).replace('\\n',' '))
data['safe']=data.text.map(safe_tweet)
data=data[data.safe.ne('')].reset_index(drop=True)
data['minimal']=data.safe.str.casefold()
def tweet_recipe(text):
    text=re.sub(r'https?://\S+|www\.\S+','HTTPURL',text.lower())
    text=re.sub(r'(?<!\w)@\w+','@USER',text)
    text=re.sub(r'\buser\b','@USER',text);text=re.sub(r'\burl\b','HTTPURL',text)
    return clean(emoji.demojize(text))
data['bert']=data.safe.map(tweet_recipe)
data['group']=pd.factorize(data.minimal.str.replace(r'\s+',' ',regex=True).str.strip())[0]
show('Before/after',data[[TEXT_COL,'safe','bert']].head(5))
''')
section('03 · Iterative stratification pada grup','Profil grup memakai any-positive dan any-known per label. Unknown diberi 0 hanya pada profil stratifikasi, tidak pada target pelatihan/evaluasi. Target yang konflik pada teks identik ditampilkan, bukan diubah ke mayoritas diam-diam.',r'''
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit,MultilabelStratifiedKFold
group_positive=data.groupby('group')[LABEL_COLS].max().fillna(0)
group_known=data[LABEL_COLS].notna().groupby(data.group).max().astype(int)
profile=np.hstack([group_positive.to_numpy(),group_known.to_numpy()]);ids=group_positive.index.to_numpy()
rest,held=next(MultilabelStratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(ids,profile))
fit,dev=next(MultilabelStratifiedShuffleSplit(n_splits=1,test_size=.25,random_state=SEED+1).split(ids[rest],profile[rest]))
train=data[data.group.isin(ids[rest][fit])].copy();val=data[data.group.isin(ids[rest][dev])].copy();test=data[data.group.isin(ids[held])].copy()
for a,b in [(train,val),(train,test),(val,test)]:assert set(a.group).isdisjoint(b.group)
Y=train[LABEL_COLS].to_numpy(float);V=val[LABEL_COLS].to_numpy(float);T=test[LABEL_COLS].to_numpy(float);L=len(LABEL_COLS)
show('Positif per split',pd.DataFrame({name:frame[LABEL_COLS].eq(1).sum() for name,frame in [('train',train),('validation',val),('test',test)]}))
show('Konflik label dalam grup teks',data.groupby('group')[LABEL_COLS].nunique().gt(1).sum())
split=pd.concat([f[['row_id','group']].assign(split=name) for name,f in [('train',train),('validation',val),('test',test)]])
split.to_csv(OUT/'split.csv',index=False)
''')
section('04 · EDA wajib multilabel','Persen aktif memakai label diketahui per kolom. Kardinalitas/density/subset yang membutuhkan semua label hanya dihitung pada baris lengkap. Ko-occurence memakai penyebut pairwise-known; missing tidak dijadikan negatif.',r'''
text_eda(train.safe,'Train')
show('Support per label',pd.DataFrame({'known':train[LABEL_COLS].notna().sum(),'missing':train[LABEL_COLS].isna().sum(),
    'positif':train[LABEL_COLS].eq(1).sum(),'negatif':train[LABEL_COLS].eq(0).sum(),'%_aktif_known':train[LABEL_COLS].mean()*100}))
complete=train[LABEL_COLS].dropna()
show('Kardinalitas baris lengkap',distribution(complete.sum(axis=1)))
print('n complete:',len(complete),'| mean cardinality:',complete.sum(axis=1).mean(),'| density:',complete.to_numpy().mean())
co=pd.DataFrame(index=LABEL_COLS,columns=LABEL_COLS,dtype=float)
for a in LABEL_COLS:
    for b in LABEL_COLS:
        known=train[[a,b]].notna().all(axis=1)
        co.loc[a,b]=(train.loc[known,a].eq(1)&train.loc[known,b].eq(1)).mean()*100
show('Co-occurence · % pairwise-known',co)
show('Kombinasi label lengkap teratas',complete.value_counts().head(TOP).rename('n').to_frame())
''')
section('05 · Metrik masked & sanity checks','Macro-F1 per label diketahui, micro-F1 seluruh sel diketahui, sample-F1/exact match hanya baris complete. Label tanpa positif membuat AP undefined; laporkan NaN, jangan menyebut AP tinggi pada label yang tak bisa dinilai.',r'''
from sklearn.metrics import average_precision_score,precision_recall_fscore_support
def targets(y):
    y=np.asarray(y,dtype=float).copy();y[y<0]=np.nan
    return y
def evaluate(y,p,threshold=.5):
    y=targets(y);known=np.isfinite(y);pred=p>=threshold;per=[];ap=[]
    for j in range(y.shape[1]):
        mask=known[:,j]
        per.append(f1_score(y[mask,j],pred[mask,j],zero_division=0) if mask.any() else np.nan)
        ap.append(average_precision_score(y[mask,j],p[mask,j]) if np.any(y[mask,j]==1) else np.nan)
    complete=known.all(axis=1)
    return {'macro_f1':float(np.nanmean(per)),'micro_f1':f1_score(y[known],pred[known],zero_division=0),
        'macro_AP':float(np.nanmean(ap)) if np.isfinite(ap).any() else np.nan,
        'Hamming_known':float(np.mean(y[known]!=pred[known])),
        'samples_f1_complete':f1_score(y[complete],pred[complete],average='samples',zero_division=0) if complete.any() else np.nan,
        'exact_match_complete':float(np.all(y[complete]==pred[complete],axis=1).mean()) if complete.any() else np.nan}
toy=np.array([[1.,np.nan],[0.,1.]])
assert evaluate(toy,np.array([[.9,.9],[.1,.9]]))['micro_f1']==1
assert evaluate(np.array([[1.,0.],[0.,1.]]),np.array([[.9,.1],[.1,.9]]))['macro_f1']==1
print('Sanity check masking: PASS')
''')
section('06 · Binary relevance: diketahui per label','Setiap classifier hanya memakai baris dengan target label tersebut diketahui. Label train yang konstan memakai predictor konstan; label tanpa observasi memakai prior 0,5 dan ditandai, bukan dianggap sudah dipelajari.',r'''
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_union
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
def fit_binary(x,y):
    fitted=[]
    for j in range(y.shape[1]):
        known=np.isfinite(y[:,j]);values=np.unique(y[known,j])
        if not len(values):fitted.append(None);continue
        model=DummyClassifier(strategy='constant',constant=int(values[0])) if len(values)==1 else LogisticRegression(C=2,max_iter=1200,class_weight='balanced',random_state=SEED)
        fitted.append(model.fit(x[known],y[known,j].astype(int)))
    return fitted
def predict_binary(models,x):
    result=np.zeros((x.shape[0],len(models)))
    for j,model in enumerate(models):
        if model is None:result[:,j]=.5
        elif 1 in model.classes_:result[:,j]=model.predict_proba(x)[:,list(model.classes_).index(1)]
    return result
vp={'01_all_negative':np.zeros_like(V)};hp={'01_all_negative':np.zeros_like(T)};times=[]
print('Label train tanpa positif:',[LABEL_COLS[j] for j in range(L) if not np.any(Y[:,j]==1)])
print('Label train tanpa known:',[LABEL_COLS[j] for j in range(L) if not np.isfinite(Y[:,j]).any()])
''')
section('07 · Model 2–3: TF-IDF word & word+char','Vocabulary hanya fit pada train. Word+char menangkap obfuscation/ejaan dan emoji tanpa memaksa kamus koreksi. Dua representasi memakai binary relevance LR yang sama; perbandingan ini bukan eksperimen algoritme saja.',r'''
word=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=60000,sublinear_tf=True)
word_char=make_union(TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=50000,sublinear_tf=True),
    TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=80000,sublinear_tf=True))
for name,vectorizer in [('02_word_lr',word),('03_word_char_lr',word_char)]:
    start=time.perf_counter();x=vectorizer.fit_transform(train.minimal)
    models=fit_binary(x,Y)
    vp[name]=predict_binary(models,vectorizer.transform(val.minimal));hp[name]=predict_binary(models,vectorizer.transform(test.minimal))
    times.append({'model':name,'fit_and_predict_seconds':time.perf_counter()-start})
''')
section('08 · Model 4: E5 + binary relevance LR','E5 di-freeze; classifier per label tetap fit hanya train dengan mask target. Tidak memakai class softmax karena label dapat aktif bersamaan.',r'''
start=time.perf_counter()
train_e,val_e,test_e=[embed(frame.safe) for frame in [train,val,test]]
models=fit_binary(train_e,Y)
vp['04_e5_lr']=predict_binary(models,val_e);hp['04_e5_lr']=predict_binary(models,test_e)
times.append({'model':'04_e5_lr','fit_and_predict_seconds':time.perf_counter()-start})
release_encoder()
''')
section('09 · Model 5: IndoBERTweet masked BCE','Unknown menjadi sentinel -1 pada batch dan tidak ikut loss. Pos_weight berasal dari jumlah positif/negatif diketahui di train, akar rasio dibatasi 20. Checkpoint dipilih validation macro-F1 pada threshold 0,5. Threshold dipelajari sesudah training.',r'''
from transformers import Trainer,TrainingArguments,DataCollatorWithPadding,set_seed
from datasets import Dataset,disable_progress_bars
disable_progress_bars();set_seed(SEED)
path_model=locate('indobertweet','indolem/indobertweet-base-uncased')
tok=AutoTokenizer.from_pretrained(path_model)
lengths=pd.Series([len(row) for row in tok(train.bert.tolist(),truncation=False)['input_ids']])
show('Token model train',lengths.describe(percentiles=[.5,.9,.95,.99]));print('Terpotong (%):',lengths.gt(MAX_LEN).mean()*100)
pos=np.sum(Y==1,axis=0);neg=np.sum(Y==0,axis=0);pos_weight=np.sqrt(np.clip(neg/np.maximum(pos,1),1,20))
show('Loss weight per label',pd.Series(pos_weight,index=LABEL_COLS))
def encoded(frame,labels=True):
    payload={'text':frame.bert.tolist()}
    if labels:payload['labels']=np.where(frame[LABEL_COLS].isna(),-1,frame[LABEL_COLS]).astype('float32').tolist()
    return Dataset.from_dict(payload).map(lambda b:tok(b['text'],truncation=True,max_length=MAX_LEN),batched=True,remove_columns='text')
model=AutoModelForSequenceClassification.from_pretrained(path_model,num_labels=L,problem_type='multi_label_classification',ignore_mismatched_sizes=True)
model.config.id2label=dict(enumerate(LABEL_COLS));model.config.label2id={name:i for i,name in enumerate(LABEL_COLS)}
''',r'''
class MaskedTrainer(Trainer):
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        y=inputs.pop('labels').float();mask=y>=0;result=model(**inputs)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(result.logits.float(),y.clamp(0,1),
            pos_weight=torch.tensor(pos_weight,dtype=torch.float32,device=y.device),reduction='none')
        loss=(loss*mask).sum()/mask.sum().clamp(min=1)
        return (loss,result) if return_outputs else loss
from scipy.special import expit
args=TrainingArguments(output_dir=str(OUT/'checkpoints'),num_train_epochs=EPOCHS,learning_rate=2e-5,
    per_device_train_batch_size=16,per_device_eval_batch_size=32,fp16=True,weight_decay=.01,
    eval_strategy='epoch',save_strategy='epoch',save_only_model=True,save_total_limit=1,load_best_model_at_end=True,
    metric_for_best_model='macro_f1',greater_is_better=True,logging_strategy='epoch',disable_tqdm=True,report_to='none',seed=SEED)
trainer=MaskedTrainer(model=model,args=args,train_dataset=encoded(train),eval_dataset=encoded(val),
    data_collator=DataCollatorWithPadding(tok),compute_metrics=lambda p:evaluate(p.label_ids,expit(p.predictions)))
trainer.model_accepts_loss_kwargs=False
start=time.perf_counter();trainer.train()
vp['05_indobertweet']=expit(trainer.predict(encoded(val,False)).predictions)
hp['05_indobertweet']=expit(trainer.predict(encoded(test,False)).predictions)
times.append({'model':'05_indobertweet','fit_and_predict_seconds':time.perf_counter()-start})
trainer.save_model(str(OUT/'model_indobertweet'));tok.save_pretrained(str(OUT/'model_indobertweet'))
assert len(vp)==5 and all(p.shape==V.shape and np.isfinite(p).all() for p in vp.values())
del trainer,model;gc.collect();torch.cuda.empty_cache()
''')
section('10 · Threshold global/per-label & ensemble','Ensemble empat model non-dummy. Cross-fit validation dilakukan pada grup teks yang sama. Label tune dengan support kurang memakai threshold 0,5; tie memilih threshold dekat 0,5. Tidak memaksa satu label aktif, karena semua-negatif dapat valid.',r'''
names=[name for name in vp if name!='01_all_negative']
vp['06_ensemble']=sum(vp[name] for name in names)/len(names);hp['06_ensemble']=sum(hp[name] for name in names)/len(names)
def fit_thresholds(prob,y):
    thresholds=np.full(L,.5)
    for j in range(L):
        known=np.isfinite(y[:,j]);target=y[known,j]
        if (target==1).sum()<THRESHOLD_MIN_POS or (target==0).sum()<THRESHOLD_MIN_NEG:continue
        thresholds[j]=max(GRID,key=lambda c:(f1_score(target,prob[known,j]>=c,zero_division=0),-abs(c-.5)))
    return thresholds
def fit_global(prob,y):return max(GRID,key=lambda c:(evaluate(y,prob,c)['macro_f1'],-abs(c-.5)))
profile=val.groupby('group')[LABEL_COLS].max().fillna(0);gids=profile.index.to_numpy()
folds=[]
for fit_ids,check_ids in MultilabelStratifiedKFold(3,shuffle=True,random_state=SEED).split(gids,profile.to_numpy()):
    folds.append((np.flatnonzero(val.group.isin(gids[fit_ids])),np.flatnonzero(val.group.isin(gids[check_ids]))))
board=[];oof={};tuning={}
''',r'''
for name,prob in vp.items():
    global_prob=np.empty_like(prob);label_prob=np.empty_like(prob)
    # Encode threshold decisions as 0/1 scores only for F1/Hamming comparison; AP is reported separately from raw probabilities.
    for a,b in folds:
        global_prob[b]=(prob[b]>=fit_global(prob[a],V[a])).astype(float)
        label_prob[b]=(prob[b]>=fit_thresholds(prob[a],V[a])).astype(float)
    for route,scores in [('raw_05',prob),('global',global_prob),('per_label',label_prob)]:
        key_name=name+'__'+route;oof[key_name]=scores>=.5
        row=evaluate(V,scores);row['macro_AP']=evaluate(V,prob)['macro_AP']
        board.append({'candidate':key_name,'model':name,'route':route,**row})
    tuning[name]={'global':float(fit_global(prob,V)),'per_label':fit_thresholds(prob,V)}
board=pd.DataFrame(board).sort_values('macro_f1',ascending=False)
show('Validation cross-fit · AP always raw probability',board)
best=board.iloc[0];chosen,route=best.model,best.route
cut=.5 if route=='raw_05' else tuning[chosen][route]
print('Selected:',chosen,route,'| thresholds:',cut)
''')
section('11 · Hierarki HS · kondisional','HS_* positif dapat mengimplikasikan HS, tetapi Abusive berdiri sendiri. Audit konsistensi anotasi train dahulu. Tidak memaksa kategori/severity saling eksklusif tanpa bukti. Aturan dipakai hanya bila anotasi train konsisten dan cross-fit validation membaik.',r'''
parent=LABEL_COLS.index('HS') if 'HS' in LABEL_COLS else None
children=[i for i,name in enumerate(LABEL_COLS) if name.startswith('HS_')]
def hierarchy(pred):
    out=pred.copy()
    if parent is not None and children:out[:,parent]|=out[:,children].any(axis=1)
    return out
violations=0
if parent is not None:
    for child in children:violations+=int(np.sum((Y[:,parent]==0)&(Y[:,child]==1)))
base_oof=oof[best.candidate];fixed_oof=hierarchy(base_oof)
use_hierarchy=violations==0 and evaluate(V,fixed_oof.astype(float))['macro_f1']>evaluate(V,base_oof.astype(float))['macro_f1']+MIN_GAIN
print('Known train hierarchy violations:',violations,'| hierarchy selected:',use_hierarchy)
policy={'model':chosen,'route':route,'thresholds':np.broadcast_to(cut,(L,)).tolist(),
    'hierarchy':bool(use_hierarchy),'labels':LABEL_COLS,'unknown':'NaN / -1 batch sentinel; never counted as zero'}
(OUT/'selection.json').write_text(json.dumps(policy,indent=2))
print('Frozen before test:',policy)
''')
section('12 · Holdout & kelemahan tiap label','Semua keputusan dibekukan. Laporan memperlihatkan support diketahui, precision/recall/F1/AP per label, false positive/negative, semua-negatif dan jumlah label aktif. AP bukan skor dari keputusan threshold.',r'''
rows=[]
for name,prob in hp.items():
    for how,threshold in [('raw_05',.5),('per_label',tuning[name]['per_label'])]:
        rows.append({'model':name,'route':how,**evaluate(T,prob,threshold)})
final=hp[chosen]>=cut
if use_hierarchy:final=hierarchy(final)
score=evaluate(T,final.astype(float));score['macro_AP']=evaluate(T,hp[chosen])['macro_AP']
rows.append({'model':'FINAL','route':route,**score})
show('Holdout comparison',pd.DataFrame(rows));pd.DataFrame(rows).to_csv(OUT/'models.csv',index=False)
per_label=[]
for j,label in enumerate(LABEL_COLS):
    known=np.isfinite(T[:,j]);target=T[known,j].astype(int);p=final[known,j]
    precision,recall,f1,_=precision_recall_fscore_support(target,p,average='binary',zero_division=0)
    per_label.append({'label':label,'known':known.sum(),'positive':target.sum(),'threshold':np.broadcast_to(cut,(L,))[j],
        'precision':precision,'recall':recall,'f1':f1,'AP':average_precision_score(target,hp[chosen][known,j]) if target.sum() else np.nan,
        'FP':int(np.sum((target==0)&p)),'FN':int(np.sum((target==1)&~p))})
show('Per label',pd.DataFrame(per_label).sort_values('positive'))
''',r'''
known=np.isfinite(T);errors=(T!=final)&known
analysis=test[['row_id',TEXT_COL]].copy()
analysis['true_known']=[[LABEL_COLS[j] for j in range(L) if T[i,j]==1] for i in range(len(test))]
analysis['predicted']=[[LABEL_COLS[j] for j in range(L) if final[i,j]] for i in range(len(test))]
analysis['n_label_errors']=errors.sum(axis=1)
analysis['n_unknown']=(~known).sum(axis=1)
show('Kesalahan multilabel terbanyak',analysis.nlargest(TOP,'n_label_errors'))
show('Label aktif prediksi',distribution(pd.Series(final.sum(axis=1))))
slice_rows=[];length=test.safe.str.split().str.len()
conditions={'<=5_kata':length.le(5),'>40_kata':length.gt(40)}
for name in ['negasi','slang_contoh','emoji/simbol']:conditions[name]=test.safe.str.count(NOISE[name],flags=re.I).gt(0)
for name,mask in conditions.items():
    ids=mask.to_numpy()
    if ids.any():slice_rows.append({'slice':name,'n':int(ids.sum()),**evaluate(T[ids],final[ids].astype(float))})
show('Kelemahan kondisi teks · AP slice tidak ditafsirkan dari keputusan 0/1',pd.DataFrame(slice_rows).drop(columns=['macro_AP']))
pd.DataFrame(per_label).to_csv(OUT/'per_label.csv',index=False);analysis.to_csv(OUT/'errors.csv',index=False)
board.to_csv(OUT/'validation.csv',index=False);pd.DataFrame(times).to_csv(OUT/'timing.csv',index=False)
np.savez_compressed(OUT/'predictions.npz',probabilities=hp[chosen],decisions=final,truth=T,labels=np.array(LABEL_COLS))
''')
section('13 · Ringkasan & sumber','Data beranotasi task hate speech tidak otomatis mewakili domain baru. Rare label perlu jumlah positif yang cukup pada validation/test; angka kecil bukan bukti model sudah stabil. [IndoBERTweet recipe](https://huggingface.co/indolem/indobertweet-base-uncased), [E5](https://huggingface.co/intfloat/multilingual-e5-base), [iterative stratification](https://github.com/trent-b/iterative-stratification).',r'''
manifest={'task':RESULT_NAME,'rows':len(data),'splits':{'train':len(train),'validation':len(val),'test':len(test)},
    'labels':LABEL_COLS,'models':len(hp)-1,'selection':policy,'final':score,
    'gpu':torch.cuda.get_device_name(0),'seconds':time.perf_counter()-START}
assert len(hp)==6
(OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2,default=float));print(manifest)
''')

for c in cells:
    if c.cell_type=='code':ast.parse(c.source)
notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
nb.validate(notebook);nb.write(notebook,DEST)
kernel=ROOT/'_build/multilabel_review_kaggle';kernel.mkdir(exist_ok=True)
payload=base64.b64encode(nb.writes(notebook).encode()).decode()
runner=f'''import base64,pathlib,papermill as pm
source=pathlib.Path('/kaggle/working/_source.ipynb')
source.write_bytes(base64.b64decode({payload!r}))
try:
    pm.execute_notebook(str(source),'/kaggle/working/12_multi_label_review.ipynb',kernel_name='python3',cwd='/kaggle/working',progress_bar=False,log_output=True)
finally:source.unlink(missing_ok=True)
'''
nb.write(nb.v4.new_notebook(cells=[nb.v4.new_code_cell(runner)],metadata=notebook.metadata),kernel/'runner.ipynb')
meta={'id':'ivanwllm/gemastik-multilabel-review','title':'Gemastik Multilabel Review','code_file':'runner.ipynb','language':'python','kernel_type':'notebook',
    'is_private':True,'enable_gpu':True,'machine_shape':'NvidiaTeslaT4','enable_internet':True,
    'dataset_sources':['ilhamfp31/indonesian-abusive-and-hate-speech-twitter-text'],
    'kernel_sources':['ivanwllm/gemastik-nlp-00-setup-offline'],'competition_sources':[],'model_sources':[]}
(kernel/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print('Wrote',DEST,'| code cells',sum(c.cell_type=='code' for c in cells))
