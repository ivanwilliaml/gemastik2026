"""Prepare remote-only execution of the actual short cells, not mirrored implementations."""
import base64
import json
from pathlib import Path
from build_cheatsheet import build, ROOT


SMOKE=r'''
import json,re,traceback,time,os
import numpy as np,pandas as pd
from pathlib import Path
root=Path('/kaggle/working/cheatsheet_smoke');root.mkdir(exist_ok=True)
themes=['Program sehat sekolah siswa bermanfaat bagus mendukung',
        'Program buruk korupsi dana anggaran terlambat menolak',
        'Program informasi jadwal sekolah pemerintah daerah netral']
def fixture(task):
    n=120
    raw=pd.DataFrame({'text':[themes[i%3]+' kode'+chr(97+i//26)+chr(97+i%26) for i in range(n)],
        'label':[str(i%3) for i in range(n)],'HS':[i%2 for i in range(n)],
        'Abusive':[np.nan if i%5==0 else (i//2)%2 for i in range(n)],
        'menu':['baik','buruk',None]*(n//3),'distribusi':['baik','buruk','netral']*(n//3),
        'published_at':pd.date_range('2025-01-01',periods=n,freq='D').astype(str),
        'author_id':[str(i//2) for i in range(n)]})
    raw['premise']=[themes[i%3]+' premise'+str(i) for i in range(n)]
    raw['hypothesis']=[themes[(i//2)%3]+' hypothesis'+str(i) for i in range(n)]
    raw['text_1'],raw['text_2']=raw.premise,raw.hypothesis
    raw['query']=[f'apa program sekolah kode{i//4}' for i in range(n)]
    raw['document']=raw.text;raw['qid']=np.arange(n)//4;raw['relevance']=np.arange(n)%3
    raw['question']=[f'Siapa menerima program kota{i}?' for i in range(n)]
    raw['context']=[f'Program di kota{i} dimulai besok. Siswa sekolah menerima makanan sehat.' for i in range(n)]
    raw['answer']='Siswa sekolah'
    raw['tokens']=[json.dumps(['Siswa','sekolah','makan',str(i)]) for i in range(n)]
    raw['tags']=[json.dumps(['NOUN','NOUN','VERB','NUM'] if task=='pos' else ['B-ORG','I-ORG','O','O']) for i in range(n)]
    if task in {'regression','similarity'}:raw['label']=np.arange(n)%6+.1
    raw.to_csv('train.csv',index=False)
    corpus=pd.DataFrame({'doc_id':[f'{i:04}' for i in range(30)],'text':[themes[i%3]+f' wilayah kota{i}' for i in range(30)]})
    corpus.to_csv('corpus.csv',index=False)
    queries=pd.DataFrame({'query':[f'program kota{i}' for i in range(20)],'relevant_ids':[json.dumps([f'{i:04}',f'{i+1:04}']) for i in range(20)]})
    queries.to_csv('queries.csv',index=False)
    raw.head(9).assign(id=[f'{i:04}' for i in range(9)]).to_csv('test.csv',index=False)
    return raw
reports=[];spaces={}
def check(name,fn):
    start=time.monotonic()
    try:fn();row={'test':name,'ok':True,'seconds':time.monotonic()-start}
    except Exception:row={'test':name,'ok':False,'error':traceback.format_exc()}
    reports.append(row);print(json.dumps(row),flush=True)
    (root/'validation.json').write_text(json.dumps({'results':reports,'all_passed':all(r['ok'] for r in reports)},indent=2))
def run(flow):
    folder=root/flow['id'];folder.mkdir(exist_ok=True);os.chdir(folder);fixture(flow['id'])
    ns={}
    for key in flow['core']:
        exec(compile(payload['cards'][key]['code'],'<cheat:'+key+'>','exec'),ns)
    spaces[flow['id']]=ns
for flow in payload['flows']:check(flow['id'],lambda flow=flow:run(flow))
def alternatives():
    ns=spaces['classification'];os.chdir(root/'classification')
    for key in ['dummy','char','lr','union','lr','svm','evaluate','tfidf','lr','threshold','class_bias','calibrate','word2vec','lr']:
        if key=='threshold':
            # Two classes required; select the existing binary probability head on same features.
            ns['y']=np.where(ns['y']=='0','yes','no')
            exec(payload['cards']['lr']['code'],ns)
        exec(compile(payload['cards'][key]['code'],'<cheat:'+key+'>','exec'),ns)
    # Restore sparse representation for final TF-IDF inference.
    exec(payload['cards']['tfidf']['code'],ns);exec(payload['cards']['lr']['code'],ns)
    ns['te']=pd.read_csv('test.csv',dtype={'id':str});ns['t']=ns['te']['text'].fillna('').astype(str)
    exec(payload['cards']['final']['code'],ns)
    assert pd.read_csv('submission.csv',dtype=str)['id'].iloc[0]=='0000'
check('classification_alternatives_and_final_ids',alternatives)
def partial_labels():
    ns=spaces['multilabel'];os.chdir(root/'multilabel')
    exec(payload['cards']['ml_threshold']['code'],ns);exec(payload['cards']['ml_eval']['code'],ns)
    ns['te']=pd.read_csv('test.csv',dtype={'id':str});ns['t']=ns['te']['text'].fillna('').astype(str)
    exec(payload['cards']['ml_final']['code'],ns)
    assert ns['submission'].shape==(9,3)
check('partial_labels_threshold_final',partial_labels)
def user_group():
    ns=spaces['classification'];exec(payload['cards']['user_groups']['code'],ns)
    assert set(ns['groups'][ns['tr']]).isdisjoint(ns['groups'][ns['va']])
check('user_and_duplicate_groups',user_group)
def neural():
    from huggingface_hub import snapshot_download
    folder=snapshot_download('indolem/indobertweet-base-uncased',cache_dir='/tmp/cheat_models')
    ns=spaces['classification'];os.chdir(root/'classification')
    ns['tr'],ns['va']=ns['tr'][:12],ns['va'][:8]
    for key in ['bert_setup','bert_train']:
        source=payload['cards'][key]['code'].replace('C:/gemastik/models/indobertweet',folder)
        exec(compile(source,'<cheat:'+key+'>','exec'),ns)
    assert len(ns['pred'])==len(ns['va'])
    folder=snapshot_download('intfloat/multilingual-e5-small',cache_dir='/tmp/cheat_models')
    exec(payload['cards']['embed']['code'].replace('C:/gemastik/models/e5-small',folder),ns)
    vectors=ns['embed'](['makan sehat sekolah','tidak setuju program'])
    assert np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-4)
check('local_only_transformer_and_e5',neural)
print('CHEATSHEET_SMOKE_COMPLETE',len(reports),sum(r['ok'] for r in reports),flush=True)
assert all(r['ok'] for r in reports),'Baca cheatsheet_smoke/validation.json'
'''


def main():
    data=build();folder=ROOT/'tools/cheatsheet_kaggle';folder.mkdir(exist_ok=True)
    payload=base64.b64encode(json.dumps({'cards':data['cards'],'flows':data['flows']}).encode()).decode()
    setup="import os,sys,subprocess,json,base64\nos.environ.update(USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false')\n"
    setup+="subprocess.run([sys.executable,'-m','pip','install','-q','transformers==5.22.1','scikit-learn==1.9.1','gensim==4.4.0','rank-bm25','sklearn-crfsuite','seqeval','langid','lightgbm','accelerate'],check=True)\n"
    setup+=f'payload=json.loads(base64.b64decode({payload!r}))\n'
    book={'nbformat':4,'nbformat_minor':5,'cells':[{'id':'run','cell_type':'code','metadata':{},'source':setup+SMOKE,'outputs':[],'execution_count':None}],
        'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}}}
    (folder/'runner.ipynb').write_text(json.dumps(book),encoding='utf-8')
    meta={'id':'ivanwllm/gemastik-cheatsheet-audit','title':'Gemastik Cheatsheet Audit','code_file':'runner.ipynb','language':'python','kernel_type':'notebook',
        'is_private':True,'enable_gpu':True,'machine_shape':'NvidiaTeslaT4','enable_internet':True,'dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(folder)


if __name__=='__main__':main()
