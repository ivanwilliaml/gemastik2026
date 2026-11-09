"""Satu sumber perakit CLI dan browser: kode blok + kontrak + resep."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent

SOURCES = {
    'classification':'notebook/b_klasifikasi/10_multi_kelas/10_multi_kelas_v2_pretrained.ipynb',
    'multilabel':'notebook/b_klasifikasi/12_multi_label/12_multi_label_review.ipynb',
    'similarity':'notebook/e_pasangan_teks/20_kemiripan/20_kemiripan_review.ipynb',
    'nli':'notebook/e_pasangan_teks/21_nli/21_nli_review.ipynb',
    'retrieval':'notebook/e_pasangan_teks/22_retrieval/22_retrieval_review.ipynb',
    'topic':'notebook/a_tanpa_label/01_topic_modeling/01_topic_modeling_review.ipynb',
    'clustering':'notebook/a_tanpa_label/03_clustering/03_clustering_review.ipynb',
    'temporal':'notebook/a_tanpa_label/02_tren_topik_waktu/02_tren_topik_waktu_review.ipynb'
}

BASE_CONFIG = {
    'data_file':'/kaggle/input/GANTI_DATASET/train.csv', 'encoding':'utf-8',
    'text_col':'text','label_col':'label','label_cols':['HS','Abusive'],
    'pair_cols':['premise','hypothesis'],'group_col':None,
    'seed':42,'out':'/kaggle/working/hasil_toolkit','offline':False,
    'preprocess':'native','slang':{},'stopwords':[],
    'eda_sample':5000,'eda_top':15,'eda_languages':False,
    'max_features':60000,'max_len':192,'epochs':3,'learning_rate':2e-5,'batch_size':16,
    'threshold_min_support':5,'score_range':[0,5],
    'e5_model':'intfloat/multilingual-e5-base','model_paths':{},
    'nli_label_map':{'e':'entailment','n':'neutral','c':'contradiction','entailment':'entailment',
        'neutral':'neutral','contradiction':'contradiction'},
    'corpus_file':'/kaggle/input/GANTI_DATASET/corpus.csv','queries_file':'/kaggle/input/GANTI_DATASET/queries.csv',
    'doc_id_col':'doc_id','doc_text_col':'text','query_col':'query','relevant_col':'relevant_ids',
    'k_values':[4,6,8],'fit_rows':5000,
    'time_col':'published_at','timezone':'Asia/Jakarta','naive_timezone':'UTC','period':'W',
    'min_period_rows':30,'sentiment_sample_per_period':100,'sentiment_pretrained':False,
    'positive_words':['bagus','baik','senang','bermanfaat'], 'negative_words':['buruk','kecewa','mahal','gagal']
}


def library():
    components = {}
    def add(key, title, file=None, needs=(), packages=(), stage='helper', note='', example='', gpu=False, **extra):
        code = (ROOT/'blocks'/file).read_text(encoding='utf-8') if file else ''
        ast.parse(code)
        components[key] = dict(id=key,title=title,file='blocks/'+file if file else None,needs=list(needs),
            packages=list(packages),stage=stage,note=note,example=example,code=code,gpu=gpu,**extra)
    add('core','Kontrak, data loading & penyimpanan','core.py',packages=['numpy','pandas','scikit-learn','scipy'])
    add('preprocess','Preprocessing native / normalized / BoW / IndoBERTweet','preprocess.py',['core'],['ftfy>=6.3,<7','emoji>=2,<3'],
        stage='preprocessing',note='Slang hanya dari CFG["slang"]. Native mempertahankan negasi, emoji, angka, case. Uji normalisasi sebagai eksperimen.',
        example="clean = make_cleaner(CFG)\ndf['clean'] = df[CFG['text_col']].map(clean)")
    add('eda','EDA angka: missing → vocabulary → noise','eda.py',['preprocess'],stage='eda',
        note='Basic/noise seluruh baris; vocab/ngram memakai sampel eksplisit. Persentase noise boleh overlap. Bahasa adalah estimasi sampel.',
        example="tables = numeric_eda(df, CFG['text_col'], sample=5000)\nfor name, table in tables.items(): show_table(name, table)")
    add('groups','Grup pengguna, duplicate & connected endpoint','groups.py',['preprocess'],stage='validation')
    add('split','Split grup & partial multilabel','split.py',['groups'],stage='validation',
        note='Kelompokkan pengguna + native duplicate + collision normalisasi. Pair memakai connected endpoints. NLI: e/n/c eksplisit. Unknown label tetap unknown.',
        example='prepared = prepare_supervised(df, CFG)\ndata = split_supervised(prepared, CFG)')
    add('metrics','Metrik klasifikasi, partial multilabel & STS','metrics.py',['core'],stage='evaluation',
        example="metrics = task_metrics(y_true, probabilities, CFG['task'])")
    add('error_analysis','Support, FP/FN, confusion & slice','error_analysis.py',['metrics','eda'],stage='evaluation')
    add('post','Cross-fit threshold / bias / calibration / ensemble','postprocess.py',['metrics','preprocess'],stage='postprocessing',
        note='Parameter memakai validation grup. Holdout tidak boleh memilih model/aturan. Duplicate consistency, audited rules, dan hierarki tersedia sebagai fungsi opsional.',
        example='policy, board = select_policy(validation_predictions, y_val, val_groups, CFG)')
    add('linear','Head sparse/dense dan fitur pasangan','linear.py',['core'],stage='model')
    add('encoder','E5 masked mean pooling + prefix','embedding.py',['core'],['transformers==5.20.2','accelerate>=1,<2','sentencepiece>=0.2,<1','torch'],stage='representation',gpu=True)
    add('transformer','Trainer classification / regression / masked BCE','transformer.py',['metrics','preprocess'],
        ['transformers==5.20.2','accelerate>=1,<2','datasets>=4,<5','sentencepiece>=0.2,<1','torch'],stage='model',gpu=True,
        note='Masked BCE untuk unknown; sigmoid multilabel. Model checkpoint hanya dari validation; NLI mempertahankan arah premise → hypothesis.')
    add('supervised_run','Jalankan model terpilih → freeze → audit holdout','run_supervised.py',['post','split','error_analysis'],stage='evaluation')
    add('retrieval','Retrieval multi-relevance + matching/ranking audit','retrieval.py',['preprocess','core'],['rank-bm25==0.2.2'],stage='model',
        note='MRR/Recall/NDCG benar untuk banyak dokumen relevan. Corpus index bukan supervised train. Query validation/test dipisahkan menurut exact query.',
        example='data = prepare_retrieval(corpus, queries, CFG)\nresult = run_retrieval(data, CFG)')
    # Retrieval/unsupervised need GroupShuffleSplit without copying the entire supervised contract.
    components['retrieval']['code'] = 'from sklearn.model_selection import GroupShuffleSplit\n'+components['retrieval']['code']
    add('unsupervised','Frozen topic/cluster fit + validation quality','unsupervised.py',['groups','eda'],stage='model',
        note='Fit vocabulary/model pada train unik, pilih k di validation, audit holdout. NPMI/silhouette tidak menggantikan audit semantik.',
        example='data = prepare_unsupervised(df, CFG)\nresult = run_unsupervised(data, CFG)')
    add('time','Proporsi waktu, shift & sentimen','temporal.py',['unsupervised'],stage='evaluation',
        note='Early/middle/late, timezone eksplisit, volume duplicate tetap, trailing mean, support kecil. Sentimen tanpa gold hanya estimasi/agreement.',
        example="temporal_report(result['assignments'], CFG)\nsentiment_over_time(result['assignments'], CFG)")
    add('model.dummy','Baseline prior / mean / semua-negatif','baseline.py',['core'],stage='model',model='dummy')
    for key,title in [('tfidf_word','TF-IDF word + LR / Ridge'),('tfidf_union','TF-IDF word+char + LR / Ridge')]:
        add('model.'+key,title,needs=['linear'],stage='model',model=key)
    add('model.word2vec','Word2Vec train-only + LR / Ridge','word2vec.py',['linear'],['gensim>=4.4,<5'],stage='model',model='word2vec')
    add('model.e5_head','Frozen E5 + LR / Ridge','embedding_head.py',needs=['encoder','linear'],stage='model',model='e5_head',gpu=True)
    names = {'indobertweet':'IndoBERTweet Base','indobert_p1':'IndoBERT Base P1','xlmr':'XLM-RoBERTa Base','indolem':'IndoBERT IndoLEM','nusabert':'NusaBERT Base'}
    for key,title in names.items():
        add('model.'+key,title,needs=['transformer'],stage='model',model=key,gpu=True,
            note='Bisa ditukar tanpa mengubah evaluator. IndoBERTweet memakai @USER/HTTPURL dan emoji demojize; yang lain native/normalisasi pilihan Anda.')
    add('model.nli_pretrained','mDeBERTa pretrained NLI 3 kelas','nli_pretrained.py',['core'],
        ['transformers==5.20.2','sentencepiece>=0.2,<1','torch'],stage='model',model='nli_pretrained',gpu=True)
    for key,title in [('tfidf','TF-IDF cosine'),('bm25','BM25'),('e5','E5 chunked dense'),('rrf','BM25 + E5 reciprocal rank fusion'),('reranker','Multilingual cross-encoder reranker')]:
        needs = ['retrieval']+(['encoder'] if key in {'e5','rrf'} else [])
        if key=='rrf': needs+=['model.bm25','model.e5']
        if key=='reranker': needs+=['model.bm25']
        packages = ['transformers==5.20.2','sentencepiece>=0.2,<1','torch'] if key=='reranker' else []
        add('model.'+key,title,needs=needs,packages=packages,stage='model',model=key,gpu=key in {'e5','rrf','reranker'})
    for key,title in [('nmf','NMF'),('lda','LDA'),('kmeans','TF-IDF + SVD + MiniBatchKMeans'),('e5_kmeans','E5 + MiniBatchKMeans')]:
        add('model.'+key,title,needs=['unsupervised']+(['encoder'] if key=='e5_kmeans' else []),stage='model',model=key,gpu=key=='e5_kmeans')
    add('sentiment.pretrained','Pretrained sentimen per waktu',needs=['time'],
        packages=['transformers==5.20.2','sentencepiece>=0.2,<1','torch'],stage='postprocessing',gpu=True)
    recipes = {}
    supervised_models = ['dummy','tfidf_word','tfidf_union','word2vec','e5_head',*names]
    def recipe(task,title,models,defaults,required,aliases=(),config=None,note=''):
        recipes[task] = dict(id=task,title=title,choices=['model.'+m for m in models],defaults=['model.'+m for m in defaults],
            required=required,aliases=list(aliases),source=SOURCES[task],config={**BASE_CONFIG,'task':task,**(config or {})},note=note)
    recipe('classification','Klasifikasi multiclass / biner',supervised_models,
        ['dummy','tfidf_word','tfidf_union','word2vec','indobertweet'],['eda','supervised_run'],
        ['multiclass','binary','sentiment','spam','intent','emotion','stance','fakenews'],
        {'ensemble':True,'post':['class_bias','temperature']},'Class-specific bias dan scalar temperature berbeda tujuan; temperature tidak mengubah argmax.')
    recipe('multilabel','Multilabel (partial labels)',supervised_models,
        ['dummy','tfidf_word','tfidf_union','e5_head','indobertweet'],['eda','supervised_run'],['multilabel'],
        {'ensemble':True,'post':['global_threshold','label_threshold']},'Unknown bukan 0. Threshold rare label fallback .5; tidak memaksa satu label aktif.')
    recipe('similarity','Semantic similarity / STS',supervised_models,
        ['dummy','tfidf_word','tfidf_union','e5_head','indobert_p1'],['eda','supervised_run'],['semsim','doc_sim'],
        {'pair_cols':['text_1','text_2'],'label_col':'score','ensemble':True,'post':['isotonic']},'Regresi kontinu, bukan NLI. E5 query di kedua sisi; fitur dan inference STS simetris.')
    recipe('nli','NLI entailment / neutral / contradiction',supervised_models+['nli_pretrained'],
        ['dummy','tfidf_word','e5_head','nli_pretrained','indobert_p1'],['eda','supervised_run'],['pair','consistency'],
        {'ensemble':True,'post':['class_bias','temperature']},'NLI tiga kelas, arah premise → hypothesis. Label numerik harus dipetakan eksplisit di nli_label_map.')
    recipe('retrieval','Text matching / retrieval',['tfidf','bm25','e5','rrf','reranker'],
        ['tfidf','bm25','e5','rrf','reranker'],['eda','retrieval'],['ir','ranking','dupq'],
        note='Queries perlu relevant_ids JSON list. Benchmark proxy ringkasan-artikel terlalu mudah; gunakan judgment kueri nyata untuk keputusan final.')
    for task,title,aliases in [('topic','Topic modeling',['topic']),('clustering','Clustering teks',['doc_cluster','emb_cluster']),('temporal','Tren topik / sentimen per waktu',['trend'])]:
        recipe(task,title,['nmf','lda','kmeans','e5_kmeans'],['nmf','lda','kmeans'],
            ['eda','unsupervised']+(['time'] if task=='temporal' else []),aliases,
            {'preprocess':'bow'},'Vocabulary dan topic ID dibekukan. Baca contoh dokumen per topik; metrik internal tidak mengukur akurasi sentimen.')
        if task=='temporal':
            recipes[task]['choices'].append('sentiment.pretrained')
    common = {'seed','out','encoding','offline','preprocess','slang','stopwords','eda_sample','eda_top','eda_languages','max_features'}
    supervised = {'data_file','text_col','group_col','max_len','epochs','learning_rate','batch_size','e5_model','model_paths','ensemble','post'}
    for task,recipe in recipes.items():
        if task in {'classification','multilabel','similarity','nli'}:
            keys = common | supervised
            keys |= {'label_cols','threshold_min_support'} if task=='multilabel' else {'label_col'}
            if task in {'nli','similarity'}: keys |= {'pair_cols'}
            if task=='nli': keys |= {'nli_label_map'}
            if task=='similarity': keys |= {'score_range'}
        elif task=='retrieval':
            keys = common | {'corpus_file','queries_file','doc_id_col','doc_text_col','query_col','relevant_col','e5_model','max_len'}
        else:
            keys = common | {'data_file','text_col','group_col','k_values','fit_rows','e5_model','max_len'}
            if task=='temporal':
                keys |= {'time_col','timezone','naive_timezone','period','min_period_rows','sentiment_sample_per_period','sentiment_pretrained','positive_words','negative_words'}
        recipe['config'] = {key:value for key,value in recipe['config'].items() if key in keys or key=='task'}
    provenance = {}
    for task,path in SOURCES.items():
        source = REPO/path
        assert source.exists(), source
        book = json.loads(source.read_text(encoding='utf-8'))
        code = '\n'.join(''.join(c['source']) if isinstance(c['source'],list) else c['source'] for c in book['cells'] if c['cell_type']=='code')
        provenance[task] = {'path':path,'code_sha256':hashlib.sha256(code.encode()).hexdigest(),
            'relationship':'Adaptasi kontrak dan pola review; bukan klaim salinan sel atau hasil run yang identik.'}
    return {'version':2,'components':components,'recipes':recipes,'provenance':provenance}


def resolve(lib, selected):
    ordered,active = [],set()
    def visit(key):
        if key in active: raise ValueError(f'Siklus dependensi: {key}')
        if key in ordered: return
        if key not in lib['components']: raise ValueError(f'Komponen tidak dikenal: {key}')
        active.add(key)
        for dep in lib['components'][key]['needs']: visit(dep)
        active.remove(key);ordered.append(key)
    for key in selected: visit(key)
    return ordered


def execution_steps(task):
    """Sel eksekusi sengaja pendek; fungsi penyusun ditempatkan terpisah."""
    if task in {'classification','multilabel','similarity','nli'}:
        return [
            ('Data & EDA raw',"raw = read_table(CFG['data_file'], CFG['encoding'])\ntables = eda_for_task(raw, CFG)"),
            ('Preprocess & split',"prepared = prepare_supervised(raw, CFG)\ndata = split_supervised(prepared, CFG)\nif CFG['task']=='multilabel':\n    tables = multilabel_eda(data['train'], CFG['label_cols'])\n    for name, table in tables.items(): show_table(name, table)\n    save_tables(tables, CFG['out'], 'train_eda_')"),
            ('Model → post → evaluasi',"result = run_supervised(data, CFG)")]
    if task=='retrieval':
        return [('Corpus & queries',"corpus = read_table(CFG['corpus_file'], CFG['encoding'], string_cols=[CFG['doc_id_col']])\nqueries = read_table(CFG['queries_file'], CFG['encoding'])\ntables = eda_for_task(corpus, CFG)\ndata = prepare_retrieval(corpus, queries, CFG)"),
            ('Ranking → validation → holdout',"result = run_retrieval(data, CFG)")]
    steps = [('EDA raw',"raw = read_table(CFG['data_file'], CFG['encoding'])\ntables = eda_for_task(raw, CFG)"),
        ('Frozen topic / cluster models',"data = prepare_unsupervised(raw, CFG)\nresult = run_unsupervised(data, CFG)")]
    if task=='temporal':
        steps += [('Proporsi topik & sentimen',"time_tables = temporal_report(result['assignments'], CFG)\nsentiment_tables = sentiment_over_time(result['assignments'], CFG)")]
    return steps


def build_notebook(lib, task, selected=None, overrides=None, install=True):
    recipe = lib['recipes'][task]
    selected = recipe['defaults'] if selected is None else list(selected)
    assert selected and set(selected).issubset(recipe['choices']), 'Pilihan tidak cocok dengan task / model kosong'
    chain = resolve(lib,recipe['required']+selected)
    # Model prerequisite RRF/reranker ikut sebagai model aktif, bukan hanya definisi helper.
    models = [lib['components'][key]['model'] for key in chain if lib['components'][key].get('model')]
    assert models, 'Pilih minimal satu model selain sentimen'
    cfg = {**recipe['config'],'models':models,'sentiment_pretrained':'sentiment.pretrained' in selected,**(overrides or {})}
    assert cfg['task']==task and cfg['models']==models, 'task/models mengikuti pilihan perakit; gunakan --models untuk mengubah model'
    packages = sorted(set(p for key in chain for p in lib['components'][key]['packages']))
    if task=='multilabel': packages.append('iterative-stratification>=0.1,<1')
    if cfg['eda_languages']: packages.append('langid==1.1.6')
    cells = []
    def md(text): cells.append({'cell_type':'markdown','metadata':{},'source':text})
    def code(text):
        ast.parse(text)
        cells.append({'cell_type':'code','metadata':{},'source':text,'outputs':[],'execution_count':None})
    md('# '+recipe['title']+' · toolkit modular v2\n\n'+recipe['note']+'\n\nUbah konfigurasi. Hapus/tambah blok melalui perakit, bukan mengubah puluhan sel. Semua model menggunakan split/urutan label yang sama. Hasil harus dihitung ulang di Kaggle; skor notebook sumber tidak diwariskan.\n\nSumber adaptasi: `'+recipe['source']+'`.')
    md('## 1 · Konfigurasi\n\nFile, kolom, preprocessing dan model hanya di sini. Group ID pengguna opsional. Untuk offline, set offline=True dan ganti model path ke bundle Kaggle input.')
    # repr menghasilkan Python True/False/None, bukan literal JSON.
    import pprint
    code('CFG = '+pprint.pformat(cfg,sort_dicts=False,width=110))
    if install:
        md('## 2 · Lingkungan Kaggle\n\nPasang hanya dependensi komponen terpilih. Untuk bundle offline: rakit dengan --no-install atau hapus sel instalasi.')
        code("import os, sys, subprocess\nfrom pathlib import Path\nassert Path('/kaggle/input').exists(), 'Jalankan di Kaggle.'\nos.environ.update(USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '-q', "+', '.join(repr(p) for p in packages)+"], check=True)")
    md('## 3 · Fungsi komponen\n\nDefinisi sekali; sel pemakaian pendek berada di bawah. Semua dependensi disertakan, tanpa import dari folder toolkit.')
    for key in chain:
        component = lib['components'][key]
        if component['code'].strip():
            md('### '+component['title']+'\n\n'+component['note'])
            code(component['code'])
    md('## 4 · Jalankan\n\nHoldout dibuka setelah pemilihan validation dibekukan. Selisih kecil perlu repeated group split. EDA angka tersedia sebelum modeling.')
    code(f"require_kaggle(gpu={any(lib['components'][key]['gpu'] for key in chain)})\nseed_all(CFG['seed'])")
    for title,source in execution_steps(task):
        md('### '+title);code(source)
    for i,cell in enumerate(cells): cell['id']='tk-'+str(i)
    return {'nbformat':4,'nbformat_minor':5,'cells':cells,'metadata':{
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'toolkit':{'version':2,'task':task,'tasks':recipe['aliases'],'selected':selected,'resolved':chain,'provenance':lib['provenance'][task]}}}
