"""Khusus Kaggle: uji perilaku kontrak + mini neural smoke, bukan benchmark skor."""
def smoke_test(blocks, configs, advanced=True):
    import json
    import traceback
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import time
    ns = {}
    for code in blocks:
        exec(compile(code,'<toolkit-block>','exec'),ns)
    ns['require_kaggle'](gpu=advanced)
    out = Path('/kaggle/working/composer_smoke');out.mkdir(exist_ok=True)
    results = []
    def check(name, fn):
        start=time.perf_counter()
        try:
            fn();row={'test':name,'ok':True,'seconds':time.perf_counter()-start}
        except Exception:
            row={'test':name,'ok':False,'seconds':time.perf_counter()-start,'error':traceback.format_exc()}
        results.append(row)
        print(json.dumps(row),flush=True)
        (out/'validation.json').write_text(json.dumps({'results':results,'all_passed':all(r['ok'] for r in results)},indent=2))
    def assert_true(value):
        assert value
    check('masked_unknown_not_negative',lambda:assert_true(ns['masked_multilabel_metrics'](np.array([[1,np.nan],[0,1]]),np.array([[.9,.99],[.1,.9]]))['micro_f1']==1))
    check('masked_no_known',lambda:assert_true(np.isnan(ns['masked_multilabel_metrics'](np.array([[np.nan,np.nan]]),np.ones((1,2)))['micro_f1'])))
    check('negation_preserved',lambda:assert_true('tidak' in ns['preprocess_text']('Tidak bagus!!! #MBG 😀','bow',stopwords=['tidak'])))
    def grouping():
        raw=pd.DataFrame({'left':['A','B','Z'],'right':['B','C','Y']})
        groups=ns['connected_groups'](raw,['left','right'],shared_endpoints=True)
        assert groups[0]==groups[1] and groups[2]!=groups[0]
        raw=pd.DataFrame({'text':['ga baik','tidak baik'],'a':['tidak baik','tidak baik'],'user':[1,2]})
        assert len(set(ns['connected_groups'](raw,['text'],'user',['a'])))==1
    check('connected_endpoints_and_normalization_collision',grouping)
    data_cache={}
    def supervised(task):
        cfg={**configs[task],'out':str(out/task),'models':['dummy','tfidf_word','tfidf_union'],'epochs':1,'batch_size':8,'max_len':48,'max_features':300}
        labels=['entailment','neutral','contradiction']
        themes=['Program baik bermanfaat siswa senang','Program buruk mahal gagal korupsi','Program mulai besok sekolah wilayah']
        raw=pd.DataFrame({'text':[themes[i%3]+f' kode{i}' for i in range(120)],'label':[str(i%3) for i in range(120)],
            'premise':[themes[i%3]+f' pernyataan{i}' for i in range(120)],'hypothesis':[themes[i%3]+f' hipotesis{i}' for i in range(120)],
            'text_1':[themes[i%3]+f' kalimat{i}' for i in range(120)],'text_2':[themes[(i//2)%3]+f' pasangan{i}' for i in range(120)],
            'score':[float(i%6) for i in range(120)],'HS':[i%2 for i in range(120)],'Abusive':[np.nan if i%5==0 else (i//2)%2 for i in range(120)]})
        if task=='nli':raw['label']=[labels[i%3] for i in range(len(raw))]
        tables=ns['numeric_eda'](raw,cfg['pair_cols'][0] if task in {'nli','similarity'} else 'text',sample=30)
        assert {'quality','noise','vocabulary','ngram_3'}.issubset(tables)
        data=ns['split_supervised'](ns['prepare_supervised'](raw,cfg),cfg)
        for a,b in [('train','val'),('train','test'),('val','test')]:assert set(data[a]._group).isdisjoint(data[b]._group)
        result=ns['run_supervised'](data,cfg)
        assert (Path(cfg['out'])/'selection_frozen_before_test.json').is_file()
        assert result['tables']['holdout'].shape[0]==4
        data_cache[task]=(data,cfg)
    for task in ['classification','multilabel','similarity','nli']:
        check(task+'_linear_pipeline',lambda task=task:supervised(task))
    def retrieval():
        cfg={**configs['retrieval'],'out':str(out/'retrieval'),'models':['tfidf','bm25']}
        corpus=pd.DataFrame({'doc_id':[f'd{i}' for i in range(30)],'text':[f'program makan anak sekolah wilayah kota{i} menu lauk sehat {i}' for i in range(30)]})
        queries=pd.DataFrame({'query':[f'menu sekolah kota{i}' for i in range(12)],'relevant_ids':[json.dumps([f'd{i}',f'd{i+1}']) for i in range(12)]})
        data=ns['prepare_retrieval'](corpus,queries,cfg)
        result=ns['run_retrieval'](data,cfg)
        assert result['holdout']['Recall@10'].between(0,1).all()
        known=ns['retrieval_metrics'](np.array([[0,1,2]]),[{'a','c'}],['a','b','c'])
        assert known['Recall@1']==.5 and known['Recall@5']==1 and known['MRR@10']==1
    check('multi_relevance_retrieval',retrieval)
    def unsup(task):
        cfg={**configs[task],'out':str(out/task),'models':['nmf','lda','kmeans'],'k_values':[2],'fit_rows':150,'max_features':1000}
        themes=['makan sehat siswa dapur sekolah murid','anggaran korupsi dana proyek pemerintah audit','komentar bagus senang baik manfaat dukung']
        raw=pd.DataFrame({'text':[themes[i%3]+' wilayah kode'+chr(97+i//26)+chr(97+i%26) for i in range(180)],
            'published_at':pd.date_range('2025-01-01',periods=180,freq='D').astype(str)})
        data=ns['prepare_unsupervised'](raw,cfg)
        result=ns['run_unsupervised'](data,cfg)
        assert len(result['assignments'])==len(raw)
        if task=='temporal':
            ns['temporal_report'](result['assignments'],cfg)
            ns['sentiment_over_time'](result['assignments'],cfg)
    for task in ['topic','clustering','temporal']:
        check(task+'_pipeline',lambda task=task:unsup(task))
    def calibration():
        p=np.array([[.8,.1,.1],[.2,.7,.1],[.1,.1,.8]])
        changed=ns['apply_post'](p,'temperature',2.)
        assert np.array_equal(p.argmax(1),changed.argmax(1))
        fixed=ns['duplicate_consistency'](np.array([[.2,.8],[.8,.2]]),['Sama','sama'])
        assert np.allclose(fixed,.5)
        y=np.array([[1,0],[0,1],[1,np.nan]])
        cuts=ns['fit_post'](np.full((3,2),.2),y,'label_threshold',{'threshold_min_support':5})
        assert np.allclose(cuts,.5)
    check('calibration_duplicate_rare_threshold',calibration)
    def word_vectors():
        data,cfg=data_cache['classification']
        vp,hp=ns['fit_word2vec'](data,cfg)
        assert vp.shape[1]==len(data['labels']) and np.isfinite(hp).all()
    check('word2vec_train_only',word_vectors)
    if advanced:
        for model in ['indobertweet','indobert_p1','xlmr','indolem','nusabert']:
            def neural(model=model,task='classification'):
                data,cfg=data_cache[task]
                small={**data,**{k:data[k].iloc[:12].copy() for k in ['train','val','test']}}
                cfg={**cfg,'out':str(out/('neural_'+task+'_'+model))}
                result=ns['fit_pretrained'](small,cfg,model)
                assert all(np.isfinite(p).all() for p in result)
                if task!='similarity':assert result[0].shape==(12,len(data['labels']))
            check('neural_'+model,neural)
        for task in ['multilabel','similarity']:
            def head(task=task):
                data,cfg=data_cache[task]
                small={**data,**{k:data[k].iloc[:12].copy() for k in ['train','val','test']}}
                result=ns['fit_pretrained'](small,{**cfg,'out':str(out/('neural_'+task))},'indobert_p1')
                assert all(np.isfinite(p).all() for p in result)
            check('neural_'+task+'_loss',head)
        def embed():
            e=ns['encode_e5'](['makanan sekolah sehat','anggaran pemerintah'],configs['classification'],'query')
            assert e.shape[0]==2 and np.allclose(np.linalg.norm(e,axis=1),1,atol=1e-4)
        check('e5_pooling_prefix',embed)
        def pretrained_nli():
            data,cfg=data_cache['nli']
            small={**data,**{k:data[k].iloc[:2].copy() for k in ['val','test']}}
            p,_=ns['fit_nli_pretrained'](small,cfg)
            assert p.shape==(2,3) and np.allclose(p.sum(1),1)
        check('nli_pretrained_label_mapping',pretrained_nli)
        def sentiment():
            labels=ns['pretrained_sentiment'](['Program ini bagus','Saya kecewa'],configs['temporal'])
            assert set(labels).issubset({'positive','neutral','negative'})
        check('sentiment_pretrained_label_mapping',sentiment)
        def reranker():
            rank=ns['rerank_candidates'](['program sekolah baik','anggaran pemerintah'],['makan siswa'],np.array([[0,1]]),configs['retrieval'],top=2)
            assert set(rank[0])=={0,1}
        check('cross_encoder_retrieval',reranker)
    print('COMPOSER_SMOKE_COMPLETE',len(results),'passed',sum(r['ok'] for r in results),flush=True)
    return results
