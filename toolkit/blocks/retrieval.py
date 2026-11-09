from sklearn.feature_extraction.text import TfidfVectorizer


def prepare_retrieval(corpus, queries, cfg):
    c = corpus.rename(columns={cfg['doc_id_col']:'doc_id',cfg['doc_text_col']:'text'}).copy()
    q = queries.rename(columns={cfg['query_col']:'query',cfg['relevant_col']:'relevant_ids'}).copy()
    assert {'doc_id','text'}.issubset(c) and {'query','relevant_ids'}.issubset(q)
    assert c.doc_id.notna().all() and c.doc_id.astype(str).is_unique, 'doc_id harus unik dan terisi'
    c['doc_id'] = c.doc_id.astype(str)
    c['text'] = c.text.map(safe_text)
    assert c.text.ne('').all(), 'Dokumen kosong perlu ditangani dahulu'
    q['query'] = q['query'].map(safe_text)
    assert q['query'].ne('').all(), 'Query kosong tidak dapat dievaluasi'
    def parse(value):
        value = json.loads(value) if isinstance(value,str) else value
        assert isinstance(value,list), 'relevant_ids harus JSON array, contoh ["d1","d2"]'
        return set(map(str,value))
    q['gold'] = q.relevant_ids.map(parse)
    assert q.gold.map(bool).all(), 'Query tanpa judgment positif belum dapat dinilai'
    assert set().union(*q.gold).issubset(set(c.doc_id)), 'Gold doc tidak ada dalam corpus'
    q['_group'] = pd.factorize(q['query'].str.casefold())[0]
    val,test = next(GroupShuffleSplit(1,test_size=.5,random_state=cfg['seed']).split(q,groups=q._group))
    print('Corpus:',len(c),'| query validation/test:',len(val),len(test))
    print('Duplikat dokumen:',c.text.str.casefold().duplicated().sum(),'| judgment per query:',q.gold.map(len).describe())
    return {'corpus':c.reset_index(drop=True),'val':q.iloc[val].reset_index(drop=True),'test':q.iloc[test].reset_index(drop=True)}


def retrieval_metrics(ranking, gold, doc_ids):
    rows = []
    ids = np.asarray(doc_ids).astype(str)
    for order,relevant in zip(ranking,gold):
        retrieved = ids[order].tolist()
        hits = np.array([d in relevant for d in retrieved],dtype=float)
        positions = np.flatnonzero(hits[:10])
        ideal = np.sum(1/np.log2(np.arange(min(len(relevant),10))+2))
        rows.append({'MRR@10':1/(positions[0]+1) if len(positions) else 0,
            **{f'Recall@{k}':len(set(retrieved[:k])&relevant)/len(relevant) for k in [1,5,10]},
            'NDCG@10':float(np.sum(hits[:10]/np.log2(np.arange(len(hits[:10]))+2))/ideal)})
    return pd.DataFrame(rows).mean().to_dict()


def lexical_rankings(corpus, queries, method='bm25', top=100):
    if method=='bm25':
        from rank_bm25 import BM25Okapi
        tokenize = lambda t:re.findall(r'[^\W_]+',t.casefold())
        index = BM25Okapi([tokenize(t) for t in corpus])
        return np.array([np.argsort(-index.get_scores(tokenize(t)),kind='stable')[:top] for t in queries])
    vec = TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=80000).fit(corpus)
    docs = vec.transform(corpus)
    return np.array([np.argsort(-(vec.transform([q])@docs.T).toarray()[0],kind='stable')[:top] for q in queries])


def rrf_rankings(first, second, alpha=.5, top=100, constant=60):
    merged = []
    for a,b in zip(first,second):
        scores = {}
        for ranking,weight in [(a,1-alpha),(b,alpha)]:
            for rank,doc in enumerate(ranking):
                scores[int(doc)] = scores.get(int(doc),0)+weight/(constant+rank+1)
        merged.append(sorted(scores,key=lambda doc:(-scores[doc],doc))[:top])
    return np.asarray(merged)


def dense_rankings(corpus, queries, cfg, top=100):
    chunks,owners = [],[]
    for i,text in enumerate(corpus):
        words = text.split()
        for start in range(0,min(len(words),cfg.get('max_chunks',6)*120),120):
            chunks.append(' '.join(words[start:start+120]));owners.append(i)
    emb = encode_e5(chunks,cfg,'passage')
    q = encode_e5(queries,cfg,'query')
    rankings = []
    for vector in q:
        scores = np.full(len(corpus),-np.inf)
        np.maximum.at(scores,owners,emb@vector)
        rankings.append(np.argsort(-scores,kind='stable')[:top])
    return np.array(rankings)


def rerank_candidates(corpus, queries, rankings, cfg, top=20):
    require_kaggle(gpu=True)
    import gc
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    repo = cfg.get('reranker','cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    tok = AutoTokenizer.from_pretrained(repo,local_files_only=cfg.get('offline',False))
    model = AutoModelForSequenceClassification.from_pretrained(repo,local_files_only=cfg.get('offline',False)).cuda().eval()
    assert model.config.num_labels==1, 'Reranker ini mengharapkan satu skor relevansi'
    result = []
    with torch.inference_mode():
        for query,ranking in zip(queries,rankings):
            candidates = ranking[:top]
            enc = tok([query]*len(candidates),[corpus[int(i)] for i in candidates],padding=True,truncation=True,max_length=cfg['max_len'],return_tensors='pt').to('cuda')
            scores = model(**enc).logits.float().cpu().numpy().ravel()
            result.append(np.r_[candidates[np.argsort(-scores,kind='stable')],ranking[top:]])
    del model
    gc.collect();torch.cuda.empty_cache()
    return np.array(result)


def run_retrieval(data, cfg):
    c = data['corpus'];doc_ids=c.doc_id.to_numpy();top=min(100,len(c))
    rankings = {split:{} for split in ['val','test']}
    for split in rankings:
        q = data[split]['query'].tolist()
        for method in [m for m in cfg['models'] if m in {'tfidf','bm25'}]:
            rankings[split][method] = lexical_rankings(c.text.tolist(),q,method,top)
        if 'e5' in cfg['models']:
            rankings[split]['e5'] = dense_rankings(c.text.tolist(),q,cfg,top)
    if 'rrf' in cfg['models']:
        assert all(name in rankings['val'] for name in ['bm25','e5']), 'RRF memerlukan BM25 dan E5'
        alpha = max([0,.25,.5,.75,1],key=lambda a:retrieval_metrics(rrf_rankings(rankings['val']['bm25'],rankings['val']['e5'],a,top),data['val'].gold,doc_ids)['MRR@10'])
        for split in rankings:
            rankings[split]['rrf'] = rrf_rankings(rankings[split]['bm25'],rankings[split]['e5'],alpha,top)
    if 'reranker' in cfg['models']:
        base = 'rrf' if 'rrf' in rankings['val'] else 'bm25'
        assert base in rankings['val'], 'Reranker memerlukan kandidat BM25 atau RRF'
        for split in rankings:
            rankings[split]['reranker'] = rerank_candidates(c.text.tolist(),data[split]['query'].tolist(),rankings[split][base],cfg)
    validation = pd.DataFrame([{'model':name,**retrieval_metrics(order,data['val'].gold,doc_ids)} for name,order in rankings['val'].items()])
    assert len(validation), 'Pilih minimal satu model retrieval'
    chosen = validation.sort_values('MRR@10',ascending=False,kind='stable').iloc[0].model
    out = Path(cfg['out']);out.mkdir(parents=True,exist_ok=True)
    (out/'selection_frozen_before_test.json').write_text(json.dumps({'model':chosen,'rrf_alpha':alpha if 'rrf' in cfg['models'] else None}))
    holdout = pd.DataFrame([{'model':name,**retrieval_metrics(order,data['test'].gold,doc_ids)} for name,order in rankings['test'].items()])
    rows = data['test'][['query']].copy()
    rows['gold'] = data['test'].gold.map(lambda v:sorted(v))
    rows['top10'] = [[doc_ids[i] for i in r[:10]] for r in rankings['test'][chosen]]
    rows['hit_at_1'] = [str(doc_ids[r[0]]) in g for r,g in zip(rankings['test'][chosen],data['test'].gold)]
    tables = {'validation':validation,'holdout':holdout,'errors_and_rankings':rows}
    save_tables(tables,out)
    for name,table in tables.items():
        show_table(name,table.head(20))
    print('FINAL (validation):',chosen,'| Judgment yang tidak lengkap dapat menurunkan Recall/NDCG secara semu.')
    return tables
