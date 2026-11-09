from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score


def prepare_unsupervised(raw, cfg):
    frame = raw.copy().reset_index(drop=True)
    frame['_row_id'] = np.arange(len(frame))
    frame['native'] = frame[cfg['text_col']].map(safe_text)
    frame['a'] = frame[cfg['text_col']].map(make_cleaner(cfg))
    frame = frame[frame.a.ne('')].reset_index(drop=True)
    frame['_group'] = connected_groups(frame,[cfg['text_col']],cfg.get('group_col'),['a'])
    if cfg['task']=='temporal':
        source = frame[cfg['time_col']]
        def timestamp(value):
            try:
                t = pd.Timestamp(value)
                return t.tz_localize(cfg['naive_timezone']) if t.tzinfo is None else t
            except (ValueError,TypeError):
                return pd.NaT
        utc = pd.to_datetime(source.map(timestamp),utc=True,errors='coerce')
        frame['timestamp'] = utc.dt.tz_convert(cfg['timezone'])
        print('Tanggal invalid dibuang:',int(frame.timestamp.isna().sum()))
        frame = frame[frame.timestamp.notna()].sort_values('timestamp').reset_index(drop=True)
        frame['period'] = frame.timestamp.dt.tz_localize(None).dt.to_period(cfg['period']).dt.start_time
        periods = sorted(frame.period.unique())
        assert len(periods)>=5, 'Perlu minimal lima periode untuk early/middle/late'
        a,b = max(1,int(len(periods)*.6)),max(2,int(len(periods)*.8))
        parts = {'train':frame[frame.period.isin(periods[:a])], 'val':frame[frame.period.isin(periods[a:b])], 'test':frame[frame.period.isin(periods[b:])]}
        # Tidak membuang volume berulang. Audit recurrence: group overlap lintas waktu dilaporkan.
        for split in ['val','test']:
            print('Recurring exact-text rows vs early,',split,':',parts[split]._group.isin(parts['train']._group).sum())
    else:
        rest,test = next(GroupShuffleSplit(1,test_size=.2,random_state=cfg['seed']).split(frame,groups=frame._group))
        train,val = next(GroupShuffleSplit(1,test_size=.25,random_state=cfg['seed']+1).split(frame.iloc[rest],groups=frame.iloc[rest]._group))
        parts = {'train':frame.iloc[rest[train]],'val':frame.iloc[rest[val]],'test':frame.iloc[test]}
    parts = {k:v.reset_index(drop=True) for k,v in parts.items()}
    fit = parts['train'].drop_duplicates('a')
    if cfg.get('fit_rows'):
        fit = fit.sample(min(len(fit),cfg['fit_rows']),random_state=cfg['seed'])
    parts['fit'] = fit.reset_index(drop=True)
    assert all(len(parts[k]) for k in ['fit','val','test'])
    return parts


def npmi_quality(texts, topics):
    terms = sorted({word for topic in topics for word in topic})
    if not terms or not len(texts):
        return np.nan
    vec = CountVectorizer(vocabulary=terms,binary=True)
    x = vec.transform(texts).astype('float64')
    prob = np.asarray(x.mean(0)).ravel()
    joint = (x.T@x).toarray()/len(texts)
    vocab = vec.vocabulary_;scores=[]
    for topic in topics:
        values = []
        for i,a in enumerate(topic):
            for b in topic[i+1:]:
                pa,pb,pab = prob[vocab[a]],prob[vocab[b]],joint[vocab[a],vocab[b]]
                values.append(np.log(pab/(pa*pb))/-np.log(pab) if 0<pab<1 and pa*pb>0 else (1. if pab==1 else -1.))
        if values:
            scores.append(np.mean(values))
    return float(np.mean(scores)) if scores else np.nan


def fit_topic_candidate(data, cfg, method, k):
    from scipy.sparse import csr_matrix
    assert len(data['fit'])>k, 'Teks unik train terlalu sedikit; kurangi k atau tambah data.'
    vec = CountVectorizer(min_df=2,max_df=1.,max_features=cfg['max_features'])
    counts = vec.fit_transform(data['fit'].a)
    assert min(counts.shape)>k, 'Kurangi k atau tambah ukuran train/vocabulary'
    tfidf = TfidfTransformer().fit(counts)
    dense = method=='e5_kmeans'
    reducer = None
    if method=='nmf':
        x = tfidf.transform(counts)
        model = NMF(k,init='nndsvda',max_iter=300,random_state=cfg['seed']).fit(x)
    elif method=='lda':
        x = counts
        model = LatentDirichletAllocation(k,max_iter=10,learning_method='batch',random_state=cfg['seed']).fit(x)
    else:
        if dense:
            x = encode_e5(data['fit'].native,cfg,'query')
        else:
            n = min(100,min(counts.shape)-1)
            reducer = TruncatedSVD(n,random_state=cfg['seed']).fit(tfidf.transform(counts))
            x = normalize(reducer.transform(tfidf.transform(counts)))
        model = MiniBatchKMeans(k,n_init=10,batch_size=512,random_state=cfg['seed']).fit(x)
    def represent(frame):
        if dense:
            return encode_e5(frame.native,cfg,'query')
        c = vec.transform(frame.a)
        if method=='lda':
            return c
        return normalize(reducer.transform(tfidf.transform(c))) if reducer is not None else tfidf.transform(c)
    def assign(frame):
        z = represent(frame)
        return model.transform(z).argmax(1) if method in {'nmf','lda'} else model.predict(z)
    if method in {'nmf','lda'}:
        weights = model.components_
    else:
        grouped = np.array([np.asarray(counts[model.labels_==j].sum(0)).ravel() for j in range(k)])
        weights = TfidfTransformer().fit_transform(csr_matrix(grouped)).toarray()
    words = np.array(vec.get_feature_names_out())
    topics = [words[np.argsort(-row)[:10]].tolist() for row in weights]
    return {'model':model,'assign':assign,'represent':represent,'topics':topics,'vectorizer':vec,'k':k,'method':method}


def run_unsupervised(data, cfg):
    candidates = {};rows=[]
    for method in cfg['models']:
        for k in cfg['k_values']:
            fitted = fit_topic_candidate(data,cfg,method,k)
            name = f'{method}_k{k}';candidates[name]=fitted
            val_ids = fitted['assign'](data['val'])
            row = {'model':name,'val_npmi':npmi_quality(data['val'].a,fitted['topics']),
                'val_active_groups':len(np.unique(val_ids)), 'topic_diversity':len(set(sum(fitted['topics'],[])))/sum(map(len,fitted['topics']))}
            if cfg['task']=='clustering':
                z = fitted['represent'](data['val'])
                row['val_silhouette'] = silhouette_score(z,val_ids,sample_size=min(1000,len(val_ids)),random_state=cfg['seed']) if 1<len(np.unique(val_ids))<len(val_ids) else np.nan
            rows.append(row)
    board = pd.DataFrame(rows)
    metric = 'val_silhouette' if cfg['task']=='clustering' else 'val_npmi'
    assert board[metric].notna().any(), f'Tidak ada kandidat valid untuk {metric}'
    chosen = board.sort_values(metric,ascending=False,kind='stable').iloc[0].model
    fitted = candidates[chosen]
    out = Path(cfg['out']);out.mkdir(parents=True,exist_ok=True)
    (out/'selection_frozen_before_late.json').write_text(json.dumps({'model':chosen,'selection_metric':metric}))
    assignments = pd.concat([data[s].assign(split=s,topic=fitted['assign'](data[s])) for s in ['train','val','test']],ignore_index=True)
    topics = pd.DataFrame({'topic':range(fitted['k']),'words':[' | '.join(t) for t in fitted['topics']]})
    examples = assignments.groupby(['split','topic'],group_keys=False).head(5)
    tables = {'validation':board,'topics':topics,'examples_for_human_audit':examples,
        'assignments':assignments,'late_quality':pd.Series({'NPMI':npmi_quality(data['test'].a,fitted['topics'])}).to_frame('value')}
    save_tables(tables,out)
    for name in ['validation','topics','late_quality']:
        show_table(name,tables[name])
    print('FINAL:',chosen,'| NPMI/silhouette bukan bukti kebenaran semantik; audit contoh per topik/cluster.')
    return {'tables':tables,'assignments':assignments,'fitted':fitted}
