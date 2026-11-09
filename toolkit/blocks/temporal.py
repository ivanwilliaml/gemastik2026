from scipy.spatial.distance import jensenshannon


def temporal_report(assignments, cfg):
    counts = pd.crosstab(assignments.period,assignments.topic)
    shares = counts.div(counts.sum(1),axis=0)
    n = counts.sum(1)
    rows = []
    for i,period in enumerate(shares.index):
        rows.append({'period':period,'n':int(n.loc[period]),'low_support':bool(n.loc[period]<cfg.get('min_period_rows',30)),
            'JS_distance_vs_previous':float(jensenshannon(shares.iloc[i-1],shares.iloc[i],base=2)) if i else np.nan})
    smooth = shares.rolling(3,min_periods=1).mean()  # Trailing saja, tidak mengintip masa depan.
    tables = {'period_topic_counts':counts,'period_topic_share':shares,'trailing_3_period_share':smooth,'period_shift_support':pd.DataFrame(rows)}
    save_tables(tables,cfg['out'])
    show_table('Proporsi topik dan support per periode',shares.join(n.rename('n')))
    show_table('Perubahan distribusi, bukan uji kausalitas',tables['period_shift_support'])
    return tables


def lexicon_sentiment(texts, positive, negative):
    positive,negative = set(positive),set(negative)
    results = []
    for text in texts:
        words = re.findall(r'[^\W\d_]+',text.casefold());score=0
        for i,w in enumerate(words):
            value = int(w in positive)-int(w in negative)
            if set(words[max(0,i-3):i])&NEGATIONS:
                value = -value
            score += value
        results.append('positive' if score>0 else 'negative' if score<0 else 'neutral')
    return np.array(results)


def pretrained_sentiment(texts, cfg):
    require_kaggle(gpu=True)
    import gc
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    repo = cfg.get('sentiment_model','w11wo/indonesian-roberta-base-sentiment-classifier')
    tok = AutoTokenizer.from_pretrained(repo,local_files_only=cfg.get('offline',False))
    model = AutoModelForSequenceClassification.from_pretrained(repo,local_files_only=cfg.get('offline',False)).cuda().eval()
    mapping = {int(i):str(label).casefold() for i,label in model.config.id2label.items()}
    assert set(mapping.values())=={'positive','neutral','negative'}, f'Tetapkan mapping sentimen eksplisit: {mapping}'
    texts = list(texts);results=[]
    with torch.inference_mode():
        for start in range(0,len(texts),32):
            batch = tok(texts[start:start+32],padding=True,truncation=True,max_length=cfg['max_len'],return_tensors='pt').to('cuda')
            results.extend(mapping[int(i)] for i in model(**batch).logits.argmax(1).cpu().numpy())
    del model
    gc.collect();torch.cuda.empty_cache()
    return np.array(results)


def sentiment_over_time(assignments, cfg):
    per = cfg.get('sentiment_sample_per_period',100)
    sample = pd.concat([g.sample(min(per,len(g)),random_state=cfg['seed']) for _,g in assignments.groupby('period')])
    sample = sample.copy()
    sample['lexicon'] = lexicon_sentiment(sample.native,cfg.get('positive_words',[]),cfg.get('negative_words',[]))
    if cfg.get('sentiment_pretrained',False):
        sample['pretrained'] = pretrained_sentiment(sample.native,cfg)
    tables = {'sentiment_sample_predictions':sample}
    for method in ['lexicon']+(['pretrained'] if 'pretrained' in sample else []):
        counts = pd.crosstab(sample.period,sample[method]).reindex(columns=['positive','neutral','negative'],fill_value=0)
        share = counts.div(counts.sum(1),axis=0)
        share['n_sample'] = counts.sum(1)
        tables[f'sentiment_{method}_share_sample'] = share
        show_table(method+' · proporsi sampel, bukan accuracy',share)
    if 'pretrained' in sample:
        tables['sentiment_agreement_not_accuracy'] = pd.crosstab(sample.lexicon,sample.pretrained)
    save_tables(tables,cfg['out'])
    print('Tanpa gold: agreement bukan accuracy. Sampel per periode tidak boleh digabung tanpa bobot volume.')
    return tables
