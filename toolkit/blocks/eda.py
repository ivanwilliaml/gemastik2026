from collections import Counter

NOISE_PATTERNS = {
    'url': r'https?://|www\.', 'mention': r'@\w+', 'hashtag': r'#\w+', 'angka': r'\d',
    'emoji_simbol': r'[\U0001F300-\U0001FAFF\u2600-\u27BF]', 'huruf_ulang': r'([^\W\d_])\1{2,}',
    'baca_ulang': r'[!?.,]{2,}', 'html': r'</?[a-zA-Z][^>]*>', 'encoding': r'\ufffd|Ã.|Â.',
    'non_ascii': r'[^\x00-\x7f]', 'spasi_ulang': r' {2,}', 'invisible': r'[\u200b\ufeff\u2060]',
    'kapital': r'(?-i:\b[A-Z]{3,}\b)', 'huruf_angka': r'\b(?=\w*[a-z])(?=\w*\d)\w+\b',
    'slang_contoh': r'\b(?:gk|gak|nggak|ga|bgt|yg|utk|udh|klo|jg)\b',
    'negasi': r'\b(?:tidak|tak|bukan|belum|jangan|tanpa|gak|nggak|ga|not|never)\b'
}


def numeric_eda(frame, text_col, label_cols=(), top=15, sample=5000, seed=42, languages=False):
    """Hitungan dasar seluruh baris; vocabulary/ngram pada sampel yang ukurannya dilaporkan."""
    s = frame[text_col]
    text = s.where(s.map(lambda x: isinstance(x, str)), '').fillna('')
    nonempty = text.str.strip().ne('')
    pseudo = text.str.strip().str.casefold().isin({'nan', 'none', 'null', 'n/a', 'na', '-', '--'})
    quality = pd.Series({'rows': len(s), 'missing': s.isna().sum(), 'blank': (~nonempty).sum(),
        'pseudo_null': pseudo.sum(), 'non_string_nonmissing': (s.notna() & ~s.map(lambda x: isinstance(x, str))).sum(),
        'unique_nonempty': text[nonempty].nunique(), 'duplicate_extra_nonempty': text[nonempty].duplicated().sum(),
        'duplicate_members_nonempty': text[nonempty].duplicated(keep=False).sum()})
    subset = text[nonempty].sample(min(int(sample), int(nonempty.sum())), random_state=seed)
    tokens = subset.str.casefold().str.findall(r'[^\W\d_]+')
    counts = pd.Series(Counter(w for row in tokens for w in row), dtype='int64').sort_values(ascending=False)
    n_tokens = int(counts.sum())
    tables = {'quality': quality.to_frame('n'),
        'missing_unique': pd.DataFrame({'missing': frame.isna().sum(), 'missing_pct': frame.isna().mean()*100,
            'unique': frame.apply(lambda col:col.map(lambda value:json.dumps(value,sort_keys=True) if isinstance(value,(list,dict)) else value).nunique())}),
        'length': pd.DataFrame({'char': text.str.len(), 'word': text.str.split().str.len()}).describe(percentiles=[.01,.5,.9,.95,.99]).T,
        'vocabulary': pd.Series({'sample_rows': len(subset), 'tokens': n_tokens, 'types': len(counts),
            'hapax_types': counts.eq(1).sum(), 'hapax_pct_types': counts.eq(1).mean()*100,
            'hapax_pct_tokens': counts.eq(1).sum()/max(n_tokens,1)*100}).to_frame('value'),
        'top_words': counts.head(top).to_frame('n').assign(pct_tokens=lambda d: d.n/max(n_tokens,1)*100)}
    for n in (2,3):
        grams = Counter(' '.join(row[i:i+n]) for row in tokens for i in range(len(row)-n+1))
        tables[f'ngram_{n}'] = pd.DataFrame(grams.most_common(top), columns=['term', 'n'])
    flags = pd.DataFrame({name: text.str.count(pattern, flags=re.I).gt(0) for name,pattern in NOISE_PATTERNS.items()})
    tables['noise'] = pd.DataFrame({'n_rows': flags.sum(), 'pct_all_rows_overlap_allowed': flags.mean()*100})
    tables['odd_words_candidates'] = counts[[len(w)>=20 or bool(re.search(r'([^\W\d_])\1{2,}',w)) for w in counts.index]].head(top).to_frame('n')
    for label in label_cols:
        known = frame[label].dropna()
        count = known.value_counts()
        tables[f'label_{label}'] = count.to_frame('n').assign(pct_known=lambda d:d.n/max(len(known),1)*100)
        tables[f'conflict_{label}'] = frame.assign(_key=text.str.casefold()).loc[nonempty].groupby('_key')[label].nunique().value_counts().to_frame('text_groups')
    if languages and len(subset):
        from langid.langid import LanguageIdentifier, model
        detector = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        small = subset.sample(min(150,len(subset)),random_state=seed)
        pairs = small.map(lambda t: detector.classify(t[:3000]) if len(t.split())>=5 else ('uncertain',0))
        lang = pairs.map(lambda p:p[0] if p[1]>=.8 else 'uncertain').value_counts()
        tables['language_estimate_sample_only'] = lang.to_frame('n').assign(pct_sample=lang/len(small)*100)
    return tables


def multilabel_eda(frame, labels):
    y = frame[labels].astype(float)
    complete = y.dropna()
    co = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for a in labels:
        for b in labels:
            known = y[[a,b]].notna().all(axis=1)
            co.loc[a,b] = (y.loc[known,a].eq(1)&y.loc[known,b].eq(1)).mean()*100
    return {'label_support': pd.DataFrame({'known':y.notna().sum(), 'unknown':y.isna().sum(),
                'positive':y.eq(1).sum(), 'negative':y.eq(0).sum(), 'positive_pct_known':y.mean()*100}),
        'cardinality_complete_rows': complete.sum(axis=1).value_counts().sort_index().to_frame('n'),
        'label_combinations_complete_rows': complete.value_counts().head(15).to_frame('n'),
        'cooccurrence_pct_pairwise_known': co}


def pair_eda(frame, a, b, target):
    left,right = frame[a].map(safe_text),frame[b].map(safe_text)
    pairs = pd.DataFrame({'a':left.str.casefold(),'b':right.str.casefold()})
    lset = pairs.a.map(lambda t:set(re.findall(r'[^\W\d_]+',t)))
    rset = pairs.b.map(lambda t:set(re.findall(r'[^\W\d_]+',t)))
    overlap = pd.Series([len(x&y)/max(len(x|y),1) for x,y in zip(lset,rset)])
    reversed_keys = set(zip(pairs.b,pairs.a))
    numbers = pd.Series([set(re.findall(r'\d+',x))==set(re.findall(r'\d+',y)) for x,y in zip(left,right)])
    detail = pd.DataFrame({'label_or_score':frame[target].to_numpy(),'jaccard':overlap,
        'abs_word_length_difference':(left.str.split().str.len()-right.str.split().str.len()).abs(),
        'identical':pairs.a.eq(pairs.b),'same_numbers':numbers})
    return {'pair_quality':pd.Series({'rows':len(frame),'identical_sides':pairs.a.eq(pairs.b).sum(),
            'duplicate_ordered_pairs_extra':pairs.duplicated().sum(),
            'pairs_with_reverse_present':sum((x,y) in reversed_keys for x,y in zip(pairs.a,pairs.b)),
            'shared_endpoints':len(set(pairs.a)&set(pairs.b))}).to_frame('n'),
        'pair_relation_statistics':detail.drop(columns='label_or_score').describe().T,
        'pair_relation_by_target':detail.groupby('label_or_score').agg(n=('jaccard','size'),mean_jaccard=('jaccard','mean'),
            mean_length_diff=('abs_word_length_difference','mean'),identical_pct=('identical',lambda v:v.mean()*100))}


def eda_for_task(frame, cfg):
    task = cfg['task'];tables={}
    columns = cfg['pair_cols'] if task in {'nli','similarity'} else [cfg.get('doc_text_col',cfg.get('text_col'))]
    labels = cfg['label_cols'] if task=='multilabel' else [cfg['label_col']] if task in {'classification','nli','similarity'} else []
    for i,col in enumerate(columns):
        part = numeric_eda(frame,col,labels if i==0 else (),cfg.get('eda_top',15),cfg['eda_sample'],cfg['seed'],cfg['eda_languages'])
        tables.update({str(col)+'_'+name:table for name,table in part.items()})
    if task in {'nli','similarity'}:
        tables.update(pair_eda(frame,*cfg['pair_cols'],cfg['label_col']))
    for name,table in tables.items():
        show_table(name,table)
    save_tables(tables,cfg['out'],'raw_eda_')
    return tables
