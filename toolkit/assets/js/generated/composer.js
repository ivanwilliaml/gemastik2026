window.TK = window.TK || {};
TK.composer = {
"version": 2,
"components": {
"core": {
"id": "core",
"title": "Kontrak, data loading & penyimpanan",
"file": "blocks/core.py",
"needs": [],
"packages": [
"numpy",
"pandas",
"scikit-learn",
"scipy"
],
"stage": "helper",
"note": "",
"example": "",
"code": "\"\"\"Kontrak bersama. Tidak memasang paket atau menjalankan model saat disalin.\"\"\"\nimport json\nimport random\nimport re\nimport time\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\n\nMODEL_RUNNERS = {}\nENCODERS = {}\n\n\ndef encode_e5(texts, cfg, role='query'):\n    assert 'e5' in ENCODERS, 'Pasang komponen encoder E5 melalui perakit.'\n    return ENCODERS['e5'](texts,cfg,role)\n\n\ndef register(name):\n    def decorate(fn):\n        MODEL_RUNNERS[name] = fn\n        return fn\n    return decorate\n\n\ndef read_table(path, encoding=\"utf-8\", string_cols=()):\n    path = Path(path)\n    if not path.is_file():\n        raise FileNotFoundError(f\"Isi CFG['data_file'] dengan file Kaggle yang ada: {path}\")\n    if path.suffix == \".parquet\":\n        return pd.read_parquet(path)\n    if path.suffix in {\".jsonl\", \".ndjson\"}:\n        return pd.read_json(path, lines=True)\n    return pd.read_csv(path, sep=\"\\t\" if path.suffix == \".tsv\" else \",\", encoding=encoding,\n        dtype={col:str for col in string_cols} or None)\n\n\ndef seed_all(seed=42):\n    random.seed(seed)\n    np.random.seed(seed)\n\n\ndef target_array(frame, cfg):\n    return frame[cfg['label_cols']].to_numpy(float) if cfg['task']=='multilabel' else frame.y.to_numpy()\n\n\ndef require_kaggle(gpu=False):\n    assert Path('/kaggle/input').exists(), \"Jalankan eksperimen di Kaggle.\"\n    if gpu:\n        import torch\n        assert torch.cuda.is_available(), \"Aktifkan GPU Kaggle.\"\n\n\ndef show_table(name, table):\n    from IPython.display import display\n    print(\"\\n\" + name)\n    display(table.to_frame() if isinstance(table, pd.Series) else table)\n\n\ndef save_tables(tables, out, prefix=\"\"):\n    out = Path(out)\n    out.mkdir(parents=True, exist_ok=True)\n    for name, table in tables.items():\n        table.to_csv(out / f\"{prefix}{name}.csv\", index=True)\n",
"gpu": false
},
"preprocess": {
"id": "preprocess",
"title": "Preprocessing native / normalized / BoW / IndoBERTweet",
"file": "blocks/preprocess.py",
"needs": [
"core"
],
"packages": [
"ftfy>=6.3,<7",
"emoji>=2,<3"
],
"stage": "preprocessing",
"note": "Slang hanya dari CFG[\"slang\"]. Native mempertahankan negasi, emoji, angka, case. Uji normalisasi sebagai eksperimen.",
"example": "clean = make_cleaner(CFG)\ndf['clean'] = df[CFG['text_col']].map(clean)",
"code": "import html\nimport unicodedata\n\nNEGATIONS = {'tidak', 'tak', 'bukan', 'belum', 'jangan', 'tanpa', 'gak', 'nggak', 'ga', 'not', 'never'}\n\n\ndef safe_text(value):\n    \"\"\"Missing tetap menjadi kosong; angka, emoji, negasi, tanda baca dipertahankan.\"\"\"\n    if not isinstance(value, str):\n        return ''\n    import ftfy\n    text = ftfy.fix_text(value, unescape_html=False, uncurl_quotes=False,\n        fix_latin_ligatures=False, fix_character_width=False, remove_control_chars=False, normalization=None)\n    text = unicodedata.normalize('NFC', html.unescape(text))\n    text = text.translate(dict.fromkeys(map(ord, '\\ufeff\\u200b\\u2060\\x00'), None))\n    text = re.sub(r'[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]', ' ', text)\n    return re.sub(r'\\s+', ' ', text).strip()\n\n\ndef preprocess_text(value, preset='native', slang=None, stopwords=()):\n    text = safe_text(value)\n    if preset == 'native':\n        return text\n    if preset == 'indobertweet':\n        import emoji\n        text = text.lower()\n        text = re.sub(r'https?://\\S+|www\\.\\S+|\\burl\\b', 'HTTPURL', text)\n        text = re.sub(r'(?<!\\w)@\\w+|\\buser\\b', '@USER', text)\n        return safe_text(emoji.demojize(text))\n    if preset not in {'normalized', 'bow'}:\n        raise ValueError(f'Preset tidak dikenal: {preset}')\n    text = text.casefold()\n    if slang:\n        text = re.sub(r'\\b[^\\W_]+\\b', lambda m: slang.get(m[0], m[0]), text)\n    if preset == 'bow':\n        text = re.sub(r'https?://\\S+|www\\.\\S+|@\\w+', ' ', text)\n        # Stopword diberikan pengguna, negasi selalu dilindungi; tidak ada stemming otomatis.\n        blocked = set(stopwords) - NEGATIONS\n        text = ' '.join(w for w in re.findall(r'[^\\W\\d_]+', text) if w not in blocked)\n    return safe_text(text)\n\n\ndef make_cleaner(cfg):\n    return lambda value: preprocess_text(value, cfg['preprocess'], cfg.get('slang'), cfg.get('stopwords', ()))\n",
"gpu": false
},
"eda": {
"id": "eda",
"title": "EDA angka: missing → vocabulary → noise",
"file": "blocks/eda.py",
"needs": [
"preprocess"
],
"packages": [],
"stage": "eda",
"note": "Basic/noise seluruh baris; vocab/ngram memakai sampel eksplisit. Persentase noise boleh overlap. Bahasa adalah estimasi sampel.",
"example": "tables = numeric_eda(df, CFG['text_col'], sample=5000)\nfor name, table in tables.items(): show_table(name, table)",
"code": "from collections import Counter\n\nNOISE_PATTERNS = {\n    'url': r'https?://|www\\.', 'mention': r'@\\w+', 'hashtag': r'#\\w+', 'angka': r'\\d',\n    'emoji_simbol': r'[\\U0001F300-\\U0001FAFF\\u2600-\\u27BF]', 'huruf_ulang': r'([^\\W\\d_])\\1{2,}',\n    'baca_ulang': r'[!?.,]{2,}', 'html': r'</?[a-zA-Z][^>]*>', 'encoding': r'\\ufffd|Ã.|Â.',\n    'non_ascii': r'[^\\x00-\\x7f]', 'spasi_ulang': r' {2,}', 'invisible': r'[\\u200b\\ufeff\\u2060]',\n    'kapital': r'(?-i:\\b[A-Z]{3,}\\b)', 'huruf_angka': r'\\b(?=\\w*[a-z])(?=\\w*\\d)\\w+\\b',\n    'slang_contoh': r'\\b(?:gk|gak|nggak|ga|bgt|yg|utk|udh|klo|jg)\\b',\n    'negasi': r'\\b(?:tidak|tak|bukan|belum|jangan|tanpa|gak|nggak|ga|not|never)\\b'\n}\n\n\ndef numeric_eda(frame, text_col, label_cols=(), top=15, sample=5000, seed=42, languages=False):\n    \"\"\"Hitungan dasar seluruh baris; vocabulary/ngram pada sampel yang ukurannya dilaporkan.\"\"\"\n    s = frame[text_col]\n    text = s.where(s.map(lambda x: isinstance(x, str)), '').fillna('')\n    nonempty = text.str.strip().ne('')\n    pseudo = text.str.strip().str.casefold().isin({'nan', 'none', 'null', 'n/a', 'na', '-', '--'})\n    quality = pd.Series({'rows': len(s), 'missing': s.isna().sum(), 'blank': (~nonempty).sum(),\n        'pseudo_null': pseudo.sum(), 'non_string_nonmissing': (s.notna() & ~s.map(lambda x: isinstance(x, str))).sum(),\n        'unique_nonempty': text[nonempty].nunique(), 'duplicate_extra_nonempty': text[nonempty].duplicated().sum(),\n        'duplicate_members_nonempty': text[nonempty].duplicated(keep=False).sum()})\n    subset = text[nonempty].sample(min(int(sample), int(nonempty.sum())), random_state=seed)\n    tokens = subset.str.casefold().str.findall(r'[^\\W\\d_]+')\n    counts = pd.Series(Counter(w for row in tokens for w in row), dtype='int64').sort_values(ascending=False)\n    n_tokens = int(counts.sum())\n    tables = {'quality': quality.to_frame('n'),\n        'missing_unique': pd.DataFrame({'missing': frame.isna().sum(), 'missing_pct': frame.isna().mean()*100,\n            'unique': frame.apply(lambda col:col.map(lambda value:json.dumps(value,sort_keys=True) if isinstance(value,(list,dict)) else value).nunique())}),\n        'length': pd.DataFrame({'char': text.str.len(), 'word': text.str.split().str.len()}).describe(percentiles=[.01,.5,.9,.95,.99]).T,\n        'vocabulary': pd.Series({'sample_rows': len(subset), 'tokens': n_tokens, 'types': len(counts),\n            'hapax_types': counts.eq(1).sum(), 'hapax_pct_types': counts.eq(1).mean()*100,\n            'hapax_pct_tokens': counts.eq(1).sum()/max(n_tokens,1)*100}).to_frame('value'),\n        'top_words': counts.head(top).to_frame('n').assign(pct_tokens=lambda d: d.n/max(n_tokens,1)*100)}\n    for n in (2,3):\n        grams = Counter(' '.join(row[i:i+n]) for row in tokens for i in range(len(row)-n+1))\n        tables[f'ngram_{n}'] = pd.DataFrame(grams.most_common(top), columns=['term', 'n'])\n    flags = pd.DataFrame({name: text.str.count(pattern, flags=re.I).gt(0) for name,pattern in NOISE_PATTERNS.items()})\n    tables['noise'] = pd.DataFrame({'n_rows': flags.sum(), 'pct_all_rows_overlap_allowed': flags.mean()*100})\n    tables['odd_words_candidates'] = counts[[len(w)>=20 or bool(re.search(r'([^\\W\\d_])\\1{2,}',w)) for w in counts.index]].head(top).to_frame('n')\n    for label in label_cols:\n        known = frame[label].dropna()\n        count = known.value_counts()\n        tables[f'label_{label}'] = count.to_frame('n').assign(pct_known=lambda d:d.n/max(len(known),1)*100)\n        tables[f'conflict_{label}'] = frame.assign(_key=text.str.casefold()).loc[nonempty].groupby('_key')[label].nunique().value_counts().to_frame('text_groups')\n    if languages and len(subset):\n        from langid.langid import LanguageIdentifier, model\n        detector = LanguageIdentifier.from_modelstring(model, norm_probs=True)\n        small = subset.sample(min(150,len(subset)),random_state=seed)\n        pairs = small.map(lambda t: detector.classify(t[:3000]) if len(t.split())>=5 else ('uncertain',0))\n        lang = pairs.map(lambda p:p[0] if p[1]>=.8 else 'uncertain').value_counts()\n        tables['language_estimate_sample_only'] = lang.to_frame('n').assign(pct_sample=lang/len(small)*100)\n    return tables\n\n\ndef multilabel_eda(frame, labels):\n    y = frame[labels].astype(float)\n    complete = y.dropna()\n    co = pd.DataFrame(index=labels, columns=labels, dtype=float)\n    for a in labels:\n        for b in labels:\n            known = y[[a,b]].notna().all(axis=1)\n            co.loc[a,b] = (y.loc[known,a].eq(1)&y.loc[known,b].eq(1)).mean()*100\n    return {'label_support': pd.DataFrame({'known':y.notna().sum(), 'unknown':y.isna().sum(),\n                'positive':y.eq(1).sum(), 'negative':y.eq(0).sum(), 'positive_pct_known':y.mean()*100}),\n        'cardinality_complete_rows': complete.sum(axis=1).value_counts().sort_index().to_frame('n'),\n        'label_combinations_complete_rows': complete.value_counts().head(15).to_frame('n'),\n        'cooccurrence_pct_pairwise_known': co}\n\n\ndef pair_eda(frame, a, b, target):\n    left,right = frame[a].map(safe_text),frame[b].map(safe_text)\n    pairs = pd.DataFrame({'a':left.str.casefold(),'b':right.str.casefold()})\n    lset = pairs.a.map(lambda t:set(re.findall(r'[^\\W\\d_]+',t)))\n    rset = pairs.b.map(lambda t:set(re.findall(r'[^\\W\\d_]+',t)))\n    overlap = pd.Series([len(x&y)/max(len(x|y),1) for x,y in zip(lset,rset)])\n    reversed_keys = set(zip(pairs.b,pairs.a))\n    numbers = pd.Series([set(re.findall(r'\\d+',x))==set(re.findall(r'\\d+',y)) for x,y in zip(left,right)])\n    detail = pd.DataFrame({'label_or_score':frame[target].to_numpy(),'jaccard':overlap,\n        'abs_word_length_difference':(left.str.split().str.len()-right.str.split().str.len()).abs(),\n        'identical':pairs.a.eq(pairs.b),'same_numbers':numbers})\n    return {'pair_quality':pd.Series({'rows':len(frame),'identical_sides':pairs.a.eq(pairs.b).sum(),\n            'duplicate_ordered_pairs_extra':pairs.duplicated().sum(),\n            'pairs_with_reverse_present':sum((x,y) in reversed_keys for x,y in zip(pairs.a,pairs.b)),\n            'shared_endpoints':len(set(pairs.a)&set(pairs.b))}).to_frame('n'),\n        'pair_relation_statistics':detail.drop(columns='label_or_score').describe().T,\n        'pair_relation_by_target':detail.groupby('label_or_score').agg(n=('jaccard','size'),mean_jaccard=('jaccard','mean'),\n            mean_length_diff=('abs_word_length_difference','mean'),identical_pct=('identical',lambda v:v.mean()*100))}\n\n\ndef eda_for_task(frame, cfg):\n    task = cfg['task'];tables={}\n    columns = cfg['pair_cols'] if task in {'nli','similarity'} else [cfg.get('doc_text_col',cfg.get('text_col'))]\n    labels = cfg['label_cols'] if task=='multilabel' else [cfg['label_col']] if task in {'classification','nli','similarity'} else []\n    for i,col in enumerate(columns):\n        part = numeric_eda(frame,col,labels if i==0 else (),cfg.get('eda_top',15),cfg['eda_sample'],cfg['seed'],cfg['eda_languages'])\n        tables.update({str(col)+'_'+name:table for name,table in part.items()})\n    if task in {'nli','similarity'}:\n        tables.update(pair_eda(frame,*cfg['pair_cols'],cfg['label_col']))\n    for name,table in tables.items():\n        show_table(name,table)\n    save_tables(tables,cfg['out'],'raw_eda_')\n    return tables\n",
"gpu": false
},
"groups": {
"id": "groups",
"title": "Grup pengguna, duplicate & connected endpoint",
"file": "blocks/groups.py",
"needs": [
"preprocess"
],
"packages": [],
"stage": "validation",
"note": "",
"example": "",
"code": "from sklearn.model_selection import GroupShuffleSplit\n\ndef connected_groups(frame, text_cols, group_col=None, normalized_cols=(), shared_endpoints=False):\n    \"\"\"Gabung pengguna, duplikat native, dan collision normalisasi sebelum split.\"\"\"\n    parent = np.arange(len(frame))\n    def root(i):\n        while parent[i] != i:\n            parent[i] = parent[parent[i]]\n            i = parent[i]\n        return i\n    seen = {}\n    for i,(_,row) in enumerate(frame.iterrows()):\n        keys = [(('endpoint' if shared_endpoints else col), safe_text(row[col]).casefold()) for col in text_cols]\n        keys += [('endpoint' if shared_endpoints else 'normalized_'+col, str(row[col]).casefold()) for col in normalized_cols]\n        if group_col and pd.notna(row[group_col]):\n            keys.append(('user',str(row[group_col])))\n        for key in keys:\n            if not key[1]:\n                continue\n            if key in seen:\n                parent[root(i)] = root(seen[key])\n            else:\n                seen[key] = i\n    return np.array([root(i) for i in range(len(frame))])\n\n",
"gpu": false
},
"split": {
"id": "split",
"title": "Split grup & partial multilabel",
"file": "blocks/split.py",
"needs": [
"groups"
],
"packages": [],
"stage": "validation",
"note": "Kelompokkan pengguna + native duplicate + collision normalisasi. Pair memakai connected endpoints. NLI: e/n/c eksplisit. Unknown label tetap unknown.",
"example": "prepared = prepare_supervised(df, CFG)\ndata = split_supervised(prepared, CFG)",
"code": "from sklearn.model_selection import GroupShuffleSplit\n\n\n\n\ndef prepare_supervised(raw, cfg):\n    cleaner = make_cleaner(cfg)\n    frame = raw.copy().reset_index(drop=True)\n    frame['_row_id'] = np.arange(len(frame))\n    cols = cfg['pair_cols'] if cfg['task'] in {'nli','similarity'} else [cfg['text_col']]\n    assert set(cols).issubset(frame), f'Kolom teks perlu {cols}; tersedia {frame.columns.tolist()}'\n    for col, dest in zip(cols, ['a','b']):\n        frame[dest] = frame[col].map(cleaner)\n    valid = frame['a'].ne('')\n    if len(cols)==2:\n        valid &= frame.b.ne('')\n    labels = cfg['label_cols'] if cfg['task']=='multilabel' else [cfg['label_col']]\n    assert set(labels).issubset(frame), f'Kolom target perlu {labels}'\n    if cfg['task']=='multilabel':\n        for col in labels:\n            values = pd.to_numeric(frame[col],errors='coerce')\n            assert not (frame[col].notna() & values.isna()).any(), f'Label {col} tidak numerik'\n            assert values.dropna().isin([0,1]).all(), f'Label {col} bukan 0/1/missing'\n            frame[col] = values\n        valid &= frame[labels].notna().any(axis=1)\n    elif cfg['task']=='similarity':\n        frame['y'] = pd.to_numeric(frame[cfg['label_col']],errors='coerce')\n        valid &= np.isfinite(frame.y)\n        assert frame.loc[valid,'y'].between(*cfg['score_range']).all(), 'Skor di luar score_range'\n    else:\n        valid &= frame[cfg['label_col']].notna()\n        frame['label_name'] = frame[cfg['label_col']].astype(str).str.strip()\n        if cfg['task']=='nli':\n            mapping = cfg['nli_label_map']\n            frame['label_name'] = frame.label_name.str.casefold().map(mapping)\n            assert frame.loc[valid,'label_name'].notna().all(), 'Lengkapi nli_label_map untuk label data Anda'\n    print('Baris invalid dibuang:',int((~valid).sum()))\n    frame = frame.loc[valid].reset_index(drop=True)\n    normalized = ['a','b'] if len(cols)==2 else ['a']\n    frame['_group'] = connected_groups(frame, cols, cfg.get('group_col'), normalized, shared_endpoints=len(cols)==2)\n    assert frame._group.nunique()>=5, 'Komponen grup terlalu sedikit; audit shared endpoint/duplikat/pengguna.'\n    return frame\n\n\ndef split_supervised(frame, cfg):\n    seed = cfg['seed']\n    if cfg['task']=='multilabel':\n        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit\n        pos = frame.groupby('_group')[cfg['label_cols']].max().fillna(0)\n        known = frame[cfg['label_cols']].notna().groupby(frame._group).max()\n        ids, profile = pos.index.to_numpy(), np.hstack([pos, known.astype(int)])\n        first = MultilabelStratifiedShuffleSplit(1,test_size=.2,random_state=seed)\n        rest, test = next(first.split(ids,profile))\n        fit, val = next(MultilabelStratifiedShuffleSplit(1,test_size=.25,random_state=seed+1).split(ids[rest],profile[rest]))\n        partitions = [ids[rest][fit],ids[rest][val],ids[test]]\n    else:\n        rest, test = next(GroupShuffleSplit(1,test_size=.2,random_state=seed).split(frame,groups=frame._group))\n        fit, val = next(GroupShuffleSplit(1,test_size=.25,random_state=seed+1).split(frame.iloc[rest],groups=frame.iloc[rest]._group))\n        partitions = [frame.iloc[rest[fit]]._group.unique(),frame.iloc[rest[val]]._group.unique(),frame.iloc[test]._group.unique()]\n    parts = {name:frame[frame._group.isin(ids)].reset_index(drop=True) for name,ids in zip(['train','val','test'],partitions)}\n    if cfg['task'] in {'classification','nli'}:\n        names = ['entailment','neutral','contradiction'] if cfg['task']=='nli' else sorted(parts['train'].label_name.unique())\n        assert len(names)>=2, 'Train memerlukan minimal dua kelas'\n        for name,part in parts.items():\n            part['y'] = part.label_name.map({v:i for i,v in enumerate(names)})\n            assert part.y.notna().all(), f'{name} mengandung kelas yang tidak ada di train; atur ulang split/grup.'\n            part['y'] = part.y.astype(int)\n        assert set(parts['train'].y)==set(range(len(names))), 'Train tidak mencakup semua kelas NLI'\n    else:\n        names = cfg['label_cols'] if cfg['task']=='multilabel' else ['score']\n    parts['labels'] = list(names)\n    for a,b in [('train','val'),('train','test'),('val','test')]:\n        assert set(parts[a]._group).isdisjoint(parts[b]._group)\n    print('Split baris:',{k:len(parts[k]) for k in ['train','val','test']}, '| label order:',names)\n    return parts\n\n\n",
"gpu": false
},
"metrics": {
"id": "metrics",
"title": "Metrik klasifikasi, partial multilabel & STS",
"file": "blocks/metrics.py",
"needs": [
"core"
],
"packages": [],
"stage": "evaluation",
"note": "",
"example": "metrics = task_metrics(y_true, probabilities, CFG['task'])",
"code": "from sklearn.metrics import (f1_score, log_loss, classification_report, confusion_matrix,\n    average_precision_score, precision_recall_fscore_support, mean_absolute_error, mean_squared_error)\nfrom scipy.stats import pearsonr, spearmanr\n\n\ndef masked_multilabel_metrics(y, p, threshold=.5):\n    y = np.asarray(y,float).copy()\n    y[y<0] = np.nan\n    p = np.asarray(p,float)\n    assert p.shape==y.shape and np.isfinite(p).all()\n    known = np.isfinite(y)\n    pred = p>=threshold\n    per, ap = [], []\n    for j in range(y.shape[1]):\n        k = known[:,j]\n        per.append(f1_score(y[k,j],pred[k,j],zero_division=0) if k.any() else np.nan)\n        ap.append(average_precision_score(y[k,j],p[k,j]) if np.any(y[k,j]==1) else np.nan)\n    complete = known.all(axis=1)\n    return {'macro_f1':float(np.nanmean(per)) if np.isfinite(per).any() else np.nan,\n        'micro_f1':f1_score(y[known],pred[known],zero_division=0) if known.any() else np.nan,\n        'macro_AP':float(np.nanmean(ap)) if np.isfinite(ap).any() else np.nan,\n        'hamming_known':float(np.mean(y[known]!=pred[known])) if known.any() else np.nan,\n        'exact_match_complete':float(np.all(y[complete]==pred[complete],axis=1).mean()) if complete.any() else np.nan,\n        'n_complete':int(complete.sum()), 'n_known_cells':int(known.sum())}\n\n\ndef task_metrics(y, p, task):\n    p = np.asarray(p)\n    assert len(y)==len(p) and np.isfinite(p).all()\n    if task=='similarity':\n        y, p = np.asarray(y).ravel(), p.ravel()\n        variable = len(y)>1 and np.std(y)>0 and np.std(p)>0\n        return {'pearson':float(pearsonr(y,p).statistic) if variable else np.nan,\n            'spearman':float(spearmanr(y,p).statistic) if variable else np.nan,\n            'MAE':mean_absolute_error(y,p), 'RMSE':float(np.sqrt(mean_squared_error(y,p)))}\n    if task=='multilabel':\n        return masked_multilabel_metrics(y,p)\n    assert p.ndim==2 and np.allclose(p.sum(1),1,atol=1e-4)\n    pred = p.argmax(1)\n    return {'macro_f1':f1_score(y,pred,labels=np.arange(p.shape[1]),average='macro',zero_division=0),\n        'accuracy':float(np.mean(y==pred)), 'log_loss':log_loss(y,p,labels=np.arange(p.shape[1]))}\n",
"gpu": false
},
"error_analysis": {
"id": "error_analysis",
"title": "Support, FP/FN, confusion & slice",
"file": "blocks/error_analysis.py",
"needs": [
"metrics",
"eda"
],
"packages": [],
"stage": "evaluation",
"note": "",
"example": "",
"code": "\"\"\"Audit kesalahan; dipasang terpisah dari hitungan metrik/post-processing.\"\"\"\n\ndef error_tables(frame, y, p, cfg, labels):\n    task = cfg['task']\n    rows = frame[['_row_id','a']+(['b'] if 'b' in frame else [])].copy()\n    if task=='multilabel':\n        known = np.isfinite(y)\n        pred = p>=.5\n        detail = []\n        for j,label in enumerate(labels):\n            k = known[:,j]\n            pr,rc,f1,_ = precision_recall_fscore_support(y[k,j],pred[k,j],average='binary',zero_division=0) if k.any() else (np.nan,)*4\n            detail.append({'label':label,'known':int(k.sum()),'positive':int(np.sum(y[k,j]==1)),\n                'precision':pr,'recall':rc,'f1':f1,\n                'FP':int(np.sum((y[k,j]==0)&pred[k,j])), 'FN':int(np.sum((y[k,j]==1)&~pred[k,j]))})\n        rows['truth'] = [[labels[j] for j in range(len(labels)) if y[i,j]==1] for i in range(len(y))]\n        rows['predicted'] = [[labels[j] for j in range(len(labels)) if pred[i,j]] for i in range(len(y))]\n        rows['unknown'] = (~known).sum(1)\n        rows['error_count'] = ((y!=pred)&known).sum(1)\n        tables = {'per_label':pd.DataFrame(detail)}\n    elif task=='similarity':\n        rows['truth'],rows['predicted'] = y,p.ravel()\n        rows['absolute_error'] = np.abs(y-p.ravel())\n        tables = {}\n    else:\n        pred = p.argmax(1)\n        rows['truth'] = [labels[i] for i in y]\n        rows['predicted'] = [labels[i] for i in pred]\n        rows['confidence'] = p.max(1)\n        rows['error'] = y!=pred\n        tables = {'per_class':pd.DataFrame(classification_report(y,pred,labels=np.arange(len(labels)),target_names=labels,output_dict=True,zero_division=0)).T,\n            'confusion_counts':pd.DataFrame(confusion_matrix(y,pred,labels=np.arange(len(labels))),index=labels,columns=labels)}\n    tables['errors'] = rows\n    conditions = {'short_5_words':frame.a.str.split().str.len().le(5), 'long_40_words':frame.a.str.split().str.len().gt(40)}\n    for name in ['negasi','emoji_simbol','slang_contoh']:\n        conditions[name] = frame.a.str.count(NOISE_PATTERNS[name],flags=re.I).gt(0)\n    slices = []\n    for name,mask in conditions.items():\n        k = mask.to_numpy()\n        if k.any():\n            row = task_metrics(y[k],p[k],task)\n            if task=='multilabel':\n                row.pop('macro_AP',None)  # Decisions after threshold are not probability AP.\n            slices.append({'slice':name,'n':int(k.sum()),**row})\n    tables['slices'] = pd.DataFrame(slices)\n    return tables\n\n",
"gpu": false
},
"post": {
"id": "post",
"title": "Cross-fit threshold / bias / calibration / ensemble",
"file": "blocks/postprocess.py",
"needs": [
"metrics",
"preprocess"
],
"packages": [],
"stage": "postprocessing",
"note": "Parameter memakai validation grup. Holdout tidak boleh memilih model/aturan. Duplicate consistency, audited rules, dan hierarki tersedia sebagai fungsi opsional.",
"example": "policy, board = select_policy(validation_predictions, y_val, val_groups, CFG)",
"code": "from sklearn.model_selection import GroupKFold\nfrom scipy.special import softmax\nfrom scipy.optimize import minimize_scalar\n\n\ndef fit_post(p, y, route, cfg):\n    if route=='identity':\n        return None\n    if route=='isotonic':\n        from sklearn.isotonic import IsotonicRegression\n        return IsotonicRegression(out_of_bounds='clip').fit(p.ravel(),y)\n    if route=='temperature':\n        z = np.log(np.clip(p,1e-7,1))\n        return float(minimize_scalar(lambda t:log_loss(y,softmax(z/t,axis=1),labels=np.arange(p.shape[1])),bounds=(.3,4),method='bounded').x)\n    if route=='class_bias':\n        bias = np.zeros(p.shape[1])\n        for _ in range(2):\n            for j in range(len(bias)):\n                best,score = bias[j],-1\n                for v in [-.8,-.4,0.,.4,.8]:\n                    trial = bias.copy();trial[j]=v\n                    s = task_metrics(y,apply_post(p,'class_bias',trial),cfg['task'])['macro_f1']\n                    if (s,-abs(v))>(score,-abs(best)):\n                        best,score = v,s\n                bias[j] = best\n        return bias\n    if route in {'global_threshold','label_threshold'}:\n        grid = [.03,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.7,.8,.9]\n        if route=='global_threshold':\n            return max(grid,key=lambda c:(masked_multilabel_metrics(y,p,c)['macro_f1'],-abs(c-.5)))\n        cuts = np.full(p.shape[1],.5)\n        for j in range(p.shape[1]):\n            k = np.isfinite(y[:,j]);target=y[k,j]\n            if min(np.sum(target==1),np.sum(target==0))<cfg.get('threshold_min_support',5):\n                continue\n            cuts[j] = max(grid,key=lambda c:(f1_score(target,p[k,j]>=c,zero_division=0),-abs(c-.5)))\n        return cuts\n    raise ValueError(f'Post-processing tidak dikenal: {route}')\n\n\ndef apply_post(p, route, fitted):\n    if route=='identity':\n        return p.copy()\n    if route=='isotonic':\n        return fitted.predict(p.ravel())\n    if route=='temperature':\n        return softmax(np.log(np.clip(p,1e-7,1))/fitted,axis=1)\n    if route=='class_bias':\n        return softmax(np.log(np.clip(p,1e-7,1))+fitted,axis=1)\n    if route in {'global_threshold','label_threshold'}:\n        return (p>=fitted).astype(float)\n    raise ValueError(route)\n\n\ndef select_policy(vp, y, groups, cfg):\n    \"\"\"Cross-fit parameter post di validation, kemudian refit; holdout tidak dibaca.\"\"\"\n    candidates = dict(vp)\n    base = [name for name in vp if name!='dummy']\n    if cfg.get('ensemble',False) and len(base)>1:\n        candidates['ensemble'] = np.mean([vp[name] for name in base],axis=0)\n    nfold = min(3,len(np.unique(groups)))\n    routes = ['identity']+cfg.get('post',[])\n    allowed = {'identity','isotonic'} if cfg['task']=='similarity' else {'identity','global_threshold','label_threshold'} if cfg['task']=='multilabel' else {'identity','temperature','class_bias'}\n    assert set(routes)<=allowed, f'Post-processing tidak cocok dengan {cfg[\"task\"]}: {set(routes)-allowed}'\n    if nfold<2 and len(routes)>1:\n        print('Post dinonaktifkan: validation kurang dari dua grup.')\n        routes = ['identity']\n    folds = list(GroupKFold(n_splits=nfold).split(y,groups=groups)) if nfold>=2 else []\n    board = []\n    for name,p in candidates.items():\n        for route in routes:\n            cv = p.copy()\n            if route!='identity':\n                for a,b in folds:\n                    cv[b] = apply_post(p[b],route,fit_post(p[a],y[a],route,cfg))\n            metrics = task_metrics(y,cv,cfg['task'])\n            if cfg['task']=='multilabel':\n                metrics['macro_AP'] = masked_multilabel_metrics(y,p)['macro_AP']\n            board.append({'model':name,'route':route,**metrics})\n    board = pd.DataFrame(board)\n    metric,ascending = ('RMSE',True) if cfg['task']=='similarity' else ('macro_f1',False)\n    # Identitas menang saat seri; temperature dapat dipilih untuk log-loss yang membaik dengan F1 sama.\n    sort = [metric]+(['log_loss'] if 'log_loss' in board else [])\n    board = board.sort_values(sort,ascending=[ascending]+([True] if len(sort)>1 else []),kind='stable')\n    best = board.iloc[0]\n    params = fit_post(candidates[best.model],y,best.route,cfg)\n    return {'model':best.model,'route':best.route,'params':params,'members':base},board\n\n\ndef predict_policy(hp, policy):\n    p = np.mean([hp[name] for name in policy['members']],axis=0) if policy['model']=='ensemble' else hp[policy['model']]\n    return apply_post(p,policy['route'],policy['params']),p\n\n\ndef duplicate_consistency(probabilities, texts):\n    \"\"\"Opsional: rata-rata prediksi exact key dalam satu batch inference, tanpa label.\"\"\"\n    keys = pd.Series([safe_text(t).casefold() for t in texts])\n    values = pd.DataFrame(probabilities)\n    assert len(keys)==len(values)\n    return values.groupby(keys).transform('mean').to_numpy()\n\n\ndef audited_rules(probabilities, texts, rules):\n    \"\"\"rules=[{'pattern':..., 'class_index':..., 'audited':True}]. Jangan belajar aturan dari holdout.\"\"\"\n    p = probabilities.copy()\n    for rule in rules:\n        assert rule.get('audited') is True, 'Aturan harus diaudit dari train dahulu.'\n        for i,text in enumerate(texts):\n            if re.search(rule['pattern'],text,flags=re.I):\n                p[i] = 0\n                p[i,rule['class_index']] = 1\n    return p\n\n\ndef enforce_hierarchy(decisions, parent, children):\n    \"\"\"Opsional multilabel. Aktifkan hanya setelah audit anotasi train dan validasi grup.\"\"\"\n    result = decisions.astype(bool).copy()\n    result[:,parent] |= result[:,children].any(axis=1)\n    return result\n",
"gpu": false
},
"linear": {
"id": "linear",
"title": "Head sparse/dense dan fitur pasangan",
"file": "blocks/linear.py",
"needs": [
"core"
],
"packages": [],
"stage": "model",
"note": "",
"example": "",
"code": "from scipy.sparse import hstack\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.pipeline import FeatureUnion\nfrom sklearn.linear_model import LogisticRegression, Ridge\nfrom sklearn.dummy import DummyClassifier\n\n\ndef vectorizer(kind='word', features=60000):\n    word = TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=features,sublinear_tf=True)\n    if kind=='word':\n        return word\n    return FeatureUnion([('word',word),('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=features,sublinear_tf=True))])\n\n\ndef pair_matrix(a, b, symmetric=False):\n    delta = abs(a-b)\n    return hstack([delta,a.multiply(b)] if symmetric else [a,b,delta,a.multiply(b)],format='csr')\n\n\ndef binary_heads(x, y, v, h, seed=42):\n    vp,hp = np.zeros((v.shape[0],y.shape[1])),np.zeros((h.shape[0],y.shape[1]))\n    for j in range(y.shape[1]):\n        k = np.isfinite(y[:,j]);values = np.unique(y[k,j])\n        if not len(values):\n            print('Label tanpa known train, prior .5:',j)\n            vp[:,j],hp[:,j] = .5,.5\n            continue\n        model = DummyClassifier(strategy='constant',constant=values[0]) if len(values)==1 else LogisticRegression(C=2,max_iter=1200,class_weight='balanced',random_state=seed)\n        model.fit(x[k],y[k,j])\n        if 1 in model.classes_:\n            pos = list(model.classes_).index(1)\n            vp[:,j],hp[:,j] = model.predict_proba(v)[:,pos],model.predict_proba(h)[:,pos]\n    return vp,hp\n\n\ndef fit_head(x, v, h, data, cfg):\n    y = target_array(data['train'],cfg)\n    if cfg['task']=='multilabel':\n        return binary_heads(x,y,v,h,cfg['seed'])\n    if cfg['task']=='similarity':\n        model = Ridge(alpha=5).fit(x,y)\n        return tuple(np.clip(model.predict(m),*cfg['score_range']) for m in [v,h])\n    model = LogisticRegression(C=2,max_iter=1200,class_weight='balanced',random_state=cfg['seed']).fit(x,y)\n    assert np.array_equal(model.classes_,np.arange(len(data['labels'])))\n    return model.predict_proba(v),model.predict_proba(h)\n\n\ndef fit_tfidf(data, cfg, kind):\n    vec = vectorizer(kind,cfg.get('max_features',60000))\n    pair = cfg['task'] in {'nli','similarity'}\n    train = data['train']\n    vec.fit(pd.concat([train.a,train.b]) if pair else train.a)\n    matrices = []\n    for name in ['train','val','test']:\n        frame = data[name]\n        a = vec.transform(frame.a)\n        matrices.append(pair_matrix(a,vec.transform(frame.b),cfg['task']=='similarity') if pair else a)\n    return fit_head(*matrices,data,cfg)\n\n\n@register('tfidf_word')\ndef fit_word(data, cfg):\n    return fit_tfidf(data,cfg,'word')\n\n\n@register('tfidf_union')\ndef fit_word_char(data, cfg):\n    return fit_tfidf(data,cfg,'union')\n",
"gpu": false
},
"encoder": {
"id": "encoder",
"title": "E5 masked mean pooling + prefix",
"file": "blocks/embedding.py",
"needs": [
"core"
],
"packages": [
"transformers==5.20.2",
"accelerate>=1,<2",
"sentencepiece>=0.2,<1",
"torch"
],
"stage": "representation",
"note": "",
"example": "",
"code": "def e5_encode(texts, cfg, role='query'):\n    \"\"\"E5: masked mean pooling, L2 normalisasi, prefix query/passage yang eksplisit.\"\"\"\n    require_kaggle(gpu=True)\n    import gc\n    import torch\n    import torch.nn.functional as F\n    from transformers import AutoTokenizer, AutoModel\n    assert role in {'query','passage'}\n    repo = cfg.get('e5_model','intfloat/multilingual-e5-base')\n    tok = AutoTokenizer.from_pretrained(repo,local_files_only=cfg.get('offline',False))\n    model = AutoModel.from_pretrained(repo,local_files_only=cfg.get('offline',False)).cuda().eval()\n    rows = []\n    texts = list(texts)\n    with torch.inference_mode():\n        for start in range(0,len(texts),cfg.get('embed_batch',32)):\n            batch = [role+': '+str(t) for t in texts[start:start+cfg.get('embed_batch',32)]]\n            enc = tok(batch,padding=True,truncation=True,max_length=cfg.get('max_len',256),return_tensors='pt').to('cuda')\n            hidden = model(**enc).last_hidden_state.float()\n            pooled = (hidden*enc.attention_mask.unsqueeze(-1)).sum(1)/enc.attention_mask.sum(1,keepdim=True).clamp(min=1)\n            rows.append(F.normalize(pooled,p=2,dim=1).cpu().numpy())\n    result = np.vstack(rows) if rows else np.empty((0,model.config.hidden_size),dtype='float32')\n    del model\n    gc.collect();torch.cuda.empty_cache()\n    return result\n\n\nENCODERS['e5'] = e5_encode\n",
"gpu": true
},
"transformer": {
"id": "transformer",
"title": "Trainer classification / regression / masked BCE",
"file": "blocks/transformer.py",
"needs": [
"metrics",
"preprocess"
],
"packages": [
"transformers==5.20.2",
"accelerate>=1,<2",
"datasets>=4,<5",
"sentencepiece>=0.2,<1",
"torch"
],
"stage": "model",
"note": "Masked BCE untuk unknown; sigmoid multilabel. Model checkpoint hanya dari validation; NLI mempertahankan arah premise → hypothesis.",
"example": "",
"code": "PRETRAINED_MODELS = {\n    'indobertweet':'indolem/indobertweet-base-uncased',\n    'indobert_p1':'indobenchmark/indobert-base-p1',\n    'xlmr':'FacebookAI/xlm-roberta-base',\n    'indolem':'indolem/indobert-base-uncased',\n    'nusabert':'LazarusNLP/NusaBERT-base'\n}\n\n\ndef fit_pretrained(data, cfg, name):\n    require_kaggle(gpu=True)\n    import gc\n    import torch\n    from scipy.special import expit, softmax\n    from datasets import Dataset, Value, Sequence\n    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, set_seed\n    set_seed(cfg['seed'])\n    repo = cfg.get('model_paths',{}).get(name,PRETRAINED_MODELS[name])\n    offline = cfg.get('offline',False)\n    tok = AutoTokenizer.from_pretrained(repo,local_files_only=offline)\n    task = cfg['task'];multi=task=='multilabel';reg=task=='similarity'\n    problem = 'multi_label_classification' if multi else 'regression' if reg else 'single_label_classification'\n    model = AutoModelForSequenceClassification.from_pretrained(repo,local_files_only=offline,\n        num_labels=1 if reg else len(data['labels']),problem_type=problem,ignore_mismatched_sizes=True)\n    if not reg:\n        model.config.id2label = dict(enumerate(data['labels']))\n        model.config.label2id = {name:i for i,name in enumerate(data['labels'])}\n    def input_text(series):\n        return series.map(lambda t:preprocess_text(t,'indobertweet')).tolist() if name=='indobertweet' else series.tolist()\n    def encoded(frame, labels=True):\n        a = input_text(frame.a)\n        payload = tok(a,input_text(frame.b) if 'b' in frame else None,truncation=True,max_length=cfg['max_len'])\n        if labels:\n            y = target_array(frame,cfg)\n            payload['labels'] = np.where(np.isfinite(y),y,-1).astype('float32') if multi else y.astype('float32' if reg else 'int64')\n        ds = Dataset.from_dict(payload)\n        if labels and reg:\n            ds = ds.cast_column('labels',Value('float32'))\n        if labels and multi:\n            ds = ds.cast_column('labels',Sequence(Value('float32')))\n        return ds\n    y = target_array(data['train'],cfg)\n    pos_weight = np.sqrt(np.clip(np.sum(y==0,axis=0)/np.maximum(np.sum(y==1,axis=0),1),1,20)) if multi else None\n    class MaskedTrainer(Trainer):\n        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):\n            labels = inputs.pop('labels').float();mask=labels>=0\n            output = model(**inputs)\n            loss = torch.nn.functional.binary_cross_entropy_with_logits(output.logits.float(),labels.clamp(0,1),\n                pos_weight=torch.as_tensor(pos_weight,device=labels.device,dtype=torch.float32),reduction='none')\n            loss = (loss*mask).sum()/mask.sum().clamp(min=1)\n            return (loss,output) if return_outputs else loss\n    def probability(z):\n        return z.ravel() if reg else expit(z) if multi else softmax(z,axis=1)\n    def metric(prediction):\n        score = task_metrics(prediction.label_ids,probability(prediction.predictions),task)\n        return {'selection':-score['RMSE'] if reg else score['macro_f1']}\n    out = Path(cfg['out'])/('model_'+name)\n    args = TrainingArguments(output_dir=str(out),num_train_epochs=cfg['epochs'],learning_rate=cfg['learning_rate'],\n        per_device_train_batch_size=cfg['batch_size'],per_device_eval_batch_size=cfg['batch_size']*2,\n        gradient_accumulation_steps=cfg.get('accumulation',1),fp16=True,weight_decay=.01,\n        eval_strategy='epoch',save_strategy='epoch',save_only_model=True,save_total_limit=1,\n        load_best_model_at_end=True,metric_for_best_model='selection',greater_is_better=True,\n        logging_strategy='epoch',disable_tqdm=True,report_to='none',seed=cfg['seed'])\n    trainer_class = MaskedTrainer if multi else Trainer\n    fit_frame = data['train']\n    if reg:\n        swapped = fit_frame.copy()\n        swapped['a'],swapped['b'] = fit_frame.b,fit_frame.a\n        fit_frame = pd.concat([fit_frame,swapped],ignore_index=True)\n    trainer = trainer_class(model=model,args=args,train_dataset=encoded(fit_frame),eval_dataset=encoded(data['val']),\n        data_collator=DataCollatorWithPadding(tok),compute_metrics=metric)\n    if multi:\n        trainer.model_accepts_loss_kwargs = False\n    trainer.train()\n    results = []\n    for split in ['val','test']:\n        frame = data[split]\n        scores = probability(trainer.predict(encoded(frame,False)).predictions)\n        if reg:\n            swapped = frame.copy()\n            swapped['a'],swapped['b'] = frame.b,frame.a\n            scores = (scores+probability(trainer.predict(encoded(swapped,False)).predictions))/2\n        results.append(scores)\n    results = tuple(results)\n    if reg:\n        results = tuple(np.clip(p,*cfg['score_range']) for p in results)\n    trainer.save_model(str(out));tok.save_pretrained(str(out))\n    (out/'model_provenance.json').write_text(json.dumps({'repo':repo,'commit':getattr(model.config,'_commit_hash',None),'labels':data['labels']},indent=2))\n    del trainer,model\n    gc.collect();torch.cuda.empty_cache()\n    return results\n\n\nfor _model_name in PRETRAINED_MODELS:\n    MODEL_RUNNERS[_model_name] = lambda data,cfg,name=_model_name:fit_pretrained(data,cfg,name)\n",
"gpu": true
},
"supervised_run": {
"id": "supervised_run",
"title": "Jalankan model terpilih → freeze → audit holdout",
"file": "blocks/run_supervised.py",
"needs": [
"post",
"split",
"error_analysis"
],
"packages": [],
"stage": "evaluation",
"note": "",
"example": "",
"code": "def run_supervised(data, cfg):\n    \"\"\"Model plug-in semuanya menghasilkan (validation, holdout) dengan urutan label yang sama.\"\"\"\n    seed_all(cfg['seed'])\n    out = Path(cfg['out']);out.mkdir(parents=True,exist_ok=True)\n    vp,hp,times = {},{},[]\n    for name in cfg['models']:\n        assert name in MODEL_RUNNERS, f'Blok model {name} belum disalin'\n        start = time.perf_counter()\n        vp[name],hp[name] = MODEL_RUNNERS[name](data,cfg)\n        for split,p in [('val',vp[name]),('test',hp[name])]:\n            assert len(p)==len(data[split]) and np.isfinite(p).all(), f'Output {name}/{split} invalid'\n            if cfg['task']=='multilabel':\n                assert p.shape==(len(data[split]),len(data['labels']))\n        times.append({'model':name,'seconds':time.perf_counter()-start})\n    yval = target_array(data['val'],cfg)\n    policy,board = select_policy(vp,yval,data['val']._group.to_numpy(),cfg)\n    show_table('Validation cross-fit (pengembangan, bukan test)',board)\n    summary = {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in policy.items() if k!='params'}\n    params = policy['params']\n    if policy['route']=='isotonic':\n        summary['params'] = {'x':params.X_thresholds_.tolist(),'y':params.y_thresholds_.tolist()}\n    else:\n        summary['params'] = params.tolist() if isinstance(params,np.ndarray) else params\n    (out/'selection_frozen_before_test.json').write_text(json.dumps(summary,indent=2))\n    # Baru sekarang baca target holdout. Tidak ada pemilihan ulang berdasarkan tabel berikut.\n    ytest = target_array(data['test'],cfg)\n    final,raw = predict_policy(hp,policy)\n    rows = [{'model':name,**task_metrics(ytest,p,cfg['task'])} for name,p in hp.items()]\n    score = task_metrics(ytest,final,cfg['task'])\n    if cfg['task']=='multilabel':\n        score['macro_AP'] = masked_multilabel_metrics(ytest,raw)['macro_AP']\n    rows.append({'model':'FINAL',**score})\n    tables = {'validation':board,'holdout':pd.DataFrame(rows),'timing':pd.DataFrame(times)}\n    tables.update(error_tables(data['test'],ytest,final,cfg,data['labels']))\n    if cfg['task']=='multilabel':\n        for j,label in enumerate(data['labels']):\n            k = np.isfinite(ytest[:,j])\n            tables['per_label'].loc[j,'AP_raw_probability'] = average_precision_score(ytest[k,j],raw[k,j]) if np.any(ytest[k,j]==1) else np.nan\n    save_tables(tables,out)\n    np.savez_compressed(out/'predictions.npz',truth=ytest,probability=raw,decision_or_score=final,labels=np.array(data['labels']))\n    pd.concat([data[k][['_row_id','_group']].assign(split=k) for k in ['train','val','test']]).to_csv(out/'split.csv',index=False)\n    (out/'config.json').write_text(json.dumps(cfg,indent=2,default=str))\n    show_table('Holdout · pilihan sudah dibekukan',tables['holdout'])\n    show_table('Kelemahan per kondisi',tables['slices'])\n    return {'policy':policy,'tables':tables,'validation_predictions':vp,'holdout_predictions':hp}\n",
"gpu": false
},
"retrieval": {
"id": "retrieval",
"title": "Retrieval multi-relevance + matching/ranking audit",
"file": "blocks/retrieval.py",
"needs": [
"preprocess",
"core"
],
"packages": [
"rank-bm25==0.2.2"
],
"stage": "model",
"note": "MRR/Recall/NDCG benar untuk banyak dokumen relevan. Corpus index bukan supervised train. Query validation/test dipisahkan menurut exact query.",
"example": "data = prepare_retrieval(corpus, queries, CFG)\nresult = run_retrieval(data, CFG)",
"code": "from sklearn.model_selection import GroupShuffleSplit\nfrom sklearn.feature_extraction.text import TfidfVectorizer\n\n\ndef prepare_retrieval(corpus, queries, cfg):\n    c = corpus.rename(columns={cfg['doc_id_col']:'doc_id',cfg['doc_text_col']:'text'}).copy()\n    q = queries.rename(columns={cfg['query_col']:'query',cfg['relevant_col']:'relevant_ids'}).copy()\n    assert {'doc_id','text'}.issubset(c) and {'query','relevant_ids'}.issubset(q)\n    assert c.doc_id.notna().all() and c.doc_id.astype(str).is_unique, 'doc_id harus unik dan terisi'\n    c['doc_id'] = c.doc_id.astype(str)\n    c['text'] = c.text.map(safe_text)\n    assert c.text.ne('').all(), 'Dokumen kosong perlu ditangani dahulu'\n    q['query'] = q['query'].map(safe_text)\n    assert q['query'].ne('').all(), 'Query kosong tidak dapat dievaluasi'\n    def parse(value):\n        value = json.loads(value) if isinstance(value,str) else value\n        assert isinstance(value,list), 'relevant_ids harus JSON array, contoh [\"d1\",\"d2\"]'\n        return set(map(str,value))\n    q['gold'] = q.relevant_ids.map(parse)\n    assert q.gold.map(bool).all(), 'Query tanpa judgment positif belum dapat dinilai'\n    assert set().union(*q.gold).issubset(set(c.doc_id)), 'Gold doc tidak ada dalam corpus'\n    q['_group'] = pd.factorize(q['query'].str.casefold())[0]\n    val,test = next(GroupShuffleSplit(1,test_size=.5,random_state=cfg['seed']).split(q,groups=q._group))\n    print('Corpus:',len(c),'| query validation/test:',len(val),len(test))\n    print('Duplikat dokumen:',c.text.str.casefold().duplicated().sum(),'| judgment per query:',q.gold.map(len).describe())\n    return {'corpus':c.reset_index(drop=True),'val':q.iloc[val].reset_index(drop=True),'test':q.iloc[test].reset_index(drop=True)}\n\n\ndef retrieval_metrics(ranking, gold, doc_ids):\n    rows = []\n    ids = np.asarray(doc_ids).astype(str)\n    for order,relevant in zip(ranking,gold):\n        retrieved = ids[order].tolist()\n        hits = np.array([d in relevant for d in retrieved],dtype=float)\n        positions = np.flatnonzero(hits[:10])\n        ideal = np.sum(1/np.log2(np.arange(min(len(relevant),10))+2))\n        rows.append({'MRR@10':1/(positions[0]+1) if len(positions) else 0,\n            **{f'Recall@{k}':len(set(retrieved[:k])&relevant)/len(relevant) for k in [1,5,10]},\n            'NDCG@10':float(np.sum(hits[:10]/np.log2(np.arange(len(hits[:10]))+2))/ideal)})\n    return pd.DataFrame(rows).mean().to_dict()\n\n\ndef lexical_rankings(corpus, queries, method='bm25', top=100):\n    if method=='bm25':\n        from rank_bm25 import BM25Okapi\n        tokenize = lambda t:re.findall(r'[^\\W_]+',t.casefold())\n        index = BM25Okapi([tokenize(t) for t in corpus])\n        return np.array([np.argsort(-index.get_scores(tokenize(t)),kind='stable')[:top] for t in queries])\n    vec = TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=80000).fit(corpus)\n    docs = vec.transform(corpus)\n    return np.array([np.argsort(-(vec.transform([q])@docs.T).toarray()[0],kind='stable')[:top] for q in queries])\n\n\ndef rrf_rankings(first, second, alpha=.5, top=100, constant=60):\n    merged = []\n    for a,b in zip(first,second):\n        scores = {}\n        for ranking,weight in [(a,1-alpha),(b,alpha)]:\n            for rank,doc in enumerate(ranking):\n                scores[int(doc)] = scores.get(int(doc),0)+weight/(constant+rank+1)\n        merged.append(sorted(scores,key=lambda doc:(-scores[doc],doc))[:top])\n    return np.asarray(merged)\n\n\ndef dense_rankings(corpus, queries, cfg, top=100):\n    chunks,owners = [],[]\n    for i,text in enumerate(corpus):\n        words = text.split()\n        for start in range(0,min(len(words),cfg.get('max_chunks',6)*120),120):\n            chunks.append(' '.join(words[start:start+120]));owners.append(i)\n    emb = encode_e5(chunks,cfg,'passage')\n    q = encode_e5(queries,cfg,'query')\n    rankings = []\n    for vector in q:\n        scores = np.full(len(corpus),-np.inf)\n        np.maximum.at(scores,owners,emb@vector)\n        rankings.append(np.argsort(-scores,kind='stable')[:top])\n    return np.array(rankings)\n\n\ndef rerank_candidates(corpus, queries, rankings, cfg, top=20):\n    require_kaggle(gpu=True)\n    import gc\n    import torch\n    from transformers import AutoTokenizer,AutoModelForSequenceClassification\n    repo = cfg.get('reranker','cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')\n    tok = AutoTokenizer.from_pretrained(repo,local_files_only=cfg.get('offline',False))\n    model = AutoModelForSequenceClassification.from_pretrained(repo,local_files_only=cfg.get('offline',False)).cuda().eval()\n    assert model.config.num_labels==1, 'Reranker ini mengharapkan satu skor relevansi'\n    result = []\n    with torch.inference_mode():\n        for query,ranking in zip(queries,rankings):\n            candidates = ranking[:top]\n            enc = tok([query]*len(candidates),[corpus[int(i)] for i in candidates],padding=True,truncation=True,max_length=cfg['max_len'],return_tensors='pt').to('cuda')\n            scores = model(**enc).logits.float().cpu().numpy().ravel()\n            result.append(np.r_[candidates[np.argsort(-scores,kind='stable')],ranking[top:]])\n    del model\n    gc.collect();torch.cuda.empty_cache()\n    return np.array(result)\n\n\ndef run_retrieval(data, cfg):\n    c = data['corpus'];doc_ids=c.doc_id.to_numpy();top=min(100,len(c))\n    rankings = {split:{} for split in ['val','test']}\n    for split in rankings:\n        q = data[split]['query'].tolist()\n        for method in [m for m in cfg['models'] if m in {'tfidf','bm25'}]:\n            rankings[split][method] = lexical_rankings(c.text.tolist(),q,method,top)\n        if 'e5' in cfg['models']:\n            rankings[split]['e5'] = dense_rankings(c.text.tolist(),q,cfg,top)\n    if 'rrf' in cfg['models']:\n        assert all(name in rankings['val'] for name in ['bm25','e5']), 'RRF memerlukan BM25 dan E5'\n        alpha = max([0,.25,.5,.75,1],key=lambda a:retrieval_metrics(rrf_rankings(rankings['val']['bm25'],rankings['val']['e5'],a,top),data['val'].gold,doc_ids)['MRR@10'])\n        for split in rankings:\n            rankings[split]['rrf'] = rrf_rankings(rankings[split]['bm25'],rankings[split]['e5'],alpha,top)\n    if 'reranker' in cfg['models']:\n        base = 'rrf' if 'rrf' in rankings['val'] else 'bm25'\n        assert base in rankings['val'], 'Reranker memerlukan kandidat BM25 atau RRF'\n        for split in rankings:\n            rankings[split]['reranker'] = rerank_candidates(c.text.tolist(),data[split]['query'].tolist(),rankings[split][base],cfg)\n    validation = pd.DataFrame([{'model':name,**retrieval_metrics(order,data['val'].gold,doc_ids)} for name,order in rankings['val'].items()])\n    assert len(validation), 'Pilih minimal satu model retrieval'\n    chosen = validation.sort_values('MRR@10',ascending=False,kind='stable').iloc[0].model\n    out = Path(cfg['out']);out.mkdir(parents=True,exist_ok=True)\n    (out/'selection_frozen_before_test.json').write_text(json.dumps({'model':chosen,'rrf_alpha':alpha if 'rrf' in cfg['models'] else None}))\n    holdout = pd.DataFrame([{'model':name,**retrieval_metrics(order,data['test'].gold,doc_ids)} for name,order in rankings['test'].items()])\n    rows = data['test'][['query']].copy()\n    rows['gold'] = data['test'].gold.map(lambda v:sorted(v))\n    rows['top10'] = [[doc_ids[i] for i in r[:10]] for r in rankings['test'][chosen]]\n    rows['hit_at_1'] = [str(doc_ids[r[0]]) in g for r,g in zip(rankings['test'][chosen],data['test'].gold)]\n    tables = {'validation':validation,'holdout':holdout,'errors_and_rankings':rows}\n    save_tables(tables,out)\n    for name,table in tables.items():\n        show_table(name,table.head(20))\n    print('FINAL (validation):',chosen,'| Judgment yang tidak lengkap dapat menurunkan Recall/NDCG secara semu.')\n    return tables\n",
"gpu": false
},
"unsupervised": {
"id": "unsupervised",
"title": "Frozen topic/cluster fit + validation quality",
"file": "blocks/unsupervised.py",
"needs": [
"groups",
"eda"
],
"packages": [],
"stage": "model",
"note": "Fit vocabulary/model pada train unik, pilih k di validation, audit holdout. NPMI/silhouette tidak menggantikan audit semantik.",
"example": "data = prepare_unsupervised(df, CFG)\nresult = run_unsupervised(data, CFG)",
"code": "from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer\nfrom sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD\nfrom sklearn.cluster import MiniBatchKMeans\nfrom sklearn.preprocessing import normalize\nfrom sklearn.metrics import silhouette_score\n\n\ndef prepare_unsupervised(raw, cfg):\n    frame = raw.copy().reset_index(drop=True)\n    frame['_row_id'] = np.arange(len(frame))\n    frame['native'] = frame[cfg['text_col']].map(safe_text)\n    frame['a'] = frame[cfg['text_col']].map(make_cleaner(cfg))\n    frame = frame[frame.a.ne('')].reset_index(drop=True)\n    frame['_group'] = connected_groups(frame,[cfg['text_col']],cfg.get('group_col'),['a'])\n    if cfg['task']=='temporal':\n        source = frame[cfg['time_col']]\n        def timestamp(value):\n            try:\n                t = pd.Timestamp(value)\n                return t.tz_localize(cfg['naive_timezone']) if t.tzinfo is None else t\n            except (ValueError,TypeError):\n                return pd.NaT\n        utc = pd.to_datetime(source.map(timestamp),utc=True,errors='coerce')\n        frame['timestamp'] = utc.dt.tz_convert(cfg['timezone'])\n        print('Tanggal invalid dibuang:',int(frame.timestamp.isna().sum()))\n        frame = frame[frame.timestamp.notna()].sort_values('timestamp').reset_index(drop=True)\n        frame['period'] = frame.timestamp.dt.tz_localize(None).dt.to_period(cfg['period']).dt.start_time\n        periods = sorted(frame.period.unique())\n        assert len(periods)>=5, 'Perlu minimal lima periode untuk early/middle/late'\n        a,b = max(1,int(len(periods)*.6)),max(2,int(len(periods)*.8))\n        parts = {'train':frame[frame.period.isin(periods[:a])], 'val':frame[frame.period.isin(periods[a:b])], 'test':frame[frame.period.isin(periods[b:])]}\n        # Tidak membuang volume berulang. Audit recurrence: group overlap lintas waktu dilaporkan.\n        for split in ['val','test']:\n            print('Recurring exact-text rows vs early,',split,':',parts[split]._group.isin(parts['train']._group).sum())\n    else:\n        rest,test = next(GroupShuffleSplit(1,test_size=.2,random_state=cfg['seed']).split(frame,groups=frame._group))\n        train,val = next(GroupShuffleSplit(1,test_size=.25,random_state=cfg['seed']+1).split(frame.iloc[rest],groups=frame.iloc[rest]._group))\n        parts = {'train':frame.iloc[rest[train]],'val':frame.iloc[rest[val]],'test':frame.iloc[test]}\n    parts = {k:v.reset_index(drop=True) for k,v in parts.items()}\n    fit = parts['train'].drop_duplicates('a')\n    if cfg.get('fit_rows'):\n        fit = fit.sample(min(len(fit),cfg['fit_rows']),random_state=cfg['seed'])\n    parts['fit'] = fit.reset_index(drop=True)\n    assert all(len(parts[k]) for k in ['fit','val','test'])\n    return parts\n\n\ndef npmi_quality(texts, topics):\n    terms = sorted({word for topic in topics for word in topic})\n    if not terms or not len(texts):\n        return np.nan\n    vec = CountVectorizer(vocabulary=terms,binary=True)\n    x = vec.transform(texts).astype('float64')\n    prob = np.asarray(x.mean(0)).ravel()\n    joint = (x.T@x).toarray()/len(texts)\n    vocab = vec.vocabulary_;scores=[]\n    for topic in topics:\n        values = []\n        for i,a in enumerate(topic):\n            for b in topic[i+1:]:\n                pa,pb,pab = prob[vocab[a]],prob[vocab[b]],joint[vocab[a],vocab[b]]\n                values.append(np.log(pab/(pa*pb))/-np.log(pab) if 0<pab<1 and pa*pb>0 else (1. if pab==1 else -1.))\n        if values:\n            scores.append(np.mean(values))\n    return float(np.mean(scores)) if scores else np.nan\n\n\ndef fit_topic_candidate(data, cfg, method, k):\n    from scipy.sparse import csr_matrix\n    assert len(data['fit'])>k, 'Teks unik train terlalu sedikit; kurangi k atau tambah data.'\n    vec = CountVectorizer(min_df=2,max_df=1.,max_features=cfg['max_features'])\n    counts = vec.fit_transform(data['fit'].a)\n    assert min(counts.shape)>k, 'Kurangi k atau tambah ukuran train/vocabulary'\n    tfidf = TfidfTransformer().fit(counts)\n    dense = method=='e5_kmeans'\n    reducer = None\n    if method=='nmf':\n        x = tfidf.transform(counts)\n        model = NMF(k,init='nndsvda',max_iter=300,random_state=cfg['seed']).fit(x)\n    elif method=='lda':\n        x = counts\n        model = LatentDirichletAllocation(k,max_iter=10,learning_method='batch',random_state=cfg['seed']).fit(x)\n    else:\n        if dense:\n            x = encode_e5(data['fit'].native,cfg,'query')\n        else:\n            n = min(100,min(counts.shape)-1)\n            reducer = TruncatedSVD(n,random_state=cfg['seed']).fit(tfidf.transform(counts))\n            x = normalize(reducer.transform(tfidf.transform(counts)))\n        model = MiniBatchKMeans(k,n_init=10,batch_size=512,random_state=cfg['seed']).fit(x)\n    def represent(frame):\n        if dense:\n            return encode_e5(frame.native,cfg,'query')\n        c = vec.transform(frame.a)\n        if method=='lda':\n            return c\n        return normalize(reducer.transform(tfidf.transform(c))) if reducer is not None else tfidf.transform(c)\n    def assign(frame):\n        z = represent(frame)\n        return model.transform(z).argmax(1) if method in {'nmf','lda'} else model.predict(z)\n    if method in {'nmf','lda'}:\n        weights = model.components_\n    else:\n        grouped = np.array([np.asarray(counts[model.labels_==j].sum(0)).ravel() for j in range(k)])\n        weights = TfidfTransformer().fit_transform(csr_matrix(grouped)).toarray()\n    words = np.array(vec.get_feature_names_out())\n    topics = [words[np.argsort(-row)[:10]].tolist() for row in weights]\n    return {'model':model,'assign':assign,'represent':represent,'topics':topics,'vectorizer':vec,'k':k,'method':method}\n\n\ndef run_unsupervised(data, cfg):\n    candidates = {};rows=[]\n    for method in cfg['models']:\n        for k in cfg['k_values']:\n            fitted = fit_topic_candidate(data,cfg,method,k)\n            name = f'{method}_k{k}';candidates[name]=fitted\n            val_ids = fitted['assign'](data['val'])\n            row = {'model':name,'val_npmi':npmi_quality(data['val'].a,fitted['topics']),\n                'val_active_groups':len(np.unique(val_ids)), 'topic_diversity':len(set(sum(fitted['topics'],[])))/sum(map(len,fitted['topics']))}\n            if cfg['task']=='clustering':\n                z = fitted['represent'](data['val'])\n                row['val_silhouette'] = silhouette_score(z,val_ids,sample_size=min(1000,len(val_ids)),random_state=cfg['seed']) if 1<len(np.unique(val_ids))<len(val_ids) else np.nan\n            rows.append(row)\n    board = pd.DataFrame(rows)\n    metric = 'val_silhouette' if cfg['task']=='clustering' else 'val_npmi'\n    assert board[metric].notna().any(), f'Tidak ada kandidat valid untuk {metric}'\n    chosen = board.sort_values(metric,ascending=False,kind='stable').iloc[0].model\n    fitted = candidates[chosen]\n    out = Path(cfg['out']);out.mkdir(parents=True,exist_ok=True)\n    (out/'selection_frozen_before_late.json').write_text(json.dumps({'model':chosen,'selection_metric':metric}))\n    assignments = pd.concat([data[s].assign(split=s,topic=fitted['assign'](data[s])) for s in ['train','val','test']],ignore_index=True)\n    topics = pd.DataFrame({'topic':range(fitted['k']),'words':[' | '.join(t) for t in fitted['topics']]})\n    examples = assignments.groupby(['split','topic'],group_keys=False).head(5)\n    tables = {'validation':board,'topics':topics,'examples_for_human_audit':examples,\n        'assignments':assignments,'late_quality':pd.Series({'NPMI':npmi_quality(data['test'].a,fitted['topics'])}).to_frame('value')}\n    save_tables(tables,out)\n    for name in ['validation','topics','late_quality']:\n        show_table(name,tables[name])\n    print('FINAL:',chosen,'| NPMI/silhouette bukan bukti kebenaran semantik; audit contoh per topik/cluster.')\n    return {'tables':tables,'assignments':assignments,'fitted':fitted}\n",
"gpu": false
},
"time": {
"id": "time",
"title": "Proporsi waktu, shift & sentimen",
"file": "blocks/temporal.py",
"needs": [
"unsupervised"
],
"packages": [],
"stage": "evaluation",
"note": "Early/middle/late, timezone eksplisit, volume duplicate tetap, trailing mean, support kecil. Sentimen tanpa gold hanya estimasi/agreement.",
"example": "temporal_report(result['assignments'], CFG)\nsentiment_over_time(result['assignments'], CFG)",
"code": "from scipy.spatial.distance import jensenshannon\n\n\ndef temporal_report(assignments, cfg):\n    counts = pd.crosstab(assignments.period,assignments.topic)\n    shares = counts.div(counts.sum(1),axis=0)\n    n = counts.sum(1)\n    rows = []\n    for i,period in enumerate(shares.index):\n        rows.append({'period':period,'n':int(n.loc[period]),'low_support':bool(n.loc[period]<cfg.get('min_period_rows',30)),\n            'JS_distance_vs_previous':float(jensenshannon(shares.iloc[i-1],shares.iloc[i],base=2)) if i else np.nan})\n    smooth = shares.rolling(3,min_periods=1).mean()  # Trailing saja, tidak mengintip masa depan.\n    tables = {'period_topic_counts':counts,'period_topic_share':shares,'trailing_3_period_share':smooth,'period_shift_support':pd.DataFrame(rows)}\n    save_tables(tables,cfg['out'])\n    show_table('Proporsi topik dan support per periode',shares.join(n.rename('n')))\n    show_table('Perubahan distribusi, bukan uji kausalitas',tables['period_shift_support'])\n    return tables\n\n\ndef lexicon_sentiment(texts, positive, negative):\n    positive,negative = set(positive),set(negative)\n    results = []\n    for text in texts:\n        words = re.findall(r'[^\\W\\d_]+',text.casefold());score=0\n        for i,w in enumerate(words):\n            value = int(w in positive)-int(w in negative)\n            if set(words[max(0,i-3):i])&NEGATIONS:\n                value = -value\n            score += value\n        results.append('positive' if score>0 else 'negative' if score<0 else 'neutral')\n    return np.array(results)\n\n\ndef pretrained_sentiment(texts, cfg):\n    require_kaggle(gpu=True)\n    import gc\n    import torch\n    from transformers import AutoTokenizer,AutoModelForSequenceClassification\n    repo = cfg.get('sentiment_model','w11wo/indonesian-roberta-base-sentiment-classifier')\n    tok = AutoTokenizer.from_pretrained(repo,local_files_only=cfg.get('offline',False))\n    model = AutoModelForSequenceClassification.from_pretrained(repo,local_files_only=cfg.get('offline',False)).cuda().eval()\n    mapping = {int(i):str(label).casefold() for i,label in model.config.id2label.items()}\n    assert set(mapping.values())=={'positive','neutral','negative'}, f'Tetapkan mapping sentimen eksplisit: {mapping}'\n    texts = list(texts);results=[]\n    with torch.inference_mode():\n        for start in range(0,len(texts),32):\n            batch = tok(texts[start:start+32],padding=True,truncation=True,max_length=cfg['max_len'],return_tensors='pt').to('cuda')\n            results.extend(mapping[int(i)] for i in model(**batch).logits.argmax(1).cpu().numpy())\n    del model\n    gc.collect();torch.cuda.empty_cache()\n    return np.array(results)\n\n\ndef sentiment_over_time(assignments, cfg):\n    per = cfg.get('sentiment_sample_per_period',100)\n    sample = pd.concat([g.sample(min(per,len(g)),random_state=cfg['seed']) for _,g in assignments.groupby('period')])\n    sample = sample.copy()\n    sample['lexicon'] = lexicon_sentiment(sample.native,cfg.get('positive_words',[]),cfg.get('negative_words',[]))\n    if cfg.get('sentiment_pretrained',False):\n        sample['pretrained'] = pretrained_sentiment(sample.native,cfg)\n    tables = {'sentiment_sample_predictions':sample}\n    for method in ['lexicon']+(['pretrained'] if 'pretrained' in sample else []):\n        counts = pd.crosstab(sample.period,sample[method]).reindex(columns=['positive','neutral','negative'],fill_value=0)\n        share = counts.div(counts.sum(1),axis=0)\n        share['n_sample'] = counts.sum(1)\n        tables[f'sentiment_{method}_share_sample'] = share\n        show_table(method+' · proporsi sampel, bukan accuracy',share)\n    if 'pretrained' in sample:\n        tables['sentiment_agreement_not_accuracy'] = pd.crosstab(sample.lexicon,sample.pretrained)\n    save_tables(tables,cfg['out'])\n    print('Tanpa gold: agreement bukan accuracy. Sampel per periode tidak boleh digabung tanpa bobot volume.')\n    return tables\n",
"gpu": false
},
"model.dummy": {
"id": "model.dummy",
"title": "Baseline prior / mean / semua-negatif",
"file": "blocks/baseline.py",
"needs": [
"core"
],
"packages": [],
"stage": "model",
"note": "",
"example": "",
"code": "@register('dummy')\ndef fit_dummy(data, cfg):\n    y = target_array(data['train'],cfg)\n    if cfg['task']=='multilabel':\n        return tuple(np.zeros((len(data[s]),len(data['labels']))) for s in ['val','test'])\n    if cfg['task']=='similarity':\n        return tuple(np.full(len(data[s]),float(np.mean(y))) for s in ['val','test'])\n    prior = np.bincount(y,minlength=len(data['labels']))/len(y)\n    return tuple(np.tile(prior,(len(data[s]),1)) for s in ['val','test'])\n",
"gpu": false,
"model": "dummy"
},
"model.tfidf_word": {
"id": "model.tfidf_word",
"title": "TF-IDF word + LR / Ridge",
"file": null,
"needs": [
"linear"
],
"packages": [],
"stage": "model",
"note": "",
"example": "",
"code": "",
"gpu": false,
"model": "tfidf_word"
},
"model.tfidf_union": {
"id": "model.tfidf_union",
"title": "TF-IDF word+char + LR / Ridge",
"file": null,
"needs": [
"linear"
],
"packages": [],
"stage": "model",
"note": "",
"example": "",
"code": "",
"gpu": false,
"model": "tfidf_union"
},
"model.word2vec": {
"id": "model.word2vec",
"title": "Word2Vec train-only + LR / Ridge",
"file": "blocks/word2vec.py",
"needs": [
"linear"
],
"packages": [
"gensim>=4.4,<5"
],
"stage": "model",
"note": "",
"example": "",
"code": "@register('word2vec')\ndef fit_word2vec(data, cfg):\n    from gensim.models import Word2Vec\n    pair = cfg['task'] in {'nli','similarity'}\n    def tokenize(values):\n        return [re.findall(r'[^\\W\\d_]+',str(t).casefold()) for t in values]\n    train = data['train']\n    corpus = tokenize(pd.concat([train.a,train.b]) if pair else train.a)\n    model = Word2Vec(corpus,vector_size=100,min_count=1,window=5,workers=1,seed=cfg['seed'],epochs=10)\n    def encode(values):\n        out = []\n        for tokens in tokenize(values):\n            vectors = [model.wv[w] for w in tokens if w in model.wv]\n            out.append(np.mean(vectors,axis=0) if vectors else np.zeros(100))\n        return np.asarray(out)\n    matrices = []\n    for split in ['train','val','test']:\n        f = data[split];a=encode(f.a)\n        if pair:\n            b = encode(f.b)\n            a = np.hstack([np.abs(a-b),a*b]) if cfg['task']=='similarity' else np.hstack([a,b,np.abs(a-b),a*b])\n        matrices.append(a)\n    return fit_head(*matrices,data,cfg)\n",
"gpu": false,
"model": "word2vec"
},
"model.e5_head": {
"id": "model.e5_head",
"title": "Frozen E5 + LR / Ridge",
"file": "blocks/embedding_head.py",
"needs": [
"encoder",
"linear"
],
"packages": [],
"stage": "model",
"note": "",
"example": "",
"code": "\"\"\"Head E5 terpisah; retrieval cukup mengambil encoder tanpa supervised head.\"\"\"\n\n@register('e5_head')\ndef fit_e5(data, cfg):\n    matrices = []\n    for split in ['train','val','test']:\n        frame = data[split]\n        a = e5_encode(frame.a,cfg,'query')\n        if cfg['task'] in {'nli','similarity'}:\n            b = e5_encode(frame.b,cfg,'query')\n            a = (a*b).sum(1,keepdims=True) if cfg['task']=='similarity' else np.hstack([a,b,np.abs(a-b),a*b])\n        matrices.append(a)\n    return fit_head(*matrices,data,cfg)\n\n",
"gpu": true,
"model": "e5_head"
},
"model.indobertweet": {
"id": "model.indobertweet",
"title": "IndoBERTweet Base",
"file": null,
"needs": [
"transformer"
],
"packages": [],
"stage": "model",
"note": "Bisa ditukar tanpa mengubah evaluator. IndoBERTweet memakai @USER/HTTPURL dan emoji demojize; yang lain native/normalisasi pilihan Anda.",
"example": "",
"code": "",
"gpu": true,
"model": "indobertweet"
},
"model.indobert_p1": {
"id": "model.indobert_p1",
"title": "IndoBERT Base P1",
"file": null,
"needs": [
"transformer"
],
"packages": [],
"stage": "model",
"note": "Bisa ditukar tanpa mengubah evaluator. IndoBERTweet memakai @USER/HTTPURL dan emoji demojize; yang lain native/normalisasi pilihan Anda.",
"example": "",
"code": "",
"gpu": true,
"model": "indobert_p1"
},
"model.xlmr": {
"id": "model.xlmr",
"title": "XLM-RoBERTa Base",
"file": null,
"needs": [
"transformer"
],
"packages": [],
"stage": "model",
"note": "Bisa ditukar tanpa mengubah evaluator. IndoBERTweet memakai @USER/HTTPURL dan emoji demojize; yang lain native/normalisasi pilihan Anda.",
"example": "",
"code": "",
"gpu": true,
"model": "xlmr"
},
"model.indolem": {
"id": "model.indolem",
"title": "IndoBERT IndoLEM",
"file": null,
"needs": [
"transformer"
],
"packages": [],
"stage": "model",
"note": "Bisa ditukar tanpa mengubah evaluator. IndoBERTweet memakai @USER/HTTPURL dan emoji demojize; yang lain native/normalisasi pilihan Anda.",
"example": "",
"code": "",
"gpu": true,
"model": "indolem"
},
"model.nusabert": {
"id": "model.nusabert",
"title": "NusaBERT Base",
"file": null,
"needs": [
"transformer"
],
"packages": [],
"stage": "model",
"note": "Bisa ditukar tanpa mengubah evaluator. IndoBERTweet memakai @USER/HTTPURL dan emoji demojize; yang lain native/normalisasi pilihan Anda.",
"example": "",
"code": "",
"gpu": true,
"model": "nusabert"
},
"model.nli_pretrained": {
"id": "model.nli_pretrained",
"title": "mDeBERTa pretrained NLI 3 kelas",
"file": "blocks/nli_pretrained.py",
"needs": [
"core"
],
"packages": [
"transformers==5.20.2",
"sentencepiece>=0.2,<1",
"torch"
],
"stage": "model",
"note": "",
"example": "",
"code": "@register('nli_pretrained')\ndef fit_nli_pretrained(data, cfg):\n    require_kaggle(gpu=True)\n    import gc\n    import torch\n    from scipy.special import softmax\n    from transformers import AutoTokenizer, AutoModelForSequenceClassification\n    repo = cfg.get('nli_model','MoritzLaurer/mDeBERTa-v3-base-mnli-xnli')\n    tok = AutoTokenizer.from_pretrained(repo,local_files_only=cfg.get('offline',False))\n    model = AutoModelForSequenceClassification.from_pretrained(repo,local_files_only=cfg.get('offline',False)).cuda().eval()\n    mapping = {str(label).casefold():int(i) for i,label in model.config.id2label.items()}\n    assert set(data['labels']).issubset(mapping), f'Mapping label model perlu eksplisit: {mapping}'\n    order = [mapping[label] for label in data['labels']]\n    results = []\n    with torch.inference_mode():\n        for split in ['val','test']:\n            rows = []\n            for start in range(0,len(data[split]),32):\n                f = data[split].iloc[start:start+32]\n                # Premise -> hypothesis. Tidak ada swap pada NLI.\n                enc = tok(f.a.tolist(),f.b.tolist(),padding=True,truncation=True,max_length=cfg['max_len'],return_tensors='pt').to('cuda')\n                rows.append(softmax(model(**enc).logits.float().cpu().numpy()[:,order],axis=1))\n            results.append(np.vstack(rows))\n    del model\n    gc.collect();torch.cuda.empty_cache()\n    return tuple(results)\n",
"gpu": true,
"model": "nli_pretrained"
},
"model.tfidf": {
"id": "model.tfidf",
"title": "TF-IDF cosine",
"file": null,
"needs": [
"retrieval"
],
"packages": [],
"stage": "model",
"note": "",
"example": "",
"code": "",
"gpu": false,
"model": "tfidf"
},
"model.bm25": {
"id": "model.bm25",
"title": "BM25",
"file": null,
"needs": [
"retrieval"
],
"packages": [],
"stage": "model",
"note": "",
"example": "",
"code": "",
"gpu": false,
"model": "bm25"
},
"model.e5": {
"id": "model.e5",
"title": "E5 chunked dense",
"file": null,
"needs": [
"retrieval",
"encoder"
],
"packages": [],
"stage": "model",
"note": "",
"example": "",
"code": "",
"gpu": true,
"model": "e5"
},
"model.rrf": {
"id": "model.rrf",
"title": "BM25 + E5 reciprocal rank fusion",
"file": null,
"needs": [
"retrieval",
"encoder",
"model.bm25",
"model.e5"
],
"packages": [],
"stage": "model",
"note": "",
"example": "",
"code": "",
"gpu": true,
"model": "rrf"
},
"model.reranker": {
"id": "model.reranker",
"title": "Multilingual cross-encoder reranker",
"file": null,
"needs": [
"retrieval",
"model.bm25"
],
"packages": [
"transformers==5.20.2",
"sentencepiece>=0.2,<1",
"torch"
],
"stage": "model",
"note": "",
"example": "",
"code": "",
"gpu": true,
"model": "reranker"
},
"model.nmf": {
"id": "model.nmf",
"title": "NMF",
"file": null,
"needs": [
"unsupervised"
],
"packages": [],
"stage": "model",
"note": "",
"example": "",
"code": "",
"gpu": false,
"model": "nmf"
},
"model.lda": {
"id": "model.lda",
"title": "LDA",
"file": null,
"needs": [
"unsupervised"
],
"packages": [],
"stage": "model",
"note": "",
"example": "",
"code": "",
"gpu": false,
"model": "lda"
},
"model.kmeans": {
"id": "model.kmeans",
"title": "TF-IDF + SVD + MiniBatchKMeans",
"file": null,
"needs": [
"unsupervised"
],
"packages": [],
"stage": "model",
"note": "",
"example": "",
"code": "",
"gpu": false,
"model": "kmeans"
},
"model.e5_kmeans": {
"id": "model.e5_kmeans",
"title": "E5 + MiniBatchKMeans",
"file": null,
"needs": [
"unsupervised",
"encoder"
],
"packages": [],
"stage": "model",
"note": "",
"example": "",
"code": "",
"gpu": true,
"model": "e5_kmeans"
},
"sentiment.pretrained": {
"id": "sentiment.pretrained",
"title": "Pretrained sentimen per waktu",
"file": null,
"needs": [
"time"
],
"packages": [
"transformers==5.20.2",
"sentencepiece>=0.2,<1",
"torch"
],
"stage": "postprocessing",
"note": "",
"example": "",
"code": "",
"gpu": true
}
},
"recipes": {
"classification": {
"id": "classification",
"title": "Klasifikasi multiclass / biner",
"choices": [
"model.dummy",
"model.tfidf_word",
"model.tfidf_union",
"model.word2vec",
"model.e5_head",
"model.indobertweet",
"model.indobert_p1",
"model.xlmr",
"model.indolem",
"model.nusabert"
],
"defaults": [
"model.dummy",
"model.tfidf_word",
"model.tfidf_union",
"model.word2vec",
"model.indobertweet"
],
"required": [
"eda",
"supervised_run"
],
"aliases": [
"multiclass",
"binary",
"sentiment",
"spam",
"intent",
"emotion",
"stance",
"fakenews"
],
"source": "notebook/b_klasifikasi/10_multi_kelas/10_multi_kelas_v2_pretrained.ipynb",
"config": {
"data_file": "/kaggle/input/GANTI_DATASET/train.csv",
"encoding": "utf-8",
"text_col": "text",
"label_col": "label",
"group_col": null,
"seed": 42,
"out": "/kaggle/working/hasil_toolkit",
"offline": false,
"preprocess": "native",
"slang": {},
"stopwords": [],
"eda_sample": 5000,
"eda_top": 15,
"eda_languages": false,
"max_features": 60000,
"max_len": 192,
"epochs": 3,
"learning_rate": 2e-05,
"batch_size": 16,
"e5_model": "intfloat/multilingual-e5-base",
"model_paths": {},
"task": "classification",
"ensemble": true,
"post": [
"class_bias",
"temperature"
]
},
"note": "Class-specific bias dan scalar temperature berbeda tujuan; temperature tidak mengubah argmax."
},
"multilabel": {
"id": "multilabel",
"title": "Multilabel (partial labels)",
"choices": [
"model.dummy",
"model.tfidf_word",
"model.tfidf_union",
"model.word2vec",
"model.e5_head",
"model.indobertweet",
"model.indobert_p1",
"model.xlmr",
"model.indolem",
"model.nusabert"
],
"defaults": [
"model.dummy",
"model.tfidf_word",
"model.tfidf_union",
"model.e5_head",
"model.indobertweet"
],
"required": [
"eda",
"supervised_run"
],
"aliases": [
"multilabel"
],
"source": "notebook/b_klasifikasi/12_multi_label/12_multi_label_review.ipynb",
"config": {
"data_file": "/kaggle/input/GANTI_DATASET/train.csv",
"encoding": "utf-8",
"text_col": "text",
"label_cols": [
"HS",
"Abusive"
],
"group_col": null,
"seed": 42,
"out": "/kaggle/working/hasil_toolkit",
"offline": false,
"preprocess": "native",
"slang": {},
"stopwords": [],
"eda_sample": 5000,
"eda_top": 15,
"eda_languages": false,
"max_features": 60000,
"max_len": 192,
"epochs": 3,
"learning_rate": 2e-05,
"batch_size": 16,
"threshold_min_support": 5,
"e5_model": "intfloat/multilingual-e5-base",
"model_paths": {},
"task": "multilabel",
"ensemble": true,
"post": [
"global_threshold",
"label_threshold"
]
},
"note": "Unknown bukan 0. Threshold rare label fallback .5; tidak memaksa satu label aktif."
},
"similarity": {
"id": "similarity",
"title": "Semantic similarity / STS",
"choices": [
"model.dummy",
"model.tfidf_word",
"model.tfidf_union",
"model.word2vec",
"model.e5_head",
"model.indobertweet",
"model.indobert_p1",
"model.xlmr",
"model.indolem",
"model.nusabert"
],
"defaults": [
"model.dummy",
"model.tfidf_word",
"model.tfidf_union",
"model.e5_head",
"model.indobert_p1"
],
"required": [
"eda",
"supervised_run"
],
"aliases": [
"semsim",
"doc_sim"
],
"source": "notebook/e_pasangan_teks/20_kemiripan/20_kemiripan_review.ipynb",
"config": {
"data_file": "/kaggle/input/GANTI_DATASET/train.csv",
"encoding": "utf-8",
"text_col": "text",
"label_col": "score",
"pair_cols": [
"text_1",
"text_2"
],
"group_col": null,
"seed": 42,
"out": "/kaggle/working/hasil_toolkit",
"offline": false,
"preprocess": "native",
"slang": {},
"stopwords": [],
"eda_sample": 5000,
"eda_top": 15,
"eda_languages": false,
"max_features": 60000,
"max_len": 192,
"epochs": 3,
"learning_rate": 2e-05,
"batch_size": 16,
"score_range": [
0,
5
],
"e5_model": "intfloat/multilingual-e5-base",
"model_paths": {},
"task": "similarity",
"ensemble": true,
"post": [
"isotonic"
]
},
"note": "Regresi kontinu, bukan NLI. E5 query di kedua sisi; fitur dan inference STS simetris."
},
"nli": {
"id": "nli",
"title": "NLI entailment / neutral / contradiction",
"choices": [
"model.dummy",
"model.tfidf_word",
"model.tfidf_union",
"model.word2vec",
"model.e5_head",
"model.indobertweet",
"model.indobert_p1",
"model.xlmr",
"model.indolem",
"model.nusabert",
"model.nli_pretrained"
],
"defaults": [
"model.dummy",
"model.tfidf_word",
"model.e5_head",
"model.nli_pretrained",
"model.indobert_p1"
],
"required": [
"eda",
"supervised_run"
],
"aliases": [
"pair",
"consistency"
],
"source": "notebook/e_pasangan_teks/21_nli/21_nli_review.ipynb",
"config": {
"data_file": "/kaggle/input/GANTI_DATASET/train.csv",
"encoding": "utf-8",
"text_col": "text",
"label_col": "label",
"pair_cols": [
"premise",
"hypothesis"
],
"group_col": null,
"seed": 42,
"out": "/kaggle/working/hasil_toolkit",
"offline": false,
"preprocess": "native",
"slang": {},
"stopwords": [],
"eda_sample": 5000,
"eda_top": 15,
"eda_languages": false,
"max_features": 60000,
"max_len": 192,
"epochs": 3,
"learning_rate": 2e-05,
"batch_size": 16,
"e5_model": "intfloat/multilingual-e5-base",
"model_paths": {},
"nli_label_map": {
"e": "entailment",
"n": "neutral",
"c": "contradiction",
"entailment": "entailment",
"neutral": "neutral",
"contradiction": "contradiction"
},
"task": "nli",
"ensemble": true,
"post": [
"class_bias",
"temperature"
]
},
"note": "NLI tiga kelas, arah premise → hypothesis. Label numerik harus dipetakan eksplisit di nli_label_map."
},
"retrieval": {
"id": "retrieval",
"title": "Text matching / retrieval",
"choices": [
"model.tfidf",
"model.bm25",
"model.e5",
"model.rrf",
"model.reranker"
],
"defaults": [
"model.tfidf",
"model.bm25",
"model.e5",
"model.rrf",
"model.reranker"
],
"required": [
"eda",
"retrieval"
],
"aliases": [
"ir",
"ranking",
"dupq"
],
"source": "notebook/e_pasangan_teks/22_retrieval/22_retrieval_review.ipynb",
"config": {
"encoding": "utf-8",
"seed": 42,
"out": "/kaggle/working/hasil_toolkit",
"offline": false,
"preprocess": "native",
"slang": {},
"stopwords": [],
"eda_sample": 5000,
"eda_top": 15,
"eda_languages": false,
"max_features": 60000,
"max_len": 192,
"e5_model": "intfloat/multilingual-e5-base",
"corpus_file": "/kaggle/input/GANTI_DATASET/corpus.csv",
"queries_file": "/kaggle/input/GANTI_DATASET/queries.csv",
"doc_id_col": "doc_id",
"doc_text_col": "text",
"query_col": "query",
"relevant_col": "relevant_ids",
"task": "retrieval"
},
"note": "Queries perlu relevant_ids JSON list. Benchmark proxy ringkasan-artikel terlalu mudah; gunakan judgment kueri nyata untuk keputusan final."
},
"topic": {
"id": "topic",
"title": "Topic modeling",
"choices": [
"model.nmf",
"model.lda",
"model.kmeans",
"model.e5_kmeans"
],
"defaults": [
"model.nmf",
"model.lda",
"model.kmeans"
],
"required": [
"eda",
"unsupervised"
],
"aliases": [
"topic"
],
"source": "notebook/a_tanpa_label/01_topic_modeling/01_topic_modeling_review.ipynb",
"config": {
"data_file": "/kaggle/input/GANTI_DATASET/train.csv",
"encoding": "utf-8",
"text_col": "text",
"group_col": null,
"seed": 42,
"out": "/kaggle/working/hasil_toolkit",
"offline": false,
"preprocess": "bow",
"slang": {},
"stopwords": [],
"eda_sample": 5000,
"eda_top": 15,
"eda_languages": false,
"max_features": 60000,
"max_len": 192,
"e5_model": "intfloat/multilingual-e5-base",
"k_values": [
4,
6,
8
],
"fit_rows": 5000,
"task": "topic"
},
"note": "Vocabulary dan topic ID dibekukan. Baca contoh dokumen per topik; metrik internal tidak mengukur akurasi sentimen."
},
"clustering": {
"id": "clustering",
"title": "Clustering teks",
"choices": [
"model.nmf",
"model.lda",
"model.kmeans",
"model.e5_kmeans"
],
"defaults": [
"model.nmf",
"model.lda",
"model.kmeans"
],
"required": [
"eda",
"unsupervised"
],
"aliases": [
"doc_cluster",
"emb_cluster"
],
"source": "notebook/a_tanpa_label/03_clustering/03_clustering_review.ipynb",
"config": {
"data_file": "/kaggle/input/GANTI_DATASET/train.csv",
"encoding": "utf-8",
"text_col": "text",
"group_col": null,
"seed": 42,
"out": "/kaggle/working/hasil_toolkit",
"offline": false,
"preprocess": "bow",
"slang": {},
"stopwords": [],
"eda_sample": 5000,
"eda_top": 15,
"eda_languages": false,
"max_features": 60000,
"max_len": 192,
"e5_model": "intfloat/multilingual-e5-base",
"k_values": [
4,
6,
8
],
"fit_rows": 5000,
"task": "clustering"
},
"note": "Vocabulary dan topic ID dibekukan. Baca contoh dokumen per topik; metrik internal tidak mengukur akurasi sentimen."
},
"temporal": {
"id": "temporal",
"title": "Tren topik / sentimen per waktu",
"choices": [
"model.nmf",
"model.lda",
"model.kmeans",
"model.e5_kmeans",
"sentiment.pretrained"
],
"defaults": [
"model.nmf",
"model.lda",
"model.kmeans"
],
"required": [
"eda",
"unsupervised",
"time"
],
"aliases": [
"trend"
],
"source": "notebook/a_tanpa_label/02_tren_topik_waktu/02_tren_topik_waktu_review.ipynb",
"config": {
"data_file": "/kaggle/input/GANTI_DATASET/train.csv",
"encoding": "utf-8",
"text_col": "text",
"group_col": null,
"seed": 42,
"out": "/kaggle/working/hasil_toolkit",
"offline": false,
"preprocess": "bow",
"slang": {},
"stopwords": [],
"eda_sample": 5000,
"eda_top": 15,
"eda_languages": false,
"max_features": 60000,
"max_len": 192,
"e5_model": "intfloat/multilingual-e5-base",
"k_values": [
4,
6,
8
],
"fit_rows": 5000,
"time_col": "published_at",
"timezone": "Asia/Jakarta",
"naive_timezone": "UTC",
"period": "W",
"min_period_rows": 30,
"sentiment_sample_per_period": 100,
"sentiment_pretrained": false,
"positive_words": [
"bagus",
"baik",
"senang",
"bermanfaat"
],
"negative_words": [
"buruk",
"kecewa",
"mahal",
"gagal"
],
"task": "temporal"
},
"note": "Vocabulary dan topic ID dibekukan. Baca contoh dokumen per topik; metrik internal tidak mengukur akurasi sentimen."
}
},
"provenance": {
"classification": {
"path": "notebook/b_klasifikasi/10_multi_kelas/10_multi_kelas_v2_pretrained.ipynb",
"code_sha256": "ca45335e26d0cbb8739c20c83d3cb3cb300160896dd47eaef87646a787ee478f",
"relationship": "Adaptasi kontrak dan pola review; bukan klaim salinan sel atau hasil run yang identik."
},
"multilabel": {
"path": "notebook/b_klasifikasi/12_multi_label/12_multi_label_review.ipynb",
"code_sha256": "d24fe31c3b0bb6f85482736e7a630cb5ec9ec15ebdb9194c89d0c92a0be1a2cd",
"relationship": "Adaptasi kontrak dan pola review; bukan klaim salinan sel atau hasil run yang identik."
},
"similarity": {
"path": "notebook/e_pasangan_teks/20_kemiripan/20_kemiripan_review.ipynb",
"code_sha256": "4f7f2c85df50e644c3b52645715bd14776e7bd776a631170ebcfadf11436391f",
"relationship": "Adaptasi kontrak dan pola review; bukan klaim salinan sel atau hasil run yang identik."
},
"nli": {
"path": "notebook/e_pasangan_teks/21_nli/21_nli_review.ipynb",
"code_sha256": "356e7327d2db4ba8c264ac4024352d3c50b2d89d00130a54ba3bf6c614eae933",
"relationship": "Adaptasi kontrak dan pola review; bukan klaim salinan sel atau hasil run yang identik."
},
"retrieval": {
"path": "notebook/e_pasangan_teks/22_retrieval/22_retrieval_review.ipynb",
"code_sha256": "2a5d9812e1a2b3eb4eabd7792aa3a51b0e4cc7d43f7beea6c239ff1d9b18d179",
"relationship": "Adaptasi kontrak dan pola review; bukan klaim salinan sel atau hasil run yang identik."
},
"topic": {
"path": "notebook/a_tanpa_label/01_topic_modeling/01_topic_modeling_review.ipynb",
"code_sha256": "13db76fe5e1043a4fcf773ff425b60876bc7295f29c84b6a01ad53bf40f7409f",
"relationship": "Adaptasi kontrak dan pola review; bukan klaim salinan sel atau hasil run yang identik."
},
"clustering": {
"path": "notebook/a_tanpa_label/03_clustering/03_clustering_review.ipynb",
"code_sha256": "d10bcea740958d471a5e9fc266438ad64ee7504a010c05122c55272bf68e9cf0",
"relationship": "Adaptasi kontrak dan pola review; bukan klaim salinan sel atau hasil run yang identik."
},
"temporal": {
"path": "notebook/a_tanpa_label/02_tren_topik_waktu/02_tren_topik_waktu_review.ipynb",
"code_sha256": "2e0f4a9645d9d3af97117a81705e0ecfa379bb2ef7075d27c359547384d3532a",
"relationship": "Adaptasi kontrak dan pola review; bukan klaim salinan sel atau hasil run yang identik."
}
},
"steps": {
"classification": [
[
"Data & EDA raw",
"raw = read_table(CFG['data_file'], CFG['encoding'])\ntables = eda_for_task(raw, CFG)"
],
[
"Preprocess & split",
"prepared = prepare_supervised(raw, CFG)\ndata = split_supervised(prepared, CFG)\nif CFG['task']=='multilabel':\n    tables = multilabel_eda(data['train'], CFG['label_cols'])\n    for name, table in tables.items(): show_table(name, table)\n    save_tables(tables, CFG['out'], 'train_eda_')"
],
[
"Model → post → evaluasi",
"result = run_supervised(data, CFG)"
]
],
"multilabel": [
[
"Data & EDA raw",
"raw = read_table(CFG['data_file'], CFG['encoding'])\ntables = eda_for_task(raw, CFG)"
],
[
"Preprocess & split",
"prepared = prepare_supervised(raw, CFG)\ndata = split_supervised(prepared, CFG)\nif CFG['task']=='multilabel':\n    tables = multilabel_eda(data['train'], CFG['label_cols'])\n    for name, table in tables.items(): show_table(name, table)\n    save_tables(tables, CFG['out'], 'train_eda_')"
],
[
"Model → post → evaluasi",
"result = run_supervised(data, CFG)"
]
],
"similarity": [
[
"Data & EDA raw",
"raw = read_table(CFG['data_file'], CFG['encoding'])\ntables = eda_for_task(raw, CFG)"
],
[
"Preprocess & split",
"prepared = prepare_supervised(raw, CFG)\ndata = split_supervised(prepared, CFG)\nif CFG['task']=='multilabel':\n    tables = multilabel_eda(data['train'], CFG['label_cols'])\n    for name, table in tables.items(): show_table(name, table)\n    save_tables(tables, CFG['out'], 'train_eda_')"
],
[
"Model → post → evaluasi",
"result = run_supervised(data, CFG)"
]
],
"nli": [
[
"Data & EDA raw",
"raw = read_table(CFG['data_file'], CFG['encoding'])\ntables = eda_for_task(raw, CFG)"
],
[
"Preprocess & split",
"prepared = prepare_supervised(raw, CFG)\ndata = split_supervised(prepared, CFG)\nif CFG['task']=='multilabel':\n    tables = multilabel_eda(data['train'], CFG['label_cols'])\n    for name, table in tables.items(): show_table(name, table)\n    save_tables(tables, CFG['out'], 'train_eda_')"
],
[
"Model → post → evaluasi",
"result = run_supervised(data, CFG)"
]
],
"retrieval": [
[
"Corpus & queries",
"corpus = read_table(CFG['corpus_file'], CFG['encoding'], string_cols=[CFG['doc_id_col']])\nqueries = read_table(CFG['queries_file'], CFG['encoding'])\ntables = eda_for_task(corpus, CFG)\ndata = prepare_retrieval(corpus, queries, CFG)"
],
[
"Ranking → validation → holdout",
"result = run_retrieval(data, CFG)"
]
],
"topic": [
[
"EDA raw",
"raw = read_table(CFG['data_file'], CFG['encoding'])\ntables = eda_for_task(raw, CFG)"
],
[
"Frozen topic / cluster models",
"data = prepare_unsupervised(raw, CFG)\nresult = run_unsupervised(data, CFG)"
]
],
"clustering": [
[
"EDA raw",
"raw = read_table(CFG['data_file'], CFG['encoding'])\ntables = eda_for_task(raw, CFG)"
],
[
"Frozen topic / cluster models",
"data = prepare_unsupervised(raw, CFG)\nresult = run_unsupervised(data, CFG)"
]
],
"temporal": [
[
"EDA raw",
"raw = read_table(CFG['data_file'], CFG['encoding'])\ntables = eda_for_task(raw, CFG)"
],
[
"Frozen topic / cluster models",
"data = prepare_unsupervised(raw, CFG)\nresult = run_unsupervised(data, CFG)"
],
[
"Proporsi topik & sentimen",
"time_tables = temporal_report(result['assignments'], CFG)\nsentiment_tables = sentiment_over_time(result['assignments'], CFG)"
]
]
}
};
