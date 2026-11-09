from sklearn.model_selection import GroupShuffleSplit




def prepare_supervised(raw, cfg):
    cleaner = make_cleaner(cfg)
    frame = raw.copy().reset_index(drop=True)
    frame['_row_id'] = np.arange(len(frame))
    cols = cfg['pair_cols'] if cfg['task'] in {'nli','similarity'} else [cfg['text_col']]
    assert set(cols).issubset(frame), f'Kolom teks perlu {cols}; tersedia {frame.columns.tolist()}'
    for col, dest in zip(cols, ['a','b']):
        frame[dest] = frame[col].map(cleaner)
    valid = frame['a'].ne('')
    if len(cols)==2:
        valid &= frame.b.ne('')
    labels = cfg['label_cols'] if cfg['task']=='multilabel' else [cfg['label_col']]
    assert set(labels).issubset(frame), f'Kolom target perlu {labels}'
    if cfg['task']=='multilabel':
        for col in labels:
            values = pd.to_numeric(frame[col],errors='coerce')
            assert not (frame[col].notna() & values.isna()).any(), f'Label {col} tidak numerik'
            assert values.dropna().isin([0,1]).all(), f'Label {col} bukan 0/1/missing'
            frame[col] = values
        valid &= frame[labels].notna().any(axis=1)
    elif cfg['task']=='similarity':
        frame['y'] = pd.to_numeric(frame[cfg['label_col']],errors='coerce')
        valid &= np.isfinite(frame.y)
        assert frame.loc[valid,'y'].between(*cfg['score_range']).all(), 'Skor di luar score_range'
    else:
        valid &= frame[cfg['label_col']].notna()
        frame['label_name'] = frame[cfg['label_col']].astype(str).str.strip()
        if cfg['task']=='nli':
            mapping = cfg['nli_label_map']
            frame['label_name'] = frame.label_name.str.casefold().map(mapping)
            assert frame.loc[valid,'label_name'].notna().all(), 'Lengkapi nli_label_map untuk label data Anda'
    print('Baris invalid dibuang:',int((~valid).sum()))
    frame = frame.loc[valid].reset_index(drop=True)
    normalized = ['a','b'] if len(cols)==2 else ['a']
    frame['_group'] = connected_groups(frame, cols, cfg.get('group_col'), normalized, shared_endpoints=len(cols)==2)
    assert frame._group.nunique()>=5, 'Komponen grup terlalu sedikit; audit shared endpoint/duplikat/pengguna.'
    return frame


def split_supervised(frame, cfg):
    seed = cfg['seed']
    if cfg['task']=='multilabel':
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        pos = frame.groupby('_group')[cfg['label_cols']].max().fillna(0)
        known = frame[cfg['label_cols']].notna().groupby(frame._group).max()
        ids, profile = pos.index.to_numpy(), np.hstack([pos, known.astype(int)])
        first = MultilabelStratifiedShuffleSplit(1,test_size=.2,random_state=seed)
        rest, test = next(first.split(ids,profile))
        fit, val = next(MultilabelStratifiedShuffleSplit(1,test_size=.25,random_state=seed+1).split(ids[rest],profile[rest]))
        partitions = [ids[rest][fit],ids[rest][val],ids[test]]
    else:
        rest, test = next(GroupShuffleSplit(1,test_size=.2,random_state=seed).split(frame,groups=frame._group))
        fit, val = next(GroupShuffleSplit(1,test_size=.25,random_state=seed+1).split(frame.iloc[rest],groups=frame.iloc[rest]._group))
        partitions = [frame.iloc[rest[fit]]._group.unique(),frame.iloc[rest[val]]._group.unique(),frame.iloc[test]._group.unique()]
    parts = {name:frame[frame._group.isin(ids)].reset_index(drop=True) for name,ids in zip(['train','val','test'],partitions)}
    if cfg['task'] in {'classification','nli'}:
        names = ['entailment','neutral','contradiction'] if cfg['task']=='nli' else sorted(parts['train'].label_name.unique())
        assert len(names)>=2, 'Train memerlukan minimal dua kelas'
        for name,part in parts.items():
            part['y'] = part.label_name.map({v:i for i,v in enumerate(names)})
            assert part.y.notna().all(), f'{name} mengandung kelas yang tidak ada di train; atur ulang split/grup.'
            part['y'] = part.y.astype(int)
        assert set(parts['train'].y)==set(range(len(names))), 'Train tidak mencakup semua kelas NLI'
    else:
        names = cfg['label_cols'] if cfg['task']=='multilabel' else ['score']
    parts['labels'] = list(names)
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert set(parts[a]._group).isdisjoint(parts[b]._group)
    print('Split baris:',{k:len(parts[k]) for k in ['train','val','test']}, '| label order:',names)
    return parts


