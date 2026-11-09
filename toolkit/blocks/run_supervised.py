def run_supervised(data, cfg):
    """Model plug-in semuanya menghasilkan (validation, holdout) dengan urutan label yang sama."""
    seed_all(cfg['seed'])
    out = Path(cfg['out']);out.mkdir(parents=True,exist_ok=True)
    vp,hp,times = {},{},[]
    for name in cfg['models']:
        assert name in MODEL_RUNNERS, f'Blok model {name} belum disalin'
        start = time.perf_counter()
        vp[name],hp[name] = MODEL_RUNNERS[name](data,cfg)
        for split,p in [('val',vp[name]),('test',hp[name])]:
            assert len(p)==len(data[split]) and np.isfinite(p).all(), f'Output {name}/{split} invalid'
            if cfg['task']=='multilabel':
                assert p.shape==(len(data[split]),len(data['labels']))
        times.append({'model':name,'seconds':time.perf_counter()-start})
    yval = target_array(data['val'],cfg)
    policy,board = select_policy(vp,yval,data['val']._group.to_numpy(),cfg)
    show_table('Validation cross-fit (pengembangan, bukan test)',board)
    summary = {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in policy.items() if k!='params'}
    params = policy['params']
    if policy['route']=='isotonic':
        summary['params'] = {'x':params.X_thresholds_.tolist(),'y':params.y_thresholds_.tolist()}
    else:
        summary['params'] = params.tolist() if isinstance(params,np.ndarray) else params
    (out/'selection_frozen_before_test.json').write_text(json.dumps(summary,indent=2))
    # Baru sekarang baca target holdout. Tidak ada pemilihan ulang berdasarkan tabel berikut.
    ytest = target_array(data['test'],cfg)
    final,raw = predict_policy(hp,policy)
    rows = [{'model':name,**task_metrics(ytest,p,cfg['task'])} for name,p in hp.items()]
    score = task_metrics(ytest,final,cfg['task'])
    if cfg['task']=='multilabel':
        score['macro_AP'] = masked_multilabel_metrics(ytest,raw)['macro_AP']
    rows.append({'model':'FINAL',**score})
    tables = {'validation':board,'holdout':pd.DataFrame(rows),'timing':pd.DataFrame(times)}
    tables.update(error_tables(data['test'],ytest,final,cfg,data['labels']))
    if cfg['task']=='multilabel':
        for j,label in enumerate(data['labels']):
            k = np.isfinite(ytest[:,j])
            tables['per_label'].loc[j,'AP_raw_probability'] = average_precision_score(ytest[k,j],raw[k,j]) if np.any(ytest[k,j]==1) else np.nan
    save_tables(tables,out)
    np.savez_compressed(out/'predictions.npz',truth=ytest,probability=raw,decision_or_score=final,labels=np.array(data['labels']))
    pd.concat([data[k][['_row_id','_group']].assign(split=k) for k in ['train','val','test']]).to_csv(out/'split.csv',index=False)
    (out/'config.json').write_text(json.dumps(cfg,indent=2,default=str))
    show_table('Holdout · pilihan sudah dibekukan',tables['holdout'])
    show_table('Kelemahan per kondisi',tables['slices'])
    return {'policy':policy,'tables':tables,'validation_predictions':vp,'holdout_predictions':hp}
