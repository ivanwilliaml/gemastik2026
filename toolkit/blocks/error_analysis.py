"""Audit kesalahan; dipasang terpisah dari hitungan metrik/post-processing."""

def error_tables(frame, y, p, cfg, labels):
    task = cfg['task']
    rows = frame[['_row_id','a']+(['b'] if 'b' in frame else [])].copy()
    if task=='multilabel':
        known = np.isfinite(y)
        pred = p>=.5
        detail = []
        for j,label in enumerate(labels):
            k = known[:,j]
            pr,rc,f1,_ = precision_recall_fscore_support(y[k,j],pred[k,j],average='binary',zero_division=0) if k.any() else (np.nan,)*4
            detail.append({'label':label,'known':int(k.sum()),'positive':int(np.sum(y[k,j]==1)),
                'precision':pr,'recall':rc,'f1':f1,
                'FP':int(np.sum((y[k,j]==0)&pred[k,j])), 'FN':int(np.sum((y[k,j]==1)&~pred[k,j]))})
        rows['truth'] = [[labels[j] for j in range(len(labels)) if y[i,j]==1] for i in range(len(y))]
        rows['predicted'] = [[labels[j] for j in range(len(labels)) if pred[i,j]] for i in range(len(y))]
        rows['unknown'] = (~known).sum(1)
        rows['error_count'] = ((y!=pred)&known).sum(1)
        tables = {'per_label':pd.DataFrame(detail)}
    elif task=='similarity':
        rows['truth'],rows['predicted'] = y,p.ravel()
        rows['absolute_error'] = np.abs(y-p.ravel())
        tables = {}
    else:
        pred = p.argmax(1)
        rows['truth'] = [labels[i] for i in y]
        rows['predicted'] = [labels[i] for i in pred]
        rows['confidence'] = p.max(1)
        rows['error'] = y!=pred
        tables = {'per_class':pd.DataFrame(classification_report(y,pred,labels=np.arange(len(labels)),target_names=labels,output_dict=True,zero_division=0)).T,
            'confusion_counts':pd.DataFrame(confusion_matrix(y,pred,labels=np.arange(len(labels))),index=labels,columns=labels)}
    tables['errors'] = rows
    conditions = {'short_5_words':frame.a.str.split().str.len().le(5), 'long_40_words':frame.a.str.split().str.len().gt(40)}
    for name in ['negasi','emoji_simbol','slang_contoh']:
        conditions[name] = frame.a.str.count(NOISE_PATTERNS[name],flags=re.I).gt(0)
    slices = []
    for name,mask in conditions.items():
        k = mask.to_numpy()
        if k.any():
            row = task_metrics(y[k],p[k],task)
            if task=='multilabel':
                row.pop('macro_AP',None)  # Decisions after threshold are not probability AP.
            slices.append({'slice':name,'n':int(k.sum()),**row})
    tables['slices'] = pd.DataFrame(slices)
    return tables

