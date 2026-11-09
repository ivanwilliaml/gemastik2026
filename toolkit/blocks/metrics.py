from sklearn.metrics import (f1_score, log_loss, classification_report, confusion_matrix,
    average_precision_score, precision_recall_fscore_support, mean_absolute_error, mean_squared_error)
from scipy.stats import pearsonr, spearmanr


def masked_multilabel_metrics(y, p, threshold=.5):
    y = np.asarray(y,float).copy()
    y[y<0] = np.nan
    p = np.asarray(p,float)
    assert p.shape==y.shape and np.isfinite(p).all()
    known = np.isfinite(y)
    pred = p>=threshold
    per, ap = [], []
    for j in range(y.shape[1]):
        k = known[:,j]
        per.append(f1_score(y[k,j],pred[k,j],zero_division=0) if k.any() else np.nan)
        ap.append(average_precision_score(y[k,j],p[k,j]) if np.any(y[k,j]==1) else np.nan)
    complete = known.all(axis=1)
    return {'macro_f1':float(np.nanmean(per)) if np.isfinite(per).any() else np.nan,
        'micro_f1':f1_score(y[known],pred[known],zero_division=0) if known.any() else np.nan,
        'macro_AP':float(np.nanmean(ap)) if np.isfinite(ap).any() else np.nan,
        'hamming_known':float(np.mean(y[known]!=pred[known])) if known.any() else np.nan,
        'exact_match_complete':float(np.all(y[complete]==pred[complete],axis=1).mean()) if complete.any() else np.nan,
        'n_complete':int(complete.sum()), 'n_known_cells':int(known.sum())}


def task_metrics(y, p, task):
    p = np.asarray(p)
    assert len(y)==len(p) and np.isfinite(p).all()
    if task=='similarity':
        y, p = np.asarray(y).ravel(), p.ravel()
        variable = len(y)>1 and np.std(y)>0 and np.std(p)>0
        return {'pearson':float(pearsonr(y,p).statistic) if variable else np.nan,
            'spearman':float(spearmanr(y,p).statistic) if variable else np.nan,
            'MAE':mean_absolute_error(y,p), 'RMSE':float(np.sqrt(mean_squared_error(y,p)))}
    if task=='multilabel':
        return masked_multilabel_metrics(y,p)
    assert p.ndim==2 and np.allclose(p.sum(1),1,atol=1e-4)
    pred = p.argmax(1)
    return {'macro_f1':f1_score(y,pred,labels=np.arange(p.shape[1]),average='macro',zero_division=0),
        'accuracy':float(np.mean(y==pred)), 'log_loss':log_loss(y,p,labels=np.arange(p.shape[1]))}
