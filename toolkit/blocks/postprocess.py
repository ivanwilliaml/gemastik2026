from sklearn.model_selection import GroupKFold
from scipy.special import softmax
from scipy.optimize import minimize_scalar


def fit_post(p, y, route, cfg):
    if route=='identity':
        return None
    if route=='isotonic':
        from sklearn.isotonic import IsotonicRegression
        return IsotonicRegression(out_of_bounds='clip').fit(p.ravel(),y)
    if route=='temperature':
        z = np.log(np.clip(p,1e-7,1))
        return float(minimize_scalar(lambda t:log_loss(y,softmax(z/t,axis=1),labels=np.arange(p.shape[1])),bounds=(.3,4),method='bounded').x)
    if route=='class_bias':
        bias = np.zeros(p.shape[1])
        for _ in range(2):
            for j in range(len(bias)):
                best,score = bias[j],-1
                for v in [-.8,-.4,0.,.4,.8]:
                    trial = bias.copy();trial[j]=v
                    s = task_metrics(y,apply_post(p,'class_bias',trial),cfg['task'])['macro_f1']
                    if (s,-abs(v))>(score,-abs(best)):
                        best,score = v,s
                bias[j] = best
        return bias
    if route in {'global_threshold','label_threshold'}:
        grid = [.03,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.7,.8,.9]
        if route=='global_threshold':
            return max(grid,key=lambda c:(masked_multilabel_metrics(y,p,c)['macro_f1'],-abs(c-.5)))
        cuts = np.full(p.shape[1],.5)
        for j in range(p.shape[1]):
            k = np.isfinite(y[:,j]);target=y[k,j]
            if min(np.sum(target==1),np.sum(target==0))<cfg.get('threshold_min_support',5):
                continue
            cuts[j] = max(grid,key=lambda c:(f1_score(target,p[k,j]>=c,zero_division=0),-abs(c-.5)))
        return cuts
    raise ValueError(f'Post-processing tidak dikenal: {route}')


def apply_post(p, route, fitted):
    if route=='identity':
        return p.copy()
    if route=='isotonic':
        return fitted.predict(p.ravel())
    if route=='temperature':
        return softmax(np.log(np.clip(p,1e-7,1))/fitted,axis=1)
    if route=='class_bias':
        return softmax(np.log(np.clip(p,1e-7,1))+fitted,axis=1)
    if route in {'global_threshold','label_threshold'}:
        return (p>=fitted).astype(float)
    raise ValueError(route)


def select_policy(vp, y, groups, cfg):
    """Cross-fit parameter post di validation, kemudian refit; holdout tidak dibaca."""
    candidates = dict(vp)
    base = [name for name in vp if name!='dummy']
    if cfg.get('ensemble',False) and len(base)>1:
        candidates['ensemble'] = np.mean([vp[name] for name in base],axis=0)
    nfold = min(3,len(np.unique(groups)))
    routes = ['identity']+cfg.get('post',[])
    allowed = {'identity','isotonic'} if cfg['task']=='similarity' else {'identity','global_threshold','label_threshold'} if cfg['task']=='multilabel' else {'identity','temperature','class_bias'}
    assert set(routes)<=allowed, f'Post-processing tidak cocok dengan {cfg["task"]}: {set(routes)-allowed}'
    if nfold<2 and len(routes)>1:
        print('Post dinonaktifkan: validation kurang dari dua grup.')
        routes = ['identity']
    folds = list(GroupKFold(n_splits=nfold).split(y,groups=groups)) if nfold>=2 else []
    board = []
    for name,p in candidates.items():
        for route in routes:
            cv = p.copy()
            if route!='identity':
                for a,b in folds:
                    cv[b] = apply_post(p[b],route,fit_post(p[a],y[a],route,cfg))
            metrics = task_metrics(y,cv,cfg['task'])
            if cfg['task']=='multilabel':
                metrics['macro_AP'] = masked_multilabel_metrics(y,p)['macro_AP']
            board.append({'model':name,'route':route,**metrics})
    board = pd.DataFrame(board)
    metric,ascending = ('RMSE',True) if cfg['task']=='similarity' else ('macro_f1',False)
    # Identitas menang saat seri; temperature dapat dipilih untuk log-loss yang membaik dengan F1 sama.
    sort = [metric]+(['log_loss'] if 'log_loss' in board else [])
    board = board.sort_values(sort,ascending=[ascending]+([True] if len(sort)>1 else []),kind='stable')
    best = board.iloc[0]
    params = fit_post(candidates[best.model],y,best.route,cfg)
    return {'model':best.model,'route':best.route,'params':params,'members':base},board


def predict_policy(hp, policy):
    p = np.mean([hp[name] for name in policy['members']],axis=0) if policy['model']=='ensemble' else hp[policy['model']]
    return apply_post(p,policy['route'],policy['params']),p


def duplicate_consistency(probabilities, texts):
    """Opsional: rata-rata prediksi exact key dalam satu batch inference, tanpa label."""
    keys = pd.Series([safe_text(t).casefold() for t in texts])
    values = pd.DataFrame(probabilities)
    assert len(keys)==len(values)
    return values.groupby(keys).transform('mean').to_numpy()


def audited_rules(probabilities, texts, rules):
    """rules=[{'pattern':..., 'class_index':..., 'audited':True}]. Jangan belajar aturan dari holdout."""
    p = probabilities.copy()
    for rule in rules:
        assert rule.get('audited') is True, 'Aturan harus diaudit dari train dahulu.'
        for i,text in enumerate(texts):
            if re.search(rule['pattern'],text,flags=re.I):
                p[i] = 0
                p[i,rule['class_index']] = 1
    return p


def enforce_hierarchy(decisions, parent, children):
    """Opsional multilabel. Aktifkan hanya setelah audit anotasi train dan validasi grup."""
    result = decisions.astype(bool).copy()
    result[:,parent] |= result[:,children].any(axis=1)
    return result
