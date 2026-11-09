@register('dummy')
def fit_dummy(data, cfg):
    y = target_array(data['train'],cfg)
    if cfg['task']=='multilabel':
        return tuple(np.zeros((len(data[s]),len(data['labels']))) for s in ['val','test'])
    if cfg['task']=='similarity':
        return tuple(np.full(len(data[s]),float(np.mean(y))) for s in ['val','test'])
    prior = np.bincount(y,minlength=len(data['labels']))/len(y)
    return tuple(np.tile(prior,(len(data[s]),1)) for s in ['val','test'])
