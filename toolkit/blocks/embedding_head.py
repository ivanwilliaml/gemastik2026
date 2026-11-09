"""Head E5 terpisah; retrieval cukup mengambil encoder tanpa supervised head."""

@register('e5_head')
def fit_e5(data, cfg):
    matrices = []
    for split in ['train','val','test']:
        frame = data[split]
        a = e5_encode(frame.a,cfg,'query')
        if cfg['task'] in {'nli','similarity'}:
            b = e5_encode(frame.b,cfg,'query')
            a = (a*b).sum(1,keepdims=True) if cfg['task']=='similarity' else np.hstack([a,b,np.abs(a-b),a*b])
        matrices.append(a)
    return fit_head(*matrices,data,cfg)

