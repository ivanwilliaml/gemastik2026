@register('word2vec')
def fit_word2vec(data, cfg):
    from gensim.models import Word2Vec
    pair = cfg['task'] in {'nli','similarity'}
    def tokenize(values):
        return [re.findall(r'[^\W\d_]+',str(t).casefold()) for t in values]
    train = data['train']
    corpus = tokenize(pd.concat([train.a,train.b]) if pair else train.a)
    model = Word2Vec(corpus,vector_size=100,min_count=1,window=5,workers=1,seed=cfg['seed'],epochs=10)
    def encode(values):
        out = []
        for tokens in tokenize(values):
            vectors = [model.wv[w] for w in tokens if w in model.wv]
            out.append(np.mean(vectors,axis=0) if vectors else np.zeros(100))
        return np.asarray(out)
    matrices = []
    for split in ['train','val','test']:
        f = data[split];a=encode(f.a)
        if pair:
            b = encode(f.b)
            a = np.hstack([np.abs(a-b),a*b]) if cfg['task']=='similarity' else np.hstack([a,b,np.abs(a-b),a*b])
        matrices.append(a)
    return fit_head(*matrices,data,cfg)
