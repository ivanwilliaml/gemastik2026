from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.dummy import DummyClassifier


def vectorizer(kind='word', features=60000):
    word = TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=features,sublinear_tf=True)
    if kind=='word':
        return word
    return FeatureUnion([('word',word),('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=features,sublinear_tf=True))])


def pair_matrix(a, b, symmetric=False):
    delta = abs(a-b)
    return hstack([delta,a.multiply(b)] if symmetric else [a,b,delta,a.multiply(b)],format='csr')


def binary_heads(x, y, v, h, seed=42):
    vp,hp = np.zeros((v.shape[0],y.shape[1])),np.zeros((h.shape[0],y.shape[1]))
    for j in range(y.shape[1]):
        k = np.isfinite(y[:,j]);values = np.unique(y[k,j])
        if not len(values):
            print('Label tanpa known train, prior .5:',j)
            vp[:,j],hp[:,j] = .5,.5
            continue
        model = DummyClassifier(strategy='constant',constant=values[0]) if len(values)==1 else LogisticRegression(C=2,max_iter=1200,class_weight='balanced',random_state=seed)
        model.fit(x[k],y[k,j])
        if 1 in model.classes_:
            pos = list(model.classes_).index(1)
            vp[:,j],hp[:,j] = model.predict_proba(v)[:,pos],model.predict_proba(h)[:,pos]
    return vp,hp


def fit_head(x, v, h, data, cfg):
    y = target_array(data['train'],cfg)
    if cfg['task']=='multilabel':
        return binary_heads(x,y,v,h,cfg['seed'])
    if cfg['task']=='similarity':
        model = Ridge(alpha=5).fit(x,y)
        return tuple(np.clip(model.predict(m),*cfg['score_range']) for m in [v,h])
    model = LogisticRegression(C=2,max_iter=1200,class_weight='balanced',random_state=cfg['seed']).fit(x,y)
    assert np.array_equal(model.classes_,np.arange(len(data['labels'])))
    return model.predict_proba(v),model.predict_proba(h)


def fit_tfidf(data, cfg, kind):
    vec = vectorizer(kind,cfg.get('max_features',60000))
    pair = cfg['task'] in {'nli','similarity'}
    train = data['train']
    vec.fit(pd.concat([train.a,train.b]) if pair else train.a)
    matrices = []
    for name in ['train','val','test']:
        frame = data[name]
        a = vec.transform(frame.a)
        matrices.append(pair_matrix(a,vec.transform(frame.b),cfg['task']=='similarity') if pair else a)
    return fit_head(*matrices,data,cfg)


@register('tfidf_word')
def fit_word(data, cfg):
    return fit_tfidf(data,cfg,'word')


@register('tfidf_union')
def fit_word_char(data, cfg):
    return fit_tfidf(data,cfg,'union')
