# title: Vocabulary, kata langka, OOV, n-gram teratas
# when: Menentukan min_df, max_features, dan perlu tidaknya char n-gram.
# tags: vocabulary oov kata_langka ngram stopword
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def vocab_stats(train, test=None):
    v = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = v.fit_transform(train)
    df_ = np.asarray((X > 0).sum(0)).ravel()
    out = {"vocab": X.shape[1], "df1_pct": round(float((df_ == 1).mean()), 3), "token_per_doc": round(float(X.sum(1).mean()), 1)}
    if test is not None:
        V = CountVectorizer(token_pattern=r"(?u)\b\w+\b").fit(test)
        tr = set(v.vocabulary_)
        cnt = np.asarray(V.transform(test).sum(0)).ravel()
        names = V.get_feature_names_out()
        oov = np.array([w not in tr for w in names])
        out.update(oov_unik=round(float(oov.mean()), 3), oov_token=round(float(cnt[oov].sum() / cnt.sum()), 3))
    return out

def top_ngrams(texts, k=30, ngram=(1, 1), stop_words=None, min_df=2):
    v = CountVectorizer(ngram_range=ngram, stop_words=stop_words, min_df=min_df, token_pattern=r"(?u)\b\w+\b")
    c = np.asarray(v.fit_transform(texts).sum(0)).ravel()
    n = v.get_feature_names_out()
    return [(n[i], int(c[i])) for i in c.argsort()[::-1][:k]]
