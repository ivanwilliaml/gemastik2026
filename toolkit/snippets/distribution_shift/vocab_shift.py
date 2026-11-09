# title: Vocabulary shift
# when: AUC adversarial tinggi. Temukan kata artefak.
# avoid: Membuang kata hanya karena beda: bisa tema yang sah.
# tags: vocabulary shift log_odds
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def vocab_shift(train, test, k=30, min_df=3):
    v = CountVectorizer(min_df=min_df, token_pattern=r"(?u)\b\w+\b", binary=True)
    X = v.fit_transform(list(train) + list(test))
    a = np.asarray(X[:len(train)].mean(0)).ravel()
    b = np.asarray(X[len(train):].mean(0)).ravel()
    d = pd.DataFrame({"kata": v.get_feature_names_out(), "train": a, "test": b, "log_rasio": np.log((b + 1e-4) / (a + 1e-4))})
    return d.reindex(d.log_rasio.abs().sort_values(ascending=False).index).head(k).round(4)
