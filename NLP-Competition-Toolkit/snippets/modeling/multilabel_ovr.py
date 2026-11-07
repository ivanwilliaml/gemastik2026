# title: One-vs-Rest dan Classifier Chain
# when: Multilabel. Binary relevance kuat; lanjutkan dengan threshold per label.
# avoid: Threshold 0,5 pada label jarang.
# tags: multilabel ovr chain binary_relevance
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain

def make_ovr(C=4.0, n_jobs=4):
    return OneVsRestClassifier(LogisticRegression(C=C, max_iter=2000), n_jobs=n_jobs)

def make_chain(C=4.0, seed=42):
    return ClassifierChain(LogisticRegression(C=C, max_iter=2000), order="random", random_state=seed)
