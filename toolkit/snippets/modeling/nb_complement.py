# title: MultinomialNB dan ComplementNB
# when: Baseline tercepat. ComplementNB untuk imbalance, spam, dan intent.
# avoid: Fitur negatif (SVD, embedding).
# tags: naive_bayes complement spam
from sklearn.naive_bayes import ComplementNB, MultinomialNB

def make_nb(kind="complement", alpha=.3):
    return (ComplementNB if kind == "complement" else MultinomialNB)(alpha=alpha)
