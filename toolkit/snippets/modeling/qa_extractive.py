# title: Baseline QA: kalimat terbaik
# when: Baseline QA dalam menit, sekaligus membaca format.
# avoid: Teks tanpa tanda baca: pakai jendela kata.
# tags: qa ekstraktif tfidf
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def best_sentence(question, context, window=25):
    s = [x for x in re.split(r"(?<=[.!?])\s+", str(context)) if x.strip()]
    if len(s) <= 1:
        w = str(context).split()
        s = [" ".join(w[i:i + window]) for i in range(0, max(len(w), 1), max(window // 2, 1))]
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(s + [question])
    return s[int((v.transform(s) @ v.transform([question]).T).toarray().ravel().argmax())]

def answer_tail(question, sentence):
    """Sisa kata pada kalimat yang tidak ada di pertanyaan: jawaban kasar."""
    q = set(re.findall(r"\w+", question.lower()))
    return " ".join(w for w in sentence.split() if re.sub(r"\W", "", w.lower()) not in q)
