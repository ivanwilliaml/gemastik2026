# title: Baca CoNLL/BIO dan ubah offset ke BIO
# when: Task per token: NER, POS, term aspek, keyphrase.
# tags: ner bio conll offset pos
def read_conll(path, sep=None, tag_col=-1, encoding="utf-8"):
    """Kembalikan (sents, tags): list token dan list tag per kalimat."""
    sents, tags, s, t = [], [], [], []
    for line in open(path, encoding=encoding):
        line = line.strip()
        if not line or line.startswith("-DOCSTART-"):
            if s:
                sents.append(s); tags.append(t); s, t = [], []
            continue
        parts = line.split(sep)
        s.append(parts[0]); t.append(parts[tag_col])
    if s:
        sents.append(s); tags.append(t)
    return sents, tags

def spans_to_bio(tokens_offsets, spans):
    """tokens_offsets: [(start,end)], spans: [(start,end,label)] menjadi tag BIO per token."""
    tags = ["O"] * len(tokens_offsets)
    for s, e, lab in spans:
        first = True
        for i, (a, b) in enumerate(tokens_offsets):
            if a < e and b > s:
                tags[i] = ("B-" if first else "I-") + lab
                first = False
    return tags
