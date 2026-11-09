# title: Bi-encoder kosinus
# when: Similarity, atau sebagai fitur tambahan. Murah.
# avoid: Kontradiksi: mirip bukan berarti entailment.
# tags: biencoder kosinus similarity
# needs: feature_engineering/sbert_encode
def biencoder_cos(model_path, a, b, **kw):
    Ea, Eb = encode_st(model_path, a, **kw), encode_st(model_path, b, **kw)
    return (Ea * Eb).sum(1)
