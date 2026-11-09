# title: Cross-encoder pasangan
# when: NLI, parafrase, duplikat, relevansi. Dua teks dalam satu input.
# avoid: Uji simetri bila urutan tak bermakna.
# tags: pair cross_encoder nli
# needs: modeling/transformer_trainer
def train_pair(model_path, a, b, y, va=None, vb=None, vy=None, **kw):
    return train_transformer(model_path, a, y, va, vy, texts2=b, valid_texts2=vb, **kw)

def predict_pair(model, tok, a, b, symmetric=False, **kw):
    p = predict_transformer(model, tok, a, b, **kw)
    return (p + predict_transformer(model, tok, b, a, **kw)) / 2 if symmetric else p
