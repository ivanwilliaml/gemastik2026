# title: Fine-tuning multilabel
# when: Multilabel dengan transformer. Lanjutkan dengan threshold per label.
# avoid: Target integer: fungsi ini mengubahnya ke float.
# tags: multilabel bce transformer
# needs: modeling/transformer_trainer
def train_multilabel(model_path, texts, Y, valid_texts=None, valid_Y=None, **kw):
    return train_transformer(model_path, texts, Y, valid_texts, valid_Y, problem="multi", **kw)

def predict_multilabel(model, tok, texts, **kw):
    return predict_transformer(model, tok, texts, problem="multi", **kw)
