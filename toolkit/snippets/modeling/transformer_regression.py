# title: Fine-tuning regresi
# when: Target numerik atau skor kemiripan. Standarkan target dulu.
# tags: regresi transformer
# needs: modeling/transformer_trainer
def train_regression(model_path, texts, y, valid_texts=None, valid_y=None, texts2=None, valid_texts2=None, **kw):
    y = np.asarray(y, dtype="float32")
    mu, sd = y.mean(), y.std() + 1e-6
    vy = None if valid_y is None else (np.asarray(valid_y, dtype="float32") - mu) / sd
    m, t = train_transformer(model_path, texts, ((y - mu) / sd)[:, None], valid_texts, None if vy is None else vy[:, None], texts2, valid_texts2, problem="reg", num_labels=1, **kw)
    return m, t, (float(mu), float(sd))

def predict_regression(model, tok, texts, scale, texts2=None, **kw):
    return predict_transformer(model, tok, texts, texts2, problem="reg", **kw).ravel() * scale[1] + scale[0]
