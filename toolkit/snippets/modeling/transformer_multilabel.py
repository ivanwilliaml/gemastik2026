# title: Fine-tuning multilabel
# when: Multilabel dengan transformer. Lanjutkan dengan threshold per label.
# avoid: Target integer: fungsi ini mengubahnya ke float.
# tags: multilabel bce transformer
# needs: modeling/transformer_trainer
def train_multilabel(model_path, texts, Y, valid_texts=None, valid_Y=None, **kw):
    import numpy as np
    import torch
    Y = np.asarray(Y, dtype=float)
    assert np.isin(Y[np.isfinite(Y)], [0, 1]).all(), 'Target harus 0/1/NaN'
    def masked_bce(logits, labels):
        mask = labels >= 0
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels.float().clamp(0,1), reduction='none')
        return (loss*mask).sum()/mask.sum().clamp(min=1)
    Y = np.where(np.isfinite(Y), Y, -1)
    if valid_Y is not None:
        valid_Y = np.asarray(valid_Y, dtype=float)
        assert np.isin(valid_Y[np.isfinite(valid_Y)], [0, 1]).all(), 'Target validation harus 0/1/NaN'
        valid_Y = np.where(np.isfinite(valid_Y), valid_Y, -1)
    kw.setdefault('loss_fn', masked_bce)
    return train_transformer(model_path, texts, Y, valid_texts, valid_Y, problem="multi", **kw)

def predict_multilabel(model, tok, texts, **kw):
    return predict_transformer(model, tok, texts, problem="multi", **kw)
