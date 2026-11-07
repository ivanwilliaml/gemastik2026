# title: Focal loss
# when: Transformer dan kelas minor tetap rendah. Berikan sebagai loss_fn ke train_transformer.
# avoid: Sebelum decision layer dicoba.
# tags: focal loss gamma
import torch
import torch.nn.functional as F

def focal_loss(gamma=2.0, weight=None):
    """fn = focal_loss(); train_transformer(..., loss_fn=fn)"""
    def fn(logits, labels):
        ce = F.cross_entropy(logits, labels, reduction="none", weight=None if weight is None else torch.as_tensor(weight, device=logits.device, dtype=logits.dtype))
        pt = torch.exp(-F.cross_entropy(logits, labels, reduction="none"))
        return ((1 - pt) ** gamma * ce).mean()
    return fn
