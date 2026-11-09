# title: Metrik klasifikasi dan multilabel
# when: Klasifikasi, pasangan, ABSA. Tulis ulang rumus panitia dan uji.
# avoid: evaluate.load: butuh internet.
# tags: f1 macro auc logloss hamming topk
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, hamming_loss, log_loss, precision_score, recall_score, roc_auc_score, top_k_accuracy_score

def cls_report(y, proba=None, pred=None):
    """proba: (n, k). pred otomatis argmax bila kosong."""
    y = np.asarray(y)
    pred = pred if pred is not None else np.asarray(proba).argmax(1)
    r = dict(acc=accuracy_score(y, pred), macro_f1=f1_score(y, pred, average="macro", zero_division=0), weighted_f1=f1_score(y, pred, average="weighted", zero_division=0),
             precision=precision_score(y, pred, average="macro", zero_division=0), recall=recall_score(y, pred, average="macro", zero_division=0))
    if proba is not None:
        p = np.asarray(proba)
        r["log_loss"] = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=np.arange(p.shape[1]))
        if p.shape[1] == 2:
            r.update(f1_pos=f1_score(y, pred, zero_division=0), roc_auc=roc_auc_score(y, p[:, 1]), pr_auc=average_precision_score(y, p[:, 1]))
        else:
            r["roc_auc_ovr"] = roc_auc_score(y, p, multi_class="ovr", labels=np.arange(p.shape[1]))
            r["top2"] = top_k_accuracy_score(y, p, k=2, labels=np.arange(p.shape[1]))
    return {k: round(float(v), 4) for k, v in r.items()}

def multilabel_report(Y, pred):
    Y, pred = np.asarray(Y), np.asarray(pred)
    return {k: round(float(v), 4) for k, v in dict(micro_f1=f1_score(Y, pred, average="micro", zero_division=0), macro_f1=f1_score(Y, pred, average="macro", zero_division=0),
            samples_f1=f1_score(Y, pred, average="samples", zero_division=0), hamming=hamming_loss(Y, pred), subset_acc=accuracy_score(Y, pred)).items()}
