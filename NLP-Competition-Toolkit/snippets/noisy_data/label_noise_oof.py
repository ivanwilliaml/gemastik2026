# title: Kandidat noise: OOF dan cleanlab
# when: Prioritas baca saja.
# avoid: Menghapus otomatis. Prediksi data latih, bukan OOF.
# tags: cleanlab noise oof
import numpy as np
import pandas as pd

def noise_candidates(texts, y, oof, k=50):
    """Urutkan menurut probabilitas label yang diberikan: terendah paling mencurigakan."""
    y, oof = np.asarray(y), np.asarray(oof)
    d = pd.DataFrame({"text": list(texts), "y": y, "pred": oof.argmax(1), "p_label": oof[np.arange(len(y)), y], "conf": oof.max(1)})
    return d[d.y != d.pred].sort_values("p_label").head(k)

def cleanlab_rank(y, oof):
    from cleanlab.filter import find_label_issues
    return find_label_issues(labels=np.asarray(y), pred_probs=np.asarray(oof), return_indices_ranked_by="self_confidence", n_jobs=1)
