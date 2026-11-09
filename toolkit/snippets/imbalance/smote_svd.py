# title: SMOTE di fitur padat
# when: Fitur padat dan kelas minor kecil. Pipeline imblearn memastikan hanya di fold latih.
# avoid: TF-IDF mentah.
# tags: smote imblearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imb_pipeline
from sklearn.linear_model import LogisticRegression

def make_smote_lr(k=5, C=1.0, seed=42):
    """Input fitur padat. Bisa langsung masuk oof_predict sebagai model."""
    return imb_pipeline(SMOTE(k_neighbors=k, random_state=seed), LogisticRegression(C=C, max_iter=2000))
