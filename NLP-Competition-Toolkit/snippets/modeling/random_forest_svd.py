# title: Random Forest di SVD atau meta features
# when: Fitur padat kecil, untuk diversity.
# avoid: TF-IDF sparse besar.
# tags: random_forest
from sklearn.ensemble import RandomForestClassifier

def make_rf(n=300, leaf=2, n_jobs=8, seed=42):
    return RandomForestClassifier(n_estimators=n, min_samples_leaf=leaf, n_jobs=n_jobs, random_state=seed)
