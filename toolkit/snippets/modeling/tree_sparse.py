# title: LightGBM dan XGBoost dengan early stopping
# when: Fitur pasangan, meta, SVD, atau sparse terbatas. Tanpa early stopping bisa langsung masuk oof_predict.
# avoid: early_stopping_rounds di fit: sudah hilang.
# tags: lightgbm xgboost tree early_stopping
import lightgbm as lgb
from xgboost import XGBClassifier

def make_lgbm(n=500, lr=.05, leaves=63, seed=42, n_jobs=8):
    return lgb.LGBMClassifier(n_estimators=n, learning_rate=lr, num_leaves=leaves, subsample=.8, subsample_freq=1,
                              colsample_bytree=.5, random_state=seed, n_jobs=n_jobs, verbose=-1)

def fit_lgbm_es(model, Xtr, ytr, Xva, yva, patience=50):
    return model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0)])

def make_xgb(n=500, lr=.05, depth=6, device="cpu", patience=None, seed=42, eval_metric=None):
    """device='cuda' di GPU. patience: wajib beri eval_set saat fit. eval_metric kosong: logloss atau mlogloss otomatis."""
    return XGBClassifier(n_estimators=n, learning_rate=lr, max_depth=depth, tree_method="hist", device=device, subsample=.8,
                         colsample_bytree=.5, early_stopping_rounds=patience, eval_metric=eval_metric, random_state=seed)
