# %% [markdown]
# **Model dengan cross-validation.** `cv_tfidf` adalah baseline word dan char TF-IDF + regresi
# logistik, sedangkan `cv_transformer` melatih tiap kombinasi model dan seed di semua fold. Hasilnya
# probabilitas OOF dan rata-rata probabilitas holdout.

# %%
from scipy.sparse import hstack  # noqa: E402
from sklearn.multiclass import OneVsRestClassifier  # noqa: E402


def cv_tfidf(train_texts, target, holdout_texts, folds, multilabel=False):
    """Baseline TF-IDF. Vektorisasi difit di dalam fold supaya tidak bocor."""
    target = np.asarray(target)
    shape = target.shape[1] if multilabel else int(target.max()) + 1
    oof = np.zeros((len(target), shape))
    holdout = np.zeros((len(holdout_texts), shape))
    train_texts = pd.Series(list(train_texts))
    holdout_series = pd.Series(list(holdout_texts))
    for fold in np.unique(folds):
        fit, val = folds != fold, folds == fold
        word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
        x_fit = hstack([word.fit_transform(train_texts[fit]), char.fit_transform(train_texts[fit])])
        base = LogisticRegression(C=3.0, max_iter=2000, class_weight="balanced")
        model = OneVsRestClassifier(base) if multilabel else base
        model.fit(x_fit.tocsr(), target[fit])
        for mask_texts, store, where in ((train_texts[val], oof, val), (holdout_series, holdout, None)):
            x_pred = hstack([word.transform(mask_texts), char.transform(mask_texts)]).tocsr()
            probs = model.predict_proba(x_pred)
            if where is None:
                store += probs / len(np.unique(folds))
            else:
                store[where] = probs
    return oof, holdout


def cv_transformer(train_texts, target, holdout_texts, folds, model_runs, epochs, problem="ce",
                   balanced=True, max_len=128):
    """Latih tiap (model, seed) di semua fold, lalu rata-ratakan probabilitasnya."""
    target = np.asarray(target)
    train_texts = np.array(list(train_texts), dtype=object)
    holdout_texts = list(holdout_texts)
    n_out = target.shape[1] if problem == "bce" else (1 if problem == "mse" else int(target.max()) + 1)
    oof = np.zeros((len(target), n_out))
    holdout = np.zeros((len(holdout_texts), n_out))
    n_folds = len(np.unique(folds))
    for model_name, seed in model_runs:
        for fold in range(n_folds):
            fit, val = folds != fold, folds == fold
            weights = None
            if problem == "ce" and balanced:
                weights = len(target[fit]) / (n_out * np.bincount(target[fit], minlength=n_out))
            elif problem == "bce" and balanced:
                positives = np.maximum(target[fit].sum(0), 1)
                weights = np.clip((len(target[fit]) - positives) / positives, 1, 20)
            pred_val, pred_hold = train_predict(
                model_name, train_texts[fit].tolist(), target[fit],
                [train_texts[val].tolist(), holdout_texts], problem=problem,
                num_labels=n_out if problem != "bce" else None, class_weights=weights,
                epochs=epochs, seed=seed + fold, max_len=max_len,
            )
            oof[val] += pred_val.reshape(val.sum(), -1) / len(model_runs)
            holdout += pred_hold.reshape(len(holdout_texts), -1) / (n_folds * len(model_runs))
    return oof, holdout
