# title: eda_report(): satu fungsi EDA lengkap
# when: Selalu. Profil, kosong, duplikat, panjang, target, kosakata, OOV, kata khas, overlap, adversarial, dan saran otomatis.
# avoid: Melewatkan membaca 50 teks mentah: fungsi ini mencetak contohnya.
# tags: eda profil duplikat panjang target oov adversarial saran overlap imbalance
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

def _nk(s):
    return s.fillna("").astype(str).str.lower().str.replace(r"\W+", " ", regex=True).str.strip()

def eda_report(train, test=None, text="text", target=None, time=None, n_show=8, sample=20000, seed=42):
    """Kembalikan dict ringkasan berisi saran dan rekomendasi max_words. target: nama kolom atau daftar kolom."""
    tips, out = [], {}
    print("=== PROFIL ===\ntrain", train.shape, "| test", None if test is None else test.shape)
    print(pd.DataFrame({"dtype": train.dtypes.astype(str), "null": train.isna().mean().round(4), "nunique": train.nunique()}).to_string())
    if test is not None:
        print("kolom hanya di train:", sorted(set(train.columns) - set(test.columns)))
    tx = train[text].fillna("").astype(str)
    print("\n=== CONTOH ===")
    for t in train[text].sample(min(n_show, len(train)), random_state=seed):
        print(" -", str(t)[:180])
    key = _nk(train[text])
    out["kosong"], out["duplikat"] = round(float(key.eq("").mean()), 4), round(float(key.duplicated().mean()), 4)
    print("\n=== KUALITAS ===\nkosong", out["kosong"], "| duplikat ternormalisasi", out["duplikat"], "| punya URL", round(tx.str.contains(r"https?://").mean(), 3),
          "| mention", round(tx.str.contains(r"@\w").mean(), 3), "| tanda baca akhir kalimat", round(tx.str.contains(r"[.!?]").mean(), 3))
    print("teks paling sering:", key.value_counts().head(3).to_dict())
    if out["duplikat"] > .01:
        tips.append("Duplikat > 1%: pakai group fold dan laporkan skor random serta grup.")
    w = tx.str.split().str.len()
    q = w.quantile([.5, .9, .95, .99]).round(0)
    out["max_words"] = int(q[.95])
    print("\n=== PANJANG (kata) ===\n", q.to_dict(), "| char p50", int(tx.str.len().median()))
    if q[.5] < 5:
        tips.append("Teks sangat pendek: utamakan TF-IDF karakter.")
    if q[.95] > 256:
        tips.append("Teks panjang: pertimbangkan head+tail dan chunk.")
    if target is not None:
        print("\n=== TARGET ===")
        cols = target if isinstance(target, (list, tuple)) else [target]
        if len(cols) > 1:
            Y = train[cols].to_numpy()
            print(pd.DataFrame({"pct": train[cols].mean().round(4)}).T.to_string(), "\nlabel per baris", round(float(Y.sum(1).mean()), 3))
        else:
            s = train[cols[0]]
            if pd.api.types.is_numeric_dtype(s) and s.nunique() > 20:
                print(s.describe().round(3).to_dict(), "skew", round(s.skew(), 3))
                if s.skew() > 1 and s.min() >= 0:
                    tips.append("Target berekor panjang: latih pada log1p.")
            else:
                vc = s.value_counts(dropna=False)
                print((vc / len(s)).round(4).to_dict())
                out["rasio_imbalance"] = round(float(vc.max() / max(vc.min(), 1)), 1)
                if out["rasio_imbalance"] > 20:
                    tips.append("Imbalance parah: metrik macro-F1/PR-AUC, decision layer dari OOF.")
                if s.nunique() <= 50 and s.notna().all():
                    v = TfidfVectorizer(max_features=30000, min_df=3, sublinear_tf=True)
                    X = v.fit_transform(tx.iloc[:sample])
                    m = LogisticRegression(C=4, max_iter=1000).fit(X, s.iloc[:sample].astype(str))
                    names, coef = v.get_feature_names_out(), m.coef_ if m.coef_.shape[0] > 1 else np.vstack([-m.coef_[0], m.coef_[0]])
                    print("kata khas (cek artefak):")
                    for j, c in enumerate(m.classes_):
                        print(f"  {c}:", [names[i] for i in coef[j].argsort()[::-1][:10]])
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b", binary=True)
    X = cv.fit_transform(tx.sample(min(sample, len(tx)), random_state=seed))
    df_ = np.asarray(X.sum(0)).ravel()
    print("\n=== KOSAKATA ===\nvocab", X.shape[1], "| kata muncul sekali", round(float((df_ == 1).mean()), 3))
    if test is not None and text in test:
        te = test[text].fillna("").astype(str)
        vt = CountVectorizer(token_pattern=r"(?u)\b\w+\b").fit(te.sample(min(sample, len(te)), random_state=seed))
        cnt = np.asarray(vt.transform(te.sample(min(sample, len(te)), random_state=seed)).sum(0)).ravel()
        oov = np.array([x not in cv.vocabulary_ for x in vt.get_feature_names_out()])
        out["oov"] = round(float(cnt[oov].sum() / max(cnt.sum(), 1)), 3)
        tk = set(key)
        out["overlap"] = round(float(_nk(te).isin(tk).mean()), 4)
        print("OOV token test", out["oov"], "| teks test ada di train", out["overlap"])
        if out["oov"] > .2:
            tips.append("OOV tinggi: TF-IDF karakter atau FastText.")
        n = min(5000, len(tx), len(te))
        t2 = list(tx.sample(n, random_state=seed)) + list(te.sample(n, random_state=seed))
        lab = np.r_[np.zeros(n), np.ones(n)]
        Xa = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000).fit_transform(t2)
        out["adversarial_auc"] = round(float(roc_auc_score(lab, cross_val_predict(LogisticRegression(max_iter=500), Xa, lab, cv=3, method="predict_proba")[:, 1])), 4)
        print("AUC adversarial", out["adversarial_auc"], "| panjang p50 test", int(te.str.split().str.len().median()))
        if out["adversarial_auc"] > .7:
            tips.append("Train dan test berbeda: cek kata pembeda, pertimbangkan split waktu/sumber.")
    if time and time in train:
        t = pd.to_datetime(train[time], errors="coerce")
        print("\n=== WAKTU ===\nrentang", t.min(), "s.d.", t.max(), "| gagal parse", round(float(t.isna().mean()), 3))
        if test is not None and time in test:
            tt = pd.to_datetime(test[time], errors="coerce")
            print("test", tt.min(), "s.d.", tt.max())
            if tt.min() > t.median():
                tips.append("Test lebih baru dari sebagian besar train: pakai time split.")
    out["saran"] = tips
    print("\n=== SARAN ===", *(tips or ["Tidak ada peringatan khusus."]), sep="\n - ")
    return out
