d = pd.DataFrame({"text": texts + texts[:30], "label": list(y) + list(y[:30]), "time": pd.date_range("2025-01-01", periods=len(texts) + 30).astype(str)})
o = eda_report(d, pd.DataFrame({"text": texts_test}), target="label", time="time"); assert 0 < o["duplikat"] and "oov" in o and "adversarial_auc" in o and isinstance(o["saran"], list)
eda_report(pd.DataFrame({"text": texts, "y": y.astype(float) * 1.5 + np.arange(len(y))}), target="y")
eda_report(pd.DataFrame({"text": ml_texts, "a": Y[:, 0], "b": Y[:, 1]}), target=["a", "b"])
