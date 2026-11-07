d = pd.DataFrame({"id": range(5), "text": ["kalimat yang cukup panjang sekali"] * 5, "y": [0, 1, 0, 1, 0]})
o = detect_columns(d, test=d.drop(columns="y")).set_index("col")
assert o.loc["text", "tebakan"] == "teks" and o.loc["y", "tebakan"] == "target?"
