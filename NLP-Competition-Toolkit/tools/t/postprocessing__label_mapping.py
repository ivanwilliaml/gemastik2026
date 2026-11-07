yi, c = encode_labels(["b", "a", "b"]); assert c[yi].tolist() == ["b", "a", "b"]
pd.DataFrame({"id": [1, 2], "label": ["x", "x"]}).to_csv("sample.csv", index=False)
assert write_submission("sample.csv", ["a", "b"], allowed=["a", "b"]).label.tolist() == ["a", "b"]
try:
    write_submission("sample.csv", ["a"]); raise SystemExit("seharusnya gagal")
except AssertionError:
    pass
