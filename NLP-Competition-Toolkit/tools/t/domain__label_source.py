a, k, m = _agreement([0, 0, 1, 1], ["x", "x", "y", "y"])
assert a == 1.0 and m == {"0": "x", "1": "y"}
import tempfile, pathlib
d = pathlib.Path(tempfile.mkdtemp())
(d / "positive.tsv").write_text("word\tweight\nbagus\t3\n", encoding="utf-8")
(d / "negative.tsv").write_text("word\tweight\nburuk\t-3\n", encoding="utf-8")
r = label_fingerprint(["bagus sekali", "buruk sekali", "bagus", "buruk"], ["p", "n", "p", "n"], None, names=(), inset_dir=d, n=10)
assert r.setuju.max() == 1.0
if "sentiment" in MODELS:
    import pathlib as _p
    r2 = label_fingerprint(texts[:40], y[:40], _p.Path(MODELS["sentiment"]).parent, names=(_p.Path(MODELS["sentiment"]).name,), n=40)
    assert len(r2) == 2
