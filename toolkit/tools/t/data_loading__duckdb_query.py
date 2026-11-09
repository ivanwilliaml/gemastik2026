import tempfile, os, pandas as pd
f = os.path.join(tempfile.mkdtemp(), "a.parquet")
pd.DataFrame({"text": ["a", "a", None, "bcd"], "label": [1, 1, 0, 0]}).to_parquet(f)
assert sql(f"SELECT label, COUNT(*) n FROM '{f}' GROUP BY 1 ORDER BY 1").n.tolist() == [2, 2]
r = profile_file(f)
assert r.baris[0] == 4 and r.unik[0] == 2 and r.kosong[0] == 1
