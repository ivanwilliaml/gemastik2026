d = pd.DataFrame({"text": ["Halo!", "halo", "nan", None, "dunia baru sekali ini", "dunia baru sekali ini ya"], "y": [1, 0, 1, 1, 1, 1]})
c = clean_dataset(d, target="y"); assert len(c) == 3 and c.y[0] in (0, 1)
c2 = clean_dataset(d, target="y", conflict="drop"); assert "halo" not in c2.text.str.lower().tolist() and len(c2) == 2
c3 = clean_dataset(d, near_dup=True, dedup=False, near_threshold=.5); assert c3._group.iloc[-1] == c3._group.iloc[-2]
assert clean_test(d).text.isna().sum() == 0
