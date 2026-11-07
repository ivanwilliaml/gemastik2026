m = meta_features(["Halo!! http://a.id @x #y", ""]); assert m.loc[0, "excl"] == 2 and m.loc[0, "url"] == 1 and m.shape[0] == 2
t = time_features(["2025-09-17 10:00", "2025-09-17 11:00"], {"klb": "2025-09-24"}); assert t.vol_hari[0] == 2 and t.hari_ke_klb[0] < 0
assert lexicon_score(["enak basi"], ["enak"], ["basi"]).tolist() == [[1, 1]]
