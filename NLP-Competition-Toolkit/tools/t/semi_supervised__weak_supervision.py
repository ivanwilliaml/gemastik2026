H = keyword_lf(["makanan basi sekali", "anggaran triliun", "gitu aja", "nasi basi anggaran"], {"pangan": ["basi", "racun"], "dana": ["anggaran"], "git": ["git"]})
assert H.git.sum() == 0 and H.pangan[0] == 1 and vote_lf(H).tolist() == [0, 1, -1, 0]
assert lf_accuracy([0, -1], [0, 1]) == dict(coverage=.5, precision=1.0) and len(audit_lf(H, ["a", "b", "c", "d"], 1)["pangan"]) == 1
