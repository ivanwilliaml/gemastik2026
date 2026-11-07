assert length_table(texts).loc[.5, "word"] > 0
assert length_table(texts, by=y).shape[0] == 3
class _T:
    def __call__(self, t, truncation=False): return {"input_ids": [x.split() for x in t]}
assert 1 <= suggest_max_len(_T(), texts) <= 512
