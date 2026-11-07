st = {}
assert normalize_slang("yg gitu bgt", stats=st) == "yang gitu banget" and st == {"yg": 1, "bgt": 1}
import pandas as _pd
_pd.DataFrame({"slang": ["wkwk"], "formal": ["haha"]}).to_csv("lex.csv", index=False)
assert load_slang("lex.csv")["wkwk"] == "haha"
