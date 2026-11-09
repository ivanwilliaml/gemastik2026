assert clean_indobertweet("@A cek http://x.id 😀") .startswith("@USER cek HTTPURL")
import pandas as _pd
assert join_columns(_pd.DataFrame({"a": ["x"], "b": ["y"]}), ["a", "b"]) == ["x [SEP] y"]
assert clean_transformer("a\x01  b") == "a b"
