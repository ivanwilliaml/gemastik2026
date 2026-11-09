d = pd.DataFrame({"text": ["halo", None, "nan", "-"], "y": [1, 2, 3, 4]})
assert profile(d, "text").loc["text", "null"] == .25
assert clean_text_col(d.text).tolist() == ["halo", "", "", ""]
