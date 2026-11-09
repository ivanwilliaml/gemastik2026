r = qa_report(["Jakarta, Indonesia", "x"], [["jakarta indonesia"], "y"]); assert r["em"] == .5 and r["f1"] == .5
assert f1_token("presiden jokowi", "jokowi") == 2 / 3
