assert tokenize("Anak-anak MAKAN 2 kali!") == ["anak-anak", "makan", "2", "kali"]
assert stem_corpus(["memakan nasi yang enak"], stop=["yang"]) == ["makan nasi enak"]
