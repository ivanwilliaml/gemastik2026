d = "Anak makan nasi. Nasi itu enak. Anak suka nasi enak. Burung terbang tinggi."
assert lead_n(d, 1) == "Anak makan nasi." and "Burung" not in textrank(d, 2) and len(split_sents("a " * 60, window=30)) == 2
