H = tag_terms(["keracunan di sekolah, menunya basi", "anggaran triliun", "halo"])
assert H.keamanan_pangan.tolist() == [True, False, False] and H.kualitas_menu.tolist()[0] and H.anggaran.tolist()[1] and not H.iloc[2].any()
assert not tag_terms(["kenalan baru"]).any(axis=None), "kata di dalam kata tidak boleh cocok"
assert term_hits(["menu basi"]).query("istilah == 'basi'").hit.iloc[0] == 1
assert aspect_sentiment(H, ["negatif", "netral", "positif"]).loc["keamanan_pangan", "negatif"] == 1
assert audit_aspects(["menu basi", "menu enak"], tag_terms(["menu basi", "menu enak"]), 1)["kualitas_menu"]
