f = text_flags(["😀😀", "[Sticker]", "kalimat biasa saja", "http://a.id", ""]); assert f.non_huruf[0] and f.sticker[1] and f.url_saja[3] and f.kosong[4]
assert domain_noise(["slot gacor hari ini", "slotnya kosong"], ["slot gacor"]).tolist() == [True, False]
