assert parse_money("Rp10.000 dan Rp174 triliun, 15rb, tahun 2025, 12 anak") == [10000.0, 1.74e14, 15000.0]
assert parse_money("Rp 270,2 T") == [2.702e14]
assert parse_quantity("3 siswa, ratusan murid, 500 porsi") == [("3", "siswa"), ("ratusan", "murid"), ("500", "porsi")]
assert parse_dates("kemarin keracunan", "2025-09-25") == [("kemarin", "2025-09-24")]
assert parse_dates("6 November 2025", None)[0][1] == "2025-11-06"
assert parse_dates("24/09/2025", None)[0][1] == "2025-09-24"
r = extract_records(["kemarin 3 siswa keracunan ayam, mual. Rp10.000 di SPPG BGN", "halo", None], ["2025-09-25", None, None])
assert r.gejala[0] == ["mual"] and r.insiden[0] == ["keracunan"] and r.makanan[0] == ["ayam"] and r.uang[0] == [10000.0]
assert r.tanggal[0] == ["2025-09-24"] and set(r.entitas[0]) == {"BGN", "SPPG"} and r.gejala[1] == []
if "ner" in MODELS:
    assert len(ner_entities(["Bupati Garut datang ke Bandung"], MODELS["ner"])[0]) >= 1
