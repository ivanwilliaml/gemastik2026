assert find_locations("keracunan di bandung barat") == [("Bandung Barat", "Jawa Barat")]
assert find_locations("nasib malang sekali") == [] and find_locations("sekolah di malang")[0][0] == "Malang"
assert find_locations("lempar batu") == [] and find_locations("jaman sby dulu") == []
assert find_locations("kabupaten garut jabar")[0][0] == "Garut"
L = tag_locations(["di garut", "halo", None])
assert L.provinsi.iloc[0] == "Jawa Barat" and L.provinsi.isna().sum() == 2
assert resolve_location("kab bandung brat")[0] == "Bandung Barat" and resolve_location("xyzabc")[1] is None
import tempfile, os
assert os.path.exists(plot_province_map({"Jawa Barat": 3, "Papua": 1}, None, os.path.join(tempfile.mkdtemp(), "m.png")))
assert len(WILAYAH) == 38 and sum(len(v.split("|")) for v in WILAYAH.values()) >= 480
