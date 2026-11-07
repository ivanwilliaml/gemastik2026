# title: Wilayah: deteksi kabupaten/kota, provinsi, dan peta slide
# when: Teks menyebut lokasi kejadian. Cepat dan tanpa model. 514 kab/kota, 38 provinsi menurut Kepmendagri 2025.
# avoid: Membaca hasilnya sebagai lokasi kejadian tanpa audit: 'orang Garut' belum tentu kejadian di Garut. Geocoding online tidak bisa dipakai saat lomba.
# needs: domain/mbg_lexicon
# tags: lokasi wilayah kabupaten kota provinsi gazetteer peta geopandas entity resolution rapidfuzz
import re
from pathlib import Path
import ahocorasick
import numpy as np
import pandas as pd

WILAYAH = {
    "Aceh": "Aceh Barat|Aceh Barat Daya|Aceh Besar|Aceh Jaya|Aceh Selatan|Aceh Singkil|Aceh Tamiang|Aceh Tengah|Aceh Tenggara|Aceh Timur|Aceh Utara|Banda Aceh|Bener Meriah|Bireuen|Gayo Lues|Langsa|Lhokseumawe|Nagan Raya|Pidie|Pidie Jaya|Sabang|Simeulue|Subulussalam",
    "Bali": "Badung|Bangli|Buleleng|Denpasar|Gianyar|Jembrana|Karangasem|Klungkung|Tabanan",
    "Banten": "Cilegon|Lebak|Pandeglang|Serang|Tangerang|Tangerang Selatan",
    "Bengkulu": "Bengkulu|Bengkulu Selatan|Bengkulu Tengah|Bengkulu Utara|Kaur|Kepahiang|Lebong|Mukomuko|Rejang Lebong|Seluma",
    "Daerah Istimewa Yogyakarta": "Bantul|Gunungkidul|Kulon Progo|Sleman|Yogyakarta",
    "Daerah Khusus Ibukota Jakarta": "Jakarta Barat|Jakarta Pusat|Jakarta Selatan|Jakarta Timur|Jakarta Utara|Kepulauan Seribu",
    "Gorontalo": "Boalemo|Bone Bolango|Gorontalo|Gorontalo Utara|Pohuwato",
    "Jambi": "Batanghari|Bungo|Jambi|Kerinci|Merangin|Muaro Jambi|Sarolangun|Sungai Penuh|Tanjung Jabung Barat|Tanjung Jabung Timur|Tebo",
    "Jawa Barat": "Bandung|Bandung Barat|Banjar|Bekasi|Bogor|Ciamis|Cianjur|Cimahi|Cirebon|Depok|Garut|Indramayu|Karawang|Kuningan|Majalengka|Pangandaran|Purwakarta|Subang|Sukabumi|Sumedang|Tasikmalaya",
    "Jawa Tengah": "Banjarnegara|Banyumas|Batang|Blora|Boyolali|Brebes|Cilacap|Demak|Grobogan|Jepara|Karanganyar|Kebumen|Kendal|Klaten|Kudus|Magelang|Pati|Pekalongan|Pemalang|Purbalingga|Purworejo|Rembang|Salatiga|Semarang|Sragen|Sukoharjo|Surakarta|Tegal|Temanggung|Wonogiri|Wonosobo",
    "Jawa Timur": "Bangkalan|Banyuwangi|Batu|Blitar|Bojonegoro|Bondowoso|Gresik|Jember|Jombang|Kediri|Lamongan|Lumajang|Madiun|Magetan|Malang|Mojokerto|Nganjuk|Ngawi|Pacitan|Pamekasan|Pasuruan|Ponorogo|Probolinggo|Sampang|Sidoarjo|Situbondo|Sumenep|Surabaya|Trenggalek|Tuban|Tulungagung",
    "Kalimantan Barat": "Bengkayang|Kapuas Hulu|Kayong Utara|Ketapang|Kubu Raya|Landak|Melawi|Mempawah|Pontianak|Sambas|Sanggau|Sekadau|Singkawang|Sintang",
    "Kalimantan Selatan": "Balangan|Banjar|Banjarbaru|Banjarmasin|Barito Kuala|Hulu Sungai Selatan|Hulu Sungai Tengah|Hulu Sungai Utara|Kotabaru|Tabalong|Tanah Bumbu|Tanah Laut|Tapin",
    "Kalimantan Tengah": "Barito Selatan|Barito Timur|Barito Utara|Gunung Mas|Kapuas|Katingan|Kotawaringin Barat|Kotawaringin Timur|Lamandau|Murung Raya|Palangkaraya|Pulang Pisau|Seruyan|Sukamara",
    "Kalimantan Timur": "Balikpapan|Berau|Bontang|Kutai Barat|Kutai Kartanegara|Kutai Timur|Mahakam Ulu|Paser|Penajam Paser Utara|Samarinda",
    "Kalimantan Utara": "Bulungan|Malinau|Nunukan|Tana Tidung|Tarakan",
    "Kepulauan Bangka Belitung": "Bangka|Bangka Barat|Bangka Selatan|Bangka Tengah|Belitung|Belitung Timur|Pangkal Pinang",
    "Kepulauan Riau": "Batam|Bintan|Karimun|Kepulauan Anambas|Lingga|Natuna|Tanjung Pinang",
    "Lampung": "Bandar Lampung|Lampung Barat|Lampung Selatan|Lampung Tengah|Lampung Timur|Lampung Utara|Mesuji|Metro|Pesawaran|Pesisir Barat|Pringsewu|Tanggamus|Tulang Bawang|Tulang Bawang Barat|Way Kanan",
    "Maluku": "Ambon|Buru|Buru Selatan|Kepulauan Aru|Kepulauan Tanimbar|Maluku Barat Daya|Maluku Tengah|Maluku Tenggara|Seram Bagian Barat|Seram Bagian Timur|Tual",
    "Maluku Utara": "Halmahera Barat|Halmahera Selatan|Halmahera Tengah|Halmahera Timur|Halmahera Utara|Kepulauan Sula|Pulau Morotai|Pulau Taliabu|Ternate|Tidore Kepulauan",
    "Nusa Tenggara Barat": "Bima|Dompu|Lombok Barat|Lombok Tengah|Lombok Timur|Lombok Utara|Mataram|Sumbawa|Sumbawa Barat",
    "Nusa Tenggara Timur": "Alor|Belu|Ende|Flores Timur|Kab Timor Tengah Selatan|Kupang|Lembata|Malaka|Manggarai|Manggarai Barat|Manggarai Timur|Nagekeo|Ngada|Rote Ndao|Sabu Raijua|Sikka|Sumba Barat|Sumba Barat Daya|Sumba Tengah|Sumba Timur|Timor Tengah Utara",
    "Papua": "Biak Numfor|Jayapura|Keerom|Kepulauan Yapen|Mamberamo Raya|Sarmi|Supiori|Waropen",
    "Papua Barat": "Fak Fak|Kaimana|Manokwari|Manokwari Selatan|Pegunungan Arfak|Teluk Bintuni|Teluk Wondama",
    "Papua Barat Daya": "Maybrat|Raja Ampat|Sorong|Sorong Selatan|Tambrauw",
    "Papua Pegunungan": "Jayawijaya|Kab Pegunungan Bintang|Lanny Jaya|Mamberamo Tengah|Nduga|Tolikara|Yahukimo|Yalimo",
    "Papua Selatan": "Asmat|Boven Digoel|Mappi|Merauke",
    "Papua Tengah": "Deiyai|Dogiyai|Intan Jaya|Mimika|Nabire|Paniai|Puncak|Puncak Jaya",
    "Riau": "Bengkalis|Dumai|Indragiri Hilir|Indragiri Hulu|Kampar|Kepulauan Meranti|Kuantan Singingi|Pekanbaru|Pelalawan|Rokan Hilir|Rokan Hulu|Siak",
    "Sulawesi Barat": "Majene|Mamasa|Mamuju|Mamuju Tengah|Pasangkayu|Polewali Mandar",
    "Sulawesi Selatan": "Bantaeng|Barru|Bone|Bulukumba|Enrekang|Gowa|Jeneponto|Kepulauan Selayar|Luwu|Luwu Timur|Luwu Utara|Makassar|Maros|Palopo|Pangkajene dan Kepulauan|Parepare|Pinrang|Sidenreng Rappang|Sinjai|Soppeng|Takalar|Tana Toraja|Toraja Utara|Wajo",
    "Sulawesi Tengah": "Banggai|Banggai Kepulauan|Banggai Laut|Buol|Donggala|Morowali|Morowali Utara|Palu|Parigi Moutong|Poso|Sigi|Tojo Una Una|Toli-Toli",
    "Sulawesi Tenggara": "Bau Bau|Bombana|Buton|Buton Selatan|Buton Tengah|Buton Utara|Kendari|Kolaka|Kolaka Timur|Kolaka Utara|Konawe|Konawe Kepulauan|Konawe Selatan|Konawe Utara|Muna|Muna Barat|Wakatobi",
    "Sulawesi Utara": "Bitung|Bolaang Mongondow|Bolaang Mongondow Selatan|Bolaang Mongondow Timur|Bolaang Mongondow Utara|Kep. Siau Tagulandang Biaro|Kepulauan Sangihe|Kepulauan Talaud|Kotamobagu|Manado|Minahasa|Minahasa Selatan|Minahasa Tenggara|Minahasa Utara|Tomohon",
    "Sumatera Barat": "Agam|Bukittinggi|Dharmasraya|Kepulauan Mentawai|Lima Puluh Kota|Padang|Padang Panjang|Padang Pariaman|Pariaman|Pasaman|Pasaman Barat|Payakumbuh|Pesisir Selatan|Sawahlunto|Sijunjung|Solok|Solok Selatan|Tanah Datar",
    "Sumatera Selatan": "Banyuasin|Empat Lawang|Lahat|Lubuk Linggau|Muara Enim|Musi Banyuasin|Musi Rawas|Musi Rawas Utara|Ogan Ilir|Ogan Komering Ilir|Ogan Komering Ulu|Ogan Komering Ulu Selatan|Ogan Komering Ulu Timur|Pagar Alam|Palembang|Penukal Abab Lematang Ilir|Prabumulih",
    "Sumatera Utara": "Asahan|Batu Bara|Binjai|Dairi|Deli Serdang|Gunungsitoli|Humbang Hasundutan|Karo|Labuhanbatu|Labuhanbatu Selatan|Labuhanbatu Utara|Langkat|Mandailing Natal|Medan|Nias|Nias Barat|Nias Selatan|Nias Utara|Padang Lawas|Padang Lawas Utara|Padangsidimpuan|Pakpak Bharat|Pematangsiantar|Samosir|Serdang Bedagai|Sibolga|Simalungun|Tanjungbalai|Tapanuli Selatan|Tapanuli Tengah|Tapanuli Utara|Tebing Tinggi|Toba"
}
# Provinsi tidak punya baris di atas sebagai nama; alias umum di media sosial.
PROV_ALIAS = {"Aceh": "aceh", "Bali": "bali", "Banten": "banten", "Daerah Istimewa Yogyakarta": "diy|jogja|jogjakarta|yogya|yogyakarta", "Daerah Khusus Ibukota Jakarta": "dki|jakarta|jkt",
              "Jawa Barat": "jawa barat|jabar", "Jawa Tengah": "jawa tengah|jateng", "Jawa Timur": "jawa timur|jatim", "Kalimantan Barat": "kalimantan barat|kalbar", "Kalimantan Selatan": "kalimantan selatan|kalsel",
              "Kalimantan Tengah": "kalimantan tengah|kalteng", "Kalimantan Timur": "kalimantan timur|kaltim", "Kalimantan Utara": "kalimantan utara|kaltara", "Kepulauan Bangka Belitung": "bangka belitung|babel",
              "Kepulauan Riau": "kepulauan riau|kepri", "Lampung": "lampung", "Maluku": "maluku", "Maluku Utara": "maluku utara|malut", "Nusa Tenggara Barat": "nusa tenggara barat|ntb", "Nusa Tenggara Timur": "nusa tenggara timur|ntt",
              "Papua": "papua", "Papua Barat": "papua barat", "Papua Barat Daya": "papua barat daya", "Papua Pegunungan": "papua pegunungan", "Papua Selatan": "papua selatan", "Papua Tengah": "papua tengah", "Riau": "riau",
              "Sulawesi Barat": "sulawesi barat|sulbar", "Sulawesi Selatan": "sulawesi selatan|sulsel", "Sulawesi Tengah": "sulawesi tengah|sulteng", "Sulawesi Tenggara": "sulawesi tenggara|sultra", "Sulawesi Utara": "sulawesi utara|sulut",
              "Sumatera Barat": "sumatera barat|sumbar", "Sumatera Selatan": "sumatera selatan|sumsel", "Sumatera Utara": "sumatera utara|sumut", "Gorontalo": "gorontalo", "Bengkulu": "bengkulu", "Jambi": "jambi"}
KAB_ALIAS = {"bdg": "Bandung", "kbb": "Bandung Barat", "smg": "Semarang", "solo": "Surakarta", "mks": "Makassar", "plg": "Palembang", "tangsel": "Tangerang Selatan", "jaksel": "Jakarta Selatan", "jakbar": "Jakarta Barat", "jaktim": "Jakarta Timur", "jakut": "Jakarta Utara", "jakpus": "Jakarta Pusat"}
# Nama yang juga kata umum: hanya dihitung bila didahului konteks lokasi.
AMBIGUOUS = set("""padang batu malang serang medan buru sabang merauke bone metro bima ketapang karo kudus puncak batang bangka banjar lebak tegal kediri sampang luwu asmat agam pati kaur alor muna solok landak
belu ende rote siak bungo sigi poso dairi balangan tojo gowa deli maros wajo pinrang soppeng barru bantaeng takalar selayar sinjai sidrap tana lingga bintan natuna anambas tolitoli buol parigi morowali konawe kolaka
solo ngawi nganjuk sumba flores sikka ngada lembata kupang timor seram tual aru dobo sula tidore ternate halmahera morotai biak nabire mimika paniai dogiyai deiyai yalimo tolikara lanny nduga puncak yahukimo boven mappi
ambon sarmi keerom waropen supiori kaimana fakfak bintuni wondama maybrat tambrauw sorong raja ampat""".split())
_CONTEXT = re.compile(r"(?:\b(?:di|ke|dari|kab|kabupaten|kota|kec|kecamatan|daerah|wilayah|warga|sppg|sd|smp|sma|smk|sdn|smpn|sman|smkn|mts|man|mi|pemkab|pemkot|bupati|walikota|wali kota)\s+)$")
_KAB = {n.lower(): (n, p) for p, s in WILAYAH.items() for n in s.split("|")}
_KAB.update({a: _KAB[n.lower()] for a, n in KAB_ALIAS.items() if n.lower() in _KAB})
_PROV = {a: p for p, s in PROV_ALIAS.items() for a in s.split("|")}
_AUTO = ahocorasick.Automaton()
for _k in set(_KAB) | set(_PROV):
    _AUTO.add_word(_k, _k)
_AUTO.make_automaton()

def find_locations(text):
    """[(kabupaten/kota atau None, provinsi)]. Pencocokan terpanjang: 'bandung barat' mengalahkan 'bandung'."""
    t = str(text).lower()
    hits = []
    for end, k in _AUTO.iter(t):
        s = end - len(k) + 1
        if (s == 0 or not t[s - 1].isalnum()) and (end + 1 == len(t) or not t[end + 1].isalnum()):
            hits.append((s, end, k))
    hits.sort(key=lambda h: (h[0], h[0] - h[1]))
    out, last = [], -1
    for s, e, k in sorted(hits, key=lambda h: h[0] - h[1]):
        if any(not (e < a or s > b) for a, b in [(x[0], x[1]) for x in out]):
            continue
        if k in AMBIGUOUS and not _CONTEXT.search(t[max(0, s - 22):s]):
            continue
        out.append((s, e, k))
    res = []
    for s, e, k in sorted(out):
        if k in _KAB:
            res.append(_KAB[k])
        else:
            res.append((None, _PROV[k]))
    return res

def tag_locations(texts):
    """DataFrame: kab_kota, provinsi (lokasi pertama yang paling spesifik), n_lokasi. Teks identik sekali."""
    texts = ["" if t is None or t != t else str(t) for t in texts]
    cache = {}
    for t in set(texts):
        r = find_locations(t)
        kab = [x for x in r if x[0]]
        first = kab[0] if kab else r[0] if r else (None, None)
        cache[t] = (first[0], first[1], len(r))
    return pd.DataFrame([cache[t] for t in texts], columns=["kab_kota", "provinsi", "n_lokasi"])

def resolve_location(name, cutoff=85):
    """Nama dari NER atau kolom metadata ke nama baku, tahan typo. Kembalikan (kab_kota, provinsi, skor)."""
    from rapidfuzz import fuzz, process
    n = re.sub(r"^(kabupaten|kab\.?|kota|provinsi|prov\.?)\s+", "", str(name).lower().strip())
    if n in _KAB:
        return (*_KAB[n], 100.0)
    if n in _PROV:
        return (None, _PROV[n], 100.0)
    m = process.extractOne(n, list(_KAB) + list(_PROV), scorer=fuzz.WRatio, score_cutoff=cutoff)
    if m is None:
        return (None, None, 0.0)
    return (*_KAB[m[0]], float(m[1])) if m[0] in _KAB else (None, _PROV[m[0]], float(m[1]))

# Nama pada geoBoundaries ADM1 (2017, 34 provinsi): empat provinsi Papua baru digabung ke induknya.
_ADM1 = {"Bali": "Bali", "Nusa Tenggara Barat": "West Nusa Tenggara", "Banten": "Banten", "Jawa Tengah": "Central Java", "Jawa Barat": "West Java", "Kalimantan Tengah": "Central Kalimantan", "Kalimantan Selatan": "South Kalimantan", "Kalimantan Barat": "West Kalimantan",
         "Sulawesi Tengah": "Central Sulawesi", "Gorontalo": "Gorontalo", "Sulawesi Utara": "North Sulawesi", "Sulawesi Selatan": "South Sulawesi", "Sulawesi Tenggara": "Southeast Sulawesi", "Sulawesi Barat": "West Sulawesi", "Aceh": "Aceh", "Bengkulu": "Bengkulu", "Jambi": "Jambi",
         "Lampung": "Lampung", "Riau": "Riau", "Sumatera Barat": "West Sumatra", "Sumatera Selatan": "South Sumatra", "Sumatera Utara": "North Sumatra", "Nusa Tenggara Timur": "East Nusa Tenggara", "Maluku": "Maluku", "Maluku Utara": "North Maluku", "Jawa Timur": "East Java",
         "Kepulauan Bangka Belitung": "Bangka-Belitung Islands", "Kepulauan Riau": "Riau Islands", "Papua": "Papua", "Papua Selatan": "Papua", "Papua Tengah": "Papua", "Papua Pegunungan": "Papua", "Papua Barat": "West Papua", "Papua Barat Daya": "West Papua",
         "Kalimantan Timur": "East Kalimantan", "Kalimantan Utara": "North Kalimantan", "Daerah Istimewa Yogyakarta": "Special Region of Yogyakarta", "Daerah Khusus Ibukota Jakarta": "Jakarta Special Capital Region"}

def plot_province_map(counts, geojson=None, path="peta.png", title="Sebaran per provinsi"):
    """counts: Series provinsi -> nilai. Tanpa file geojson atau geopandas, otomatis jadi grafik batang."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = pd.Series(counts).astype(float)
    try:
        if not geojson or not Path(geojson).exists():
            raise FileNotFoundError(geojson)
        import geopandas as gpd
        g = gpd.read_file(geojson)
        v = s.groupby(s.index.map(lambda p: _ADM1.get(p, p))).sum()
        g["nilai"] = g.shapeName.map(v).fillna(0)
        ax = g.plot(column="nilai", cmap="YlOrRd", legend=True, figsize=(11, 4.6), edgecolor="#667085", linewidth=.3)
        ax.set_axis_off()
    except Exception as e:
        print("peta dilewati, pakai grafik batang:", type(e).__name__)
        t = s.sort_values().tail(20)
        plt.figure(figsize=(8, max(3, .35 * len(t))))
        plt.barh(t.index.astype(str), t.values, color="#b42318")
    plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    return path
