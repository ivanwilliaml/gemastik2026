# title: Kamus MBG: entitas, aspek, kejadian
# when: Tema MBG. Multi-label, satu baris boleh banyak aspek. Untuk fitur, label lemah, dan bahan slide.
# avoid: Menjadikan kamus sebagai kebenaran. Audit presisi 30 sampel dan buang istilah nol hit.
# tags: mbg kamus leksikon aspek entitas sppg bgn keracunan aho-corasick multilabel ontologi
import re
import ahocorasick
import numpy as np
import pandas as pd

MBG_ENTITIES = {
    "MBG": ["mbg", "makan bergizi gratis", "makan bergizi", "makan gratis", "makan siang gratis", "program makan gratis", "embege"],
    "BGN": ["bgn", "badan gizi nasional", "badan gizi"],
    "SPPG": ["sppg", "satuan pelayanan pemenuhan gizi", "dapur mbg", "dapur sppg"],
    "Kemenkes": ["kemenkes", "kementerian kesehatan", "menkes"], "Kejagung": ["kejagung", "kejaksaan agung"], "KPK": ["kpk"],
    "DPR": ["dpr", "banggar"], "JPPI": ["jppi"], "KPAI": ["kpai"], "Ombudsman": ["ombudsman"], "ICW": ["icw"], "CISDI": ["cisdi"],
    "Prabowo": ["prabowo", "wowo"], "Gibran": ["gibran"], "Sudaryono": ["sudaryono"], "Dadan Hindayana": ["dadan", "dadan hindayana"], "Nanik Deyang": ["nanik"], "Purbaya": ["purbaya"],
}
MBG_ASPECTS = {
    "keamanan_pangan": ["keracunan", "keracunannya", "racun", "beracun", "basi", "busuk", "berjamur", "jamuran", "belatung", "ulat", "ulet", "klb", "e coli", "salmonella", "kadaluarsa", "kedaluwarsa", "tidak layak", "gak layak", "ga layak", "mual", "muntah", "diare", "pusing", "sakit perut", "dilarikan", "puskesmas", "rumah sakit", "korban", "bau", "mentah"],
    "kualitas_menu": ["menu", "ompreng", "ayam", "nasi", "sayur", "susu", "burger", "spaghetti", "spageti", "roti", "telur", "telor", "tempe", "ikan", "kacang", "buah", "pisang", "jeruk", "lauk", "enak", "hambar", "porsi", "ultra processed", "upf", "sosis", "nugget"],
    "gizi_kesehatan": ["gizi", "bergizi", "ahli gizi", "nutrisi", "stunting", "protein", "kalori", "akg", "obesitas", "sehat", "kesehatan", "dokter"],
    "anggaran": ["anggaran", "apbn", "rapbn", "pajak", "triliun", "pagu", "efisiensi", "pemborosan", "refocusing", "dana", "buang uang", "uang rakyat", "biaya", "per porsi", "15 ribu", "10 ribu", "10rb", "15rb"],
    "korupsi_tata_kelola": ["korupsi", "koruptor", "korup", "nepotisme", "kolusi", "markup", "mark up", "tersangka", "kejagung", "jual beli titik", "yayasan", "calo", "transparan", "transparansi", "proyek", "bancakan", "berjamaah", "kpk", "tata kelola", "pungli"],
    "distribusi_dapur": ["sppg", "dapur", "distribusi", "pengiriman", "terlambat", "telat", "kapasitas", "suspend", "sanitasi", "slhs", "sop", "relawan", "mitra", "juknis", "ditutup", "penerima", "belum dapat", "tidak merata", "gak merata", "merata"],
    "pendidikan_guru": ["guru", "honorer", "murid", "siswa", "sekolah", "kantin", "santri", "pendidikan", "gaji guru", "biaya sekolah", "sekolah gratis"],
    "ekonomi_umkm": ["umkm", "lapangan kerja", "pekerja", "pedagang", "jualan", "catering", "katering", "petani", "peternak", "nelayan", "susu lokal"],
    "politik": ["prabowo", "wowo", "gibran", "dpr", "tni", "polri", "janji kampanye", "militer", "purbaya", "zulhas", "sudaryono", "nanik", "dadan", "presiden", "pemerintah", "menteri"],
    "media_hoaks": ["hoax", "hoaks", "buzzer", "influencer", "narasi", "digoreng", "komdigi", "klarifikasi", "fitnah"],
    "penolakan": ["stop", "setop", "hentikan", "bubar", "bubarkan", "batalkan", "hapus", "ganti uang", "tolak", "menolak"],
    "minta_perbaikan": ["evaluasi", "perbaiki", "perbaikan", "benahi", "awasi", "diawasi", "pengawasan", "tolong", "seharusnya", "harusnya"],
    "dukungan": ["bagus", "setuju", "mendukung", "dukung", "program bagus", "bermanfaat", "terima kasih", "makasih", "alhamdulillah", "lanjutkan", "semoga"],
}
MBG_EVENTS = {"2025-01-06": "Peluncuran MBG", "2025-09-24": "KLB Bandung Barat", "2026-06-02": "Pimpinan BGN dicopot", "2026-07-22": "Kepala BGN baru", "2026-10-29": "Laporan 100 hari"}
MBG_FOOD = sorted({w for w in MBG_ASPECTS["kualitas_menu"] if w not in ("menu", "ompreng", "enak", "hambar", "porsi", "ultra processed", "upf")})
MBG_SYMPTOM = ["mual", "muntah", "diare", "pusing", "sakit perut", "demam", "lemas"]
MBG_INCIDENT = ["keracunan", "basi", "busuk", "berjamur", "belatung", "ulat", "klb", "kadaluarsa", "kedaluwarsa"]

_SUFFIX = re.compile(r"(?:nya|ny|ku|mu|lah|kah)?(?![0-9a-z])")

def build_matcher(lex):
    """lex: {kelas: [istilah]}. Kembalikan automaton, dipakai find_terms."""
    a = ahocorasick.Automaton()
    for key, terms in lex.items():
        for t in terms:
            t = t.lower()
            old = a.get(t) if t in a else (t, [])
            a.add_word(t, (t, old[1] + [key]))
    a.make_automaton()
    return a

def find_terms(text, matcher):
    """Istilah utuh dalam teks. Akhiran -nya, -ku, -mu ikut cocok: 'menunya' cocok 'menu'."""
    t = str(text).lower()
    out = []
    for end, (term, keys) in matcher.iter(t):
        start = end - len(term) + 1
        if (start == 0 or not t[start - 1].isalnum()) and _SUFFIX.match(t, end + 1):
            out.append((term, keys))
    return out

def tag_terms(texts, lex=None):
    """DataFrame bool: baris x kelas. Teks identik hanya diproses sekali."""
    lex = lex or MBG_ASPECTS
    keys, idx = list(lex), {k: i for i, k in enumerate(lex)}
    m = build_matcher(lex)
    texts = ["" if t is None or t != t else str(t) for t in texts]
    cache = {}
    for t in set(texts):
        row = np.zeros(len(keys), dtype=bool)
        for _, ks in find_terms(t, m):
            row[[idx[k] for k in ks]] = True
        cache[t] = row
    return pd.DataFrame(np.vstack([cache[t] for t in texts]) if texts else np.zeros((0, len(keys)), bool), columns=keys)

def term_hits(texts, lex=None):
    """Jumlah baris per istilah. Istilah nol hit dibuang, istilah terlalu sering diaudit."""
    lex = lex or MBG_ASPECTS
    m = build_matcher(lex)
    c = {}
    for t in texts:
        for term in {x[0] for x in find_terms(t, m)}:
            c[term] = c.get(term, 0) + 1
    rows = [(k, t, c.get(t.lower(), 0)) for k, v in lex.items() for t in v]
    return pd.DataFrame(rows, columns=["kelas", "istilah", "hit"]).sort_values(["kelas", "hit"], ascending=[True, False]).reset_index(drop=True)

def audit_aspects(texts, H, n=30, seed=42):
    """{kelas: contoh acak}. Baca dan hitung presisinya sebelum dipakai di slide."""
    texts = pd.Series(list(texts))
    return {k: texts[H[k].to_numpy()].drop_duplicates().sample(min(n, int(H[k].sum())), random_state=seed).tolist() for k in H if H[k].any()}

def aspect_sentiment(H, sentiment):
    """Tabel aspek x sentimen. Satu baris dihitung di semua aspeknya."""
    s = np.asarray(sentiment)
    return pd.DataFrame({k: pd.Series(s[H[k].to_numpy()]).value_counts() for k in H}).T.fillna(0).astype(int)
