"""Bangun alur_multi_kelas.html: diagram alur interaktif + penjelasan langkah + daftar metode + hasil run.

Satu file, tanpa CDN, bisa dibuka offline. Kode contoh diambil langsung dari notebook, hasil dari folder hasil_mc/.
"""
from pathlib import Path
import json
import pandas as pd
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "b_klasifikasi/10_multi_kelas"
NOTEBOOKS = {1: "10_multi_kelas_1_fitur_klasik.ipynb", 2: "10_multi_kelas_2_pretrained_ekstraksi.ipynb", 3: "10_multi_kelas_3_pretrained_finetune.ipynb",
             4: "10_multi_kelas_4_bert_fusi_metadata.ipynb", 5: "10_multi_kelas_5_lgbm_bert_ensemble.ipynb"}
NB_NAMES = {1: "Fitur klasik + handcrafted", 2: "Pretrained beku", 3: "Pretrained fine-tune", 4: "BERT + fusi metadata", 5: "LGBM + BERT + ensemble"}
ALL = [1, 2, 3, 4, 5]

# (nama, kapan dipakai / catatan, dipakai di notebook)
STEPS = [
    dict(id="data", no=1, title="Data dan audit", unit="pemeriksaan", lane=None,
         what="Muat CSV lalu periksa kualitasnya **sebelum** membuang apa pun. Hasilnya angka: berapa baris hilang dan karena apa. Nama kolom tidak ditebak; `assert` memberi daftar kolom bila salah.",
         snippet="Skema sebelum pembersihan",
         methods=[
             ("Skema dan tipe kolom", "Selalu. Memastikan teks benar-benar string dan label benar-benar kategori.", ALL),
             ("Missing per kolom", "Selalu. Missing di label tidak bisa dipakai training.", ALL),
             ("Teks non-string atau kosong", "Angka atau list di kolom teks tidak boleh diubah diam-diam menjadi 'nan'.", ALL),
             ("Pseudo-null ('nan', 'none', '-', 'n/a')", "Sering muncul pada CSV hasil scraping; lolos dari `isna()`.", ALL),
             ("Label kosong atau variasi penulisan", "Cek `Positif` vs `positif ` sebelum mengubah ke angka.", ALL),
             ("Duplikat baris penuh", "Duplikat murni biasanya hasil scraping ganda.", ALL),
             ("Encoding dan separator file", "Bila `read_csv` gagal atau muncul karakter rusak; coba `encoding='latin-1'` atau `sep=';'`.", []),
             ("Sampling `N_ROWS`", "Iterasi cepat saat pengembangan; `None` untuk run penuh.", ALL),
         ]),
    dict(id="prep", no=2, title="Preprocessing universal", unit="langkah", lane=None,
         what="Rantai satu arah `asli → safe → normalized → clean`, dibuat dari daftar langkah satu baris. Tiap langkah dicatat jumlah baris yang berubah, jadi langkah yang tidak relevan untuk data ini kelihatan (berubah = 0). Negasi dan istilah domain dilindungi. Dari rantai ini lahir enam **view** teks yang dipilih sesuai model.",
         snippet="SAFE = [",
         methods=[
             ("safe · perbaikan encoding (ftfy)", "Teks hasil scraping sering berisi mojibake (`Ã©`). Tidak mengubah makna.", ALL),
             ("safe · Unicode NFC", "Menyeragamkan karakter bergabung; emoji dan aksara daerah dipertahankan.", ALL),
             ("safe · HTML entity", "`&amp;` → `&`, `&#39;` → `'`.", ALL),
             ("safe · karakter tak terlihat dan kontrol", "Zero-width space dan karakter kontrol memecah token tanpa terlihat.", ALL),
             ("safe · spasi", "Spasi ganda dan newline menjadi satu spasi.", ALL),
             ("normalized · obfuscation", "`k3r4cunan` → `keracunan`; kamus kecil, perlu diisi sesuai domain.", ALL),
             ("normalized · URL, email, telepon → placeholder", "Isi tautan hampir tidak membawa sentimen, tetapi keberadaannya bisa.", ALL),
             ("normalized · mention → USER, hashtag dipecah", "Nama akun tidak berguna untuk generalisasi; hashtag camelCase dipecah menjadi kata.", ALL),
             ("normalized · huruf kecil", "TF-IDF dan Word2Vec; tidak untuk encoder yang sensitif huruf besar (`embed`).", ALL),
             ("normalized · slang (`gk` → `tidak`)", "Kamus + perlindungan kata domain dan kata ambigu (`sm`, `dr`).", [1, 3, 4, 5]),
             ("normalized · huruf berulang (`baguuus`)", "Hanya diterima bila hasilnya ada di kosakata, jadi `mbg` atau nama tidak rusak.", [1, 3, 4, 5]),
             ("normalized · tawa (`wkwkwk`, `hahaha`)", "Disatukan agar tidak jadi ratusan token unik.", [1, 3, 4, 5]),
             ("normalized · reduplikasi (`anak2`)", "Daftar putih saja agar `S2`, `GPT2`, tanggal tidak rusak.", [1, 3, 4, 5]),
             ("normalized · bahasa daerah (opsional)", "`ora`, `teu`, `indak`; aktifkan bila EDA menunjukkan campuran daerah.", []),
             ("normalized · koreksi typo (kamus)", "Kamus kosong secara default; isi dari daftar kata langka di EDA.", [1, 3, 4, 5]),
             ("normalized · tanda baca berulang", "`!!!!!` → `!!`; sisakan sinyal ekspresif.", ALL),
             ("normalized · emoji → teks", "Emoji membawa sentimen; `demojize` membuatnya terbaca oleh model kata.", ALL),
             ("clean · hapus emoji / emoticon / angka (opsional)", "Mati secara default karena emoji dan angka sering penting; aktifkan lewat `CLEAN_OPT`.", []),
             ("clean · tanda baca dan kata berulang", "Untuk bag-of-words; tidak untuk BERT.", [1]),
             ("clean · stopword (tanpa negasi)", "Daftar stopword dikurangi negasi dan istilah domain.", [1]),
             ("clean · stemming Sastrawi", "Mengecilkan vocabulary; di-cache per kata. Kadang menghapus nuansa.", [1]),
             ("view `bert` (resep IndoBERTweet)", "Huruf kecil, `HTTPURL`, `@USER`, emoji → teks. Mengikuti model card.", [2, 3, 4, 5]),
             ("view `embed` (huruf asli + placeholder)", "Untuk encoder yang dilatih pada teks natural; tidak dikecilkan hurufnya.", [2, 3, 4, 5]),
         ]),
    dict(id="split", no=3, title="Grup duplikat dan split", unit="pilihan", lane=None,
         what="Komentar yang sama atau hampir sama tidak boleh berada di dua split, kalau tidak skor bocor. Tiga sumber disatukan jadi satu grup (teks identik, pengguna sama, near-duplicate MinHash) lewat connected components. Split 60 / 20 / 20 dengan `StratifiedGroupKFold`. Validation untuk memilih, **holdout dibuka sekali**.",
         snippet="StratifiedGroupKFold(5",
         methods=[
             ("Grup teks identik", "Selalu; paling murah.", ALL),
             ("Grup pengguna / video / thread (`GROUP_COL`)", "Bila penulis yang sama cenderung menulis hal serupa.", ALL),
             ("Near-duplicate MinHash char 5-gram", "Menangkap salinan dengan beda satu-dua kata; aproksimasi, kandidat bisa terlewat.", ALL),
             ("Dedup dengan kemiripan embedding", "Lebih sensitif ke makna; lebih mahal.", []),
             ("Random split", "Hanya bila yakin tidak ada duplikat atau grup.", []),
             ("Stratified split", "Menjaga proporsi kelas, tanpa jaminan grup.", []),
             ("StratifiedGroupKFold", "Proporsi kelas terjaga dan grup tidak terbelah.", ALL),
             ("Time-based split", "Bila ada kolom waktu dan tes di masa depan.", []),
             ("K-fold CV / repeated CV", "Data kecil dan butuh estimasi variansi; lebih mahal.", []),
         ]),
    dict(id="eda", no=4, title="EDA: angka saja", unit="analisis", lane=None,
         what="Hanya train. Setiap tabel menjawab satu pertanyaan keputusan: perlu bobot kelas? perlu normalisasi? model akan belajar sentimen atau topik? Tidak ada grafik otomatis; yang ditampilkan angka dan contoh.",
         snippet="Distribusi kelas · train",
         methods=[
             ("Distribusi kelas", "Menentukan macro-F1 dan bobot kelas.", ALL),
             ("Panjang karakter dan kata (persentil)", "Menentukan `MAX_LEN` dan apakah panjang bocor jadi sinyal.", ALL),
             ("Panjang per kelas", "Bila satu kelas jauh lebih pendek, model bisa menebak dari panjang.", ALL),
             ("Duplikat (persis dan setelah normalisasi)", "Mengukur bocor potensial dan inflasi skor.", ALL),
             ("Vocabulary: TTR, hapax, kata langka", "Hapax besar = banyak ejaan unik → normalisasi atau char n-gram.", ALL),
             ("Top kata, bigram, trigram", "Melihat topik dominan dan frasa negasi.", ALL),
             ("Kata khas per kelas (log-odds)", "Bila yang khas hanya nama atau topik, model belajar topik.", ALL),
             ("Ragam noise (22 pola) dan per kelas", "Menentukan langkah preprocessing yang layak; beda antar kelas = sinyal sekaligus bias.", ALL),
             ("Konflik label (teks sama, label beda)", "Batas atas skor yang realistis.", ALL),
             ("Confidence pemberi label", "Bila label berasal dari model, skor bukan bukti sentimen manusia.", ALL),
             ("Contoh teks per kelas", "Membaca langsung; tidak bisa digantikan angka.", ALL),
             ("Estimasi bahasa (langid)", "Menentukan model monolingual atau multilingual.", ALL),
             ("Panjang token per tokenizer", "Persen terpotong pada `MAX_LEN`.", [2, 3]),
             ("OOV validation terhadap vocabulary train", "Seberapa banyak kata baru yang tidak dikenal model kata.", [1]),
         ]),
    dict(id="fe", no=5, title="Feature engineering", unit="kelompok fitur", lane=None,
         what="Fitur handcrafted dihitung dalam enam kelompok (±75 fitur; paket awal 37). Gaya penulisan dihitung dari teks **asli** (huruf besar, tanda baca, huruf berulang tidak boleh hilang); leksikon dari teks **normalisasi**. Fitur diskalakan hanya dari train. Prinsipnya: fitur yang baik menyimpan informasi yang belum dipelajari representasi teks utama, bukan fitur yang paling banyak.",
         snippet="GROUP = {",
         methods=[
             ("structural · panjang, jumlah kata, kalimat, rasio kata unik, URL, mention, hashtag, digit", "Paling murah; TTR dipengaruhi panjang dokumen, jangan dibaca sebagai kekayaan kosakata murni.", [1, 4, 5]),
             ("stylistic · kapital, huruf berulang, tanda baca, deret tanda baca terpanjang", "Prioritas tinggi untuk medsos; intensitas (jumlah, deret terpanjang) lebih informatif daripada flag 0/1.", [1, 4, 5]),
             ("expressive · emoji, emoticon, tawa", "Emoji dihitung dari teks asli dan tetap dipertahankan; jangan menganggap satu emoji selalu positif.", [1, 4, 5]),
             ("lexical · slang, singkatan, stopword, kata langka, OOV, campur Inggris, kata kasar", "Statistik korpus (frekuensi, OOV) dari train saja.", [1, 4, 5]),
             ("lexicon · negasi + cakupan, kata positif/negatif, emosi, intensifier", "Negasi dengan cakupan 3 token membedakan `tidak bagus` dari `bagus`; kamus perlu diperluas untuk domain baru.", [1, 4, 5]),
             ("discourse · kata ganti orang, kontras, sebab, kondisi, modal", "Lebih berguna pada teks panjang; pada komentar 1-3 kata hampir selalu nol.", [1, 4, 5]),
             ("fragmentasi tokenizer (subword per kata)", "Khusus BERT: penanda ejaan informal dan ketidakcocokan vocabulary.", [4]),
             ("TF-IDF-SVD (64 komponen)", "Sinyal leksikal terkompresi; pembanding fitur handcrafted.", [1, 4, 5]),
             ("Transformasi sign·log1p + standardisasi (fit di train, klip ±5)", "Fitur hitungan sangat miring ke kanan.", [1, 4, 5]),
             ("Mutual information terhadap label", "Menunjukkan kelompok fitur dengan sinyal; bukan bukti kausal.", [1, 4, 5]),
             ("Korelasi antarfitur (|r| > 0,9)", "Redundansi: kandidat dibuang.", [1, 4, 5]),
             ("Ablasi bertahap E0-E4 dan buang/acak per kelompok", "Satu-satunya cara tahu apakah sebuah kelompok benar-benar berguna.", [1]),
             ("Fitur POS (noun/verb/adjective ratio)", "Butuh tagger Indonesia yang akurat; di teks informal tagging sering keliru. Tidak dipakai.", []),
             ("Metadata platform (jam, like, reply, umur akun)", "Hanya bila tersedia saat inferensi; user_id dan engagement bisa membuat model menghafal pengguna.", []),
             ("LFTK (220+ fitur linguistik)", "Pustaka katalog fitur; berbahasa Inggris, referensi untuk memperluas kelompok.", []),
         ]),
    dict(id="rep", no=6, title="Representasi teks", unit="pilihan", lane="split",
         what="Cara mengubah teks menjadi angka. Ini keputusan terbesar. Lima notebook, lima jalur: fitur yang dipelajari dari train saja (1), vektor dari model pretrained yang dibekukan (2), representasi yang ikut dilatih (3), representasi yang dilatih bersama fitur metadata (4), dan gabungan model tabular + BERT (5).",
         snippet="REPS = {",
         methods=[
             ("Bag-of-words (count)", "Baseline paling sederhana; TF-IDF hampir selalu lebih baik.", []),
             ("TF-IDF kata 1-2 gram", "Baseline yang sulit dikalahkan pada data kecil.", [1]),
             ("TF-IDF karakter 3-5 gram", "Tahan terhadap ejaan nyeleneh dan imbuhan.", [1]),
             ("Hashing vectorizer", "Data sangat besar; tanpa vocabulary, tidak bisa dibaca balik.", []),
             ("LSA (TF-IDF + SVD)", "Mengompresi TF-IDF; kadang membantu kepala non-linear.", []),
             ("Word2Vec rata-rata (dilatih di train)", "Korpus kecil: embedding lemah, tetapi bebas bocor.", [1]),
             ("FastText rata-rata (subkata)", "Kata baru tetap dapat vektor dari potongan karakter.", [1]),
             ("Embedding statis pretrained (GloVe, fastText id)", "Butuh berkas eksternal; sering tidak tersedia offline.", []),
             ("Sentence embedding beku (e5, mpnet)", "Cepat, stabil, makna kalimat; ukuran model perlu diunduh.", [2]),
             ("BERT mean-pooling beku", "Rata-rata token; biasanya mengalahkan CLS pada BERT biasa.", [2]),
             ("BERT CLS beku", "Satu token; untuk model yang dilatih agar CLS bermakna.", [2]),
             ("Pooling beberapa layer", "Menggabungkan layer atas; sedikit tambahan skor, lebih rumit.", []),
             ("Concat banyak encoder", "Menggabungkan sudut pandang beberapa model; dimensi besar.", [2]),
             ("Representasi ikut dilatih (fine-tune)", "Seluruh bobot belajar; akurasi tertinggi, butuh GPU dan data cukup.", [3, 4, 5]),
             ("TF-IDF + fitur handcrafted (hstack berbobot)", "Menambah sinyal gaya penulisan; bobot meta menentukan seberapa kuat dibanding ribuan kolom TF-IDF.", [1]),
             ("Embedding beku + metadata (LR / LightGBM)", "Vektor 768 dimensi + ±37 fitur; embedding dibekukan dan diekstrak satu kali.", [5]),
             ("TF-IDF-SVD sebagai sinyal tambahan", "Kompresi TF-IDF; pada data berlabel sedikit sering lebih berguna daripada fitur handcrafted saja.", [4]),
         ]),
    dict(id="model", no=7, title="Model dan pengklasifikasi", unit="pilihan", lane="split",
         what="Pengklasifikasi di atas representasi. Notebook 1 dan 2 menyapu parameter `C` di validation; notebook 3 melatih seluruh jaringan dengan loop PyTorch biasa (AdamW, warmup, fp16, loss berbobot kelas).",
         snippet="CLF = {",
         methods=[
             ("Mayoritas", "Batas bawah. Model yang tidak mengalahkannya tidak berguna.", [1, 2]),
             ("Logistic Regression", "Probabilitas, cepat, mudah dibaca; pilihan pertama.", [1, 2, 5]),
             ("Linear SVM", "Sering sedikit lebih kuat di TF-IDF; hanya margin, bukan probabilitas.", [1, 2]),
             ("Naive Bayes", "Sangat cepat di bag-of-words; kuat untuk teks pendek.", []),
             ("k-NN", "Berguna untuk melihat tetangga terdekat; lambat dan sensitif skala.", []),
             ("LightGBM di atas embedding beku + metadata", "Model tabular: menerima vektor angka, bukan teks. Butuh random search dan `colsample` kecil (800 kolom, ±3.000 baris).", [5]),
             ("Fusi fitur di kepala BERT (concat + proyeksi metadata)", "Menempelkan fitur handcrafted ke vektor kalimat saat fine-tuning; diuji bertahap E5-E9.", [4]),
             ("Random Forest / Extra Trees", "Jarang unggul di teks; berguna bila ada fitur tabular tambahan.", []),
             ("MLP kecil di atas embedding", "Menangkap interaksi non-linear; tidak ada bobot kelas.", [2]),
             ("Fine-tune penuh", "Standar emas untuk teks; butuh GPU.", [3, 4, 5]),
             ("Fine-tune sebagian (bekukan layer bawah)", "Lebih hemat memori dan lebih stabil di data kecil.", []),
             ("LoRA / adapter", "Menghemat memori untuk model besar.", []),
             ("SetFit (kontrastif + kepala)", "Data berlabel sangat sedikit.", []),
             ("Zero-shot NLI", "Tanpa label sama sekali; lebih lemah dari supervised.", []),
             ("LLM few-shot", "Alternatif bila diizinkan; lambat dan mahal pada ratusan ribu baris.", []),
         ]),
    dict(id="select", no=8, title="Seleksi dan post-processing", unit="pilihan", lane=None,
         what="Model dipilih dari macro-F1 **validation**, lalu diputuskan apakah bias per kelas dipakai. Bias dinilai cross-fit (dicari di satu fold, diuji di fold lain) dan hanya dipakai bila naik lebih dari `MIN_POST_GAIN`. Keputusan dibekukan **sebelum** holdout dibuka.",
         snippet="grid = np.array(",
         methods=[
             ("Pilih macro-F1 validation tertinggi", "Metrik utama untuk kelas timpang.", ALL),
             ("Tuning `C` pada grid", "Validation optimistis karena ikut dipakai memilih.", [1, 2]),
             ("Random search hyperparameter LightGBM (OOF)", "12 trial × 5 fold; tabel menampilkan rata-rata dan terburuk trial untuk mengukur optimisme seleksi.", [5]),
             ("OOF 5-fold untuk semua cabang", "Prediksi tiap baris dari model yang tidak melihat baris itu; dasar ensemble yang jujur.", [5]),
             ("Bobot ensemble dituning (simplex 0,1)", "Dicari di OOF, dinilai nested: bobot dicari di empat fold, diuji di fold kelima.", [5]),
             ("Rata-rata dua seed per konfigurasi", "Mengurangi noise fine-tuning; selisih antar seed menjadi pembanding efek fitur.", [4]),
             ("Ensemble rata-rata log-prob", "Menurunkan variansi; tidak untuk SVM (margin).", ALL),
             ("Bias per kelas (coordinate ascent, cross-fit)", "Mengangkat kelas yang jarang diprediksi.", ALL),
             ("Threshold per kelas (one-vs-rest)", "Untuk multilabel atau kelas yang sangat jarang.", []),
             ("Kalibrasi (temperature scaling)", "Bila probabilitas dipakai langsung untuk keputusan.", []),
             ("Stacking LR di atas log-probabilitas OOF", "Dinilai nested (cross_val_predict); butuh OOF bersih, riskan overfit di data kecil.", [5]),
             ("Koreksi berbasis aturan (negasi)", "Murah, tetapi harus diukur: sering menambah kesalahan.", []),
             ("Pseudo-label / self-training", "Bila ada banyak data tanpa label.", []),
             ("Refit di train + validation untuk final", "Dilakukan setelah pemilihan; skor validation tidak dilaporkan ulang.", []),
         ]),
    dict(id="eval", no=9, title="Evaluasi dan kelemahan", unit="analisis", lane=None,
         what="Holdout dibuka sekali, lalu dibongkar: per kelas, pasangan kelas yang tertukar, kondisi teks yang membuat model gagal, salah prediksi yang paling yakin, dan interval kepercayaan. Tujuannya menunjukkan **kelemahan**, bukan hanya satu angka.",
         snippet="Holdout · semua model",
         methods=[
             ("Macro-F1 (metrik utama)", "Tiap kelas berbobot sama; jujur untuk kelas timpang.", ALL),
             ("Weighted-F1, accuracy, balanced accuracy", "Pelengkap; accuracy bisa menipu bila kelas timpang.", ALL),
             ("Precision / recall / F1 / support per kelas", "Menunjukkan kelas yang gagal di balik rata-rata.", ALL),
             ("Confusion matrix (jumlah dan %)", "Melihat arah salah, bukan hanya jumlah salah.", ALL),
             ("Pasangan kelas paling tertukar", "Petunjuk batas label yang kabur.", ALL),
             ("Slice per kondisi teks", "Panjang, negasi, slang, URL, emoji, tawa, confidence.", ALL),
             ("Salah prediksi paling yakin", "Dibaca manusia: model salah atau labelnya?", ALL),
             ("Bootstrap CI per grup", "Selisih antar model lebih kecil dari lebar CI tidak bisa dipercaya.", ALL),
             ("Fitur pendorong (koefisien LR)", "Apakah model belajar kata sentimen atau kata topik.", [1]),
             ("Kurva belajar per epoch", "Overfit atau belum konvergen.", [3, 4]),
             ("Ablasi dan permutation importance per kelompok fitur", "Buang satu kelompok atau acak di validation: kelompok mana yang benar-benar memberi informasi.", [1]),
             ("Variasi antar seed", "Selisih konfigurasi lebih kecil dari simpangan baku antar seed bukan temuan.", [4]),
             ("Agreement antar cabang dan batas atas ensemble", "Ensemble hanya membantu bila cabang berbeda pendapat di baris yang berbeda.", [5]),
             ("Selisih validation → holdout", "Mengukur optimisme seleksi.", ALL),
             ("Kalibrasi (ECE)", "Bila probabilitas dipakai untuk keputusan.", []),
             ("Uji berpasangan (McNemar)", "Bila dua model berbeda tipis dan butuh uji formal.", []),
         ]),
]

RULES = [
    ("Data kecil (< 2.000 baris) atau tanpa GPU", "Notebook 1", "TF-IDF + LR/SVM sulit dikalahkan di data kecil; berjalan di CPU dalam hitungan menit."),
    ("Ingin baseline kuat tanpa tuning rumit", "Notebook 2", "Embedding beku + LR: stabil, tanpa learning rate atau epoch; GPU hanya untuk ekstraksi."),
    ("Data ≥ 2.000 baris, GPU tersedia, ingin skor tertinggi", "Notebook 3", "Fine-tune penuh biasanya yang terbaik, tetapi butuh seleksi di validation yang ketat."),
    ("Teks penuh slang dan ejaan nyeleneh", "Notebook 1 (char) atau 3 (IndoBERTweet)", "Char n-gram dan model medsos lebih tahan terhadap ejaan."),
    ("Campuran bahasa Inggris atau daerah", "Notebook 2 atau 3 (e5, XLM-R)", "Model multilingual; cek estimasi bahasa di EDA."),
    ("Ingin tahu apakah fitur gaya penulisan (kapital, emoji, negasi) menambah informasi", "Notebook 1 (E0-E4) lalu 4", "Ablasi bertahap di model klasik dulu; lanjutkan ke BERT hanya bila ada sinyal."),
    ("Sudah punya beberapa model berbeda dan ingin menekan variansi", "Notebook 5", "Ensemble dengan bobot dituning di OOF dan dinilai nested; hanya membantu bila cabang berbeda pendapat."),
    ("Satu kelas sangat kecil", "Semua, lalu baca bagian per kelas", "Aktifkan bobot kelas; macro-F1 akan memperlihatkan kegagalannya."),
]


def snippet(phrase):
    """Ambil sel kode pertama (dari notebook 1, lalu 2 dan 3) yang memuat frasa, maksimal 28 baris."""
    for k in ALL:
        if not (FOLDER / NOTEBOOKS[k]).exists():
            continue
        for c in nb.read(FOLDER / NOTEBOOKS[k], as_version=4).cells:
            if c.cell_type == "code" and phrase in c.source:
                lines = c.source.splitlines()
                return "\n".join(lines[:28]) + ("\n# ... (sel lengkap ada di notebook)" if len(lines) > 28 else ""), k
    return "", None


def results():
    out = {}
    for k in ALL:
        d = ROOT / "b_klasifikasi/10_multi_kelas/hasil_mc" / str(k)
        if not (d / "run_manifest.json").exists():
            out[k] = None
            continue
        man, sel = json.loads((d / "run_manifest.json").read_text()), json.loads((d / "selection.json").read_text())
        hold = pd.read_csv(d / "holdout_models.csv", index_col=0)
        per = pd.read_csv(d / "per_class.csv", index_col=0)
        classes = sel["kelas"]
        weak = per.loc[classes].sort_values("f1-score").iloc[0]
        out[k] = dict(model=man["final"]["model"], macro_f1=man["final"]["macro_f1"], accuracy=man["final"]["accuracy"], menit=man["detik"] / 60,
                      post=sel["post_processing"], lemah=f"{weak.name} (F1 {weak['f1-score']:.2f}, n={int(weak['support'])})", gpu=man.get("gpu"),
                      tabel=[dict(model=i, macro_f1=r.macro_f1, val=r.val_macro_f1) for i, r in hold.sort_values("val_macro_f1", ascending=False).head(8).iterrows()])
    return out


for s in STEPS:
    s["code"], s["code_nb"] = snippet(s.pop("snippet"))
    s["methods"] = [dict(name=a, note=b, used=c) for a, b, c in s["methods"]]

DATA = dict(steps=STEPS, rules=RULES, names=NB_NAMES, results=results(), files={k: v for k, v in NOTEBOOKS.items()})

TEMPLATE = r'''<!doctype html>
<html lang="id" data-theme="auto">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Alur klasifikasi multiclass</title>
<style>
:root{--bg:#f7f7f4;--surface:#fff;--ink:#1c1d20;--muted:#62666d;--line:#dcdcd6;--accent:#2b59c3;--accent-soft:#e6edfb;
--c1:#1f7a5c;--c2:#b4651b;--c3:#7a3fb0;--c1s:#e3f3ec;--c2s:#fbeede;--c3s:#f0e6f8;--c4:#1d6f96;--c4s:#e0f0f7;--c5:#b03a48;--c5s:#fae4e7;--code:#f1f1ec;--shadow:0 1px 2px rgba(0,0,0,.06)}
:root[data-theme="dark"]{--bg:#141517;--surface:#1c1e21;--ink:#ecece8;--muted:#9a9ea6;--line:#2e3136;--accent:#7ea2ff;--accent-soft:#223052;
--c1:#5ccba2;--c2:#f0a25c;--c3:#c79bf0;--c1s:#16322a;--c2s:#37281a;--c3s:#2c2038;--c4:#6cc3ea;--c4s:#15303d;--c5:#f08a97;--c5s:#3a1d22;--code:#101113;--shadow:none}
@media (prefers-color-scheme:dark){:root[data-theme="auto"]{--bg:#141517;--surface:#1c1e21;--ink:#ecece8;--muted:#9a9ea6;--line:#2e3136;--accent:#7ea2ff;--accent-soft:#223052;
--c1:#5ccba2;--c2:#f0a25c;--c3:#c79bf0;--c1s:#16322a;--c2s:#37281a;--c3s:#2c2038;--c4:#6cc3ea;--c4s:#15303d;--c5:#f08a97;--c5s:#3a1d22;--code:#101113;--shadow:none}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
header{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:20px}
h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--muted);margin:0;max-width:720px}
.tools{display:flex;gap:8px}
button{font:inherit;color:inherit;cursor:pointer}
.tbtn{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:6px 12px}
.diagram{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px;box-shadow:var(--shadow)}
.row{display:flex;align-items:stretch;gap:0;flex-wrap:nowrap}
.node{flex:1 1 0;min-width:0;background:var(--surface);border:1.5px solid var(--line);border-radius:10px;padding:10px 12px;text-align:left;position:relative;transition:border-color .15s,background .15s}
.node:hover{border-color:var(--accent)}
.node[aria-pressed="true"]{border-color:var(--accent);background:var(--accent-soft)}
.node .no{font-size:11px;color:var(--muted);font-weight:600;letter-spacing:.04em}
.node .t{display:block;font-weight:600;margin:2px 0 6px;line-height:1.25}
.chip{display:inline-block;font-size:11px;background:var(--code);border-radius:99px;padding:1px 8px;color:var(--muted)}
.arrow{flex:0 0 26px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:18px}
.down{display:flex;justify-content:center;color:var(--muted);font-size:20px;margin:4px 0;line-height:1}
.lanes{display:grid;gap:8px;position:relative}
.lane{display:grid;grid-template-columns:150px 1fr;gap:10px;align-items:stretch;border-radius:10px;padding:8px}
.lane .lbl{font-size:12px;font-weight:700;display:flex;flex-direction:column;justify-content:center}
.lane .lbl small{font-weight:400;color:var(--muted)}
.lane.l1{background:var(--c1s)}.lane.l2{background:var(--c2s)}.lane.l3{background:var(--c3s)}
.lane.l4{background:var(--c4s)}.lane.l5{background:var(--c5s)}.lane.l4 .lbl{color:var(--c4)}.lane.l5 .lbl{color:var(--c5)}.skip{flex:1 1 0;display:flex;align-items:center;justify-content:center;border:1.5px dashed var(--line);border-radius:10px;font-size:12px;color:var(--muted);padding:6px;text-align:center}
.lane.l1 .lbl{color:var(--c1)}.lane.l2 .lbl{color:var(--c2)}.lane.l3 .lbl{color:var(--c3)}
.lane .row .node{font-size:13px}
.lane .lanehint{font-size:12px;color:var(--muted);align-self:center;padding-left:6px;flex:1.4 1 0}
.panel{margin-top:16px;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:20px;box-shadow:var(--shadow)}
.panel h2{margin:0 0 4px;font-size:18px}
.panel .what{margin:6px 0 14px;max-width:820px}
code{background:var(--code);padding:1px 5px;border-radius:4px;font:12.5px ui-monospace,Menlo,Consolas,monospace}
.filters{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0}
.f{border:1px solid var(--line);background:var(--surface);border-radius:99px;padding:3px 12px;font-size:13px}
.f[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.methods{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:8px}
.m{border:1px solid var(--line);border-radius:8px;padding:9px 12px;background:var(--bg)}
.m b{font-weight:600}
.m p{margin:3px 0 6px;color:var(--muted);font-size:13.5px}
.m.off{opacity:.62}
.tag{font-size:11px;border-radius:99px;padding:1px 8px;margin-right:4px;font-weight:600}
.tag.n1{background:var(--c1s);color:var(--c1)}.tag.n2{background:var(--c2s);color:var(--c2)}.tag.n3{background:var(--c3s);color:var(--c3)}.tag.n4{background:var(--c4s);color:var(--c4)}.tag.n5{background:var(--c5s);color:var(--c5)}.tag.n0{background:var(--code);color:var(--muted);font-weight:400}
.codebox{position:relative;margin-top:16px}
.codebox pre{margin:0;background:var(--code);border-radius:8px;padding:12px 14px;overflow:auto;max-height:340px;font:12.5px/1.5 ui-monospace,Menlo,Consolas,monospace}
.copy{position:absolute;top:8px;right:8px;background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:2px 10px;font-size:12px}
h3{font-size:14px;margin:18px 0 6px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}
td.num{font-variant-numeric:tabular-nums;text-align:right}
section.block{margin-top:24px}
section.block h2{font-size:17px;margin:0 0 8px}
.res{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}
.card{border:1px solid var(--line);border-radius:10px;padding:12px 14px;background:var(--surface)}
.card h4{margin:0 0 6px;font-size:14px}
.big{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums}
.kv{color:var(--muted);font-size:13px;margin:2px 0}
a{color:var(--accent)}
@media (max-width:820px){.row{flex-direction:column}.arrow{transform:rotate(90deg);flex-basis:22px}.lane{grid-template-columns:1fr}.lane .lanehint{display:none}}
</style>
</head>
<body>
<main>
<header>
  <div>
    <h1>Alur klasifikasi multiclass</h1>
    <p class="sub">Klik kotak mana saja untuk melihat apa yang dikerjakan, berapa banyak pilihan metodenya, dan mana yang dipakai di tiap notebook. Lima notebook berbagi langkah 1-4 dan 8-9 (kode identik), lalu bercabang di langkah 5-7.</p>
  </div>
  <div class="tools"><button class="tbtn" id="theme" type="button">Terang / gelap</button></div>
</header>

<div class="diagram" id="diagram"></div>
<div class="panel" id="panel" aria-live="polite"></div>

<section class="block"><h2>Hasil run di Kaggle</h2><div class="res" id="res"></div><div id="restab"></div></section>
<section class="block"><h2>Kapan pakai yang mana</h2><table id="rules"></table></section>
<section class="block"><h2>Notebook</h2><div id="files"></div></section>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const md = s => esc(s).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
const step = id => D.steps.find(s => s.id === id);
let cur = 'prep', flt = 0;

const node = (s, nbk = 0) => `<button class="node" type="button" data-id="${s.id}" data-nb="${nbk}" aria-pressed="${s.id === cur && flt === nbk}"><span class="no">LANGKAH ${s.no}</span><span class="t">${esc(s.title)}</span><span class="chip">${s.methods.length} ${s.unit}</span></button>`;
const arrow = '<div class="arrow" aria-hidden="true">→</div>';
const chain = ids => `<div class="row">${ids.map(i => node(step(i))).join(arrow)}</div>`;
const laneMeta = {1: ['Notebook 1', 'tf-idf, w2v, fasttext + fitur handcrafted'], 2: ['Notebook 2', '5 encoder beku + LR / SVM / MLP'], 3: ['Notebook 3', '3 model × 2 view, fine-tune'], 4: ['Notebook 4', 'BERT + fusi metadata, E5-E9'], 5: ['Notebook 5', 'LightGBM + BERT + ensemble OOF']};
const laneHint = {1: 'fitur dipelajari dari train', 2: 'backbone dibekukan', 3: 'seluruh bobot dilatih', 4: 'BERT dan metadata dilatih bersama', 5: '6 cabang, OOF, bobot dituning'};
const usesFE = k => step('fe').methods.some(m => m.used.includes(k));

function drawDiagram() {
  const lanes = [1, 2, 3, 4, 5].map(k => `<div class="lane l${k}"><div class="lbl">${laneMeta[k][0]}<small>${laneMeta[k][1]}</small></div><div class="row">${usesFE(k) ? node(step('fe'), k) : '<div class="skip">tanpa fitur handcrafted</div>'}${arrow}${node(step('rep'), k)}${arrow}${node(step('model'), k)}<div class="lanehint">${laneHint[k]}</div></div></div>`).join('');
  $('#diagram').innerHTML = chain(['data', 'prep', 'split', 'eda']) + '<div class="down">↓ cabang lima notebook</div><div class="lanes">' + lanes + '</div><div class="down">↓ semua notebook bergabung</div>' + chain(['select', 'eval']);
  document.querySelectorAll('.node').forEach(b => b.onclick = () => { cur = b.dataset.id; flt = +b.dataset.nb; render(); $('#panel').scrollIntoView({behavior: 'smooth', block: 'nearest'}); });
}

function render() {
  drawDiagram();
  const s = step(cur), ms = s.methods;
  const shown = ms.filter(m => flt === 0 ? true : flt === -1 ? m.used.length === 0 : m.used.includes(flt));
  const tags = m => m.used.length ? m.used.map(k => `<span class="tag n${k}">NB${k}</span>`).join('') : '<span class="tag n0">tidak dipakai · opsi</span>';
  const used = ms.filter(m => m.used.length).length;
  const fbtn = (v, t) => `<button class="f" type="button" data-f="${v}" aria-pressed="${flt === v}">${t}</button>`;
  $('#panel').innerHTML = `<h2>${s.no}. ${esc(s.title)}</h2><span class="chip">${ms.length} ${s.unit} · ${used} dipakai · ${ms.length - used} opsi lain</span>
    <p class="what">${md(s.what)}</p>
    <div class="filters">${fbtn(0, 'Semua')}${[1, 2, 3, 4, 5].map(k => fbtn(k, 'NB' + k + ' ' + D.names[k])).join('')}${fbtn(-1, 'Opsi (tidak dipakai)')}</div>
    <div class="methods">${shown.map(m => `<div class="m ${m.used.length ? '' : 'off'}"><b>${esc(m.name)}</b><p>${md(m.note)}</p>${tags(m)}</div>`).join('')}</div>
    ${s.code ? `<h3>Kode dari notebook ${s.code_nb}</h3><div class="codebox"><button class="copy" type="button" id="copy">Salin</button><pre><code id="code">${esc(s.code)}</code></pre></div>` : ''}`;
  document.querySelectorAll('.f').forEach(b => b.onclick = () => { flt = +b.dataset.f; render(); });
  const c = $('#copy'); if (c) c.onclick = () => { navigator.clipboard.writeText(s.code).then(() => { c.textContent = 'Tersalin'; setTimeout(() => c.textContent = 'Salin', 1200); }).catch(() => {}); };
}

function drawResults() {
  const f = x => x == null ? '-' : x.toFixed(3);
  $('#res').innerHTML = [1, 2, 3, 4, 5].map(k => {
    const r = D.results[k];
    return r ? `<div class="card"><h4>NB${k} · ${esc(D.names[k])}</h4><div class="big">${f(r.macro_f1)}</div><div class="kv">macro-F1 holdout · accuracy ${f(r.accuracy)}</div><div class="kv">model: <code>${esc(r.model)}</code>${r.post ? ' + bias kelas' : ''}</div><div class="kv">kelas terlemah: ${esc(r.lemah)}</div><div class="kv">${r.menit.toFixed(1)} menit${r.gpu ? ' · ' + esc(r.gpu) : ' · CPU'}</div></div>`
             : `<div class="card"><h4>NB${k} · ${esc(D.names[k])}</h4><div class="kv">belum dijalankan</div></div>`;
  }).join('');
  const rows = [1, 2, 3, 4, 5].filter(k => D.results[k]).flatMap(k => D.results[k].tabel.map(t => ({k, ...t})));
  $('#restab').innerHTML = rows.length ? `<h3>Model teratas per notebook (urut validation; NB5 = OOF)</h3><table><tr><th>Notebook</th><th>Model</th><th class="num">Macro-F1 validation</th><th class="num">Macro-F1 holdout</th></tr>${rows.map(r => `<tr><td>NB${r.k}</td><td><code>${esc(r.model)}</code></td><td class="num">${f(r.val)}</td><td class="num">${f(r.macro_f1)}</td></tr>`).join('')}</table><p class="kv">Skor terhadap pseudo-label MBG, bukan anotasi manusia. Selisih kecil antar model berada di dalam interval kepercayaan; lihat bagian evaluasi tiap notebook.</p>` : '';
}

$('#rules').innerHTML = '<tr><th>Situasi</th><th>Pakai</th><th>Alasan</th></tr>' + D.rules.map(r => `<tr><td>${esc(r[0])}</td><td><b>${esc(r[1])}</b></td><td>${esc(r[2])}</td></tr>`).join('');
$('#files').innerHTML = '<ul>' + [1, 2, 3, 4, 5].map(k => `<li><a href="${D.files[k]}">${D.files[k]}</a> · ${esc(D.names[k])}</li>`).join('') + '</ul>';
$('#theme').onclick = () => { const r = document.documentElement, dark = getComputedStyle(r).getPropertyValue('--bg').trim() === '#141517'; r.dataset.theme = dark ? 'light' : 'dark'; try { localStorage.setItem('theme', r.dataset.theme); } catch (e) {} };
try { const t = localStorage.getItem('theme'); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
render(); drawResults();
</script>
</body>
</html>
'''

html = TEMPLATE.replace("__DATA__", json.dumps(DATA, ensure_ascii=False).replace("</", "<\\/"))
(FOLDER / "alur_multi_kelas.html").write_text(html, encoding="utf-8")
print("ditulis:", FOLDER / "alur_multi_kelas.html", "| langkah:", len(STEPS), "| metode:", sum(len(s["methods"]) for s in STEPS))
