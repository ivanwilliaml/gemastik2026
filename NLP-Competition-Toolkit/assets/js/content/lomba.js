// Fakta lomba dan domain MBG. Sumber: docs/RINGKASAN_PERSIAPAN_GEMASTIK.md (6 Nov 2026).
// Label: [Panitia], [Buku saku], [Terbit], [Ukur], [Penilaian], [Estimasi], [Belum teruji].
window.TK = window.TK || {};
TK.lomba = {
  rules: [
    "[Panitia] Data berupa teks. Tanpa akses internet.",
    "[Panitia] PC panitia per tim: i7-12700, RAM 32 GB DDR5, SSD 1 TB, RTX 3070 LHR.",
    "[Panitia] Dataset, training, dan submission hanya lewat PC panitia. **Laptop pribadi tidak boleh mengakses dataset, bahkan untuk EDA.** Laptop untuk eksplorasi terpisah, dokumentasi, dan slide.",
    "[Panitia] Konten slide harus sudah masuk sebelum kompetisi berakhir. File final paling lambat 21.00 WIB; tanggalnya belum dikonfirmasi.",
    "[Panitia] Data dan notebook utama tidak boleh dibawa pulang."
  ],
  schedule: [
    ["Selasa 10 Nov", "13.00–16.00", "Set-Up Work Station", "B0201"],
    ["Selasa 10 Nov", "16.00–17.00", "Technical Meeting", "B0201"],
    ["Rabu 11 Nov", "13.00–18.00", "Penyelesaian kasus (5 jam)", "B0201"],
    ["Kamis 12 Nov", "08.00–17.00", "Presentasi, 20 menit per tim", "A0205"],
    ["Jumat 13 Nov", "19.00–21.00", "Penutupan", "Sport Hall"]
  ],
  scheduleNote: "[Buku saku] Panduan menyebut set-up 12.00–18.00. Konfirmasi di Technical Meeting.",
  score: [["Babak penyisihan", 20], ["Skor leaderboard", 30], ["Ketepatan metode", 25], ["Presentasi", 25]],
  precedent: "[Terbit] Pada GEMASTIK XIV, penyisihan berupa klasifikasi tweet, tetapi final berubah menjadi question answering. Tim perunggu ITB baru menemukan selisih besar skor internal dan resmi 10–30 menit sebelum tenggat.",
  unknown: "Pretrained, data eksternal, dan LLM boleh atau tidak. USB/SSD boleh atau tidak. Hak admin, versi driver. Format dan batas submission. Skema dataset dan jenis label.",
  questions: [
    "Boleh mencolok SSD eksternal/USB ke PC saat Set-Up? Kalau tidak, bagaimana environment dimasukkan?",
    "Hak admin ada? Python, VC++ Redistributable, driver NVIDIA sudah terpasang? Versi `nvidia-smi`?",
    "Environment yang dipasang saat Set-Up dijamin tetap ada pada 11 November?",
    "Model pretrained, data eksternal (leksikon, stopword), dan LLM lokal boleh? Harus dideklarasikan?",
    "Dataset ada di disk lokal atau server? Format submission, batas, public/private, cara submit tanpa internet?",
    "Boleh memakai test untuk fit TF-IDF, self-training, dedup? Boleh hand-label?",
    "Kode harus bisa dijalankan ulang? Batas runtime?",
    "Slide: bagaimana memindahkan grafik? Tenggat 21.00 tanggal berapa? Rubrik dan durasi?",
    "Stop kontak, ponsel, catatan cetak, jumlah laptop per tim?",
    "Skema data, bahasa, sumber, jenis label, dan cara label dibuat?"
  ],
  dont: [
    ["Berasumsi pretrained, data eksternal, atau LLM diizinkan", "IF26 kehilangan 5 submission karena larangan pretrained"],
    ["Fine-tune 500 ribu baris, 5-fold BERT, atau zero-shot semua baris", "Waktu; LLM ≈ 139 jam pada 1 detik per baris"],
    ["Menjalankan LLM dan training GPU bersamaan", "Risiko OOM"],
    ["Memakai model hoaks siap pakai sebagai pemeriksa fakta", "ROC-AUC 0,50 pada benchmark independen"],
    ["Menghapus stopword atau stem sebelum classifier", "Uji lokal negatif"],
    ["Menghapus label mencurigakan otomatis", "Pakai cleanlab untuk prioritas baca"],
    ["Melabeli baris test atau menyetel threshold darinya", "Risiko kepatuhan"],
    ["Mengejar gerakan kecil di leaderboard", "Noise, CDW2 public LB ±0,009"],
    ["Mengedit submission di Excel", "Merusak id dan encoding"],
    ["Memasang driver atau paket baru di hari terakhir", "Crash lebih mahal dari panas"],
    ["Memakai data MBG/eksternal tanpa izin", "Aturan belum jelas"]
  ],
  gates: [
    { t: "Pretrained dilarang", d: "TF-IDF + LightGBM + word2vec, tanpa BERT. Nyalakan filter **Tanpa pretrained** di bilah atas." },
    { t: "Label banyak (≥ 20.000 setelah dedup)", d: "TF-IDF semua baris, decision layer, IndoBERTweet pada sampel, lalu blend." },
    { t: "Label sedikit (±50–5.000)", d: "Pisahkan gold; TF-IDF vs LR di atas embedding; SetFit hanya bila menang; aturan + zero-shot pada sampel; self-training 1–2 putaran; validasi pada gold." },
    { t: "Tanpa label", d: "Dedup, embedding, topik, sentimen/aspek lemah dinilai pada 150–300 gold, garis waktu, rekomendasi." },
    { t: "Pasangan teks / relasional", d: "TF-IDF biasa tidak cukup. IF26: macro-F1 0,513." }
  ],
  taskProb: [
    ["sentiment", "Sentimen / stance 3 kelas", 35], ["multilabel", "Aspek / topik", 20], ["absa", "ABSA", 10], ["binary", "Deteksi biner", 10],
    ["qa", "QA / NER / ringkasan / kata kunci", 10], ["emotion", "Emosi / toksisitas", 5], ["regression", "Regresi engagement", 5], ["report", "Tanpa label, hanya laporan", 5]
  ],
  modifiers: "[Penilaian] Pengubah yang bisa menempel: label buatan mesin 40%, label jarang (<10% baris) 35%, split test berdasarkan waktu/sumber 30%, bagian topik/lonjakan di slide 70%.",
  slides: ["Jawaban satu kalimat", "Data dan metode, termasuk dedup", "Garis waktu dengan kejadian", "Peta tema", "Heatmap aspek × sentimen", "Dua pendalaman dengan kutipan", "Validasi gold dan stabilitas", "Tiga rekomendasi dengan pemilik dan metrik", "Batasan"],
  team: "Dua orang: Modeller (GPU, validasi, decision layer) dan Analyst (EDA, gold, audit error, grafik, slide, log submission). Tiga orang: tambah Data/QA. Sinkron tiap 30 menit; orang kedua memeriksa sebelum unggah final."
};

TK.mbg = {
  regimes: ["Peluncuran 6 Jan 2025 (190 SPPG, 570.000 penerima)", "Puncak keracunan Sep–Okt 2025", "Guncangan korupsi dan kepemimpinan Jun–Jul 2026", "Gelombang kedua dan debat anggaran Agu–Okt 2026"],
  events: [
    ["17–22 Sep 2025", "Garut 569; Banggai Kepulauan 339; Bandung Barat 1.333 (hitungan lain berbeda)"],
    ["24 Sep 2025", "Bandung Barat menetapkan KLB; BGN 70 kasus, 5.914 penerima"],
    ["Okt 2025", "Puncak 85 kasus, lalu 40, 12, dan 10 pada Jan 2026"],
    ["2–3 Jun 2026", "Dadan Hindayana dan dua deputi dicopot; Kejagung menggeledah; tiga tersangka. Masih dugaan"],
    ["22 Jul 2026", "Sudaryono dilantik sebagai Kepala BGN"],
    ["Jul–Agu 2026", "Anggaran 2027: Rp174T (Banggar), Rp270,2T indikatif (BGN), Rp240T sementara"],
    ["3 Sep 2026", "5.482 korban, 71 insiden sejak 22 Jul; lebih dari 80% pelanggaran SOP"],
    ["29 Okt 2026", "Laporan 100 hari Sudaryono, 'yayasan calo', 'MBG 2.0'"]
  ],
  disputed: "Korban keracunan: BGN 11.640 (Nov 2025), Kemenkes 50.059 dalam 560 kejadian (2 Sep 2026), JPPI 43.838, KPAI 12.658 anak (2025). Tampilkan per lembaga dengan tanggal, jangan digabung. Proporsi kelas tidak bisa ditebak: Drone Emprit 50% positif setahun, INDEF 85–97% negatif pada Sep 2025. Leksikon yang menyebut Dadan sebagai kepala BGN sudah usang.",
  traps: ["Duplikat antar train/test dan template copypasta", "Baris hanya emoji dan `[Sticker]`", "Singkatan 'MBG' bermakna lain atau spam judi", "Campuran Jawa dan Sunda; deteksi bahasa gagal, pakai kata penanda", "Sarkasme dan noise label", "Di set TikTok MBG publik: 40,9% baris mengulang `comment_id`, satu video memuat 71% komentar unik [Ukur]"],
  lexNote: "Hipotesis untuk label lemah, bukan kebenaran. [D] terlihat di CSV TikTok publik; [P] dari paper/berita; [I] saran belum diverifikasi. Audit setiap aturan: jumlah hit dan presisi pada 30 sampel. Nilai pada gold yang tidak dipakai menulis aturan.",
  lexicon: [
    ["keamanan_pangan", "[D] keracunan, racun, basi, bau, korban, layak, tidak layak; [P] klb, puskesmas, sop, e coli, salmonella; [I] ulat, belatung, berjamur; mual, muntah, diare, pusing"],
    ["kualitas_menu", "[D] menu, menunya, burger, ayam, nasi, sayur, susu, kotak, roti, telur/telor, tempe, tahu, ikan, kacang, mie, gula, buah, pisang, jeruk, enak, rasa; [P] ompreng; [I] ultra processed/upf, porsi"],
    ["gizi_kesehatan", "[D] gizi, gizinya, bergizi, ahli gizi, dokter, sehat; [I] stunting, protein, kalori, akg, obesitas"],
    ["anggaran", "[D] anggaran, dana, uang, buang, pendidikan, mending; [P] apbn, rapbn, pajak, triliun, pagu, efisiensi, pemborosan, refocusing"],
    ["korupsi_tata_kelola", "[D] korupsi, ladang, lahan, ajang, berjamaah, proyek, pejabat, untung; [P] koruptor, nepotisme, kolusi, markup, tersangka, kejagung, jual beli titik, yayasan, calo, transparan"],
    ["implementasi_dapur", "[D] sppg, dapur, ditutup/tutup; [P] kepala sppg, mitra, distribusi, kapasitas, suspend, investigasi, sanitasi; [I] terlambat, tepat waktu, slhs, relawan"],
    ["pendidikan_guru", "[D] guru, gaji, honorer, murid, siswa, sekolah, kantin; [P] santri; [I] pip"],
    ["ekonomi_umkm", "[D] umkm, lapangan kerja, pekerja, pedagang, jualan, kantin, catering, harga, naik; [I] petani, peternak, susu lokal"],
    ["politik", "[D] prabowo, wowo, dpr, tni, polri, polisi, purnawirawan; [P] gibran, janji kampanye, militer, purbaya, zulhas, sudaryono, nanik, dadan, ombudsman, jppi, kpai; [I] icw, cisdi"],
    ["penolakan", "[D] stop, tutup, bubar, evaluasi, ganti, mending; [P] bubarkan, setop, hentikan"],
    ["dukungan", "[D] bagus, setuju, program bagus; [P] semoga; [I] lanjut, terima kasih/makasih, alhamdulillah"],
    ["media/hoaks/buzzer", "[D] hoax, buzzer, influencer, narasi, digoreng, pendukung, fanatik; [P] hoaks, komdigi, klarifikasi; [I] fitnah, bot"],
    ["normalisasi ejaan", "[D] yg, gk, gw, bgt, tp, klo, udh, dpt, jg, jd, bnyk, karna, krn, utk, tdk, jgn, org, dr, anak2, lgi, bnr"],
    ["jawa/sunda", "[D] ndk, madak, joss; [I] ora, ndak, mboten, wes, sing, teu, henteu, atuh, euy"]
  ],
  data: "Dataset publik MBG sekitar 100.000 baris, sebagian besar berlabel otomatis, bukan gold. Di bundle: `MBG_PROCESSED.csv` (TikTok) dan MBG X berlabel manual (~1,1k baris). **Pakai hanya kalau panitia mengizinkan data eksternal.**"
};

// Isi bundle persiapan_h1: model per profil unduh.
TK.bundleModels = [
  ["C1", "core", "intfloat/multilingual-e5-small", "Embedding", "471 MB", "topic cluster report sim qa doc"],
  ["C2", "core", "indolem/indobertweet-base-uncased", "Fine-tune teks sosial", "445 MB", "cls mlabel reg pair absa"],
  ["C3", "core", "w11wo/indonesian-roberta-base-sentiment-classifier", "Sentimen siap pakai", "499 MB", "sentiment binary report"],
  ["C4", "core", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "Zero-shot NLI", "558 MB", "cls mlabel absa report pair"],
  ["C5", "core", "cahya/NusaBert-ner-v1.3", "NER", "644 MB", "ner"],
  ["F1", "plus", "LazarusNLP/NusaBERT-base", "Basis Jawa/Sunda", "443 MB", "cls mlabel reg pair absa"],
  ["F2", "plus", "indobenchmark/indobert-base-p1", "Basis teks berita (.bin)", "498 MB", "cls mlabel reg pair absa"],
  ["F3", "plus", "Rifky/Indobert-QA", "QA Indonesia", "440 MB", "qa"],
  ["F4", "plus", "LazarusNLP/indobert-lite-base-p1-indonli-multilingual-nli-distil-mdeberta", "NLI Indonesia", "47 MB", "pair cls report"],
  ["F5", "plus", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "Embedding cadangan CPU", "590 MB", "unsup sim doc"],
  ["F8", "all", "StevenLimcorn/indonesian-roberta-base-emotion-classifier", "Emosi", "499 MB", "emotion report"],
  ["F9", "all", "intfloat/multilingual-e5-base", "Embedding lebih besar", "1,1 GB", "unsup sim"],
  ["F10", "all", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "SetFit/parafrase", "1,1 GB", "cls pair"],
  ["F11", "all", "panggi/t5-small-indonesian-summarization-cased", "Ringkasan", "242 MB", "summarization"],
  ["F12", "all", "w11wo/indonesian-roberta-base-posp-tagger", "POS", "496 MB", "ner"],
  ["F13", "all", "FacebookAI/xlm-roberta-base", "Basis multibahasa", "1,1 GB", "cls mlabel reg pair absa"],
  ["F14", "all", "timpal0l/mdeberta-v3-base-squad2", "QA multibahasa", "1,1 GB", "qa"],
  ["F15", "all", "cahya/t5-base-indonesian-summarization-cased", "Ringkasan besar", "892 MB", "summarization"]
];
TK.modelAvoid = "**Jangan dipakai:** `google/embeddinggemma-300m` (gated), `jinaai/jina-embeddings-v3` dan `tabularisai/multilingual-sentiment-analysis` (non-komersial), BGE-M3 dan e5-large (terlalu lambat untuk 500 ribu baris), fastText `.bin` (7,2 GB). Muat lewat path lokal dengan `local_files_only=True`. Jangan pakai `trust_remote_code=True`. `AutoModel` membuang head classifier.";
