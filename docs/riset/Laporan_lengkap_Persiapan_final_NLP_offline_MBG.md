# Siapkan final NLP MBG tanpa internet

Performa terbaik di hari H datang dari persiapan yang sudah teruji offline, bukan dari satu model paling hebat. Bawa satu pipeline yang bisa berganti task: **TF-IDF + model linear sebagai submission aman, `indolem/indobertweet-base-uncased` sebagai satu taruhan GPU, `intfloat/multilingual-e5-small` untuk embedding dan topik**, plus kit kode yang sudah kamu latih. Di laptop ini, fine-tuning 500.000 baris tidak realistis; fine-tune 30.000–100.000 baris saja (estimasi). Risiko terbesar bukan pilihan model, melainkan aturan yang belum jelas, RAM 8 GB, dan perubahan task mendadak. "Temasek" sangat mungkin **GEMASTIK**, tetapi ini **inferensi saya (sekitar 90%, penilaian) dan perlu kamu konfirmasi**. Semua tanggal di laporan ini hanya berlaku kalau tebakan itu benar. Tema MBG dan 500.000 baris berasal **dari kamu, bukan dari sumber publik**: panduan GEMASTIK tidak menyebut MBG dan hanya menyatakan kasus diberikan saat final ([Panduan 2026](https://kemahasiswaan.unand.ac.id/images/documents/Panduan-GEMASTIK-2026.pdf)). Panduan juga **tidak menyatakan "tanpa internet" dan tidak mengatur pretrained, data eksternal, atau LLM**, jadi ketiganya masih harus ditanyakan. Satu temuan mengubah rencana: panduan menyebut environment dipasang di **perangkat yang disediakan panitia**, sehingga spesifikasi laptop kamu mungkin tidak berlaku di hari H. Karena itu paket harus portabel.

## Enam aksi teratas, urut dari yang paling mendesak

| # | Aksi | Kapan (kalau GEMASTIK) | Hasil yang harus ada |
|---|---|---|---|
| 1 | Konfirmasi nama lomba, lalu tanya panitia: laptop sendiri atau perangkat panitia, pretrained boleh atau tidak, internet, USB | Hari ini, 5 Nov | Jawaban tertulis, kalau bisa |
| 2 | Bangun wheelhouse + venv CUDA baru di `D:\comp`, lalu uji dengan Wi-Fi mati | 5–6 Nov | `verify_offline.py` lolos setelah reboot |
| 3 | Unduh semua item **wajib** di tabel di bawah, lalu muat tiap model dengan `HF_HUB_OFFLINE=1` | 5–6 Nov | Semua model termuat dari path lokal |
| 4 | Kalau boleh memakai laptop sendiri, naikkan RAM ke 16 GB sebelum latihan | Secepatnya | Tes memori lolos |
| 5 | Tulis kit kode, ukur throughput nyata, dan uji BERTopic pada RAM 8 GB | 6–9 Nov | Tabel waktu asli menggantikan estimasi di laporan ini |
| 6 | Saat lomba: submission pertama paling lambat T70, satu taruhan GPU, freeze model di T240 | 11 Nov | Satu file valid selalu tersedia |

**Jadwal GEMASTIK XIX, Kompetisi III Penambangan Data:** setup workstation 10 Nov 2026 12:00–18:00 WIB, pengerjaan kasus 11 Nov 13:00–18:00 WIB, presentasi 12 Nov, dan penutupan malam 13 Nov ([Panduan 2026, Bab II](https://kemahasiswaan.unand.ac.id/images/documents/Panduan-GEMASTIK-2026.pdf)). Lokasinya BINUS ([ringkasan UNIMMA](https://kemahasiswaan.unimma.ac.id/pedoman/panduan-gemastik-2026/)). Sumber lain yang saya temukan tidak cocok dengan deskripsi kamu, dan tidak ada lomba yang benar-benar bernama Temasek.

## Daftar unduhan: inti dari semua persiapan

Kerjakan dari atas ke bawah, karena urutannya mengikuti dampak kalau item itu gagal. Ukuran diukur langsung dari server pada 5 Nov 2026, kecuali yang bertanda **(E)** untuk estimasi. Ukuran model adalah satu salinan bobot; tokenizer dan config menambah puluhan MB per model. "Tidak dicatat" berarti riset tidak mencatat ukurannya. Simpan semuanya di `D:\comp`.

| No | Item | ID / URL persis | Ukuran | Fungsi | Prioritas |
|---|---|---|---|---|---|
| **A** | **Installer** | | | | |
| A1 | Python 3.12.10 | https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe | 27,0 MB | Pasang Python di perangkat panitia | Wajib |
| A2 | VC++ Redistributable | https://aka.ms/vs/17/release/vc_redist.x64.exe | 25,6 MB | Mencegah WinError 126 saat import torch dan XGBoost | Wajib |
| A3 | Installer VS Code, Chrome, driver NVIDIA | Situs resmi masing-masing (URL tidak dicek saat riset) | 0,5–1 GB bersama VSIX (E) | Editor; Chrome untuk kaleido; driver hanya cadangan | Wajib |
| **B** | **Paket Python** | | | | |
| B1 | PyTorch CUDA | `torch==2.14.1+cu130` dari https://download.pytorch.org/whl/cu130 | 1.991 MB | GPU GTX 1650 (Turing) | Wajib |
| B2 | Semua paket lain | `requirements.in` di bagian runbook | sekitar 0,69 GB | Data, TF-IDF, transformer, topik, grafik | Wajib |
| B3 | PyTorch CPU cadangan | `torch==2.14.1+cpu` dari https://download.pytorch.org/whl/cpu/torch/ | tidak dicatat (2.15.0+cpu 127 MB) | Perangkat tanpa GPU NVIDIA | Wajib |
| B4 | PyTorch cadangan GPU baru | `torch==2.15.0+cu132` dari https://download.pytorch.org/whl/cu132/torch/ | 2.108 MB | Kalau cu130 gagal verifikasi | Opsional |
| B5 | PyTorch cadangan GPU lama | `torch==2.14.1+cu126` dari https://download.pytorch.org/whl/cu126/torch/ | 2.603 MB | Perangkat panitia dengan GPU di bawah Turing | Opsional |
| **C** | **Model inti** | | | | |
| C1 | Embedding utama | `intfloat/multilingual-e5-small` | 471 MB | Near-duplicate, topik, kNN, head linear, kata kunci | Wajib |
| C2 | Basis fine-tuning | `indolem/indobertweet-base-uncased` | 445 MB | Fine-tune untuk semua task berlabel di teks sosial media | Wajib |
| C3 | Sentimen siap pakai | `w11wo/indonesian-roberta-base-sentiment-classifier` | 499 MB | Baseline tanpa training, fitur, uji label mesin | Wajib |
| C4 | Zero-shot | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | 558 MB | Label lemah pada sampel, ABSA per aspek | Wajib |
| C5 | NER | `cahya/NusaBert-ner-v1.3` | 644 MB | 19 tipe entitas; item pertama yang dibuang kalau waktu unduh kurang | Wajib |
| **D** | **Leksikon dan data kecil** | | | | |
| D1 | Kamus slang | https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv | 3,09 MB | Normalisasi slang (saklar uji) | Wajib |
| D2 | Singkatan | https://raw.githubusercontent.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset/master/kamus_singkatan.csv | 18,9 KB | Normalisasi singkatan | Wajib |
| D3 | Kamus alay | https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/new_kamusalay.csv | 286 KB | Slang tambahan; baca dengan latin-1 | Wajib |
| D4 | Stopword NLTK | https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip | 37,7 KB | **Hanya** untuk kata topik dan word cloud | Wajib |
| D5 | Tokenizer NLTK | https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip | 4,3 MB | Pemotong kalimat | Wajib |
| D6 | SmSA | https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa/train_preprocess.tsv (juga `valid_preprocess.tsv`) | 2,19 MB untuk train | Latihan sentimen | Wajib |
| D7 | EmoT | https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/emot_emotion-twitter/train_preprocess.csv | 0,67 MB | Latihan; paling mirip tweet | Wajib |
| D8 | Ibrohim & Budi | https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/re_dataset.csv | 1,86 MB | Latihan multi-label; baca dengan latin-1 | Wajib |
| D9 | Komentar TikTok MBG | `MBG_PROCESSED.csv` di https://github.com/fidopecado/Indonesian-Tiktok-Comments-Sentiment-Analysis | tidak dicatat; 57.362 baris | Korpus latihan paling mirip; label otomatis, bukan gold | Wajib |
| D10 | Fakta MBG | https://github.com/andikapratamazaid/analisis-program-mbg | tidak dicatat | Bahan slide; buka sumber aslinya sebelum dipakai | Wajib |
| **E** | **Dokumentasi offline** | | | | |
| E1 | scikit-learn | https://scikit-learn.org/stable/_downloads/scikit-learn-docs.zip | 99,8 MB | Referensi | Wajib |
| E2 | pandas | https://pandas.pydata.org/docs/pandas.zip | 34,1 MB | Referensi | Wajib |
| E3 | NumPy, Matplotlib, Python | https://numpy.org/doc/stable/numpy-html.zip, https://matplotlib.org/stable/Matplotlib.pdf, https://www.python.org/ftp/python/doc/3.12.10/python-3.12.10-docs-html.zip | 36,5 + 46,5 + 13,2 MB | Referensi | Opsional |
| E4 | Docs Transformers dan sentence-transformers | `git clone --depth 1` lalu simpan folder `docs/` | tidak dicatat | Tidak ada zip resmi yang ditemukan | Opsional |
| E5 | Zeal + docset PyTorch | https://github.com/zealdocs/zeal/releases/tag/v0.9.2 | tidak dicatat | Docs PyTorch offline | Opsional |
| E6 | VSIX VS Code | `ms-python.python`, `ms-python.vscode-pylance`, `ms-python.debugpy`, `ms-toolsai.jupyter`, lewat menu Download VSIX | termasuk A3 | Ekstensi offline; JupyterLab sebagai cadangan | Opsional |
| **F** | **Model tambahan, disarankan** | | | | |
| F1 | Basis untuk Jawa/Sunda | `LazarusNLP/NusaBERT-base` | 443 MB | Fine-tune kalau campur bahasa daerah banyak | Opsional, disarankan |
| F2 | Basis umum | `indobenchmark/indobert-base-p1` | 498 MB, hanya `.bin` | Fine-tune teks berita; harus dikonversi | Opsional, disarankan |
| F3 | QA Indonesia | `Rifky/Indobert-QA` | 440 MB | Jaga-jaga kejutan seperti final 2021 | Opsional, disarankan |
| F4 | NLI Indonesia | `LazarusNLP/indobert-lite-base-p1-indonli-multilingual-nli-distil-mdeberta` | 47 MB | Pendapat kedua untuk zero-shot | Opsional, disarankan |
| F5 | Embedding cadangan CPU | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` + `model_quint8_avx2.onnx` | 471 + 118 MB | CPU AVX2 tanpa GPU | Opsional, disarankan |
| F6 | Sentimen pembanding | `crypter70/IndoBERT-Sentiment-Analysis`, `mdhugol/indonesia-bert-sentiment-classification` | 498 MB masing-masing | Menguji apakah label panitia buatan mesin | Opsional, disarankan |
| F7 | Hate speech dan profanity | `Exqrch/IndoBERTweet-HateSpeech`, `Exqrch/IndoBERTweet-Profanity` | 442 MB masing-masing | Task toksisitas | Opsional |
| F8 | Emosi | `StevenLimcorn/indonesian-roberta-base-emotion-classifier` | 499 MB | Task emosi | Opsional |
| F9 | Embedding lebih besar | `intfloat/multilingual-e5-base` | 1.112 MB | Head linear lebih kuat, 3–4x lebih lambat (E) | Opsional |
| F10 | Parafrase untuk SetFit | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 1.112 MB | SetFit pada sampel; repo penuh 10,8 GB, jadi jangan unduh semua | Opsional |
| F11 | Ringkasan | `panggi/t5-small-indonesian-summarization-cased` | 242 MB | Ringkasan sampel, tanpa skor di kartu | Opsional |
| F12 | POS | `w11wo/indonesian-roberta-base-posp-tagger` | 496 MB | Task POS | Opsional |
| F13 | Multibahasa | `FacebookAI/xlm-roberta-base` | 1.116 MB | Basis cadangan | Opsional |
| F14 | QA multibahasa | `timpal0l/mdeberta-v3-base-squad2` | 1.113 MB | QA; skornya bukan skor Indonesia | Opsional |
| F15 | Ringkasan besar | `cahya/t5-base-indonesian-summarization-cased` | 892 MB | Ringkasan; dilatih pada Liputan6 | Opsional |
| **G** | **LLM lokal, hanya kalau aturan mengizinkan** | | | | |
| G1 | Runtime CUDA | `llama-b12119-bin-win-cuda-12.4-x64.zip` + `cudart-llama-bin-win-cuda-12.4-x64.zip`, daftar aset di https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5 | 287 + 391 MB | Server GGUF, proses terpisah dari PyTorch | Opsional, disarankan |
| G2 | Runtime Vulkan | `llama-b12119-bin-win-vulkan-x64.zip`, sumber sama | 37,7 MB | Cadangan kalau CUDA gagal | Opsional, disarankan |
| G3 | LLM utama | `unsloth/gemma-4-E2B-it-GGUF` / `gemma-4-E2B-it-Q4_K_M.gguf` | 3.107 MB | Menamai topik, label sampel kecil | Opsional, disarankan |
| G4 | LLM hemat memori | repo sama / `gemma-4-E2B-it-Q3_K_M.gguf` | 2.537 MB | Kalau Q4 gagal di memori | Opsional |
| G5 | LLM pembanding | `aisingapore/Gemma-SEA-LION-v4.5-E2B-IT-GGUF` / `Gemma-SEA-LION-v4.5-E2B-IT-Q4_K_M.gguf` | 3.428 MB | Uji pada data slang; dibuang pertama kalau waktu unduh kurang | Opsional |
| G6 | Asisten kode | `Qwen/Qwen2.5-Coder-3B-Instruct-GGUF` / `qwen2.5-coder-3b-instruct-q4_k_m.gguf` | 2.105 MB | Bantuan kode offline | Opsional |
| G7 | Asisten kode kecil | `bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF` / `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf` | 986 MB | Kalau G6 terlalu berat | Opsional |
| G8 | Ekstensi Continue | VSIX `win32-x64` rilis `v1.2.23-vscode`, daftar aset di https://api.github.com/repos/continuedev/continue/releases?per_page=8 (nama file disimpulkan dari pola, cek dulu) | 74,5 MB | Menghubungkan VS Code ke server lokal | Opsional |
| **H** | **Data dan vektor tambahan** | | | | |
| H1 | Dataset IndoNLU ABSA | https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/casa_absa-prosa/train_preprocess.csv, https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/hoasa_absa-airy/train_preprocess.csv | 0,11 + 0,38 MB | Latihan ABSA | Opsional |
| H2 | IndoLEM dan NusaX | https://raw.githubusercontent.com/indolem/indolem/main/sentiment/data/train0.csv, https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/javanese/train.csv, https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/sundanese/train.csv | 430 KB; sekitar 84 KB per file | Latihan; penanda bahasa daerah | Opsional |
| H3 | IndoNLI | https://raw.githubusercontent.com/ir-nlp-csui/indonli/main/data/indonli/train.jsonl | 3,79 MB | Latihan NLI | Opsional |
| H4 | InSet | https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv, https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv | 41 + 83 KB | Fitur tambahan saja, **bukan pemberi label** | Opsional |
| H5 | Sentimen emoji | https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1048/Emoji_Sentiment_Data_v1.0.csv | 78,9 KB | Fitur emoji; emoji banyak di data komentar | Opsional |
| H6 | NRC EmoLex | https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip | 25,9 MB | Eksplorasi emosi; terjemahan Google Translate | Opsional |
| H7 | fastText | https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.vec.gz | 1.227 MB | Baseline vektor kata; jangan pakai `.bin` | Opsional |
| H8 | MBG TikTok mentah | https://www.kaggle.com/datasets/sinryurifal/dataset-komentar-tiktok-mbg-makan-bergizi-gratis | 48,4 MB | Korpus domain; CC BY-NC-SA 4.0 | Opsional |
| H9 | MBG YouTube | https://github.com/briannaufal211/mbg-youtube-sentiment-analysis, https://www.kaggle.com/datasets/dzikriraihan/indonesian-youtube-comments-mbg-program | tidak dicatat; 4,6 MB | Korpus domain; label bukan gold | Opsional |
| H10 | MBG Instagram | https://data.mendeley.com/datasets/gfsn8x62dk/1 | tidak dicatat; 14.051 komentar | Korpus domain | Opsional |
| H11 | MBG X berlabel manual | https://huggingface.co/datasets/tasyapndya/free_nutritious_meals_sentiment_x_id_2025 | sekitar 1.102 baris | Set kecil untuk uji kewarasan | Opsional |

**Total disk (estimasi).** Item wajib sekitar 6,7 GB, ditambah venv terpasang 8–11 GB (E), jadi **15–18 GB**. Ini sejalan dengan estimasi paket SSD 15–20 GB. Bila semua item opsional diunduh, totalnya sekitar **45–50 GB**. Sisihkan sekitar 10 GB lagi untuk checkpoint, embedding, dan output saat lomba, jadi **55–60 GB** paling banyak. D: dengan 283 GB aman, sedangkan C: dengan 33 GB jangan dipakai untuk gudang unduhan.

**Yang sengaja tidak masuk.** `BAAI/bge-m3` terlalu lambat, sekitar 2–4,6 jam per 500.000 baris (E). `google/embeddinggemma-300m` tertutup (gated) dan berlisensi Gemma. `jinaai/jina-embeddings-v3` berlisensi non-komersial. `cc.id.300.bin.gz` berukuran 4,5 GB terkompresi dan menjadi 7,2 GB, terlalu besar untuk RAM 8 GB. Sahabat-AI 9B Q4 berukuran 5,76 GB, juga terlalu besar. Enam ID berikut tidak bisa diakses secara anonim, jadi anggap tidak tersedia: `w11wo/indonesian-roberta-base-bapos-tagger`, `w11wo/indonesian-roberta-base-prdner-tagger`, `ayameRushia/indobert-base-p1-finetuned-indonlu-smsa`, `Exqrch/IndoBERTweet-Toxicity`, `qanastek/pos-indonesian`, `Wikidepia/IndoT5-small-paraphrase`. ID model sentimen pada repo TikTok MBG hanya disebut lewat nama akun (taufiqdp, Agufsamudra, Mdhugol, Aardiiiiy), jadi baca ID lengkapnya dari [repo itu](https://github.com/fidopecado/Indonesian-Tiktok-Comments-Sentiment-Analysis) saat mengunduh. Saya tidak menebak ID-nya.

## Sentimen atau stance paling mungkin, tetapi QA pernah muncul mendadak

Probabilitas di bawah ini **penilaian saya, bukan data**, dan tidak ada arsip kasus final resmi yang bisa saya temukan. Bobot nilai final GEMASTIK adalah **babak penyisihan 20%, skor leaderboard 30%, ketepatan metode 25%, dan presentasi 25%** ([Panduan 2026, hal. 27](https://kemahasiswaan.unand.ac.id/images/documents/Panduan-GEMASTIK-2026.pdf)). Dari sini saya menyimpulkan ada target yang bisa dinilai mesin, dan metode plus presentasi (50%) lebih berat daripada selisih kecil di leaderboard. Metrik leaderboard final GEMASTIK sebelumnya tidak ditemukan.

Preseden terpenting: pada GEMASTIK XIV, babak penyisihan berupa klasifikasi tweet vaksin, tetapi **finalnya berubah menjadi question answering**, dan tim perunggu ITB juga menemukan selisih besar antara skor internal dan skor resmi yang baru beres 10–30 menit sebelum tenggat ([ITB](https://itb.ac.id/news/persistence-become-the-success-key-for-team-nasigoreng-to-won-bronze-medal-at-gemastik-xiv/58223)). Waktu itu internet juga diizinkan. Panduan 2026 pun menyebut data terstruktur, gambar, video, dan teks, jadi premis "NLP" bersandar pada informasi kamu ([Panduan 2026](https://kemahasiswaan.unand.ac.id/images/documents/Panduan-GEMASTIK-2026.pdf)).

| Task | P | Metrik yang mungkin | Yang dilakukan |
|---|---|---|---|
| Sentimen atau stance, 3 kelas | 35% | Macro-F1 | TF-IDF word + char, decision layer per kelas, IndoBERTweet pada sampel, blend |
| Aspek atau topik, multi-class/multi-label | 20% | Macro/micro-F1, Hamming | Binary relevance, threshold per label yang disusutkan ke threshold global, leksikon aspek sebagai fitur |
| ABSA atau aspek + polaritas | 10% | Macro-F1 pasangan | Input `[aspek] [SEP] teks`, atau model per aspek |
| Deteksi biner: hoaks, relevansi, spam, buzzer | 10% | F1 kelas positif, PR-AUC | Threshold dari kurva OOF yang dihaluskan, fitur burst dan near-duplicate; jangan menuduh akun sebagai bot tanpa bukti |
| Ekstraksi/generasi: QA, NER, ringkasan, kata kunci | 10% | Exact match, entity F1, ROUGE | Baseline ekstraktif dulu (TF-IDF per kalimat), baru `Rifky/Indobert-QA` atau `cahya/NusaBert-ner-v1.3` |
| Emosi atau toksisitas | 5% | Macro-F1 | Pipeline sentimen, plus model siap pakai |
| Regresi engagement | 5% | RMSE/RMSLE, MAE | Target `log1p`, TF-IDF SVD + fitur waktu dengan LightGBM |
| Tanpa label, hanya laporan | 5% | Dinilai juri atau coherence | Embedding, topik sampel, garis waktu, rekomendasi |

Empat pengubah ini bisa menempel pada task apa pun (penilaian, tidak saling meniadakan): **label buatan mesin atau berisik 40%**, label jarang di bawah 10% baris 35%, pembagian test berbasis waktu atau sumber 30%, dan bagian topik/lonjakan yang diminta di slide 70%. Alasannya sederhana: pelabelan manual 500.000 baris tidak masuk akal, jadi label kemungkinan jarang atau dibuat mesin. Ini tebakan struktural paling penting. Tes cepatnya: latih model dangkal pada kata kunci dan bandingkan prediksi model publik dengan label panitia. Kalau kecocokannya hampir sempurna, label kemungkinan dibuat aturan atau model, dan menirunya bernilai besar ([CS-IF26](skills/ml-problem-solving/case-studies/ifest26-forged-headline.md)).

**Percabangan di gate T40** (ambang angka adalah penilaian). Periksa dulu izin pretrained, karena pelanggaran aturan pernah menghanguskan lima submission di IF26. Kalau pretrained dilarang, pakai TF-IDF, LightGBM, dan word2vec, tanpa BERT. Kalau **label banyak** (≥ 20.000 setelah dedup): TF-IDF semua baris, decision layer, IndoBERTweet pada sampel, lalu blend. Kalau **label sedikit** (sekitar 50–5.000): pisahkan gold, bandingkan TF-IDF dan LR di atas embedding, coba SetFit hanya kalau mengalahkan LR, tambahkan aturan kata kunci dan zero-shot pada sampel, self-training satu atau dua putaran, lalu latih TF-IDF pada baris yang yakin. Validasi hanya pada gold. Kalau **tanpa label**: dedup, embedding, topik, sentimen dan aspek lemah yang dinilai pada 150–300 gold buatan sendiri, garis waktu, dan rekomendasi. Pada 10 detik per baris (asumsi), 200 gold makan waktu sekitar 33 menit kalau sendirian. Kalau **pasangan teks atau task relasional**: TF-IDF biasa tidak cukup, karena di IF26 hanya memperoleh macro-F1 0,513 ([CS-IF26](skills/ml-problem-solving/case-studies/ifest26-forged-headline.md)).

## IndoBERTweet memimpin untuk teks sosial, tetapi belum ada yang diuji pada MBG

**Tidak ada model dengan skor terbit pada data MBG.** Skor di bawah berasal dari domain lain, kebanyakan dari validasi penulis kartu model, dan tidak bisa dibandingkan antarbaris kecuali dataset dan split-nya sama. Kolom "Laptop" adalah **estimasi saya, bukan pengukuran**: Y berarti bisa dijalankan pada semua baris, S berarti hanya sampel atau batch kecil.

| Task | Pilihan 1 | Cadangan | Bukti terbit | Laptop |
|---|---|---|---|---|
| Fine-tune teks sosial | `indolem/indobertweet-base-uncased` | `LazarusNLP/NusaBERT-base` | Rata-rata 86,1–86,5 di 7 dataset Twitter vs IndoBERT 83,4, mBERT 79,6 ([kartu](https://huggingface.co/indolem/indobertweet-base-uncased/raw/main/README.md)) | Y |
| Campur Jawa/Sunda | `LazarusNLP/NusaBERT-base` | IndoBERTweet | NusaX `jav`/`sun` 84,1/85,2 vs IndoBERT base 81,7/77,5 ([paper](https://ar5iv.labs.arxiv.org/html/2403.01817)) | Y |
| Teks berita | `indobenchmark/indobert-base-p1` | NusaBERT-base | SmSA 87,73, EmoT 75,48 ([IndoNLU](https://ar5iv.labs.arxiv.org/html/2009.05387)) | Y |
| Sentimen siap pakai | `w11wo/indonesian-roberta-base-sentiment-classifier` | `crypter70/IndoBERT-Sentiment-Analysis` | Test macro-F1 91,02 pada SmSA ([kartu](https://huggingface.co/w11wo/indonesian-roberta-base-sentiment-classifier/raw/main/README.md)) | Y |
| Emosi | `StevenLimcorn/indonesian-roberta-base-emotion-classifier` | `yogie27/IndoRoBERTa-Base-Emotion-Indonesian-Social-Media` | Akurasi 0,725, F1 0,7336 pada log validasi EmoT ([kartu](https://huggingface.co/StevenLimcorn/indonesian-roberta-base-emotion-classifier/raw/main/README.md)) | Y |
| Hate speech | `Exqrch/IndoBERTweet-HateSpeech` | `OwLim/indonesian-roberta-hate-speech-detection` | Akurasi 0,89, macro-F1 0,78, 10-fold ([kartu](https://huggingface.co/Exqrch/IndoBERTweet-HateSpeech/raw/main/README.md)) | Y |
| NER | `cahya/NusaBert-ner-v1.3` | `w11wo/indonesian-roberta-base-nerp-tagger` | F1 0,8171 pada NERGrit ([kartu](https://huggingface.co/cahya/NusaBert-ner-v1.3/raw/main/README.md)) | Y |
| Zero-shot | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | `LazarusNLP/indobert-lite-base-p1-indonli-multilingual-nli-distil-mdeberta` | mDeBERTa tanpa skor Indonesia; cadangan 74,69 test_lay ([kartu](https://huggingface.co/LazarusNLP/indobert-lite-base-p1-indonli-multilingual-nli-distil-mdeberta/raw/main/README.md)) | S |
| QA | `Rifky/Indobert-QA` | `timpal0l/mdeberta-v3-base-squad2` | EM 51,61, F1 69,09 pada SQuAD terjemahan ([kartu](https://huggingface.co/Rifky/Indobert-QA/raw/main/README.md)) | S |
| Ringkasan | `panggi/t5-small-indonesian-summarization-cased` | Ekstraktif | Tidak ada skor di kartu ([kartu](https://huggingface.co/panggi/t5-small-indonesian-summarization-cased/raw/main/README.md)) | S |
| Embedding | `intfloat/multilingual-e5-small` | `LazarusNLP/all-indo-e5-small-v4` | MIRACL 74,80, SmSA 83,6 ([LazarusNLP](https://raw.githubusercontent.com/LazarusNLP/indonesian-sentence-embeddings/main/README.md)) | Y |
| Hoaks | **Tidak ada** | Fine-tune kalau ada label | `theonegareth/IndoHoaxDetector`: 99,71% di split sendiri, tetapi **ROC-AUC 0,50** pada benchmark independen ([kartu](https://huggingface.co/theonegareth/IndoHoaxDetector/raw/main/README.md)) | n/a |
| ABSA | **Tidak ada** | Zero-shot pada sampel, lalu latih classifier | Hanya repo `Hamusssss12/spotify-absa-*` dengan kartu minim ([kartu](https://huggingface.co/Hamusssss12/spotify-absa-indonesian-v2/raw/main/README.md)) | S |

Abaikan skor tinggi dengan dataset tak bernama, misalnya F1 0,9646 pada `fathan/indo_sentiment_indobertweet` ([kartu](https://huggingface.co/fathan/indo_sentiment_indobertweet/raw/main/README.md)). `indobenchmark/indobert-lite-base-p1` menghemat disk dan VRAM, tetapi tidak menghemat waktu, karena ukuran hidden dan jumlah lapisannya sama dengan versi base ([config lite](https://huggingface.co/indobenchmark/indobert-lite-base-p1/raw/main/config.json)). Untuk preprocessing IndoBERTweet, ikuti cara model itu dilatih: huruf kecil, mention menjadi `@USER`, URL menjadi `HTTPURL`, emoji menjadi teks ([paper](https://ar5iv.labs.arxiv.org/html/2109.04607)).

**Berapa keunggulan BERT atas TF-IDF? Rentangnya dari sekitar +12 poin sampai negatif.** Karena itu ukur, jangan anggap pasti. Tidak ada sumber yang menunjukkan kurva belajar BERT pada teks Indonesia atau selisih TF-IDF pada 500.000 baris berlabel, dan portofolio kamu belum punya pengukuran fine-tuning BERT Indonesia yang bersih.

| Data | Klasik | Transformer | Catatan |
|---|---|---|---|
| NusaX Indonesia, macro-F1 | SVM 78,7 | IndoBERT base 90,0 | Sekitar +11 ([NusaBERT](https://ar5iv.labs.arxiv.org/html/2403.01817)) |
| SmSA, macro-F1 | fastText 82,13 (bukan TF-IDF) | Base 87,73 | +5,6 ([IndoNLU](https://ar5iv.labs.arxiv.org/html/2009.05387)) |
| Tweet MBG 54.105 baris, **akurasi** | SVM 83,39 | IndoBERT 87,60 | +4,2; sumber label tidak disebut ([JATI](https://ejournal.itn.ac.id/jati/article/view/15611)) |
| TikTok MBG, akurasi | SVM tuned 0,73 | IndoBERT 0,78 | +5; hanya dari cuplikan ([BINUS](https://journal.binus.ac.id/index.php/ijcshai/article/view/15184)) |
| Ulasan produk, macro-F1 | LinearSVC 0,5510 | IndoBERT 0,5088 | −4,2; hanya dari cuplikan ([ulasan benchmark](https://www.themoonlight.io/review/benchmarking-logistic-regression-svm-naive-bayes-and-indobert-fine-tuning-for-sentiment-analysis-on-indonesian-product-reviews)) |
| Chat slang templated (MT26) | LambdaRank 0,664 | Embedding 0,626 | Korpus template, jangan digeneralisasi ([CS-MT26](skills/ml-problem-solving/case-studies/minetoday26-ranking.md)) |

**Studi MBG terbit** dominan pada sentimen 3 kelas dan BERTopic. Ekspektasi realistis (inferensi) adalah **macro-F1 0,70–0,87 pada label bersih**, dan angka itu sering berasal dari test kecil. Contohnya IndoBERT akurasi 0,8910 dan macro-F1 0,8703 ([UPN Jatim](https://repository.upnjatim.ac.id/52506/)), lalu akurasi 70% dan macro-F1 0,7002 pada 5.509 komentar YouTube ([JSAI](https://jurnal.umb.ac.id/index.php/JSAI/article/download/10555/6282/46478)). Jangan menebak proporsi kelas: Drone Emprit melaporkan 50% positif sepanjang setahun, sementara INDEF melaporkan 85–97% negatif pada September 2025 ([kompilasi CSV, sumber sekunder](https://raw.githubusercontent.com/andikapratamazaid/analisis-program-mbg/main/data/sentimen_medsos_studi.csv)).

**Estimasi kecepatan, semuanya bukan pengukuran.** Belum ada benchmark terbit untuk GTX 1650. Skala di bawah memakai asumsi teks pendek, fp32, dan tidak ada throttling, jadi ambil batas bawahnya.

| Kerja | Setelan | 500.000 baris |
|---|---|---|
| Hashing/TF-IDF word + SGD | 1 core | 2–10 menit |
| TF-IDF char 3–5 | 2–5 GB RAM, pakai sampel | 5–20 menit |
| Embedding e5-small | len 64, batch 128–256 | 10–20 menit |
| Inferensi BERT-class | len 64, batch 64 | 35–70 menit |
| Zero-shot, 4 label | 2 juta pasangan | 4,3–9 jam: hanya sampel 20.000–50.000 |
| Fine-tune BERT-class, **50.000 baris, 1 epoch** | len 64, batch 16–32 | 12–24 menit |
| Sama | len 128, batch 8–16 | 24–46 menit |
| Fine-tune 500.000 baris | | 1,5–4 jam per epoch: **jangan** |

Estimasi LLM lokal: rencanakan **500–1.000 baris per 30 menit**. Angka terdekat yang terukur adalah 2,17 detik per panggilan pada GTX 1650 Ti Mobile dengan Ollama, bukan di laptop kamu ([DEV](https://dev.to/mrviduus/i-put-ollama-on-a-4-gb-mobile-gpu-and-got-25-heres-the-vram-math-3mhk)). Pakai LLM untuk menamai klaster dan label sampel saja. Gemma 4 E2B (64,13) dan SEA-LION v4.5 E2B (63,96) adalah model di bawah 10B dengan skor Indonesia terkuat di SEA-HELM per 18 Sep 2026, dengan asumsi kolom tanpa nama itu skor keseluruhan ([SEA-HELM](https://leaderboard.sea-lion.ai/detailed/ID)). Skor itu tidak memprediksi akurasi pada slang MBG. Hindari tag `gemma4:e2b` di Ollama, karena satu pengukuran menunjukkan 7,8 GB memori model ([DEV](https://dev.to/mrviduus/i-put-ollama-on-a-4-gb-mobile-gpu-and-got-25-heres-the-vram-math-3mhk)). Kalau LLM belum jalan setelah 30 menit, tinggalkan.

### Konflik antarcatatan dan keputusan saya

| Topik | Konflik | Pilihan | Alasan |
|---|---|---|---|
| Basis fine-tune | Satu catatan memilih IndoBERT-p1 sebagai default, catatan lain IndoBERTweet | **IndoBERTweet** | Bukti paling langsung untuk teks sosial; catatan IndoBERT-p1 sendiri menyebut ID alternatifnya belum diverifikasi. Berita: IndoBERT-p1 atau NusaBERT |
| Embedding | e5-small vs MiniLM-L12 | **e5-small** | Ukuran sama (471 MB) dan punya skor Indonesia; MiniLM tidak ada di tabel LazarusNLP, tetapi berguna sebagai cadangan CPU karena ada ONNX AVX2 |
| TF-IDF dulu vs BERT | Studi menunjukkan BERT lebih tinggi, portofolio menunjukkan keunggulan bisa hilang | **TF-IDF dulu untuk keamanan, BERT untuk peluang** | Yang pertama butuh menit dan memberi submission; yang kedua butuh GPU dan bisa gagal. Keduanya mengisi blend |
| Versi paket/torch | Dry-run BERTopic memilih torch 2.15.0 (CPU), NumPy 2.4.4, scikit-learn 1.8.0; dry-run penuh memilih NumPy 2.5.4, scikit-learn 1.9.1 | **`torch==2.14.1+cu130` + constraints + lock dari instalasi pertama** | Tanpa constraint pip mengambil torch CPU dari PyPI; lock file menjadi satu-satunya kebenaran |
| Stemmer | `Sastrawi` 1.0.1 (2016) vs `PySastrawi` 1.2.1 | **PySastrawi** | Yang ini diuji di laptop kamu |
| fastText | `fasttext-wheel` ada di daftar paket, tetapi bentrok dengan `fast-langdetect` | **Buang `fasttext-wheel`** | Uji lokal: `fast_langdetect` menolak jalan bersama paket itu; vektor cukup lewat gensim |
| Batas `.vec` | `limit=500000` vs `200000` | **200.000** | Lebih aman untuk 8 GB, 240 MB (aritmetika); naikkan kalau RAM cukup |
| Dataset HF | Satu catatan menyarankan menyimpan `indonlp/indonlu`, tetapi `load_dataset` gagal pada skrip | **GitHub raw atau parquet** | Uji lokal: `Dataset scripts are no longer supported` |
| Sampel BERTopic | 50.000–100.000 vs 30.000–50.000 vs 20.000–50.000 | **Mulai 20.000–50.000** | Tidak ada pengukuran RAM; naik hanya kalau latihan lolos |
| Freeze model | T255 vs T240 (dua catatan) vs T210 | **T240** | Slide ikut masuk berkas kumpulan dan presentasi 25%; dua catatan sudah bertemu di empat jam |
| Bobot kelas | Studi MBG memakai class weight; portofolio menunjukkan threshold lebih baik | **Mulai tanpa bobot + decision layer** | IF26: `[1,1]` + threshold mengungguli `[4,1]` sebesar +0,004 s.d. +0,011; tambahkan hanya kalau kelas langka kelaparan di pilot |
| Ukuran paket LLM | Batas "di bawah 5 GB" vs daftar minimal yang sekitar 9,4 GB | **Pakai daftar, buang SEA-LION dulu** | Disk bukan batasan di D:; waktu unduh yang lebih menentukan |
| Normalisasi slang | Studi kecil: +0,0609 macro-F1; uji lokal klasik: sekitar nol | **Saklar, satu uji saja** | Tidak ada uji transformer lokal, dan studi terbitnya kecil |

## Bangun environment dari wheelhouse, bukan dari venv riset

Rute utama adalah **wheelhouse + lock file + `pip --no-index`**, karena rute ini juga berjalan di perangkat panitia ([pip download](https://pip.pypa.io/en/stable/cli/pip_download/)). Menyalin venv hanya aman di mesin dan path yang sama. Daftar paket serupa di-resolve bersih menjadi 206 wheel, tetapi uji itu hanya membuktikan metadata cocok, bukan kode berjalan. **Daftar di bawah sudah saya ubah** (buang `snorkel`, `fasttext-wheel`, `flashtext`, `compress-fasttext`, `peft`, `optimum`, `catboost`; tambah `fast-langdetect`, `PySastrawi`), jadi hanya langkah 3 yang bisa membuktikannya. Perintah-perintah ini belum dijalankan utuh di laptop kamu.

**Langkah 0: cek hanya-baca.** Hasilnya menentukan disk mana yang dipakai dan apakah masih ada slot RAM.

```powershell
nvidia-smi
Get-PhysicalDisk | Select FriendlyName,MediaType
Get-CimInstance Win32_PhysicalMemory | Select BankLabel,Capacity,Speed,Manufacturer,PartNumber
```

**Langkah 1: susun folder dan daftar paket.**

```powershell
New-Item -ItemType Directory -Force D:\comp\wheelhouse, D:\comp\wheelhouse_alt, D:\comp\models, D:\comp\gguf, D:\comp\docs, D:\comp\installers, D:\comp\nltk_data, D:\comp\code, D:\comp\vsix, D:\comp\runs
@"
pandas
polars
pyarrow
numpy
scipy
scikit-learn
lightgbm
xgboost
transformers
datasets
accelerate
tokenizers
sentence-transformers
evaluate
setfit
bertopic
umap-learn
hdbscan
gensim
onnxruntime-gpu
cleanlab
optuna
matplotlib
seaborn
plotly
kaleido
wordcloud
networkx
jupyterlab
ipykernel
ipywidgets
tqdm
joblib
pyahocorasick
rapidfuzz
datasketch
emoji
fast-langdetect
PySastrawi
openpyxl
python-pptx
nltk
sentencepiece
tiktoken
protobuf
hf-xet
huggingface-hub
safetensors
psutil
nvidia-ml-py
"@ | Set-Content D:\comp\requirements.in -Encoding ascii
Set-Content D:\comp\constraints.txt "torch==2.14.1+cu130" -Encoding ascii
```

**Langkah 2: bangun wheelhouse (online).** Pastikan hanya ada satu `torch-*.whl` di `wheelhouse`. Tanpa constraint, pip mengambil wheel PyPI 124 MB yang menandakan build CPU, dan inilah jebakan yang menyebabkan status CPU-only sekarang.

```powershell
py -3.12 -m venv D:\comp\venv_build
D:\comp\venv_build\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip download pip setuptools wheel -d D:\comp\wheelhouse
python -m pip download "torch==2.14.1+cu130" -d D:\comp\wheelhouse --only-binary=:all: `
  --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
python -m pip download -r D:\comp\requirements.in -c D:\comp\constraints.txt `
  --find-links D:\comp\wheelhouse -d D:\comp\wheelhouse --only-binary=:all:
python -m pip download "torch==2.14.1+cpu" -d D:\comp\wheelhouse_alt --only-binary=:all: `
  --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
deactivate
```

**Langkah 3: pasang dengan Wi-Fi mati dan kabel LAN dicabut.** Setelah lock jadi, hapus `D:\comp\env`, pasang ulang dari lock, lalu reboot. Latihan baru sah kalau lolos setelah reboot.

```powershell
py -3.12 -m venv D:\comp\env
D:\comp\env\Scripts\python -m pip install --no-index --find-links D:\comp\wheelhouse --upgrade pip
D:\comp\env\Scripts\python -m pip install --no-index --find-links D:\comp\wheelhouse -r D:\comp\requirements.in -c D:\comp\constraints.txt
D:\comp\env\Scripts\python -m pip freeze > D:\comp\requirements.lock
D:\comp\env\Scripts\python -m pip check
```

Kalau verifikasi cu130 gagal, urutan cadangannya adalah 2.15.0+cu132, lalu 2.14.1+cu126. CUDA 13.x masih mencakup Turing ([PyTorch](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)). Wheel membawa DLL CUDA sendiri, jadi CUDA Toolkit tidak perlu dipasang ([ONNX Runtime docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)). Jangan mengganti driver di hari-hari terakhir.

**Langkah 4: unduh model (online, sebelum memuat `env.ps1`).** Helper di bawah memilih safetensors kalau ada, supaya tidak mengunduh dua format dan salinan TF/ONNX; ini belum pernah dijalankan.

```python
# D:\comp\dl.py
import sys
from huggingface_hub import HfApi, snapshot_download
repo, out = sys.argv[1], sys.argv[2]
files = HfApi().list_repo_files(repo)
w = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
snapshot_download(repo, local_dir=out, allow_patterns=["*.json", "*.txt", "*.model", w])
print(repo, "->", w)
```

```powershell
$py = "D:\comp\env\Scripts\python"
& $py D:\comp\dl.py intfloat/multilingual-e5-small D:\comp\models\e5-small
& $py D:\comp\dl.py indolem/indobertweet-base-uncased D:\comp\models\indobertweet
& $py D:\comp\dl.py w11wo/indonesian-roberta-base-sentiment-classifier D:\comp\models\sent-w11wo
& $py D:\comp\dl.py MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 D:\comp\models\mdeberta-nli
& $py D:\comp\dl.py cahya/NusaBert-ner-v1.3 D:\comp\models\nusabert-ner
& $py -m nltk.downloader -d D:\comp\nltk_data punkt_tab stopwords
D:\comp\env\Scripts\hf download unsloth/gemma-4-E2B-it-GGUF --include gemma-4-E2B-it-Q4_K_M.gguf --local-dir D:\comp\gguf
```

Cek tiap folder berisi `config.json`, tokenizer, dan bobot; untuk sentence-transformers juga `modules.json` dan `1_Pooling/config.json`. Kalau unduhan Xet macet, coba lagi dengan `$env:HF_HUB_DISABLE_XET="1"`.

**Langkah 5: konversi model `.bin` saat masih online.** `indobenchmark/indobert-base-p1` dan `mdhugol/indonesia-bert-sentiment-classification` tercatat hanya punya `.bin` dan `vocab.txt`; periksa juga model lain yang dicetak helper sebagai `pytorch_model.bin`. Apakah Transformers 5.22.1 masih memuat `.bin` tanpa syarat belum diperiksa, karena itu langkah ini ada. **Pakai kelas sesuai task:** `AutoModel` akan menghilangkan head classifier, jadi model sentimen harus memakai `AutoModelForSequenceClassification`.

```python
from transformers import AutoModel, AutoTokenizer
src, dst = r"D:\comp\models\indobert-base-p1", r"D:\comp\models\indobert-st"
AutoModel.from_pretrained(src).save_pretrained(dst, safe_serialization=True)
AutoTokenizer.from_pretrained(src).save_pretrained(dst)  # juga menulis tokenizer.json
```

**Langkah 6: `env.ps1`.** Muat di setiap shell dengan `. D:\comp\env.ps1`. Di notebook, set variabel yang sama di sel pertama sebelum import apa pun, karena variabel Hugging Face dibaca saat import ([HF docs](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)). Selalu muat model lewat path dengan `local_files_only=True`, jangan pakai `trust_remote_code=True`, dan gunakan `sklearn.metrics` alih-alih `evaluate.load`.

```powershell
$env:HF_HOME="D:\comp\hf"; $env:SENTENCE_TRANSFORMERS_HOME="D:\comp\hf\st"; $env:NLTK_DATA="D:\comp\nltk_data"
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"; $env:HF_DATASETS_OFFLINE="1"; $env:HF_EVALUATE_OFFLINE="1"
$env:HF_HUB_DISABLE_TELEMETRY="1"; $env:DO_NOT_TRACK="1"; $env:HF_HUB_DISABLE_UPDATE_CHECK="1"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
$env:PYTORCH_ALLOC_CONF="expandable_segments:True"; $env:TOKENIZERS_PARALLELISM="false"
```

**Langkah 7: verifikasi offline.** Socket guard membuat panggilan jaringan tersembunyi gagal dengan jelas. Kaleido butuh Chrome lokal; kalau guard mengganggu kaleido, uji kaleido terpisah dan andalkan matplotlib untuk grafik.

```python
# D:\comp\verify_offline.py
import socket
def _blocked(*a, **k): raise RuntimeError("NETWORK CALL ATTEMPTED")
socket.socket.connect = _blocked
import importlib, torch
mods = """pandas polars pyarrow numpy scipy sklearn lightgbm xgboost transformers datasets accelerate
tokenizers sentence_transformers evaluate setfit bertopic umap hdbscan gensim onnxruntime cleanlab optuna
matplotlib seaborn plotly wordcloud networkx jupyterlab ipykernel ipywidgets tqdm joblib ahocorasick
rapidfuzz datasketch emoji fast_langdetect Sastrawi openpyxl pptx kaleido nltk sentencepiece tiktoken""".split()
for m in mods:
    try: importlib.import_module(m); print("ok  ", m)
    except Exception as e: print("FAIL", m, repr(e)[:200])
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.rand(2000, 2000, device="cuda"); print("gpu matmul", (x @ x).sum().item())
from sentence_transformers import SentenceTransformer
m = SentenceTransformer(r"D:\comp\models\e5-small", local_files_only=True, device="cuda")
print(m.encode(["passage: makan bergizi gratis"]).shape)
import nltk; print(len(nltk.corpus.stopwords.words("indonesian")))
import plotly.express as px; px.bar(x=[1, 2], y=[1, 2]).write_image(r"D:\comp\kaleido_test.png"); print("kaleido ok")
```

Prefiks `query:` dan `passage:` untuk E5 berasal dari ingatan, jadi cek di kartu saat online. Tambahkan juga loop training 50 langkah dan `save_pretrained` ke latihan.

**Langkah 8: uji fp16, tetapi rencanakan fp32.** Laporan NaN fp16 terdokumentasi pada GTX 1660, bukan khusus 1650 ([PyTorch #58123](https://github.com/pytorch/pytorch/issues/58123)), dan satu pengguna GTX 1650 Ti melihat memori turun setengah tanpa tambahan kecepatan ([forum PyTorch](https://discuss.pytorch.org/t/no-speedup-with-autocast/148273)). Jadi fp16 hanya alat penghemat memori. Kartu ini juga tidak punya jalur bf16 native, jadi jangan pakai. mDeBERTa tidak mendukung fp16 ([kartu](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli)). Lulus kalau tidak ada NaN dan cosine di atas 0,999 (ambang saya), lalu 200 langkah `GradScaler` tanpa loss NaN.

```python
import torch
from transformers import AutoTokenizer, AutoModel
path = r"D:\comp\models\indobertweet"
tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModel.from_pretrained(path, local_files_only=True).cuda().eval()
batch = tok(["Makan bergizi gratis di sekolah sangat membantu orang tua."] * 16, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
with torch.inference_mode():
    a = model(**batch).last_hidden_state[:, 0].float().cpu()
    with torch.autocast("cuda", dtype=torch.float16):
        b = model(**batch).last_hidden_state[:, 0].float().cpu()
print("nan:", torch.isnan(b).any().item(), "min cosine:", torch.cosine_similarity(a, b).min().item())
```

**Langkah 9: Windows.** Jalankan sebagai admin sehari sebelumnya, dan pasang update tertunda **hari ini** supaya latihan berjalan setelah update. `LIDACTION` berasal dari ingatan, jadi cek dengan `powercfg /aliases`.

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0
powercfg /change disk-timeout-ac 0
powercfg /hibernate off
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
powercfg /setactive SCHEME_CURRENT
sc.exe stop wuauserv; sc.exe config wuauserv start= disabled   # kembalikan nanti: sc.exe config wuauserv start= demand
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks_throttle_reasons.active,power.draw,memory.used --format=csv -l 30 > D:\comp\gpu_log.csv
```

**Page file:** catatan riset menyarankan 32 GB pada SSD tercepat. **Kalau C: adalah satu-satunya SSD, jangan pakai 32 GB di sana**, karena C: hanya punya 33 GB kosong; pakai sekitar 16 GB (penilaian saya) atau biarkan system-managed. Page file mencegah crash batas commit, tetapi tidak membuat swap jadi cepat. Kalau proses aktif melakukan paging, kurangi datanya ([Microsoft](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/introduction-to-the-page-file)). Kalau D: ternyata HDD, pertimbangkan venv dan data kerja di C: (paket kerja sekitar 15 GB).

**Tambah RAM** kalau laptop kamu memang HP Pavilion Gaming 15-dk0xxx, seperti diasumsikan catatan riset (cek modelnya dulu). Jawaban komunitas HP yang konsisten adalah 2 slot SODIMM, maksimal 2 × 16 GB DDR4-2666 ([komunitas HP](https://h30434.www3.hp.com/t5/Gaming-Notebooks/Maximum-RAM-for-Upgrade/m-p/8034147)); panduan servis HP sendiri tidak berhasil diambil. Klaim 64 GB dari pemindai Crucial bertentangan, jadi abaikan. Jika hanya ada satu keping 8 GB, tambah satu keping 8 GB adalah jalan termurah ke 16 GB dual channel. Lakukan sebelum latihan, lalu jalankan Windows Memory Diagnostic. **Tunda pembelian sampai panitia menjawab soal laptop**: kalau perangkat panitia wajib dipakai, peningkatan ini tidak ada gunanya untuk hari H. RAM juga tidak menambah VRAM atau kecepatan training.

**Perangkat panitia.** Siapkan salinan di SSD eksternal dengan isi `installers`, `wheelhouse`, `wheelhouse_alt`, `models`, `nltk_data`, `docs`, `vsix`, `code`, `requirements.lock`, `env.ps1`, `verify_offline.py`. Urutan di tempat: cek `python --version`, `nvidia-smi`, RAM, dan disk; pasang Python 3.12.10 per pengguna kalau bukan 3.12; pasang lock yang sama kalau Turing atau lebih baru, cu126 kalau lebih tua, CPU kalau tanpa NVIDIA; lalu jalankan verifikasi. Tanpa GPU NVIDIA, andalkan TF-IDF linear dan embedding kecil lewat ONNX pada sampel (MiniLM AVX2, 100–300 teks/detik, estimasi keyakinan rendah). VC++ yang hilang tanpa hak admin adalah kasus tersulit, jadi tanyakan lebih awal. Kalau sempat, pinjam PC Windows kedua untuk latihan, karena hanya itu yang membuktikan tidak ada dependensi tersembunyi.

**Server LLM** berjalan sebagai proses terpisah di foldernya sendiri, jangan ditambahkan ke PATH, dan matikan sebelum training. Flag `-c`, `-t`, dan `--port` berasal dari ingatan, jadi cek `llama-server --help`; request-nya memakai `temperature: 0`, `chat_template_kwargs.enable_thinking: false`, dan `response_format` berisi JSON schema ([README llama-server](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md), [kartu SEA-LION](https://huggingface.co/aisingapore/Gemma-SEA-LION-v4.5-E2B-IT-GGUF)). Skema hanya menjamin format, bukan kebenaran, jadi sertakan kelas "tidak jelas".

```powershell
llama-server -m D:\comp\gguf\gemma-4-E2B-it-Q4_K_M.gguf -c 2048 -np 1 -t 6 --port 8080
```

## Lima jam dengan lima gate: skor aman dulu, slide tidak boleh sisa

Timeline ini **penilaian saya**, diadaptasi dari rencana catatan dengan freeze digeser ke T240. Waktu GPU dibatasi 90 menit, dan semua angka harus diperbarui setelah latihan. T adalah menit berjalan; kalau GEMASTIK, T0 adalah 13:00 WIB pada 11 Nov.

| T | Kerja | Keluaran atau gate |
|---|---|---|
| 0–10 | Baca semua dua kali, tulis kartu masalah, tanya aturan, jalankan `00_env_check`, salin data mentah read-only | Metrik, format, tenggat, batas submission tertulis |
| 10–25 | Muat ke parquet, EDA cepat, replika metrik | Sensus label, histogram panjang token |
| 25–40 | Dedup, grup, cek cara test diambil, fold/holdout | **G1 T40: pilih cabang task** |
| 40–70 | TF-IDF + linear (atau baseline cabang), validator, **submission pertama** | **G2 T70: selisih lokal vs platform masuk akal? Kalau tidak, perbaiki ini dulu** |
| 70–90 | Noise floor, decision layer, kurva belajar 5k/20k/80k/320k, baca 50 error acak; slot LLM kalau tanpa label | Batas penerimaan, file cadangan |
| 90–105 | Uji throughput BERT, pilot pada baris yang sama dengan TF-IDF | **G3 T105: lanjut, kecilkan, atau lewati BERT** |
| 105–195 | Satu run GPU dengan checkpoint; CPU di bawah 3–4 thread untuk audit error, grafik, dan draf slide | **G4 T150: proyeksi selesai tepat waktu dan validasi naik? Kalau tidak, potong langkah** |
| 195–225 | Inferensi test, blend pada holdout | **G5 T225: blend melewati batas? Kalau tidak, pakai model tunggal terbaik** |
| 225–240 | Decision layer bersarang, kandidat A (berprinsip) dan B (lindung nilai) | File tervalidasi + hash |
| **240** | **Freeze model, tidak ada model baru** | |
| 240–280 | Reproduksi pada sampel kecil, catatan metode, selesaikan slide | Kode, catatan, grafik |
| 280–290 | Unggah final, cek penerimaan, tangkapan layar, zip/rar | **Jangan menunggu menit terakhir** |
| 290–300 | Cadangan saja | Tidak mengubah kode |

**Aturan kill BERT:** kalau pada 25% anggaran waktu macro-F1 validasi di bawah TF-IDF yang dilatih pada baris yang sama dan tidak naik, kecilkan atau hentikan. Tetapi pilot hanya gerbang sumber daya, bukan vonis ilmiah, karena dry run yang dipangkas pernah membalik kesimpulan empat dari empat kali di [CDW1](skills/ml-problem-solving/knowledge/failed-experiments.md).

**Resep fine-tune (default, jangan jadikan eksperimen):** `max_len` dari histogram 20.000 sampel yang mencakup sekitar 95% baris, kemungkinan 64–128 (median komentar TikTok MBG 42 karakter, analisis sendiri); dynamic padding dan length bucketing; gradient accumulation sampai efektif 16–32; LR 2e-5; warmup 5–10%; LLRD 0,95 ([Sun dkk.](https://ar5iv.labs.arxiv.org/html/1905.05583)); validasi setiap 0,1–0,25 epoch pada 5.000–10.000 baris tetap; 1 epoch untuk ≥ 50.000 baris dan 2 hanya kalau keburu; `num_workers=0`; `save_total_limit=2`. Checkpoint dengan Adam sekitar 3× ukuran model, jadi simpan di D:. Pakai satu resep saja. Bobot, gradien, dan dua state Adam makan sekitar 4× ukuran file, yaitu sekitar 2 GB VRAM untuk IndoBERT sebelum aktivasi (aritmetika), jadi ruang tersisa sempit. Aritmetika kebutuhan waktu untuk 90 menit: 50.000 × 1 epoch butuh 9,3 baris/detik; 100.000 × 2 epoch butuh 37 baris/detik. Bandingkan dengan pengukuran kamu. Sisakan waktu inferensi, karena ukuran test belum diketahui.

**Validasi dan keputusan.** Duplikat bisa mengubah kesimpulan: pada 19 dataset sosial, peringkat model berubah pada 17 setelah penanganan duplikat ([Mu dkk.](https://ar5iv.labs.arxiv.org/html/2410.03545)). Di set TikTok MBG, **40,9% baris mengulang `comment_id` dan satu video memuat 71% komentar unik** (analisis sendiri). Laporkan skor random dan grup sejak baseline pertama, dan biarkan cek binomial pada pasangan yang muncul tepat dua kali memutuskan mana yang dipakai. Group CV bisa salah kalau test memang acak, seperti di IF26. Untuk macro-F1 multi-kelas, pakai pengali per kelas dengan coordinate ascent, bersarang, setelah blend. Di CDW2 lapisan ini ikut memberi +0,0112 jujur ([CS-CDW2](skills/ml-problem-solving/case-studies/cdw26-binary-optlevel.md)). Untuk biner, pakai threshold pada kurva OOF yang dihaluskan, dengan pemeriksaan kewarasan F1*/2 ([Lipton dkk.](https://arxiv.org/abs/1402.1892)). Blend multi-kelas dengan rata-rata probabilitas, biner dengan rata-rata rank; terima hanya kalau selisih melewati 2 SE dari noise floor. Jangan refit setelah threshold ditetapkan, dan jangan stacking kecuali metriknya log-loss.

**Jadwal latihan (kalau GEMASTIK, D-1 = 10 Nov).**

| Tanggal | Latihan | Lulus kalau |
|---|---|---|
| 5 Nov | Bangun environment, `verify_offline`, uji fp16 vs fp32 | Semua lolos dengan Wi-Fi mati |
| 6 Nov | Uji skala 500.000 baris sintetis (ulang korpus MBG dengan perturbasi, hanya untuk waktu, bukan skor); latihan 5 jam cabang berlabel pada SmSA/EmoT | Tabel waktu terisi, tanpa OOM, baseline sebelum T70 |
| 7 Nov | Latihan label sedikit: sembunyikan label, 200 gold | Tahu langkah mana yang menghasilkan |
| 8 Nov | Latihan tanpa label: topik, garis waktu, 10 slide; **uji RAM BERTopic** | Dek selesai, puncak RAM tercatat |
| 9 Nov | Task kejutan yang dipilih teman: QA, pasangan, atau regresi; bunuh dan lanjutkan training | Baseline dalam 60–90 menit; resume berhasil |
| 10 Nov | Bekukan kit, cetak runbook, uji seluruh stack di perangkat lomba | Semua model termuat dengan `local_files_only=True` |

**Kerja tim.** Kalau dua orang: Modeller pegang GPU, validasi, dan decision layer; Analyst pegang EDA, gold, audit error, grafik, slide, dan log submission. Kalau tiga orang: tambah Data/QA untuk dedup, aturan lemah, dan validator. Sinkron tiap 30 menit dan orang kedua memeriksa sebelum unggah final.

## Kit kode harus jalan satu perintah dan lolos uji Wi-Fi mati

Kit ini ada untuk menghilangkan waktu berpikir dan risiko environment. Semua script membaca satu config, dan menyimpan OOF, prediksi test, serta id supaya pertanyaan yang biasanya butuh 25 menit GPU bisa dijawab offline dalam 99 detik ([portofolio](skills/ml-problem-solving/knowledge/successful-experiments.md)). Pilihan script ini adalah penilaian; perlu diuji semua.

| Script | Isi | Uji sebelum hari H |
|---|---|---|
| `00_env_check.py` | GPU, CUDA op nyata, NaN fp16, RAM/disk, muat tiap model | Wi-Fi mati |
| `config.yaml`, `common.py` | Path, kolom, metrik, seed, logger | Dipakai semua script |
| `01_load_parquet.py` | Baca bertahap ke parquet, assert id dan duplikat | Varian CSV, JSONL, Excel |
| `02_fast_eda.py` | Null, sensus label, histogram token, probe bahasa, rentang waktu, n-gram, beda train-test | Skala 500.000 |
| `03_dedup_groups.py` | Masking, hash, MinHash LSH, konflik label, overlap train-test, cek binomial, cek klaster raksasa | Copypasta sintetis |
| `04_metric.py` | Macro/micro-F1, QWK, AUC, RMSE, NDCG, kustom | Uji vs sklearn |
| `05_tfidf_linear.py` | Word + `char_wb`, fallback hashing, LR/SGD/LinearSVC, kurva belajar, OOF | Tetap di bawah 8 GB |
| `06_finetune.py` | Loop berbatas waktu, multi-class/multi-label/regresi, **resume**, stop dini | Bunuh dan lanjutkan; pencari batch |
| `07_embed_memmap.py` | Embedding ke memmap dengan indeks selesai | Interupsi; assert kesejajaran setelah sort panjang |
| `08_head_on_embed.py` | LR/LightGBM/kNN pada embedding | Pada sampel |
| `09_weak_labels.py` | Aturan kata kunci, hit rate, template presisi, gold | Uji batas kata |
| `10_setfit_zeroshot.py` | SetFit; zero-shot sampel, fp32, hipotesis bahasa Indonesia | Muat offline |
| `11_topic.py` | Sampel lalu assign, fallback MiniBatchKMeans | Waktu dan **puncak RAM** |
| `12_decision_layer.py`, `13_blend.py` | Threshold, pengali, blend, noise floor dari `scripts/noise_floor.py` (salin, jangan ubah aslinya) | Prediksi tersimpan |
| `14_error_audit.py` | Error acak, salah yakin, shortlist cleanlab untuk **dibaca, bukan dihapus otomatis** | Output terbaca |
| `15_validate_submission.py` | Kolom, jumlah baris, urutan id, NaN, label valid, SHA-256 | Variasi format |
| `16_charts.py`, `slides_template.pptx` | Garis waktu, batang, heatmap aspek, confusion matrix; template 10 slide | Offline |
| `RUNBOOK.md` | Pohon keputusan, gate, perintah tempel, perbaikan error; **cetak** | Dipakai saat latihan |

**Baseline TF-IDF.** Lewati stopword removal: pada uji cepat di laptop kamu sendiri (bukan hasil publikasi, tanpa transformer, tanpa data MBG), stopword removal menurunkan macro-F1 pada ketiga dataset di **semua 10 fold** (−0,0493 SmSA, −0,0176 EmoT, −0,0124 Ibrohim). Sebaliknya, word + char tanpa preprocessing naik +0,0173 di EmoT dan +0,0100 di Ibrohim, sementara normalisasi slang berkisar −0,0033 sampai +0,0010 dan stemming sekitar netral. Daftar stopword NLTK identik dengan masdevid dan memuat `tidak`, `bukan`, `jangan`, `belum`, dan `kurang`. Pakai stopword dan stemming hanya untuk kata topik dan word cloud. Untuk stemming, cukup pada kosakata unik lalu dipetakan (36.346 kata sekitar 4,2 detik; pemetaan 500.000 dokumen 1,97 detik; satu run 4x lebih lambat, jadi rencanakan kasus terburuk). Ini semua uji lokal.

**500.000 baris di RAM 8 GB.** Jangan menahan DataFrame besar, LLM, dan training sekaligus; jalankan sebagai proses bergiliran. Mulai dari `HashingVectorizer(n_features=2**20, alternate_sign=False)` + `SGDClassifier(loss="log_loss")` dengan `partial_fit`, karena tidak punya kosakata dan rendah memori ([scikit-learn](https://scikit-learn.org/stable/computing/scaling_strategies.html)). Baru coba TF-IDF + `LinearSVC`, dan batasi `n_jobs=4` karena setiap proses menyalin data di Windows. Embedding 384 dimensi berukuran 768 MB (aritmetika), jadi tulis ke memmap. Pakai `Start-Process` dengan redirect log untuk job panjang, dan letakkan kode di dalam `if __name__ == '__main__':`, karena Windows memakai `spawn` ([PyTorch](https://docs.pytorch.org/docs/2.15/data.html)).

**BERTopic pada 500.000 baris belum terbukti di 8 GB.** Benchmark memori topik **ditulis tetapi tidak pernah dijalankan**, dan instalasi nlp-id, BERTopic, UMAP, serta HDBSCAN terhenti di tengah, jadi tidak ada angka RAM atau waktu lokal untuk komponen topik mana pun. Tidak ada benchmark publik pada 500.000 dokumen dan 8 GB; dokumen HDBSCAN hanya mengekstrapolasi waktu dan tidak memberi angka memori ([HDBSCAN docs](https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html)). **Jadi ini wajib dilatih, bukan diasumsikan.** Urutan yang disarankan (belum diuji): dedup, embedding sekali lalu simpan, UMAP + HDBSCAN hanya pada sampel 20.000–50.000, lalu assign sisanya. Set `low_memory=True`, `calculate_probabilities=False`, dan `min_df` tinggi, serta buang stopword setelah klaster ([BERTopic FAQ](https://maartengr.github.io/BERTopic/faq.html)). Kalau melewati batas, pakai IncrementalPCA + MiniBatchKMeans ([docs online](https://maartengr.github.io/BERTopic/getting_started/online/online.html)), meski satu sumber lemah melaporkan mutu lebih rendah. Hindari OCTIS karena mematok `scikit-learn==1.1.0` ([requirements](https://raw.githubusercontent.com/MIND-Lab/OCTIS/master/requirements.txt)), dan RAPIDS/cuML tidak tersedia di Windows 10 ([RAPIDS](https://docs.nvidia.com/datascience/install/)). Laporkan topik sebagai tema, bukan klaster tegas, karena silhouette di Compfest25 sekitar −0,005 ([portofolio](skills/ml-problem-solving/case-studies/early-and-misc-projects.md)).

**Dek 8–10 slide** (penilaian): jawaban satu kalimat; data dan metode termasuk dedup; garis waktu dengan kejadian; peta tema; heatmap aspek × sentimen; dua pendalaman dengan kutipan; validasi gold dan stabilitas; tiga rekomendasi dengan pemilik dan metrik; batasan (media sosial bukan populasi, korelasi bukan kausalitas). Sentimen + topik sudah umum pada studi MBG ([UMN](https://kc.umn.ac.id/id/eprint/47931/), [UPN](https://repository.upnjatim.ac.id/52506/)), jadi pembedanya adalah pemisahan volume asli vs copypasta, skor tervalidasi gold, dan pemisahan "menolak program" dari "minta diperbaiki" ([JSAI](https://jurnal.umb.ac.id/index.php/JSAI/article/download/10555/6282/46478)).

## Pertanyaan untuk panitia: tiga pertama bisa mengubah seluruh rencana

Kirim sekarang kalau bisa, minta jawaban tertulis, dan ulangi saat setup atau technical meeting.

| Pertanyaan | Kalau jawabannya mengubah apa |
|---|---|
| Laptop sendiri atau perangkat panitia? GPU, RAM, hak admin, Python? USB/SSD boleh? Kapan file boleh dipindah? | Jalur portabel, beli RAM atau tidak, pakai CPU atau GPU |
| Pretrained boleh (IndoBERT, sentence-transformers, LLM lokal)? Harus dideklarasikan atau publik sebelum tanggal tertentu? | Larangan menghapus seluruh cabang BERT |
| Internet benar-benar mati, atau hanya untuk submission? Bagaimana cara submit? Asisten AI boleh? | Cara mengirim, peran LLM |
| Data eksternal, leksikon, stopword, gazetteer, korpus yang sudah diunduh boleh dipakai? | Item D, H, dan label lemah |
| Test boleh dipakai untuk fit TF-IDF, self-training, dedup? Hand-label boleh, termasuk baris test? | Pseudo-label; default saya tidak melabeli baris test |
| Batas submission, pembagian public/private, submission mana yang dihitung, tie-break? | Frekuensi submit; dua file final |
| Kode harus bisa dijalankan ulang? Batas runtime? Format? | Kedisiplinan seed dan paritas |
| Rubrik, durasi presentasi, bahasa, tenggat PPT, boleh diedit semalam sebelum 12 Nov? | Bagi waktu T240–280 |
| Skema data, bahasa, sumber, jenis label, cara label dibuat? Dari mana info MBG dan 500.000 baris? | Cabang task, tes label mesin |
| Stop kontak, ekstensi, ponsel, catatan cetak, jumlah laptop tim? | Daya dan runbook |

## Domain MBG: empat rezim wacana dan leksikon yang harus diaudit

**Wacana punya perubahan rezim yang jelas, jadi model pada satu jendela waktu akan drift:** peluncuran 6 Jan 2025 dengan 190 SPPG dan 570.000 penerima ([kompilasi BGN](https://raw.githubusercontent.com/andikapratamazaid/analisis-program-mbg/main/data/perkembangan_program.csv)); puncak keracunan Sep–Okt 2025; guncangan korupsi dan kepemimpinan Jun–Jul 2026; lalu gelombang kedua dan debat anggaran Agu–Okt 2026. **Leksikon yang menyebut Dadan Hindayana sebagai kepala BGN sudah usang: kepala sekarang Sudaryono** ([Bisnis](https://kabar24.bisnis.com/read/20260722/15/1990142/resmi-dilantik-jadi-kepala-bgn-ini-langkah-awal-sudaryono)). Catatan: [Kompas](https://nasional.kompas.com/read/2026/10/30/08004121/100-hari-sudaryono-pimpin-bgn-bongkar-yayasan-calo-hingga-janji-transformasi) menulis ia menggantikan Dadan langsung, melewatkan Nanik S. Deyang yang memimpin sementara ([afu.id](https://afu.id/nasional/warisan-di-balik-kursi-bgn-untuk-sudaryono)).

| Tanggal | Peristiwa untuk fitur lonjakan | Sumber |
|---|---|---|
| 17–22 Sep 2025 | Garut 569; Banggai Kepulauan 339; Bandung Barat 1.333 (hitungan lain berbeda) | [CSV insiden](https://raw.githubusercontent.com/andikapratamazaid/analisis-program-mbg/main/data/insiden_terbesar.csv) |
| 24 Sep 2025 | Bandung Barat menetapkan KLB; BGN 70 kasus, 5.914 penerima (Jan–Sep) | [Republika](https://en.republika.co.id/berita/t377zb415/nearly-6000-students-experience-food-poisoning-during-the-free-nutritious-meal) |
| Okt 2025 | Puncak 85 kasus, lalu 40, 12, dan 10 pada Jan 2026 | [Tirto](https://tirto.id/bgn-ungkap-jumlah-keracunan-mbg-turun-hanya-10-kasus-di-januari-hpxM) |
| 2–3 Jun 2026 | Dadan dan dua deputi dicopot; Kejagung menggeledah; tiga orang jadi tersangka. **Masih dugaan** | [Jawa Pos](https://jawapos.com/kasuistika/2606030202/mantan-kepala-bgn-dadan-hindayana-resmi-jadi-tersangka-dan-ditahan-kejagung-belum-24-jam-usai-dicopot-prabowo) |
| 22 Jul 2026 | Sudaryono dilantik | [Bisnis](https://kabar24.bisnis.com/read/20260722/15/1990142/resmi-dilantik-jadi-kepala-bgn-ini-langkah-awal-sudaryono) |
| Jul–Agu 2026 | Anggaran 2027: Rp174T (Banggar), Rp270,2T indikatif (BGN), Rp240T sementara | [Bisnis](https://ekonomi.bisnis.com/read/20260706/10/1985932/banggar-dpr-estimasi-anggaran-mbg-2027-turun-35-jadi-rp174-triliun), [Bisnis](https://ekonomi.bisnis.com/read/20260721/12/1989901/bgn-sebut-pagu-anggaran-2027-rp270-triliun-belum-final), [Katadata](https://katadata.co.id/berita/nasional/6a7f145ed65f1/pemerintah-siapkan-anggaran-mbg-2027-rp-240-triliun-bidik-70-juta-penerima) |
| 3 Sep 2026 | 5.482 korban, 71 insiden sejak 22 Jul; lebih dari 80% pelanggaran SOP | [Katadata](https://katadata.co.id/berita/nasional/6a9a5b8dc9d81/5482-korban-keracunan-mbg-sejak-akhir-juli-2026-bgn-soroti-pelanggaran-sop) |
| 29 Okt 2026 | Laporan 100 hari, "yayasan calo", "MBG 2.0"; angkanya tidak saling cocok di artikel | [Kompas](https://nasional.kompas.com/read/2026/10/30/08004121/100-hari-sudaryono-pimpin-bgn-bongkar-yayasan-calo-hingga-janji-transformasi) |

**Jangan gabungkan angka yang diperselisihkan.** Korban keracunan: BGN 11.640 (Nov 2025), Kemenkes 50.059 dalam 560 kejadian (2 Sep 2026), JPPI 43.838, KPAI 12.658 anak pada 2025 ([README kompilasi](https://github.com/andikapratamazaid/analisis-program-mbg)). Tampilkan per lembaga dengan tanggal. Survei kepuasan juga bukan satu tren karena pertanyaan dan sampelnya berbeda. Bantahan hoaks seperti dana SPPG dihentikan ([Jawa Pos](https://jawapos.com/nasional/2606040257/isu-penyaluran-dana-sppg-dihentikan-dipastikan-hoaks-bgn-tegaskan-layanan-tetap-berjalan-normal)) atau siswa meninggal di Aceh ([Antara](https://www.antaranews.com/berita/5447710/hoaks-bgn-informasi-siswa-meninggal-karena-mbg-di-aceh-tidak-benar)) umumnya berasal dari pihak berkepentingan, yaitu BGN. Taksonomi 19 topik JSAI bisa dipakai untuk penamaan, misalnya keracunan siswa, tuntutan penghentian, korupsi pejabat, manajemen dapur dan yayasan, serta anggaran negara ([JSAI](https://jurnal.umb.ac.id/index.php/JSAI/article/download/10555/6282/46478)). Bukti koordinasi masih lemah: pada 12.000 komentar YouTube ditemukan 0 kandidat ketat, dan studinya menegaskan itu bukan hitungan bot ([GitHub](https://github.com/briannaufal211/mbg-youtube-sentiment-analysis)).

**Jebakan data yang perlu diduga** (inferensi): duplikat antar-train dan test, template copypasta, baris hanya emoji dan `[Sticker]`, makna lain dari singkatan "MBG" atau spam judi, campuran Jawa dan Sunda, sarkasme, dan noise label. Deteksi bahasa gagal untuk Jawa dan Sunda pada uji lokal 6 kalimat, jadi daftar kata penanda lebih masuk akal (belum diuji).

**Leksikon awal.** Ini hipotesis untuk label lemah, bukan kebenaran. Tag: **[D]** terlihat di CSV TikTok publik; **[P]** dari paper atau berita; **[I]** saran sendiri, belum diverifikasi; tanpa tag berarti begitu adanya di catatan. Setiap aturan harus diaudit dengan jumlah hit dan presisi 30 sampel acak, karena di MT26 kata `github` dan `nlp` punya 0 hit, sedangkan `git` hanya mencocokkan "gitu" ([CS-MT26](skills/ml-problem-solving/case-studies/minetoday26-ranking.md)). Jangan menilai aturan dengan label lemah lain: nilai pada gold yang tidak pernah dipakai menulis aturan.

| Aspek | Kata kunci |
|---|---|
| keamanan_pangan | [D] keracunan, racun, basi, bau, korban, layak, tidak layak, hiu (arti belum terverifikasi); [P] klb, puskesmas, sop, e coli, salmonella; [I] ulat, belatung, berjamur; tanpa tag: mual, muntah, diare, pusing |
| kualitas_menu | [D] menu, menunya, burger, ayam, nasi, sayur, susu, kotak, roti, telur/telor, tempe, tahu, ikan, kacang, mie, gula, buah, pisang, jeruk, enak, rasa; [P] ompreng; [I] ultra processed/upf, porsi |
| gizi_kesehatan | [D] gizi, gizinya, bergizi, ahli gizi, dokter, sehat; [I] stunting, protein, kalori, akg, obesitas |
| anggaran | [D] anggaran, dana, uang, buang, pendidikan, mending; [P] apbn, rapbn, pajak, triliun, pagu, efisiensi, pemborosan, refocusing |
| korupsi_tata_kelola | [D] korupsi, ladang, lahan, ajang, berjamaah, proyek, pejabat, untung; [P] koruptor, nepotisme, kolusi, markup/mark up, tersangka, kejagung, jual beli titik, titik sppg, yayasan, calo, transparan |
| implementasi_dapur | [D] sppg, dapur, ditutup/tutup; [P] kepala sppg, mitra, distribusi, kapasitas, suspend, investigasi, sanitasi; [I] terlambat, tepat waktu, slhs, relawan |
| pendidikan_guru | [D] guru, gaji, honorer, murid, siswa, sekolah, kantin, bawa pulang (D-adjacent); [P] santri; [I] pip |
| ekonomi_umkm | [D] umkm, lapangan kerja/lapangan, kerja, pekerja, pedagang, jualan, kantin, catering, harga, naik; [I] petani, peternak, susu lokal |
| politik | [D] prabowo, wowo, dpr, tni, polri, polisi, purnawirawan; [P] gibran, janji kampanye, tentara, militer, purbaya, zulhas, sudaryono, nanik, dadan, ombudsman, jppi, kpai; [I] icw, cisdi |
| penolakan | [D] stop, tutup, bubar, evaluasi, ganti, mending; [P] bubarkan; tanpa tag: setop, hentikan |
| dukungan | [D] bagus, setuju, program bagus (D-adjacent); [P] semoga; [I] lanjut, terima kasih/makasih, alhamdulillah |
| media/hoaks/buzzer | [D] hoax, buzzer, influencer, narasi, digoreng, pendukung, fanatik; [P] hoaks, komdigi, klarifikasi; [I] fitnah, bot |
| Normalisasi ejaan [D] | yg, gk, gw, bgt, tp, klo, udh, dpt, jg, jd, bnyk, bnyak, karna, krn, utk, tdk, jgn, org, dr, anak2, daerah2, dkorup, pejbat, jdi, lgi, bnr, byar |
| Jawa/Sunda | [D] ndk, madak, joss; [I] ora, ndak, mboten, wes, sing, teu, henteu, atuh, euy |

Kamus-alay mengubah 13,1% token pada EmoT dan 9,7% pada Ibrohim, tetapi hanya 0,6% pada SmSA (uji lokal). Istilah MBG seperti SPPG dan BGN kemungkinan tidak tercakup kamus lama (belum diuji), jadi buat daftar kecil dari data saat hari H. **Jangan pakai InSet sebagai pemberi label**: pada uji lokal, aturan InSet saja memberi macro-F1 0,24–0,35 pada SmSA dan melabeli 76–91% baris sebagai negatif, karena `yang`, `tidak`, dan `sangat` berbobot −5, sedangkan `enak` ada di kedua daftar. Sebagai fitur, efeknya hanya +0,0053 di SmSA dan sekitar nol di dua dataset lain. Fitur periode (sebelum Sep 2025, krisis Sep–Okt 2025, skandal Jun–Jul 2026, gelombang Agu–Okt 2026) masuk akal, tetapi cek leakage dan drift kalau test dibagi berdasarkan waktu.

## Hal yang harus dihindari

| Jangan | Alasan |
|---|---|
| Berasumsi pretrained, data eksternal, atau LLM diizinkan | Panduan tidak mengaturnya; IF26 kehilangan lima submission karena larangan pretrained |
| Fine-tune 500.000 baris, 5-fold BERT, atau zero-shot semua baris | 1,5–4 jam per epoch; 4,3–9 jam untuk 4 label (estimasi); LLM sekitar 139 jam pada 1 detik per baris (aritmetika) |
| Menjalankan LLM dan training GPU bersamaan | Risiko OOM di kedua sisi; belum ada pengukuran berbagi kartu 4 GB |
| Menganggap fp16 aman, atau memakai bf16 | NaN pada keluarga GTX 16; belum teruji di 1650 |
| Memakai model hoaks siap pakai sebagai pemeriksa fakta | ROC-AUC 0,50 pada benchmark independen; model belajar gaya penerbit |
| Menghapus stopword atau stem sebelum classifier | Uji lokal negatif; studi transformer juga mendukung pembersihan ringan ([JTIIK](https://jtiik.ub.ac.id/index.php/jtiik/article/view/8315)) |
| Menghapus label mencurigakan secara otomatis | SD26 membuang 186 gambar sulit tetapi benar ([portofolio](skills/ml-problem-solving/knowledge/failed-experiments.md)); pakai cleanlab untuk prioritas baca |
| Melabeli baris test atau menyetel threshold darinya | Risiko kepatuhan ([portofolio](skills/ml-problem-solving/knowledge/lessons-learned.md)) |
| Mengejar gerakan kecil leaderboard | Noise; CDW2 punya public LB ±0,009 |
| Mengedit submission di Excel | Merusak id dan encoding |
| Mengunduh repo utuh, atau mengandalkan `load_dataset` skrip | Repo bisa 2–10× lipat ukuran; skrip dataset sudah gagal |
| Memasang atau mengubah driver, undervolt, repaste, atau paket baru di H-1 | Crash lebih mahal daripada panas; bukti termal hanya anekdot |
| Menyebut akun sebagai bot, atau menyebut tersangka bersalah | Tidak ada bukti; kasus korupsi masih dugaan |
| Menjadikan VS Code atau LLM titik gagal tunggal | JupyterLab dan docs lokal harus cukup |

## Catatan kebersihan riset

Selama riset, venv terisolasi dan unduhan dibuat di scratchpad sesi, dan beberapa entri dataset masuk ke cache Hugging Face di `~\.cache\huggingface`. Python global tidak diubah. **Instalasi H-1 tidak boleh bergantung pada venv scratchpad itu**: bangun ulang dari wheelhouse di `D:\comp`. Entri cache boleh kamu hapus kalau mau. Saat latihan, pastikan tidak ada yang diam-diam memakai cache lama, karena itu bisa menyembunyikan file yang hilang.

## Kesimpulan

Lomba ini kemungkinan tidak dimenangkan oleh keputusan model di menit ke-200, melainkan oleh pertanyaan yang kamu kirim hari ini dan oleh latihan pekan ini. Selisih antara IndoBERTweet dan pesaing dekatnya kecil dibanding selisih antara pipeline yang teruji dan pipeline yang gagal karena satu DLL, satu ID salah, atau satu OOM. Temuan yang paling mengubah prioritas adalah kemungkinan perangkat panitia: kalau benar, investasi terbaik beralih dari menyetel laptop ke membuat paket yang berjalan di mesin asing.

Struktur nilai juga mengubah cara memaknai "performa". Dengan 80% nilai masih terbuka dan hanya 30% dari leaderboard (kalau GEMASTIK), kerja yang menaikkan skor dan kerja yang menaikkan kepercayaan juri sering sama: dedup yang jujur, validasi yang sesuai cara test diambil, gold set, dan rekomendasi yang tidak mencampur angka antarlembaga. Semua estimasi di laporan ini punya masa berlaku singkat. Latihan pertama dengan data 10.000 baris dan satu run 30 menit di bawah beban akan menggantinya dengan angka yang benar-benar milik kamu.
