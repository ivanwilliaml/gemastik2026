# Kontrak toolkit modular v2

## Memilih komponen

`index.html#/compose` dan `tools/compose.py` memakai dependency graph yang sama.
Definisi fungsi masuk sekali, sebelum pemakaian. Prasyarat model ikut aktif:
memilih RRF otomatis memasang BM25 + E5. Melepas BM25 juga melepas RRF/reranker
yang membutuhkannya. Tidak ada import `toolkit` di notebook keluaran.

`CFG` adalah satu konfigurasi. Di sel pemakaian cukup `data = split_supervised(...)`
dan `result = run_supervised(data, CFG)`. Fungsi tetap bisa diambil sendiri dengan
dependensinya. Daftar langkah tidak memaksa eksperimen berat: model yang tidak
dipilih tidak dilatih. Semua pretrained bertumpu GPU Kaggle; baseline bisa CPU Kaggle.

## Input dan output

| Kasus | Input | Kontrak output model |
|---|---|---|
| Klasifikasi | CSV text,label; group_col opsional | `(validation_probability, holdout_probability)` n×K, kelas mengikuti `data['labels']` |
| Multilabel | text dan kolom label bernilai 0/1/NaN | n×L sigmoid independen. Unknown dikecualikan dari loss/metrik |
| STS | text_1,text_2,score kontinu | n skor dalam `score_range`, tidak dikonversi menjadi kelas |
| NLI | premise,hypothesis,label | n×3: entailment, neutral, contradiction; e/n/c didukung; ID numerik harus dipetakan eksplisit |
| Retrieval | corpus doc_id,text; queries query,relevant_ids JSON list | Ranking index dokumen, deduplikasi ID wajib. Banyak positif diperhitungkan |
| Topic/clustering | text | IDs grup/topik model beku + kata kunci + contoh audit |
| Tren waktu | text,published_at | Proporsi tiap periode + support + JS distance + sentimen sampel |

Model baru: tambahkan fungsi `@register('nama')` dengan argumen `(data,cfg)`;
latih hanya dengan `data['train']`, checkpoint di `data['val']`, dan keluarkan
prediksi validation/holdout tanpa membaca target holdout. Tambahkan descriptor di
`compose_lib.py`, lalu `python toolkit/tools/build_library.py`. `model_paths`
menerima path bundle offline maupun model ID. `offline=True` mengaktifkan
`local_files_only`, bukan mengunduh diam-diam.

## Preprocessing dan validasi

- Native: perbaikan encoding terkendali, NFC, entity, invisible, whitespace. Tanpa stemming atau menghapus negasi/emoji/angka.
- IndoBERTweet: lowercase, HTTPURL, @USER, emoji demojize, sesuai [model card](https://huggingface.co/indolem/indobertweet-base-uncased).
- Normalized: hanya kamus `slang` yang diberikan. Tidak ada kamus sosial yang dianggap selalu benar untuk domain lain.
- BoW: token kata, URL/mention dibuang; `stopwords` opsional dengan negasi terlindungi. Encoder topic/sentimen menggunakan teks native.
- Group pengguna + duplikat native + collision normalisasi disatukan sebelum split. Pair memakai connected endpoints; satu teks di kedua sisi tetap satu grup. Ini dapat membentuk komponen besar: audit proporsi split, bukan memecah grup untuk mengejar ukuran.
- Multilabel: iterative stratification pada profil grup positive/known, tidak mengisi unknown target sebagai 0.
- Tren: early/middle/late berdasarkan periode. Volume duplicate tetap dihitung; recurrence lintas waktu ditampilkan. Hanya data early unik yang melatih vocabulary/model, bukan semua periode.

Tidak ada near-duplicate fuzzy otomatis di v2 modular. Untuk kampanye/parafrase
massal, tambahkan grup near-duplicate sebelum split, misalnya dari notebook review
multiclass. Exact grouping saja tidak menjamin semua kebocoran semantik hilang.

## Pemilihan dan kelemahan model

Supervised: cross-fit GroupKFold di validation untuk threshold/bias/temperature/
isotonic. Pilihan model dan checkpoint juga memakai validation sehingga skor
pengembangan tetap memiliki selection bias. Holdout baru dinilai setelah
`selection_frozen_before_test.json` ditulis. Jangan mengganti pilihan setelah
melihat tabel holdout. Ulang seed/group split untuk peningkatan kecil.

Multilabel: F1/Hamming memakai label diketahui; exact-match hanya baris complete.
AP memakai probabilitas asli, bukan keputusan threshold. Label minim support
memakai cut .5. Tidak memaksa satu label aktif. Hierarki dan aturan manual berupa
fungsi opsional; perlu audit train dan uji validation sebelum dipasang. Duplicate
consistency memakai teks dan prediksi batch saja. Scalar temperature memperbaiki
probabilitas dan tidak mengubah argmax.

Retrieval: TF-IDF, BM25, E5 chunks, RRF, reranker. Reranker memakai awal dokumen
dengan truncation; untuk relevansi di bagian akhir perlu chunk-aware reranking
seperti notebook retrieval review. Query sama dikelompokkan sebelum split.
MRR/Recall/NDCG memerlukan judgment cukup lengkap; negatif tak berlabel belum
tentu tidak relevan. Template ini mengeluarkan ranking dan hit, bukan classifier
binary matching terkalibrasi. Kueri ringkasan-artikel hanya proxy mudah.

Topic/clustering: NPMI/top-word diversity/silhouette adalah diagnostik internal.
Contoh per topik tetap harus dibaca; internal metric tidak boleh diberi nama accuracy.
Model bisa disimpan sendiri bila diperlukan; keluaran default menyimpan tabel,
assignment, kebijakan dan konfigurasi, tanpa mengharuskan paket serialization khusus.
Sentimen lexicon hanya heuristik. Pretrained ditambahkan lewat pilihan komponen;
tanpa gold hanya agreement, bukan accuracy. Sampel seragam per periode bukan
representasi volume keseluruhan tanpa pembobotan. Perubahan waktu bukan sebab-akibat.

## Jejak sumber dan uji

Setiap notebook menyimpan path dan SHA256 kode sumber review di metadata.
Hubungan adalah **adaptasi**, bukan klaim identik dengan run sumber. Build tidak
mengeksekusi generator notebook review dan tidak menghapus output aslinya.

`python toolkit/tools/check_composer.py` hanya memeriksa graph, struktur notebook,
sintaks Python, pergantian komponen dan kesetaraan rakitan JS/CLI secara lokal.
`python toolkit/tools/prepare_composer_kaggle.py` menyiapkan uji perilaku kecil
untuk Kaggle; tidak melatih/memprediksi secara lokal. Status uji remote disimpan
di `references/composer_validation.json`, terpisah dari riwayat uji pipeline klasik.

Snapshot 9 November: **292 rakitan** lolos pemeriksaan struktur, serta **25 uji
perilaku Kaggle v2** lolos: pipeline delapan kasus, masking unknown, grouping,
negasi, threshold rare label, duplicate consistency, Word2Vec, fine-tuning lima
pretrained, loss multilabel/regresi, E5, NLI pretrained, sentimen dan reranker.
Uji memakai data sintetis kecil untuk kompatibilitas, bukan estimasi kualitas model.
Pemisahan file helper diperiksa lagi melalui 292 rakitan tanpa menjalankan ML lokal.

API Trainer mengikuti versi yang dipatok `transformers==5.20.2`, dengan
`eval_strategy` dan masked BCE custom; rujukan [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer).
E5 menggunakan masked mean pooling, normalisasi dan prefix sesuai
[E5 model card](https://huggingface.co/intfloat/multilingual-e5-base).
