# Status notebook review · 9 November 2026

## Cheatsheet terpadu untuk offline

**Pembaruan terakhir:** atas permintaan pengguna, kode pendek diganti kembali
dengan kode lengkap dari pipeline/snippet asli. Alur terpadu 43 task / 25 pendekatan
tetap dipakai. Pemeriksaan sumber memastikan isi setiap tahap sama dengan pipeline
aslinya dan 142 snippet lengkap tersedia. Uji kode pendek di bawah merupakan riwayat,
bukan validasi eksekusi versi kode lengkap yang dipulihkan.

[Cheatsheet utama](../toolkit/index.html) memakai satu alur per task tanpa halaman
perakit terpisah. Sel lengkap mengikuti urutan sumber; alternatif menyebut kondisi
dan kapan dipakai. Seluruh 43 task masuk 25 pendekatan, dimulai klasifikasi,
multilabel/ABSA, topik, clustering dan tren waktu.

[Audit per file/saran](AUDIT_CHEATSHEET.md) mencakup 142 sumber snippet dan 293 saran
metode. Kode referensi panjang dipindah dari rekomendasi ketik ke referensi; bug
majority correction sebelum split, NaN multilabel, OOF kosong/label order, margin
SVM palsu probability dan cache embedding ditangani. Semua import kartu baru
dicocokkan dengan freeze bundle; tidak ada download/pip pada kode cheatsheet.

Uji sel baru berada di [Kaggle](https://www.kaggle.com/code/ivanwllm/gemastik-cheatsheet-audit),
bukan lokal. Hasil diperiksa melalui [laporan terbaru](../toolkit/references/cheatsheet_validation.json).
Itu uji kompatibilitas/skenario kecil, bukan benchmark skor pada data lomba.
Output review asli tetap utuh. Rakitan modular sebelumnya disimpan sebagai arsip.

Pemakaian: [README cheatsheet](../toolkit/README.md).

## Sudah dieksekusi dan tersimpan lokal

| Kasus | Notebook | Hasil ringkas |
|---|---|---|
| Semantic similarity | [20 review](../notebook/e_pasangan_teks/20_kemiripan/20_kemiripan_review.ipynb) | Pilihan validation: IndoBERT. Test Pearson 0,7803; Spearman 0,7527. Ensemble test Spearman lebih tinggi, tetapi pilihan tidak diganti dari hasil test. |
| NLI 3 kelas | [21 review](../notebook/e_pasangan_teks/21_nli/21_nli_review.ipynb) | Entailment/neutral/contradiction. mDeBERTa + bias kelas terpilih validation; test macro-F1 0,7670. Audit lay vs expert tersedia. |
| Retrieval/matching | [22 review](../notebook/e_pasangan_teks/22_retrieval/22_retrieval_review.ipynb) | BM25 terpilih, test MRR@10 0,9910. Kueri berasal dari ringkasan artikel: benchmark proxy sangat mudah dan bukan bukti performa pencarian manusia. |
| Tren waktu | [02 review](../notebook/a_tanpa_label/02_tren_topik_waktu/02_tren_topik_waktu_review.ipynb) | NMF k=4 dipilih pada periode tengah. Model beku diaudit pada periode akhir. Sentimen: lexicon, classifier pretrained, dan NLI; tanpa gold tidak ada klaim accuracy. |

Keempat notebook memiliki output seluruh sel kode, tanpa error. Hasil CSV/JSON tersimpan
di folder `hasil_review/` masing-masing kasus. Angka tabel ini merupakan snapshot hasil
run, bukan skor yang ditulis ke kode notebook.

## Multilabel baru

[12 review](../notebook/b_klasifikasi/12_multi_label/12_multi_label_review.ipynb) sudah
dibuat lokal dan dikirim ke [Kaggle](https://www.kaggle.com/code/ivanwllm/gemastik-multilabel-review).
Status eksekusinya perlu diperiksa sebelum mengklaim hasil final; sumber lokal awal
belum berisi output run ini.

Fokus: 12 label hate speech/abusive, unknown berbeda dari negatif, lima pendekatan,
threshold global/per-label, ensemble, hierarki HS yang kondisional, dan evaluasi label langka.

## Pekerjaan lanjutan yang bernilai

1. Unduh/audit hasil multilabel setelah tersedia, terutama recall/AP label langka dan threshold stability.
2. Ulang retrieval pada kueri manusia dan relevance judgment yang lebih lengkap; skor proxy tinggi tidak cukup.
3. Audit error NLI expert dan kasus negasi/angka; jangan menganggap kemiripan berarti entailment.
4. Ulang pembagian grup/seed bila ingin menyimpulkan selisih model kecil; jangan memilih ulang dari holdout.
5. Audit manusia pada periode/topic dan contoh model sentimen tidak sepakat sebelum mengaitkan tren dengan peristiwa.
