# Catatan perapian · 8 November 2026

## Susunan aktif

- `notebook/`: notebook universal/per-task dan library bersama.
- `toolkit/`: aplikasi HTML, pipeline, snippet, dan dataset latihan lokal.
- `persiapan_offline/`: nama baru `Persiapan Set Up/`; nama lama `persiapan_h1/` pada panduan juga diperbarui.
- `docs/`: buku saku, ringkasan, riset, dan laporan pemeliharaan.

README utama menjadi pintu masuk; setiap bagian mempunyai petunjuknya sendiri. Rujukan
`NLP-Competition-toolkit` diperbarui menjadi `toolkit`. Riwayat Git tidak dihapus dan tidak ada commit/staging otomatis.

## Preprocessing universal

Notebook utama, `stepwise.py`, dan `lexicons.py` tetap mudah diakses di folder preprocessing.
Script pembangun/pelaksana masuk `_build/`, pengujian masuk `tests/`, bukti eksekusi masuk
`reports/`, dan implementasi terdahulu masuk `legacy/`. Hasil sintetis lama disimpan dalam
`legacy/hasil_sintetis.zip`; hasil MBG terbaru tetap di `hasil_stepwise/`.

`before_after_stages.csv` dan `comments.jsonl` dikompres menjadi `.csv.gz` dan `.jsonl.gz`.
Isi hasil dekompresi dibandingkan dengan SHA-256 ekspor awal sebelum salinan mentah dihapus.
File CSV hasil akhir yang biasa tetap tersedia. Ekspor berikutnya otomatis memakai format gzip.

Cache Python/Ruff yang dapat dibuat ulang dibersihkan. Dataset, kamus, dokumen resmi,
notebook dengan output, sumber kode, dan pengujian toolkit dipertahankan. Folder `toolkit/tools/t/`
berisi kode pengujian yang dipakai, sehingga tidak diperlakukan sebagai file sementara.

## Pemeriksaan

- 71 file sumber data/resource/dokumen diperiksa dengan checksum sebelum dan setelah penataan.
- 22 pengujian preprocessing lulus setelah perpindahan file.
- Notebook dijalankan ulang pada seluruh 57.362 komentar asli: 82 sel kode, tanpa error.
- Hasil bertahap tetap identik dengan API sekali panggil.
- Tautan pada panduan utama dan aset HTML toolkit diperiksa keberadaannya.
- Input lokal dan hasil besar tetap tersedia di disk; aturan Git diperbarui agar tidak ikut terunggah.

Angka ukuran folder dan rincian pemeriksaan tersimpan di `LAPORAN_PERAPIAN.json`.

## Penyeragaman nama folder

Seluruh nama folder sekarang lowercase snake_case, termasuk notebook/, toolkit/,
persiapan_offline/, a_tanpa_label/, b_klasifikasi/, eda_universal/, dan preprocessing_universal/.
Nama berkas tidak diubah. Path pada panduan dan kode pembangun disesuaikan.
Rincian perubahan nama: PENAMAAN_FOLDER.json.
