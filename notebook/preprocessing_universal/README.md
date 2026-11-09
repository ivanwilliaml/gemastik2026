# Preprocessing Indonesia · satu metode, satu contoh

Buka Preprocessing_universal.ipynb lalu Run All. **32 metode**, masing-masing satu contoh
sintetis independen dan satu tabel before/after berisi satu baris. Tidak ada pembacaan CSV
atau pemrosesan 57 ribu komentar saat mempelajari contoh.

Urutan: cleaning teknis → elemen sosial media → normalisasi kata → cleaning agresif →
representasi alternatif/tokenisasi. Fungsi singkat langsung diikuti input dan output.
Contoh berikutnya tidak mewarisi hasil contoh sebelumnya.

Di akhir ada satu contoh gabungan untuk menggunakan API dan memverifikasi hasilnya sama
persis dengan pemanggilan fungsi bertahap. Contoh membaca data baru disediakan sebagai kode
opsional yang tidak dijalankan oleh Run All.

- stepwise.py: fungsi dan API pemakaian ulang.
- lexicons.py: adapter kamus lokal/multi-sumber.
- _build/build_examples.py: sumber notebook contoh.
- _build/methods.json: deskripsi dan satu input sintetis tiap metode.
- tests/: pengujian fungsi.
- reports/verification_examples.json: bukti eksekusi dan satu baris per tabel metode.
- legacy/Preprocessing_mbg_real.ipynb: versi terdahulu dengan output data nyata.
- data/MBG_PROCESSED.csv: 57.362 komentar asli, tetap tersedia.
- hasil_stepwise/: hasil pemrosesan data nyata terdahulu, tidak ditimpa notebook contoh.

Instal sekali: python -m pip install -r requirements.txt.
Bangun: python _build/build_notebook.py.
Jalankan dan simpan output: python _build/execute_notebook.py.
Uji: python tests/test_stepwise.py.
Semua metode demo dapat berjalan offline setelah dependensi terpasang.

Untuk dataset baru: from stepwise import preprocess_texts, lalu
preprocess_texts(df['comment'], options={'stem': False}).
Kamus, typo, bahasa daerah, penghapusan angka, stopword, dan stemming perlu disesuaikan
berdasarkan task. Tautan sumber dan alasan teknik ada di bagian akhir notebook / SOURCES.md.
