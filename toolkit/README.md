# Cheatsheet NLP offline

Buka **[index.html](index.html)** → **Cheatsheet per Task**.

Kode lengkap dari `pipelines/` dan `snippets/` digunakan kembali dalam alur
terpadu yang sekarang: 43 task, 25 pendekatan, urutan tahap yang jelas dan
referensi alternatif di bawah. Tidak ada halaman Rakit Kode terpisah.

1. Pilih task, mulai dari Setup untuk nama file, kolom dan saklar eksperimen.
2. Jalankan tahap dari atas. Fungsi pendukung dan dependensi mengikuti sumber asli.
3. Bila pretrained tidak dipakai, matikan saklar pada Setup. Tetap jalankan sel
   inisialisasi agar variabel yang dipakai tahap berikutnya tersedia.
4. Gunakan referensi alternatif sesuai kondisi data, bukan menjalankan semua snippet.
5. Unduh notebook lengkap dari halaman task bila diperlukan.

Versi kode pendek tidak lagi ditampilkan sebagai kode utama. Perbaikan bug hasil
audit tetap dipertahankan pada sumber referensi; kode tidak dipangkas untuk mengejar
jumlah baris. Catatan audit bukan jaminan model cocok untuk setiap dataset.

- [Catatan audit](../docs/AUDIT_CHEATSHEET.md)
- [Notebook dengan kode lengkap](notebooks/cheatsheet/)
- [Hasil pemeriksaan sumber dan struktur](references/cheatsheet_static.json)

Bangun ulang tampilan: `python toolkit/tools/build_library.py`.
Periksa struktur tanpa training lokal: `python toolkit/tools/check_cheatsheet.py`.
Notebook eksperimen sumber di `../notebook/` tetap dipertahankan.
