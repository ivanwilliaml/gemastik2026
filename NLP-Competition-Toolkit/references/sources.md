# Sumber dan catatan verifikasi

Diperiksa 5–7 November 2026.

| Isi | Cara verifikasi |
|---|---|
| Fakta lomba, jadwal, bobot, domain MBG, leksikon, hasil uji lokal | `docs/RINGKASAN_PERSIAPAN_GEMASTIK.md`, label kepastian dibawa apa adanya. Toolkit tidak memverifikasi ulang isinya |
| Lingkungan uji | Dipasang dari `D:\comp\bundle\wheelhouse` dengan `pip --no-index`, Python 3.12, Windows 10, torch 2.14.1+cu130, GTX 1650. Hasil `pip freeze` di `bundle_freeze.txt`; `pip check` bersih |
| Snippet | `tools/test_snippets.py`: tiap snippet dieksekusi di namespace kosong dengan data asli dan model bundle (142 dari 142 lolos, 7 Nov). Hasil di `snippet_test_result.json` |
| Pipeline per task | `tools/test_pipelines.py`: semua sel dijalankan berurutan dalam satu proses pada sampel data lokal. 22 skenario tanpa transformer dan 8 dengan transformer lolos (7 Nov), termasuk komentar TikTok MBG asli yang kotor dan pipeline ekstraksi. Hasil di `pipeline_test.json` dan `pipeline_test_bert.json` |
| Panggilan API di kartu metode | `tools/validate_api.py` mengeksekusi tiap panggilan pada lingkungan uji |
| Jebakan versi | Argumen lama dan pengganti dijalankan langsung |
| Baris, ukuran, lisensi dataset | API Hugging Face, file raw GitHub, dan penghitungan pada file terunduh |
| Pelajaran portofolio | Skill `ml-problem-solving`, ringkasan persiapan, dan notebook acuan pemilik toolkit |

## Belum diverifikasi

- **RTX 3070 dan PC panitia.** Semua uji GPU memakai GTX 1650. Jalankan ulang pengujian saat Set-Up.
- **Skala 500.000 baris.** Uji memakai sampel ratusan baris agar cepat: membuktikan kode jalan, bukan waktu atau RAM. Latih dengan data penuh sebelum lomba.
- **T5 dan ringkasan abstraktif.** Model `t5-small-sum` hanya ada di profil `all`, tidak diunduh. `load_seq2seq`, `generate_texts`, dan `train_seq2seq` belum pernah dijalankan.
- **Skor pada sampel kecil tidak bermakna.** Pengujian hanya memeriksa kode berjalan dan keluaran masuk akal.
- Prefiks `query:`/`passage:` E5 berasal dari ingatan. Cek kartu model.
- Prioritas task dan probabilitas adalah penilaian, bukan informasi panitia.
- Ambang AUC adversarial, PSI, OOV, dan rentang skala adalah aturan praktis.
- Lisensi kuning di katalog: tidak tercantum di metadata.
