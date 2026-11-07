"""Buat sumber notebook 15-18 dari satu template, karena alurnya sama."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = (HERE / "tokentask_template.txt").read_text(encoding="utf-8")

TASKS = [
    {"id": "15_ner", "num": "15", "title": "Named Entity Recognition", "key": "ner",
     "folder": "nergrit_ner-grit",
     "intro": "Menandai nama orang, tempat, dan organisasi. Contoh: dataset NERGrit.",
     "when": "soal meminta mengambil entitas dari teks, misalnya lembaga, daerah, atau tokoh di berita MBG. "
             "Metriknya biasanya span F1.",
     "metric_note": "F1 tingkat span, jadi entitas baru dianggap benar kalau batas dan jenisnya tepat."},
    {"id": "16_pos_tagging", "num": "16", "title": "POS Tagging", "key": "pos_tagging",
     "folder": "posp_pos-prosa",
     "intro": "Memberi kelas kata pada tiap token. Contoh: dataset POSP.",
     "when": "soal meminta label kelas kata, atau label per token lain yang setiap tokennya berdiri sendiri. "
             "Metriknya biasanya akurasi atau macro-F1 token.",
     "metric_note": "F1 span, dan karena tiap token berlabel `B-` nilainya setara dengan F1 mikro per token."},
    {"id": "17_term_aspek", "num": "17", "title": "Ekstraksi Term Aspek", "key": "term_aspek",
     "folder": "terma_term-extraction-airy",
     "intro": "Mengambil frasa aspek dan frasa sentimen dari ulasan. Contoh: dataset TermA.",
     "when": "soal meminta mengambil bagian teks yang menyebut aspek dan opini, misalnya `nasinya basi`. "
             "Kalau aspeknya sudah ditentukan dan hanya polaritasnya yang dicari, pakai folder `13`.",
     "metric_note": "F1 tingkat span untuk jenis ASPECT dan SENTIMENT."},
    {"id": "18_keyphrase", "num": "18", "title": "Ekstraksi Keyphrase Supervised", "key": "keyphrase",
     "folder": "keps_keyword-extraction-prosa",
     "intro": "Mengambil frasa kunci dengan contoh berlabel. Contoh: dataset KEPS.",
     "when": "ada data berlabel frasa kunci. Kalau tidak ada label sama sekali, pakai folder `05`.",
     "metric_note": "F1 tingkat span. Label KEPS hanya B, I, dan O tanpa jenis."},
]

for task in TASKS:
    body = TEMPLATE.format(**task)
    (HERE / "src" / f"{task['id']}.py").write_text(body, encoding="utf-8")
    print("tulis", task["id"])
