## Referensi dan sumber tambahan

Ditinjau 8 November 2026. Blueprint pengguna menjadi referensi kebutuhan; aturan notebook
membedakan cleaning aman, normalisasi, dan penghapusan agresif agar perubahan dapat diperiksa.

| Sumber primer | Pemakaian |
|---|---|
| [ftfy](https://ftfy.readthedocs.io/en/latest/config.html) | Repair encoding; transformasi lain dikendalikan eksplisit |
| [emoji](https://emoji-python.readthedocs.io/en/latest/api.html) | Hapus sequence utuh atau deskripsi sebagai view alternatif |
| [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) | Parser markup opt-in, bukan regex HTML |
| [Sastrawi](https://github.com/har07/PySastrawi) | Stemming per kata dengan cache pada cabang clean |
| [IndoBERTweet](https://github.com/indolem/IndoBERTweet) | Resep model: lowercase, @USER, HTTPURL, emoji shortcode |
| [indoNLP](https://hyuto.github.io/indo-nlp/api/preprocessing/) | Referensi normalisasi slang/elongation; fungsi lokal dibuat transparan |
| [Colloquial Lexicon](https://github.com/nasalsabila/kamus-alay) | Adapter CSV slang/formal, mapping luar butuh review |
| [Kamus Alay](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection) | Adapter CSV tanpa header; konflik tidak diterapkan otomatis |
| [Combined Slang](https://github.com/louisowen6/NLP_bahasa_resources) | Adapter JSON dan pencatatan sumber/hash |

Tambahan di luar blueprint: [IndoCollex](https://aclanthology.org/2021.findings-acl.280/)
sebagai benchmark normalisasi informal, dan [normalisasi code-mixed Indonesia–Inggris](https://aclanthology.org/D19-5554/)
untuk mengevaluasi campuran bahasa secara terpisah. Ini rujukan eksperimen, bukan jaminan skor MBG.

Sumber opsional untuk pengembangan: [SALTIK](https://github.com/ir-nlp-csui/saltik),
[regional lexicon](https://data.mendeley.com/datasets/mrfc9mb2yt/1),
[parallel normalization corpus](https://github.com/ir-nlp-csui/parallel-corpus-for-lexical-normalization),
[BASA](https://github.com/Muanai/basa), dan [SymSpell](https://symspellpy.readthedocs.io/en/latest/examples/lookup.html).
Tidak dianggap sebagai normalizer siap pakai tanpa evaluasi; kosakata regional dan nama sering ambigu.

Fungsi non-teks pada API lama di preprocessing.py memakai
[datasketch](https://ekzhu.com/datasketch/lsh.html),
[RapidFuzz](https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html), dan
[fastText](https://github.com/facebookresearch/fastText/blob/main/docs/language-identification.md).
Ketiganya tidak dijalankan pada alur notebook teks yang baru.

Kamus asli tidak dibundel untuk redistribusi. URL, SHA-256, mapping aktif, config, dan versi
library dicatat lokal. Periksa ketentuan sumber sebelum mendistribusikan resource.
Ablation pada data berlabel dan validasi yang sama diperlukan sebelum menyatakan skor meningkat.
