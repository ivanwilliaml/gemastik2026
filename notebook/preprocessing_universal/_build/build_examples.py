"""Notebook lesson: exactly one independent synthetic example for each method."""
from pathlib import Path
import json
import re
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
parts = re.split(r'^# %% (\w+)\s*$', (ROOT / 'stepwise.py').read_text(encoding='utf-8'), flags=re.M)
blocks = dict(zip(parts[1::2], parts[2::2]))
methods = json.loads((ROOT / '_build/methods.json').read_text(encoding='utf-8'))
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md('''# Preprocessing teks Indonesia · satu metode, satu contoh

**Setiap metode memakai satu contoh sintetis yang berdiri sendiri.** Hasil metode sebelumnya
tidak diteruskan ke contoh berikutnya. Dengan begitu, before/after menunjukkan perubahan
dari metode yang sedang dibahas saja. Semua output sudah dijalankan dan tersimpan.

Polanya selalu: **tujuan → fungsi pendek → satu input → satu output**. Tidak membaca CSV MBG
atau menampilkan sampel komentar nyata. Data aslinya tetap tersedia untuk pemakaian pipeline nanti.

| Bagian | Metode |
|---|---|
| A | Validasi teks, byte escape, encoding, Unicode, HTML/entity, invisible, whitespace |
| B | Obfuscation, URL, email, telepon, mention, hashtag, lowercase |
| C | Slang, elongation, tawa, reduplikasi, bahasa daerah, typo, repetisi punctuation |
| D | Emoji/emoticon, angka, punctuation, repetisi kata, stopword, stemming |
| E | Deskripsi emoji, resep IndoBERTweet, tokenisasi |
| F | Menggabungkan fungsi dan memakai ulang pada data baru |

Metode yang agresif tetap diperlihatkan agar efeknya jelas, bukan berarti semuanya wajib
diaktifkan. Negasi, angka, nama, emoji, dan bahasa daerah dapat membawa makna.
Versi sebelumnya dengan output 57.362 komentar nyata disimpan di
`legacy/Preprocessing_mbg_real.ipynb`.''')
md('''## Persiapan

Jalankan dari atas ke bawah. Dependensi inti contoh: pandas, ftfy, emoji, beautifulsoup4,
Sastrawi. Instal sekali memakai `python -m pip install -r requirements.txt`, lalu dapat dipakai
offline. Tidak ada instalasi atau unduhan otomatis di notebook.

Sel berikut menyiapkan pola dan kamus kecil yang dipakai beberapa fungsi. Mapping demo
dibuat eksplisit supaya hasil mudah ditulis ulang; bukan kamus bahasa Indonesia lengkap.''')
code(parts[0].strip() + '\nfrom pathlib import Path\nimport json\nfrom IPython.display import display')
code(blocks['resources'].strip() + '''

SLANG_USED = dict(SLANG)
TYPO_APPROVED = {'maknan': 'makanan', 'bergzi': 'bergizi'}  # mapping khusus contoh
HASHTAG_APPROVED = {}
example_log = []

def show_result(method, before, after):
    example_log.append({'method': method, 'before': repr(before), 'after': repr(after)})
    display(pd.DataFrame({'before': [repr(before)], 'after': [repr(after)]}))
''')

groups = {
    'A': 'A · Cleaning teknis',
    'B': 'B · Elemen media sosial',
    'C': 'C · Normalisasi kata dan ekspresi',
    'D': 'D · Cleaning agresif',
    'E': 'E · Representasi alternatif dan tokenisasi',
}
current = None
for method in methods:
    group = method['title'][0]
    if group != current:
        md('## ' + groups[group])
        current = group
    description = method['description']
    if method['key'] == 'html_markup':
        description = 'Gunakan parser hanya bila input benar-benar markup HTML. Script/style dibuang. Plain text seperti `nilai < 5 > 1` jangan diparse sembarangan.'
    elif method['key'] == 'regional':
        description = 'Contoh mapping daerah yang eksplisit, bukan penerjemah otomatis. Pada data baru, aktifkan hanya setelah memeriksa konteks.'
    elif method['key'] == 'typo':
        description = 'Terapkan hanya pasangan typo yang sudah diperiksa. Contoh ini memakai mapping `maknan→makanan` dan `bergzi→bergizi`; bukan fuzzy correction semua kata tak dikenal.'
    elif method['key'] == 'numbers':
        description = 'Contoh efek penghapusan angka. Nominal/tanggal akan kehilangan informasi; default API tidak mengaktifkannya. Angka di kode alfanumerik tetap.'
    elif method['key'] == 'escaped_bytes':
        description = 'Kembalikan escape byte UTF-8 literal menjadi emoji. Byte invalid tetap utuh. Jangan melakukan decode unicode_escape pada seluruh teks.'
    md('### ' + method['title'] + '\n\n' + description + '\n\n**Satu contoh independen:**')
    example = method['example']
    function = method['function']
    # Parentheses make lambdas immediately callable, like a named function.
    call = f'({function})(before)' if function.startswith('lambda ') else f'{function}(before)'
    code(blocks[method['key']].strip() + f'\n\nbefore = {example!r}\nafter = {call}\nshow_result({method["title"]!r}, before, after)')
    if method['key'] == 'unicode':
        md('Pada contoh Unicode, `é` dapat tampak sama di layar. Input menggunakan `e + U+0301`; output NFC menggunakan satu karakter `U+00E9`.')

md('''### E3 · Tokenisasi sederhana

Kalimat menjadi list kata menggunakan pemisah spasi. Untuk model pretrained, gunakan tokenizer
model tersebut; ini hanya contoh tokenisasi transparan pada teks clean.''')
code(blocks['tokens'].strip() + "\n\nbefore = 'menu sehat tidak merata'\nafter = tokenize(before)\nshow_result('E3 · Tokenisasi sederhana', before, after)")

md('''## F · Gabungkan dan pakai ulang

Bagian A–E memperlihatkan metode secara terpisah. API berikut menyusun **fungsi yang sama**
menjadi cabang safe → normalized → clean. Tidak ada aturan preprocessing kedua yang ditulis ulang.
Urutan langkahnya dapat dilihat pada `build_steps`; opsi agresif diatur melalui `options`.

Contoh penggunaan di bawah memakai **satu komentar sintetis gabungan** untuk memperlihatkan
cara memanggil API. Ini bagian penggunaan pipeline, bukan contoh metode tambahan.''')
code(blocks['api'])
code('''before = '  @budi MBG bgs bgt, tp gk merata!!! 😡 #TolakMBG  '
options = {'slang': SLANG_USED, 'stem': True, 'stopwords': True,
           'regional': False, 'remove_numbers': False, 'typos': {}}
pipeline_result = preprocess_texts(pd.Series([before]), options=options)
display(pipeline_result[['text_raw', 'text_safe', 'text_normalized', 'text_clean']])

# Verifikasi urutan sel manual dan API identik, pada satu input yang sama.
manual = pd.Series([before])
for branch, steps in build_steps(options).items():
    for _, function in steps:
        manual = transform_unique(manual, function)
    pd.testing.assert_series_equal(pipeline_result['text_' + branch], manual, check_names=False)
assert len(example_log) == 32
assert all(set(row) == {'method', 'before', 'after'} for row in example_log)
print('Lulus: 32 metode, masing-masing satu contoh; hasil pipeline manual = API.')''')
md('''### Pakai pada data baru nanti

Kode ini sengaja tidak dijalankan oleh Run All. Tidak ada fallback atau pembacaan data nyata
tersembunyi di contoh metode. Salin fungsi yang diperlukan, atau gunakan API `stepwise.py`.

```python
from stepwise import preprocess_texts

df = pd.read_csv('data/MBG_PROCESSED.csv', dtype='string')
result = preprocess_texts(df['comment'], options={'stem': False})
```

Kamus luar dapat disiapkan lewat `lexicons.external_lexicons(...)`; sumber asli lokal berada di
`resources/downloaded/`. Mapping yang belum disetujui tidak otomatis diterapkan. Untuk memakai
kamus tinjauan: `options={'slang': kamus_yang_disetujui}`. Opsi `typos` pada data baru harus
berasal dari review, bukan menyalin mapping demo tanpa pemeriksaan.

Notebook contoh ini tidak menimpa ekspor data nyata pada `hasil_stepwise/`. Pengujian fungsi
tersedia di `tests/test_stepwise.py`, bukti eksekusi contoh di `reports/verification_examples.json`.''')
md((ROOT / 'SOURCES.md').read_text(encoding='utf-8'))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'},
})
nbf.write(nb, ROOT / 'Preprocessing_universal.ipynb')
print(f'Built {len(cells)} cells; {len(methods)+1} independent method examples')
