"""Build the teaching notebook: one function + actual before/after per stage."""
from pathlib import Path
import re
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
parts = re.split(r'^# %% (\w+)\s*$', (ROOT / 'stepwise.py').read_text(encoding='utf-8'), flags=re.M)
blocks = dict(zip(parts[1::2], parts[2::2]))
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


def section(text):
    md(text)


def stage(key, title, description, function, example, enabled='True', target='texts'):
    md(f'### {title}\n\n{description}\n\n**Input:** hasil tahap sebelumnya. **Output:** teks tahap ini. '
       'Tabel memakai komentar asli dari CSV MBG. Jika tidak ada perubahan, tahap itu memang '
       'tidak menemukan pola yang perlu diganti pada data ini.')
    code(blocks[key].strip())
    code(f"{target} = run_step({target}, {function}, {title!r}, enabled={enabled})\n"
         f"if SHOW_RULE_EXAMPLES:\n    show_example({example!r}, {function})")


md('''# Preprocessing teks Indonesia / MBG — langkah demi langkah

**Baca dari atas ke bawah. Setiap tahap = satu fungsi pendek → satu pemanggilan → before/after.**
Semua fungsi tertanam di notebook, tanpa import file proyek. Ubah `INPUT` dan `TEXT` lalu **Run All**.
Input bawaan: **57.362 komentar asli TikTok MBG**, kolom `comment`, bukan `clean_comment`.
Seluruh baris diproses, tanpa sampling. Tidak membutuhkan internet/GPU.

| Bagian | Isi | Hasil |
|---|---|---|
| A | Input, tipe data, encoding, Unicode, HTML, karakter tak terlihat, spasi | `text_safe` |
| B | Teks tersamarkan, URL, email, telepon, mention, hashtag, lowercase | awal normalisasi |
| C | Slang, elongation, tawa, reduplikasi, daerah, typo, tanda baca berulang | `text_normalized` |
| D | Emoji/emoticon, angka opsional, tanda baca, repetisi kata, stopword, stemming | `text_clean` |
| E | Alternatif deskripsi emoji, resep IndoBERTweet, tokenisasi | versi tambahan |
| F | Perbandingan seluruh tahap, ekspor, fungsi sekali panggil, verifikasi | hasil siap dipakai |

**“Bersih maksimal” tersedia di `text_clean`.** Safe dan normalized juga disimpan sehingga
efek penghapusan dapat dibandingkan. Angka tidak dibuang secara default karena nominal/tanggal
bisa penting; ubah `REMOVE_NUMBERS=True` untuk view lebih agresif. Kata negasi serta akronim
MBG dilindungi. Stemming/stopword aktif hanya di cabang clean.

Fokus notebook ini **teks saja**. Tidak ada training, embedding, deteksi bot, atau penghapusan baris.
Raw, metadata, label, urutan, dan ID input dipertahankan. Skor model belum diukur.''')

section('''## Persiapan · dependensi dan konfigurasi

Instal sekali saat online: `python -m pip install -r requirements.txt`.
Untuk versi lengkap di notebook ini diperlukan `pandas`, `ftfy`, `emoji`, `beautifulsoup4`, dan `Sastrawi`.
Kamus GitHub dapat disiapkan sekali dan dibaca offline. Tidak ada unduhan otomatis.
CSV dibaca sebagai string agar ID/nol awal/angka teks tidak berubah karena inferensi tipe;
untuk Parquet/JSONL tipe sumber tetap dibaca dan nilai non-string ditandai pada validasi.''')
code(parts[0].strip() + '\nfrom pathlib import Path\nimport hashlib\nimport json\nimport time\nfrom IPython.display import display')
code('''INPUT = Path('../data/MBG_PROCESSED.csv')  # ganti untuk data baru
TEXT = 'comment'
SEP, ENCODING = ',', 'utf-8-sig'
OUT = Path('../hasil_stepwise')
RESOURCE_DIR = Path('../resources/downloaded')
DOWNLOAD_RESOURCES = False  # True sekali online; False untuk penggunaan offline
REVIEWED_LEXICON = Path('reviewed_lexicon.csv')
HTML_INPUT = False  # True hanya jika sumber memang markup HTML
DECODE_ESCAPED_BYTES = False  # True jika \\xf0... benar-benar emoji ter-escape
NORMALIZE_REGIONAL = False  # jangan otomatis menerjemahkan semua bahasa daerah
REMOVE_NUMBERS = False
REMOVE_STOPWORDS, STEMMING = True, True
TYPO_APPROVED = {}  # setelah review: {'maknan': 'makanan', 'bergzi': 'bergizi'}
HASHTAG_APPROVED = {}  # contoh: {'tolakmbg': 'tolak mbg'}
PREVIEW_ROWS = 8
SHOW_RULE_EXAMPLES = False  # True hanya untuk contoh aturan buatan, bukan hasil dataset
pd.set_option('display.max_colwidth', 120)''')
code(blocks['resources'])

section('''## A · Input dan cleaning teknis

Belum lowercase, belum buang emoji/angka/tanda baca. Teks raw tidak ditimpa.
Helper `run_step` hanya mengurangi kode berulang untuk pemrosesan kata unik dan tampilan:
fungsi preprocessing pada setiap sel tetap bisa disalin dan dipanggil langsung.''')
code('''def load_data(path):
    path = Path(path)
    if path.suffix.lower() in {'.csv', '.tsv'}:
        return pd.read_csv(path, sep='\\t' if path.suffix.lower() == '.tsv' else SEP,
                           encoding=ENCODING, dtype='string', keep_default_na=False, na_values=[''])
    if path.suffix.lower() in {'.parquet', '.pq'}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {'.jsonl', '.ndjson'}:
        return pd.read_json(path, lines=True)
    raise ValueError('Gunakan CSV/TSV/Parquet/JSONL.')

if INPUT is None:
    raise ValueError('Isi INPUT dengan path dataset asli. Tidak ada fallback data sintetis.')
df = load_data(INPUT)
print('Sumber data:', INPUT.resolve())
print('SHA-256:', hashlib.sha256(INPUT.read_bytes()).hexdigest())
print('Kolom teks:', TEXT, '| seluruh baris digunakan; tanpa sampling')
print('Label di file adalah keluaran model; tidak dipakai untuk preprocessing/training.')
if TEXT not in df:
    raise KeyError(f'Kolom {TEXT!r} tidak ditemukan: {list(df)}')
raw = df[TEXT].copy()
texts = raw.copy()
history, audit = {}, []
print('Baris:', len(df), '| kolom:', list(df))
display(df.head(PREVIEW_ROWS))''')
code('''def visible(value):
    return repr(value)  # newline, spasi, null, dan tipe angka terlihat jelas

def run_step(before, function, label, enabled=True):
    started = time.perf_counter()
    if enabled:
        # valid_text is handled per-row: input may be list/null, not hashable strings.
        if function.__name__ == 'valid_text':
            after = before.map(function)
        else:
            lookup = {s: function(s) for s in pd.unique(before)}
            after = before.map(lookup)
    else:
        after = before.copy()
    table = pd.DataFrame({'before': before.map(visible), 'after': after.map(visible)})
    changed = table.before.ne(table.after)
    audit.append(dict(stage=label, enabled=bool(enabled), changed=int(changed.sum()),
                      empty=int(after.eq('').sum()), seconds=round(time.perf_counter()-started, 4)))
    history[label] = after.copy()
    print(f'{label}: aktif={enabled}; berubah={changed.sum():,}/{len(before):,}; kosong={after.eq("").sum():,}')
    # Changed rows first; retain order within each set. Empty input still works.
    order = sorted(range(len(table)), key=lambda i: not changed.iloc[i])
    display(table.iloc[order].head(PREVIEW_ROWS))
    return after

def show_example(example, function):
    display(pd.DataFrame({'contoh_before': [visible(example)],
                          'contoh_after': [visible(function(example))]}))''')

stage('validate', 'A1 · Validasi tipe teks',
      'Null/angka/list menjadi string kosong pada teks kerja, bukan literal `nan` atau `123`. '
      'String asli `"nan"`/`"None"` tetap teks. Baris tidak dibuang; nilai sumber tetap ada di raw.',
      'valid_text', None)
code('''validation = pd.DataFrame({'missing': raw.isna(),
    'non_string': raw.map(lambda x: not isinstance(x, str)) & ~raw.isna(),
    'empty': texts.eq('')})
display(validation.sum().to_frame('jumlah'))''')
stage('escaped_bytes', 'A2 · Decode byte escape',
      'Opsional untuk hasil scraping dengan byte UTF-8 tersimpan literal. Byte invalid tetap utuh. '
      'Bukan decode `unicode_escape` pada seluruh teks, yang bisa merusak Unicode valid.',
      'decode_byte_escapes', r'emoji \xf0\x9f\x98\xa1', 'DECODE_ESCAPED_BYTES')
stage('encoding', 'A3 · Perbaiki encoding rusak',
      'ftfy memperbaiki mojibake. Perubahan tipografi, lebar karakter, dan HTML di ftfy dimatikan '
      'agar setiap transformasi terlihat pada tahapnya sendiri.', 'repair_encoding', 'cafÃ©, nasi sehat')
stage('unicode', 'A4 · Normalisasi Unicode NFC',
      'Samakan representasi kanonis. Tidak menghapus aksara daerah/emoji dan tidak memaksa ASCII/NFKC.',
      'normalize_unicode', 'Cafe\u0301')
stage('html_markup', 'A5 · Buang markup HTML bila ada',
      'Parser HTML hanya untuk sumber yang benar-benar markup. Plain text `nilai < 5 > 1` '
      'jangan diparse sembarangan. Script/style dibuang. Contoh kecil tetap menunjukkan parser '
      'walaupun tahap dataset dimatikan.', 'remove_html', '<p>Menu <b>sehat</b></p><script>bad()</script>', 'HTML_INPUT')
stage('entities', 'A6 · Decode HTML entities',
      '`&amp;`, `&quot;`, `&#...;` menjadi karakter aslinya. Satu lapis decoding; tidak berulang '
      'tanpa batas pada teks yang sengaja menuliskan entity.', 'decode_html_entities', 'MBG &amp; gizi &quot;baik&quot;')
stage('invisible', 'A7 · Bersihkan karakter kontrol',
      'Hapus BOM, zero-width space, word joiner, null byte; kontrol lain menjadi spasi. '
      'ZWJ/ZWNJ dan variation selectors dipertahankan agar emoji gabungan tidak pecah.',
      'remove_invisible', 'teks\u200bbaru\x00 👩‍🍳')
stage('whitespace', 'A8 · Rapikan whitespace',
      'Gabungkan tab/newline/spasi berulang dan strip tepi. Notebook ini untuk komentar satu baris; '
      'untuk dokumen berparagraf, pakai versi yang mempertahankan newline.',
      'normalize_whitespace', '  menu\tsehat\r\n  merata  ')
code('text_safe = texts.copy()  # snapshot cleaning teknis, belum normalisasi leksikal')

section('''## B · Elemen khusus media sosial

Mulai cabang normalisasi. Placeholder `URL/EMAIL/PHONE/USER` menyamakan identitas acak tanpa
mengosongkan kalimat. Setelah lowercase token menjadi huruf kecil. Ini transformasi teks,
bukan jaminan anonimisasi. Sumber asli tetap di raw.''')
stage('obfuscation', 'B1 · Normalisasi teks tersamarkan',
      'Hanya allowlist kata yang eksplisit: tidak mengubah semua 3→e/4→a karena nomor dan kode '
      'bisa rusak. Dilakukan sebelum mention supaya `ker@cunan` bisa dikenali sebagai kata.',
      'normalize_obfuscation', 'k3r4cUn4n ker@cunan koru*si S2 Rp10.000')
stage('urls', 'B2 · Ganti URL', 'URL panjang menjadi placeholder. Tanda baca penutup dipertahankan '
      'dengan heuristik sederhana; raw tetap menyimpan tautan lengkap.', 'replace_urls', 'cek https://contoh.id/x?q=gk, sekarang!')
stage('emails', 'B3 · Ganti alamat email', 'Email sebelum mention agar `@domain` tidak dianggap akun.',
      'replace_emails', 'kirim admin@contoh.id ya')
stage('phones', 'B4 · Ganti nomor telepon', 'Pola konservatif nomor seluler Indonesia. Nominal/tanggal '
      'tidak diganti. Format telepon di luar pola perlu aturan tambahan.', 'replace_phones', 'WA +62 812-3456-7890, harga Rp10.000')
stage('mentions', 'B5 · Ganti mention', 'Mention menjadi placeholder. Nama orang di kalimat biasa tetap ada.',
      'replace_mentions', '@budi dan @siti.id bahas MBG')
stage('hashtags', 'B6 · Segmentasi hashtag', 'Pisahkan CamelCase/akronim, pertahankan isi kata. '
      'Hashtag lowercase gabungan hanya dipecah lewat mapping yang disetujui.',
      'lambda s: segment_hashtags(s, HASHTAG_APPROVED)', '#TolakMBG #DukungProgramMBG #tolakmbg')
stage('lowercase', 'B7 · Lowercase', 'Casefold untuk cabang normalisasi/clean. Kapitalisasi asli tetap '
      'tersedia di safe/raw untuk model yang membutuhkannya.', 'lowercase', 'TIDAK Bagus! SPPG MBG')

section('''## C · Normalisasi kata dan ekspresi

Tidak ada fuzzy replacement global. Slang memakai batas kata utuh; kata ambigu/akronim
dilindungi. Kamus eksternal menjadi kandidat review, bukan aturan otomatis yang menimpa makna.''')
md('''### C0 · Kamus lokal dan tiga sumber GitHub

Colloquial, Kamus Alay, dan Combined Slang dibaca lokal bila sudah diunduh.
Semua mapping eksternal berstatus pending, konflik tidak diterapkan. Untuk pemakaian ulang,
isi `review_status=approved` pada pasangan yang sudah dicek lalu simpan `reviewed_lexicon.csv`.
Kamus aktif dan checksum diekspor agar hasil dapat direproduksi. Memakai versi resource berbeda
atau mengganti kamus bisa menghasilkan output berbeda.''')
old = (ROOT / 'lexicons.py').read_text(encoding='utf-8')
code(old[old.index('def read_lexicon('):].strip())
code('''lex = external_lexicons(RESOURCE_DIR, download=DOWNLOAD_RESOURCES)
OUT.mkdir(parents=True, exist_ok=True)
lex.to_csv(OUT / 'lexicon_candidates.csv', index=False)
if REVIEWED_LEXICON.exists():
    lex = pd.concat([lex, read_lexicon(REVIEWED_LEXICON)], ignore_index=True)
SLANG_USED = approved_slang(lex) if len(lex) else dict(SLANG)
print('Kandidat:', len(lex), '| mapping aktif:', len(SLANG_USED))
display(pd.DataFrame(SLANG_USED.items(), columns=['variant', 'canonical']).head(12))''')
stage('slang', 'C1 · Normalisasi slang dan singkatan informal',
      'Exact-token mapping, bukan replace substring. `gk,` tetap dikenali, isi URL/email/hashtag '
      'tidak diubah. `sm/dr/bs` tidak ditebak. Akronim MBG/SPPG/BGN tidak diekspansi otomatis.',
      'lambda s: normalize_slang(s, SLANG_USED)', 'mbg bgs bgt, tp gk merata sm dr SPPG')
stage('elongation', 'C2 · Normalisasi huruf memanjang',
      'Kurangi pengulangan 3+ hanya jika hasil ada di VOCAB. Double letter sah seperti massa/Allah '
      'tetap utuh; kata yang belum dikenal tidak dipaksa. Perluas VOCAB dari sumber yang dikurasi.',
      'normalize_elongation', 'baguuuuus bangeeet Allah massa xyztttt')
stage('laughter', 'C3 · Normalisasi tawa',
      'Kelompokkan variasi pola wkwk/haha yang eksplisit. Tawa tetap token ekspresi, tidak dibuang.',
      'normalize_laughter', 'wkwkwkwkwk hahahaha bagus')
stage('reduplication', 'C4 · Normalisasi reduplikasi angka 2',
      'Whitelist `anak2/orang2/teman2`, bukan semua token berangka. S2/GPT2/angka tanggal tetap.',
      'normalize_reduplication', 'anak2 orang2 teman2 S2 GPT2')
stage('regional', 'C5 · Normalisasi bahasa daerah opsional',
      'Dimatikan default. Contoh mapping kecil, bukan penerjemah. Aktifkan hanya setelah '
      'mengecek konteks; campuran bahasa tidak otomatis dianggap salah ketik.',
      'normalize_regional', 'ora apik dan teu enak', 'NORMALIZE_REGIONAL')
stage('typo', 'C6 · Terapkan typo yang sudah direview',
      'TYPO_APPROVED kosong secara default. Masukkan pasangan yang pasti setelah review. '
      'Kata tidak dikenal tidak otomatis dianggap typo. Nama lowercase masih bisa salah dikoreksi '
      'jika mapping sembarangan, sehingga review tetap diperlukan.',
      'lambda s: apply_reviewed_typos(s, TYPO_APPROVED)', 'maknan bergzi SPPG')
md('''Contoh buatan hanya tampil bila `SHOW_RULE_EXAMPLES=True`. Tanpa mapping typo yang sudah disetujui, teks dataset tetap utuh pada tahap ini.''')
code("if SHOW_RULE_EXAMPLES:\n    show_example('maknan bergzi', lambda s: apply_reviewed_typos(s, {'maknan': 'makanan', 'bergzi': 'bergizi'}))")
stage('punctuation_repeat', 'C7 · Ringkas tanda baca berulang',
      'Pengulangan ekstrem jadi maksimal dua: `!!!!!→!!`. Jenis tanda baca tidak disamakan. '
      'Cabang clean di bawah akan membuang tanda bacanya sepenuhnya.',
      'normalize_punctuation_repeat', 'serius????!!!! bagus.....')
code("texts = run_step(texts, normalize_whitespace, 'C8 · Rapikan spasi normalisasi')\ntext_normalized = texts.copy()")

section('''## D · Cleaning agresif: teks sebersih mungkin

Cabang ini mulai dari `text_normalized`. Emoji dan tanda baca dibuang, repetisi kata diringkas,
stopword/stemming dijalankan. **Informasi ekspresi dapat hilang**; before/after menunjukkan
persis yang berubah. Teks asli/safe/normalized tidak tertimpa. Tidak menghapus seluruh karakter
non-ASCII, tidak membuang kata pendek, dan tidak memotong komentar panjang diam-diam.''')
stage('emoji_remove', 'D1 · Hapus emoji', 'Hapus satu sequence emoji secara utuh, termasuk emoji '
      'keluarga/skin tone/flag. Alternatif deskripsi emoji tersedia terpisah di bagian E.',
      'remove_emoji', 'bagus 😍 🙏🏽 👨‍👩‍👧‍👦 🇮🇩')
stage('emoticons', 'D2 · Hapus emoticon ASCII', 'Aturan umum :) :( :)) <3, bukan semua emoticon '
      'yang pernah ada. Tanda baca sisanya dibersihkan di D4.', 'remove_emoticons', 'bagus :)) sedih :( cinta <3')
stage('numbers', 'D3 · Hapus angka opsional', 'Dimatikan default. Jika diaktifkan nominal/tanggal '
      'akan kehilangan angka; gunakan hanya jika memang tidak relevan pada task. Angka di kode '
      'alfanumerik tidak otomatis dihapus.', 'remove_numbers', '3 siswa Rp10.000 pada 10/11/2026 S2', 'REMOVE_NUMBERS')
stage('punctuation', 'D4 · Hapus tanda baca dan simbol',
      'Ubah ke spasi agar kata tidak menempel. Huruf Unicode/angka tetap. Nama dengan tanda '
      'hubung berubah menjadi kata terpisah; periksa implikasinya untuk entitas.',
      'remove_punctuation', 'anak-anak, bagus!!! tidak/buruk; 10.000')
stage('repeated_words', 'D5 · Ringkas repetisi kata berturut-turut',
      'Hanya kata identik berurutan. Ekspresi `bagus bagus` akan jadi `bagus`; '
      'normalized masih menyimpan ekspresinya.', 'reduce_repeated_words', 'bagus bagus bagus tapi tidak tidak')
stage('stopwords', 'D6 · Hapus stopword', 'Daftar kecil yang transparan dan bisa diganti. '
      'Negasi dan akronim terlindungi tidak dibuang walaupun masuk daftar tambahan.',
      'remove_stopwords', 'makanan yang bagus untuk anak dan tidak merata', 'REMOVE_STOPWORDS')
stage('stemming', 'D7 · Stemming Sastrawi',
      'Bentuk dasar per kata unik dengan cache terbatas. Negasi dan akronim terlindungi. '
      'Stemming bukan spellchecker dan dapat mengubah nama/istilah: safe/normalized tetap ada.',
      'stem_text', 'makanan membantu pemerataan dan tidak bergizi SPPG', 'STEMMING')
code("texts = run_step(texts, normalize_whitespace, 'D8 · Rapikan hasil akhir')\ntext_clean = texts.copy()")

section('''## E · Versi alternatif dan tokenisasi

Versi berikut **tidak menyambung cleaning agresif**: deskripsi emoji berangkat dari normalized,
IndoBERTweet berangkat dari safe. Tokenisasi untuk model pretrained harus memakai tokenizer
modelnya sendiri; `split()` di sini hanya tokenisasi sederhana hasil clean.''')
code('emoji_texts = text_normalized.copy()')
stage('emoji_describe', 'E1 · Emoji menjadi deskripsi', 'Shortcode Inggris dari library emoji, '
      'bukan hasil klasifikasi sentimen dan bukan terjemahan Indonesia.', 'describe_emoji', 'Menu 😍 👩‍🍳', target='emoji_texts')
code('bert_texts = text_safe.copy()')
stage('indobertweet', 'E2 · Resep khusus IndoBERTweet', 'Lowercase, mention→@USER, URL→HTTPURL, '
      'emoji shortcode mengikuti resep publik model. Jangan menambahkan stopword/stemming ke '
      'view ini tanpa eksperimen.', 'indobertweet_view', '@Budi MBG BAGUS 😍 https://contoh.id', target='bert_texts')
md('''### E3 · Tokenisasi hasil clean

Before berupa kalimat, after berupa list kata. Bukan vocabulary/embedding, belum melatih apa pun.''')
code(blocks['tokens'])
code("tokens = text_clean.map(tokenize)\ndisplay(pd.DataFrame({'before': text_clean, 'after_tokens': tokens}).head(PREVIEW_ROWS))\nif SHOW_RULE_EXAMPLES:\n    show_example('menu sehat tidak merata', tokenize)")

section('''## F · Audit, pemakaian ulang, dan ekspor

Laporan menampilkan jumlah berubah/kosong tiap tahap. Baris kosong tetap dipertahankan;
filter hanya setelah mengetahui aturan dataset/task. Tidak ada klaim skor meningkat karena
teks menjadi lebih pendek atau jumlah kata unik turun.''')
code('''result = df.copy()
for name in ['text_raw', 'text_safe', 'text_normalized', 'text_clean', 'text_emoji_described', 'text_indobertweet', 'tokens']:
    if name in result:
        raise ValueError(f'Kolom output {name} sudah ada. Gunakan input mentah.')
result['text_raw'], result['text_safe'] = raw, text_safe
result['text_normalized'], result['text_clean'] = text_normalized, text_clean
result['text_emoji_described'], result['text_indobertweet'] = emoji_texts, bert_texts
result['tokens'] = tokens
audit_table = pd.DataFrame(audit)
display(audit_table)
display(result[['text_raw', 'text_safe', 'text_normalized', 'text_clean']])
# Pilih baris untuk menelusuri seluruh transformasinya (nomor POSISI, bukan label index).
ROW = min(6, len(result)-1)
if ROW >= 0:
    display(pd.DataFrame([(label, visible(values.iloc[ROW])) for label, values in history.items()],
                         columns=['tahap', 'hasil_satu_komentar']))''')
md('''### F1 · Sekali panggil, hasil identik dengan sel bertahap

Bagian ini hanya menyusun fungsi-fungsi yang sudah dipelajari, tanpa menulis ulang aturan.
`preprocess_texts` memakai urutan dan mapping yang sama. Verifikasi otomatis di bawah membandingkan
semua output, termasuk urutan/index dan token. Untuk menyalin seluruh pipeline gunakan
`stepwise.py`; untuk satu langkah cukup salin fungsi dan dependensi yang ditampilkan pada selnya.''')
code(blocks['api'])
code('''OPTIONS = dict(html_input=HTML_INPUT, escaped_bytes=DECODE_ESCAPED_BYTES,
    regional=NORMALIZE_REGIONAL, remove_numbers=REMOVE_NUMBERS,
    stopwords=REMOVE_STOPWORDS, stem=STEMMING, slang=SLANG_USED,
    typos=TYPO_APPROVED, hashtags=HASHTAG_APPROVED)
reused = preprocess_texts(raw, options=OPTIONS)
for col in reused:
    pd.testing.assert_series_equal(result[col], reused[col], check_names=False)
pd.testing.assert_series_equal(df[TEXT], raw)
assert len(result) == len(df)
print('LULUS: hasil step-by-step = API sekali panggil; raw/urutan/jumlah baris utuh.')''')
md('''### F2 · Simpan hasil dan aturan yang digunakan

CSV untuk inspeksi, JSONL gzip untuk struktur token, `audit_stages.csv` untuk ringkasan,
`before_after_stages.csv.gz` untuk seluruh tahap dalam format lebar. Tabel lengkap tahap bersifat
opsional untuk data besar karena menggandakan penyimpanan; matikan `SAVE_ALL_STAGES`.
Manifest menyimpan versi library, konfigurasi dan mapping aktual (termasuk typo/hashtag).
Hasil identik membutuhkan input, urutan aturan, resource, dan versi library yang sama.''')
code('''import importlib.metadata
SAVE_ALL_STAGES = True  # False untuk data besar
OUT.mkdir(parents=True, exist_ok=True)
result.to_json(OUT / 'comments.jsonl.gz', orient='records', lines=True, force_ascii=False)
csv = result.copy()
csv['tokens'] = csv.tokens.map(lambda v: json.dumps(v, ensure_ascii=False))
csv.to_csv(OUT / 'comments.csv', index=False, encoding='utf-8-sig')
audit_table.to_csv(OUT / 'audit_stages.csv', index=False)
if SAVE_ALL_STAGES:
    pd.DataFrame(history).to_csv(OUT / 'before_after_stages.csv.gz', index=False, encoding='utf-8-sig')
manifest = dict(rows=len(result), input_file=str(INPUT.resolve()), input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(), text_column=TEXT, sampled=False, options=OPTIONS,
    versions={m: importlib.metadata.version(m) for m in ['pandas', 'ftfy', 'emoji', 'Sastrawi', 'beautifulsoup4']},
    dictionaries=dict(slang=SLANG_USED, vocab=sorted(VOCAB), stopwords=sorted(STOPWORDS),
                      protected=sorted(PROTECTED), negations=sorted(NEGATIONS),
                      regional=REGIONAL, obfuscation=OBFUSCATION),
    resources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in RESOURCE_DIR.glob('*') if p.is_file()},
    stage_order=[a['stage'] for a in audit],
    selected_view='text_clean', note='text-only; no row deletion; no task accuracy benchmark')
(OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print('Hasil:', OUT.resolve())''')

md('''### F3 · Data besar dan cara memilih view

Untuk ratusan ribu komentar, matikan preview/penyimpanan semua tahap atau proses CSV per batch.
Notebook ini sengaja menyimpan snapshot tiap tahap agar mudah dipelajari; mode produksi gunakan
API `stepwise.preprocess_texts` yang menyimpan hanya keluaran cabang. Kata identik diproses sekali
per tahap/batch. Stemming memakai cache per kata; pembentukan vocab dari data baru tidak otomatis.

```python
from stepwise import preprocess_texts
for i, chunk in enumerate(pd.read_csv('komentar.csv', dtype='string', chunksize=20000)):
    out = preprocess_texts(chunk['comment'], options=OPTIONS)
    out.to_json(f'hasil-{i:04d}.jsonl', orient='records', lines=True, force_ascii=False)
```

Untuk membandingkan performa model: evaluasi raw/safe/normalized/clean dengan fold, model,
seed, dan metrik yang sama. Stemming/stopword atau menghapus emoji tidak selalu membantu.
NER/QA harus menggunakan offset anotasi yang sesuai; cleaned text tidak otomatis selaras raw.
Kamus daerah/typo yang belum tervalidasi lebih baik tetap utuh daripada diubah berdasarkan tebakan.''')

sources = (ROOT / 'SOURCES.md').read_text(encoding='utf-8')
sources = sources.replace('## 15 · Sumber, pilihan metode, dan eksperimen tambahan', '## Referensi dan sumber tambahan')
md(sources + '\n\n**Catatan versi step-by-step:** notebook ini fokus pada transformasi teks; '
   'modul typo-candidate/fastText/near-duplicate yang disebut pada referensi tetap tersedia di '
   '`legacy/preprocessing.py`, tetapi tidak dijalankan oleh notebook teks ini.')

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'},
})
nbf.write(notebook, ROOT / 'legacy/Preprocessing_mbg_real.ipynb')
print(f'Built {len(cells)} cells; {sum(c.cell_type == "code" for c in cells)} code cells')
