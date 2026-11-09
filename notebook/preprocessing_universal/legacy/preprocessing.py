"""Preprocessing komentar Indonesia. Offline, tanpa training atau penghapusan baris.

Salin modul ini, atau sel fungsi di notebook. API: preprocess_frame(df, text='text').
"""
from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import emoji
import ftfy
import pandas as pd


# %% Konfigurasi dan pola (dikompilasi sekali)
@dataclass(frozen=True)
class Config:
    views: tuple = ('social', 'tfidf', 'indobertweet')
    html_input: bool = False  # True hanya bila sumber memang markup HTML
    repair_encoding: bool = True
    keep_newlines: bool = False
    elongation: bool = True
    stopwords: bool = False
    stem: bool = False
    min_chars: int = 3
    spam_repeat: float = .7
    max_chars: int = 20000  # flag; tidak memotong teks


URL = re.compile(r'https?://[^\s<>]+|www\.[^\s<>]+', re.I)
EMAIL = re.compile(r'\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b')
MENTION = re.compile(r'(?<![\w@])@[\w.]+')
HASHTAG = re.compile(r'(?<!\w)#(\w+)')
PHONE = re.compile(r'(?<!\w)(?:\+62|62|0)8[\d -]{7,14}\d(?!\w)')
WORD = re.compile(r'[^\W\d_]+', re.UNICODE)
# Social elements are indivisible: never normalize within a URL/account/email/hashtag.
TOKEN = re.compile(r'https?://[^\s<>]+|www\.[^\s<>]+|[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}|(?<!\w)[@#][\w.]+|[^\W\d_]+', re.I)
ELONG = re.compile(r'([^\W\d_])\1{2,}', re.I)
OBF = re.compile(r'(?<!\w)[a-zA-Z][a-zA-Z0-9@*$]{3,}(?!\w)')
SPACE = re.compile(r'\s+')
INVISIBLE = dict.fromkeys(map(ord, '\ufeff\u200b\u2060\x00'), None)
NEGATIONS = set('tidak tak bukan belum jangan tanpa kurang nggak gak ga gk ora teu indak'.split())
PROTECTED = set('mbg sppg bgn apbn apbd bpom bpk puskesmas umkm indonesia jakarta bandung allah massa'.split())
AMBIGUOUS = set('sm dr bs pd lg org'.split())
SLANG = dict(zip(
    'gk ga gak nggak ngga tdk blm jgn yg dgn utk krn karna bgt tp udh sdh skrg gmn klo kalo jg bgs'.split(),
    'tidak tidak tidak tidak tidak tidak belum jangan yang dengan untuk karena karena banget tapi sudah sudah sekarang bagaimana kalau kalau juga bagus'.split()))
STOPWORDS = set('yang dan di ke dari untuk pada dengan ini itu atau juga saya kami kita mereka dia oleh sebagai dalam para serta'.split()) - NEGATIONS
# Contoh kurasi lokal, bukan kamus bahasa lengkap / frekuensi korpus.
VOCAB = set('bagus banget lama makanan makan bergizi gratis program keracunan anggaran korupsi rakyat sehat sekolah anak siswa guru pemerintah merata enak tidak belum jangan bukan kurang tanpa dapur menu bantuan'.split()) | PROTECTED | set(SLANG.values())
REGIONAL = {'jav': set('ora apik tenan mangan ojo durung'.split()),
            'sun': set('teu pisan ngeunah henteu kumaha'.split()),
            'min': set('indak rancak bana lamak'.split())}
REGIONAL_MAP = {'ora': 'tidak', 'teu': 'tidak', 'indak': 'tidak', 'apik': 'bagus'}
OBF_MAP = {'k3r4cunan': 'keracunan', 'k3r4cun4n': 'keracunan',
           'ker@cunan': 'keracunan', 'koru*si': 'korupsi'}


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]


def lexical_sub(text, transform):
    """Exact word callback; social tokens tetap utuh."""
    return TOKEN.sub(lambda m: transform(m[0]) if WORD.fullmatch(m[0]) else m[0], text)


# %% Safe cleaning: NFC, encoding, HTML opt-in, whitespace
def safe_clean(text, cfg=Config()):
    if cfg.repair_encoding:
        # Only repair encoding. Disable unrelated typographic/HTML transformations.
        text = ftfy.fix_text(text, unescape_html=False, uncurl_quotes=False,
                             fix_latin_ligatures=False, fix_character_width=False,
                             remove_control_chars=False, normalization='NFC')
    if cfg.html_input:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        text = soup.get_text(' ')  # parser already decodes HTML entities
    else:
        text = html.unescape(text)
    text = unicodedata.normalize('NFC', text).translate(INVISIBLE)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Preserve ZWJ/ZWNJ, variation selectors, valid scripts and emoji sequences.
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)
    if cfg.keep_newlines:
        return re.sub(r'[^\S\n]+', ' ', text).strip()
    return SPACE.sub(' ', text).strip()


# %% Normalizers: each is an independent, reusable function
def normalize_slang(text, slang):
    def replace(word):
        key = word.casefold()
        if key in PROTECTED or key in AMBIGUOUS:
            return word
        replacement = slang.get(key, word)
        return replacement.upper() if word.isupper() else replacement
    return lexical_sub(text, replace)


def normalize_elongation(text, vocab=VOCAB):
    def replace(word):
        if word.casefold() in PROTECTED or not ELONG.search(word):
            return word
        # Collapse only 3+ repeats. Never touch natural doubles such as massa/Allah.
        candidates = [ELONG.sub(r'\1', word), ELONG.sub(r'\1\1', word)]
        valid = [w for w in candidates if w.casefold() in vocab]
        return valid[0] if valid else word  # unknown: flag only, preserve original
    return lexical_sub(text, replace)


def segment_hashtag(tag):
    # Deterministic CamelCase/acronym boundaries. Lowercase compounds need a reviewed map.
    return re.sub(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', ' ', tag)


def model_view(text, kind='social', slang=SLANG, cfg=Config()):
    if kind == 'safe':
        return text
    if kind not in {'social', 'tfidf', 'indobertweet', 'topic'}:
        raise ValueError(f'View tidak dikenal: {kind}')
    if kind == 'indobertweet':
        # Match published IndoBERTweet recipe; English emoji shortcodes.
        text = text.lower()
        text = URL.sub('HTTPURL', text)
        text = MENTION.sub('@USER', text)
        return SPACE.sub(' ', emoji.demojize(text)).strip()
    text = EMAIL.sub(' EMAIL ', text)
    text = URL.sub(' URL ', text)
    text = PHONE.sub(' PHONE ', text)
    text = MENTION.sub(' USER ', text)
    text = HASHTAG.sub(lambda m: ' ' + segment_hashtag(m[1]) + ' ', text)
    if kind in {'social', 'topic'}:
        text = normalize_slang(text, slang)
        if cfg.elongation:
            text = normalize_elongation(text)
        text = emoji.demojize(text, delimiters=(' ', ' '))
        text = re.sub(r'([!?.,])\1{2,}', r'\1\1', text)
    text = text.casefold()
    if cfg.stopwords or kind == 'topic':
        text = lexical_sub(text, lambda w: '' if w in STOPWORDS else w)
    if cfg.stem:
        text = lexical_sub(text, lambda w: w if w in PROTECTED | NEGATIONS else stem_word(w))
    return SPACE.sub(' ', text).strip()


@lru_cache(maxsize=1)
def get_stemmer():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    return StemmerFactory().create_stemmer()


@lru_cache(maxsize=50000)
def stem_word(word):
    return get_stemmer().stem(word)


# %% Metadata, quality and language indications (never auto-delete)
def inspect_text(text, cfg=Config()):
    urls = [m[0].rstrip('.,!?;)') for m in URL.finditer(text)]
    without_links = EMAIL.sub(' ', URL.sub(' ', text))
    words = [w.casefold() for w in WORD.findall(without_links)]
    counts = Counter(words)
    repeats = 1 - len(counts) / len(words) if words else 0.
    hits = {lang: sorted(set(words) & markers) for lang, markers in REGIONAL.items()}
    regional = {k: v for k, v in hits.items() if v}
    en = set(words) & set('but implementation needs improvement the this good not'.split())
    id_hits = set(words) & set('bagus tidak makan program yang ini saya'.split())
    flags = {
        'empty': not text, 'short': len(text) < cfg.min_chars,
        'long': len(text) > cfg.max_chars, 'url': bool(urls),
        'mention': bool(MENTION.search(without_links)), 'hashtag': bool(HASHTAG.search(text)),
        'emoji': bool(emoji.emoji_count(text)), 'elongation': bool(ELONG.search(text)),
        'replacement_char': '\ufffd' in text,
        'obfuscated': any(re.search(r'[0-9@*$]', m[0]) for m in OBF.finditer(without_links)),
        'repeated_phrase': bool(re.search(r'\b(\w+(?:\s+\w+){0,3})(?:\s+\1){2,}\b', text, re.I)),
    }
    return dict(urls=urls, url_hosts=[urlsplit(u if '://' in u else 'https://' + u).hostname for u in urls],
                mentions=MENTION.findall(without_links), hashtags=HASHTAG.findall(text),
                emoji_count=emoji.emoji_count(text), char_count=len(text), word_count=len(words),
                exclamation_count=text.count('!'), question_count=text.count('?'),
                uppercase_ratio=sum(c.isupper() for c in text) / max(1, sum(c.isalpha() for c in text)),
                repeated_token_ratio=repeats, language_detected='mixed/uncertain',
                language_confidence=None, regional_markers=regional,
                code_switch_candidate=bool(en and (id_hits or regional)),
                quality_flags=[k for k, v in flags.items() if v],
                is_spam_candidate=bool((len(words) >= 6 and repeats >= cfg.spam_repeat) or len(urls) >= 3))


def language_fasttext(texts, model_path, min_words=5, min_score=.8):
    """Optional local model; score is a model score, not calibrated certainty."""
    import fasttext
    model = fasttext.load_model(str(model_path))
    results = []
    for text in texts:
        if len(WORD.findall(text)) < min_words:
            results.append(('mixed/uncertain', None))
            continue
        labels, scores = model.predict(text.replace('\n', ' '), k=1)
        results.append((labels[0].replace('__label__', '') if scores[0] >= min_score else 'mixed/uncertain', float(scores[0])))
    return pd.DataFrame(results, columns=['language_detected', 'language_confidence'])


# %% Batch API: input metadata, order and raw values stay unchanged
def preprocess_frame(df, text='text', cfg=Config(), slang=None):
    if text not in df:
        raise KeyError(f'Kolom {text!r} tidak ada. Pilih dari {list(df.columns)}')
    reserved = {'text_raw', 'text_safe_cleaned', 'preprocess_row_id', 'duplicate_group_id'}
    generated = reserved | {'text_slang_normalized', 'cleaning_flags', 'is_empty_text',
                            'is_missing_text', 'is_non_string_text', 'exact_duplicate_count'}
    generated |= {'text_' + view for view in cfg.views} | set(inspect_text('').keys())
    overlap = generated & set(df.columns)
    if overlap:
        raise ValueError(f'Kolom keluaran sudah ada: {sorted(overlap)}. Pakai input mentah.')
    out = df.copy()
    raw = df[text]
    is_string = raw.map(lambda v: isinstance(v, str))
    working = raw.where(is_string, '').fillna('')
    # Unique text processed once within this call, no unbounded persistent cache.
    unique = pd.unique(working)
    safe = {s: safe_clean(s, cfg) for s in unique}
    out['text_raw'] = raw.copy()
    out['preprocess_row_id'] = range(len(df))  # positional row trace, not a source comment ID
    out['is_missing_text'] = raw.isna().to_numpy()
    out['is_non_string_text'] = (~is_string & ~raw.isna()).to_numpy()
    out['text_safe_cleaned'] = working.map(safe).to_numpy()
    cleaned = out.text_safe_cleaned
    out['is_empty_text'] = cleaned.eq('')
    lexicon = SLANG if slang is None else slang
    out['text_slang_normalized'] = cleaned.map({s: normalize_slang(s, lexicon) for s in pd.unique(cleaned)})
    for view in cfg.views:
        lookup = {s: model_view(s, view, lexicon, cfg) for s in pd.unique(cleaned)}
        out['text_' + view] = cleaned.map(lookup)
    metadata = {s: inspect_text(s, cfg) for s in pd.unique(cleaned)}
    details = pd.DataFrame([metadata[s] for s in cleaned], columns=list(inspect_text('').keys()))
    for col in details:
        out[col] = details[col].to_numpy()
    out['cleaning_flags'] = [(['technical_changed'] if a != b else []) for a, b in zip(working, cleaned)]
    # Empty/invalid records must NOT all form a duplicate cluster.
    out['duplicate_group_id'] = [digest(s) if s else None for s in cleaned]
    counts = cleaned[cleaned.ne('')].value_counts()
    out['exact_duplicate_count'] = cleaned.map(counts).fillna(0).astype('int32')
    return out


# %% Reviewed lexicons: source/conflicts never silently disappear
def read_lexicon(path, variant='variant', canonical='canonical', **read_csv_options):
    """Adapter for Kamus Alay/Colloquial/Combined/custom CSV; no forced approval."""
    lex = pd.read_csv(path, dtype=str, keep_default_na=False, **read_csv_options)
    lex = lex.rename(columns={variant: 'variant', canonical: 'canonical'})
    if not {'variant', 'canonical'} <= set(lex):
        raise ValueError('Butuh kolom variant dan canonical, atau atur nama kolom adapter.')
    for c in ['variant', 'canonical']:
        lex[c] = lex[c].str.strip().str.casefold()
    for c, default in {'source': Path(path).name, 'category': 'slang', 'language': 'id',
                       'confidence': '', 'review_status': 'pending'}.items():
        if c not in lex:
            lex[c] = default
    return lex


def approved_slang(lex):
    multiple = lex.groupby('variant').canonical.nunique()
    eligible = lex.variant.map(multiple).eq(1) & lex.review_status.eq('approved')
    eligible &= ~lex.variant.isin(PROTECTED | AMBIGUOUS) & lex.variant.str.fullmatch(r'[^\W\d_]+')
    eligible &= lex.canonical.ne('')
    return {**SLANG, **dict(lex.loc[eligible, ['variant', 'canonical']].itertuples(index=False, name=None))}


RESOURCE_URLS = {
    'colloquial.csv': 'https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv',
    'kamusalay.csv': 'https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/new_kamusalay.csv',
    'combined.json': 'https://raw.githubusercontent.com/louisowen6/NLP_bahasa_resources/master/combined_slang_words.txt',
}


def external_lexicons(folder, download=False):
    """Prepare online once; reuse offline. All community mappings remain pending.
    Original resource files stay local; review redistribution terms at source.
    """
    import json
    import urllib.request
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tables = []
    for name, url in RESOURCE_URLS.items():
        path = folder / name
        if download and not path.exists():
            data = urllib.request.urlopen(url, timeout=45).read()
            tmp = path.with_suffix('.part')
            tmp.write_bytes(data)
            tmp.replace(path)
        if not path.exists():
            continue
        if name == 'combined.json':
            values = json.loads(path.read_text(encoding='utf-8'))
            table = pd.DataFrame(values.items(), columns=['variant', 'canonical'])
        elif name == 'kamusalay.csv':
            table = pd.read_csv(path, names=['variant', 'canonical'], encoding='latin-1', keep_default_na=False)
        else:
            table = pd.read_csv(path, keep_default_na=False).rename(columns={'slang': 'variant', 'formal': 'canonical'})
        table = table[['variant', 'canonical']].copy()
        table['source'] = url
        table['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        table['category'], table['language'] = 'slang', 'id'
        table['confidence'], table['review_status'] = '', 'pending'
        tables.append(table)
    if not tables:
        return pd.DataFrame(columns=['variant', 'canonical', 'source', 'source_sha256', 'category', 'language', 'confidence', 'review_status'])
    lex = pd.concat(tables, ignore_index=True).drop_duplicates(['variant', 'canonical', 'source'])
    for c in ['variant', 'canonical']:
        lex[c] = lex[c].str.strip().str.casefold()
    lex['conflict'] = lex.variant.map(lex.groupby('variant').canonical.nunique()).gt(1)
    return lex


# %% Optional typo proposals: SymSpell once per unique lowercase word
def typo_candidates(texts, frequencies, max_distance=1):
    """frequencies={word: count} from curated/train-only vocabulary, not test labels.
    Returns suggestions only. Never changes input; uppercase names/acronyms skipped.
    """
    from symspellpy import SymSpell, Verbosity
    checker = SymSpell(max_dictionary_edit_distance=max_distance)
    for word, freq in frequencies.items():
        checker.create_dictionary_entry(word, int(freq))
    words = Counter(w for t in texts for w in WORD.findall(t)
                    if w.islower() and len(w) >= 4 and w not in PROTECTED | AMBIGUOUS)
    rows = []
    for word, count in words.items():
        if word in frequencies or word in SLANG or any(word in x for x in REGIONAL.values()):
            continue
        suggestions = checker.lookup(word, Verbosity.CLOSEST, max_edit_distance=max_distance)
        for s in suggestions[:3]:
            rows.append((word, s.term, s.distance, s.count, count, 'pending'))
    return pd.DataFrame(rows, columns=['variant', 'canonical', 'edit_distance', 'frequency', 'occurrences', 'review_status'])


# %% Optional near duplicates: LSH candidates + exact verification, capped work
def near_duplicates(texts, threshold=.85, num_perm=64, max_candidates=100, min_chars=20):
    """Unique safe texts only, char-5 shingle Jaccard; no dense NxN matrix.
    Return verified edges and workload report. LSH can miss pairs; caps reduce recall.
    Groups are intentionally not merged automatically (transitive chaining risk).
    """
    from datasketch import MinHash, MinHashLSH
    from rapidfuzz.fuzz import ratio
    if not 0 < threshold <= 1 or max_candidates < 1:
        raise ValueError('threshold harus (0,1], max_candidates >= 1')
    unique = [s for s in dict.fromkeys(texts) if len(s) >= min_chars]
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    shingles, rows, capped = [], [], 0
    for i, text in enumerate(unique):
        chars = {text[k:k + 5] for k in range(len(text) - 4)}
        sketch = MinHash(num_perm=num_perm, seed=42)
        sketch.update_batch([c.encode('utf-8') for c in sorted(chars)])
        candidates = sorted(lsh.query(sketch), key=int)
        capped += len(candidates) > max_candidates
        for j in candidates[:max_candidates]:
            other = shingles[int(j)]
            score = len(chars & other) / len(chars | other)
            if score >= threshold:
                rows.append((digest(unique[int(j)]), digest(text), score, ratio(unique[int(j)], text), unique[int(j)], text))
        lsh.insert(str(i), sketch)
        shingles.append(chars)
    return pd.DataFrame(rows, columns=['left_id', 'right_id', 'jaccard', 'fuzzy_ratio', 'left_text', 'right_text']), dict(unique_indexed=len(unique), queries_capped=capped, pairs_verified=len(rows))


# %% Audit and portable export
def audit_views(frame):
    safe = frame.text_safe_cleaned
    return pd.DataFrame([dict(view=c, changed_rows=int(frame[c].ne(safe).sum()),
                              empty_rows=int(frame[c].eq('').sum()), unique_texts=frame[c].nunique())
                         for c in frame if c.startswith('text_') and c != 'text_raw'])


def export_result(frame, folder, cfg=Config(), resource_paths=()):
    import importlib.metadata
    import json
    import platform
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # JSONL keeps lists/dicts; CSV uses explicit JSON rather than Python repr.
    frame.to_json(folder / 'comments.jsonl', orient='records', lines=True, force_ascii=False)
    csv = frame.copy()
    for col in csv:
        if csv[col].map(lambda x: isinstance(x, (dict, list))).any():
            csv[col] = csv[col].map(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)
    csv.to_csv(folder / 'comments.csv', index=False, encoding='utf-8-sig')
    audit_views(frame).to_csv(folder / 'audit_views.csv', index=False)
    manifest = dict(config=asdict(cfg), rows=len(frame), python=platform.python_version(),
                    versions={m: importlib.metadata.version(m) for m in ['pandas', 'ftfy', 'emoji']},
                    resources={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in resource_paths},
                    rules_sha256=digest(json.dumps(dict(slang=SLANG, protected=sorted(PROTECTED),
                        negations=sorted(NEGATIONS), regional={k: sorted(v) for k, v in REGIONAL.items()},
                        obfuscation=OBF_MAP), sort_keys=True)),
                    note='No row deletion. Near-duplicate pass separate; elapsed time is not a 500k benchmark.')
    (folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return folder
