"""Text-only preprocessing in small deterministic steps; also embedded in the notebook.

API: preprocess_texts(series, slang=None, options=None). Input order/index are preserved.
No downloads, no fitting, no record deletion. Each stage can be copied independently.
"""
import html
import re
import unicodedata
from functools import lru_cache

import emoji
import ftfy
import pandas as pd


# %% resources
SLANG = dict(zip(
    'gk ga gak nggak ngga tdk blm jgn yg dgn utk krn karna bgt tp udh sdh skrg gmn klo kalo jg bgs'.split(),
    'tidak tidak tidak tidak tidak tidak belum jangan yang dengan untuk karena karena banget tapi sudah sudah sekarang bagaimana kalau kalau juga bagus'.split()))
PROTECTED = set('mbg sppg bgn apbn apbd bpom bpk umkm allah massa jakarta bandung'.split())
NEGATIONS = set('tidak tak bukan belum jangan tanpa kurang nggak gak ga gk ora teu indak'.split())
AMBIGUOUS = set('sm dr bs pd lg org'.split())
VOCAB = set('bagus banget lama makanan makan bergizi gratis program keracunan anggaran korupsi sehat sekolah anak siswa pemerintah merata enak'.split()) | set(SLANG.values()) | PROTECTED
STOPWORDS = set('yang dan di ke dari untuk pada dengan ini itu atau juga saya kami kita mereka dia oleh sebagai dalam para serta adalah sebuah'.split()) - NEGATIONS - PROTECTED
REGIONAL = {'ora': 'tidak', 'teu': 'tidak', 'indak': 'tidak', 'apik': 'bagus'}
OBFUSCATION = {'k3r4cunan': 'keracunan', 'k3r4cun4n': 'keracunan',
               'ker@cunan': 'keracunan', 'koru*si': 'korupsi'}
# URL/email/account/hashtag are single tokens: normalization never enters them.
TOKEN = re.compile(r'https?://[^\s<>]+|www\.[^\s<>]+|[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}|(?<!\w)[@#][\w.]+|[^\W\d_]+', re.I)
WORD = re.compile(r'[^\W\d_]+')
URL = re.compile(r'https?://[^\s<>]+|www\.[^\s<>]+', re.I)


def map_words(text, transform):
    return TOKEN.sub(lambda m: transform(m[0]) if WORD.fullmatch(m[0]) else m[0], text)


# %% validate
def valid_text(value):
    # None/NaN/pd.NA/numbers/list are not converted to literal "nan"/"123".
    return value if isinstance(value, str) else ''


# %% escaped_bytes
def decode_byte_escapes(text):
    def decode(match):
        try:
            return bytes.fromhex(match[0].replace('\\x', '')).decode('utf-8')
        except UnicodeDecodeError:
            return match[0]  # invalid bytes remain visible for review
    return re.sub(r'(?:\\x[0-9a-fA-F]{2})+', decode, text)


# %% encoding
def repair_encoding(text):
    return ftfy.fix_text(text, unescape_html=False, uncurl_quotes=False,
                        fix_latin_ligatures=False, fix_character_width=False,
                        remove_control_chars=False, normalization=None)


# %% unicode
def normalize_unicode(text):
    return unicodedata.normalize('NFC', text)  # retain emoji/local scripts/fullwidth


# %% html_markup
def remove_html(text):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')
    for tag in soup(['script', 'style']):
        tag.decompose()
    return soup.get_text(' ')


# %% entities
def decode_html_entities(text):
    return html.unescape(text)


# %% invisible
def remove_invisible(text):
    text = text.translate(dict.fromkeys(map(ord, '\ufeff\u200b\u2060\x00'), None))
    return re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)
    # ZWJ, ZWNJ and variation selectors remain: preserve emoji and valid writing.


# %% whitespace
def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()


# %% obfuscation
def normalize_obfuscation(text, mapping=OBFUSCATION):
    return re.sub(r'(?<!\w)[a-zA-Z][a-zA-Z0-9@*$]{3,}(?!\w)',
                  lambda m: mapping.get(m[0].casefold(), m[0]), text)


# %% urls
def replace_urls(text):
    # Preserve terminal sentence punctuation; a heuristic, not a complete URL parser.
    return URL.sub(lambda m: ' URL ' + m[0][len(m[0].rstrip('.,!?;)')):], text)


# %% emails
def replace_emails(text):
    return re.sub(r'\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b', ' EMAIL ', text)


# %% phones
def replace_phones(text):
    return re.sub(r'(?<!\w)(?:\+62|62|0)[ -]?8(?:[ -]?\d){8,11}(?!\w)', ' PHONE ', text)


# %% mentions
def replace_mentions(text):
    return re.sub(r'(?<![\w@])@[\w.]+', ' USER ', text)


# %% hashtags
def segment_hashtags(text, mapping=None):
    mapping = {} if mapping is None else mapping
    def split(match):
        word = match[1]
        # #tolakmbg stays tolakmbg unless an explicit reviewed map is supplied.
        return ' ' + mapping.get(word.casefold(), re.sub(
            r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', ' ', word)) + ' '
    return re.sub(r'(?<!\w)#(\w+)', split, text)


# %% lowercase
def lowercase(text):
    return text.casefold()


# %% slang
def normalize_slang(text, mapping=SLANG):
    return map_words(text, lambda w: mapping.get(w.casefold(), w)
                     if w.casefold() not in PROTECTED | AMBIGUOUS else w)


# %% elongation
def normalize_elongation(text, vocab=VOCAB):
    def shorten(word):
        if word.casefold() in PROTECTED:
            return word
        candidates = [re.sub(r'([^\W\d_])\1{2,}', replacement, word, flags=re.I)
                      for replacement in (r'\1', r'\1\1')]
        return next((w for w in candidates if w.casefold() in vocab), word)
    return map_words(text, shorten)


# %% laughter
def normalize_laughter(text):
    def replace(word):
        return 'wkwk' if re.fullmatch(r'(?:wk){2,}|w+k+w+k+[wk]*', word, re.I) else (
            'haha' if re.fullmatch(r'(?:ha){2,}h?', word, re.I) else word)
    return map_words(text, replace)


# %% reduplication
def normalize_reduplication(text, mapping=None):
    # A whitelist avoids turning grade S2, model GPT2, dates or quantities into words.
    mapping = {'anak2': 'anak-anak', 'orang2': 'orang-orang', 'teman2': 'teman-teman'} if mapping is None else mapping
    return re.sub(r'\b\w+2\b', lambda m: mapping.get(m[0].casefold(), m[0]), text)


# %% regional
def normalize_regional(text, mapping=REGIONAL):
    return map_words(text, lambda w: mapping.get(w.casefold(), w))


# %% typo
def apply_reviewed_typos(text, mapping=None):
    mapping = {} if mapping is None else mapping
    return map_words(text, lambda w: mapping.get(w.casefold(), w)
                     if w.casefold() not in PROTECTED | AMBIGUOUS else w)


# %% punctuation_repeat
def normalize_punctuation_repeat(text):
    return re.sub(r'([!?.,])\1{2,}', r'\1\1', text)


# %% emoji_describe
def describe_emoji(text):
    return emoji.demojize(text, delimiters=(' ', ' '))


# %% emoji_remove
def remove_emoji(text):
    return emoji.replace_emoji(text, replace=' ')


# %% emoticons
def remove_emoticons(text):
    return re.sub(r"(?<!\w)(?:[:;=8][\-^']?[)(/DPp]+|<3)(?!\w)", ' ', text)


# %% numbers
def remove_numbers(text):
    # Opt-in destructive view: amounts/dates become incomplete, inspect before using.
    return re.sub(r'(?<!\w)\d+(?:[.,:/-]\d+)*(?!\w)', ' ', text)


# %% punctuation
def remove_punctuation(text):
    # Keep all scripts/letters/digits; punctuation becomes a SPACE, never concatenation.
    return re.sub(r'[^\w\s]|_', ' ', text)


# %% repeated_words
def reduce_repeated_words(text):
    # Aggressive only: even "bagus bagus" can be expressive, so retain another view.
    return re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text, flags=re.I)


# %% stopwords
def remove_stopwords(text, words=STOPWORDS):
    blocked = set(words) - NEGATIONS - PROTECTED
    return ' '.join(w for w in text.split() if w.casefold() not in blocked)


# %% stemming
@lru_cache(maxsize=1)
def get_stemmer():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    return StemmerFactory().create_stemmer()


@lru_cache(maxsize=50000)
def stem_word(word):
    return get_stemmer().stem(word)


def stem_text(text):
    return map_words(text, lambda w: w if w.casefold() in PROTECTED | NEGATIONS else stem_word(w))


# %% tokens
def tokenize(text):
    # Transparent baseline tokenization of the clean view; not a model tokenizer.
    return text.split()


# %% indobertweet
def indobertweet_view(text):
    text = URL.sub('HTTPURL', text.lower())
    text = re.sub(r'(?<![\w@])@[\w.]+', '@USER', text)
    return normalize_whitespace(emoji.demojize(text))


# %% api
DEFAULT_OPTIONS = dict(html_input=False, escaped_bytes=False, regional=False,
                       remove_numbers=False, stem=True, stopwords=True,
                       slang=None, typos=None, hashtags=None)


def build_steps(options=None):
    unknown = set(options or {}) - set(DEFAULT_OPTIONS)
    if unknown:
        raise ValueError(f'Opsi tidak dikenal: {sorted(unknown)}')
    options = {**DEFAULT_OPTIONS, **(options or {})}
    safe = [('valid', valid_text)]
    if options['escaped_bytes']:
        safe.append(('escaped_bytes', decode_byte_escapes))
    safe += [('encoding', repair_encoding), ('unicode', normalize_unicode)]
    if options['html_input']:
        safe.append(('html', remove_html))
    safe += [('entities', decode_html_entities), ('invisible', remove_invisible), ('space', normalize_whitespace)]
    normalized = [('obfuscation', normalize_obfuscation), ('urls', replace_urls),
                  ('emails', replace_emails), ('phones', replace_phones), ('mentions', replace_mentions),
                  ('hashtags', lambda s: segment_hashtags(s, options['hashtags'])), ('lowercase', lowercase),
                  ('slang', lambda s: normalize_slang(s, SLANG if options['slang'] is None else options['slang'])),
                  ('elongation', normalize_elongation), ('laughter', normalize_laughter),
                  ('reduplication', normalize_reduplication)]
    if options['regional']:
        normalized.append(('regional', normalize_regional))
    normalized += [('typo', lambda s: apply_reviewed_typos(s, options['typos'])),
                   ('punctuation_repeat', normalize_punctuation_repeat), ('space', normalize_whitespace)]
    clean = [('emoji', remove_emoji), ('emoticons', remove_emoticons)]
    if options['remove_numbers']:
        clean.append(('numbers', remove_numbers))
    clean += [('punctuation', remove_punctuation), ('repeated_words', reduce_repeated_words)]
    if options['stopwords']:
        clean.append(('stopwords', remove_stopwords))
    if options['stem']:
        clean.append(('stemming', stem_text))
    clean.append(('space', normalize_whitespace))
    return dict(safe=safe, normalized=normalized, clean=clean)


def transform_unique(texts, function):
    # Invalid values handled separately, so unhashable input never becomes a cache key.
    texts = texts.map(valid_text) if function is valid_text else texts
    cache = {text: function(text) for text in pd.unique(texts)}
    return texts.map(cache)


def preprocess_texts(texts, options=None):
    """Same functions and order as notebook. No data-dependent fitting or randomness."""
    texts = texts.copy() if isinstance(texts, pd.Series) else pd.Series(texts)
    output = pd.DataFrame({'text_raw': texts})
    current = texts
    for branch, steps in build_steps(options).items():
        for _, function in steps:
            current = transform_unique(current, function)
        output['text_' + branch] = current
    output['text_emoji_described'] = output.text_normalized.map(describe_emoji)
    output['tokens'] = output.text_clean.map(tokenize)
    output['text_indobertweet'] = output.text_safe.map(indobertweet_view)
    return output
