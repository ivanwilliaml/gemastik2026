import html
import unicodedata

NEGATIONS = {'tidak', 'tak', 'bukan', 'belum', 'jangan', 'tanpa', 'gak', 'nggak', 'ga', 'not', 'never'}


def safe_text(value):
    """Missing tetap menjadi kosong; angka, emoji, negasi, tanda baca dipertahankan."""
    if not isinstance(value, str):
        return ''
    import ftfy
    text = ftfy.fix_text(value, unescape_html=False, uncurl_quotes=False,
        fix_latin_ligatures=False, fix_character_width=False, remove_control_chars=False, normalization=None)
    text = unicodedata.normalize('NFC', html.unescape(text))
    text = text.translate(dict.fromkeys(map(ord, '\ufeff\u200b\u2060\x00'), None))
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def preprocess_text(value, preset='native', slang=None, stopwords=()):
    text = safe_text(value)
    if preset == 'native':
        return text
    if preset == 'indobertweet':
        import emoji
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+|\burl\b', 'HTTPURL', text)
        text = re.sub(r'(?<!\w)@\w+|\buser\b', '@USER', text)
        return safe_text(emoji.demojize(text))
    if preset not in {'normalized', 'bow'}:
        raise ValueError(f'Preset tidak dikenal: {preset}')
    text = text.casefold()
    if slang:
        text = re.sub(r'\b[^\W_]+\b', lambda m: slang.get(m[0], m[0]), text)
    if preset == 'bow':
        text = re.sub(r'https?://\S+|www\.\S+|@\w+', ' ', text)
        # Stopword diberikan pengguna, negasi selalu dilindungi; tidak ada stemming otomatis.
        blocked = set(stopwords) - NEGATIONS
        text = ' '.join(w for w in re.findall(r'[^\W\d_]+', text) if w not in blocked)
    return safe_text(text)


def make_cleaner(cfg):
    return lambda value: preprocess_text(value, cfg['preprocess'], cfg.get('slang'), cfg.get('stopwords', ()))
