# title: Branch C: URL, HTML, emoji, huruf berulang, angka, negasi
# when: Media sosial. Tiap saklar diuji terpisah.
# avoid: Menghapus URL pada spam dan buzzer: ganti token, jangan hapus.
# tags: url html emoji mention angka negasi
import html
import re

def clean_advanced(text, url="HTTPURL", mention="@USER", demojize=True, repeat=True, number=None, neg=False, lower=True):
    t = html.unescape(str(text))
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"https?://\S+|www\.\S+", f" {url} ", t)
    t = re.sub(r"@\w+", f" {mention} ", t)
    if demojize:
        import emoji
        t = emoji.demojize(t, delimiters=(" ", " "))
    if repeat:
        t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    if number:
        t = re.sub(r"\d+(?:[.,]\d+)*", f" {number} ", t)
    if neg:
        t = re.sub(r"\b(tidak|tak|bukan|jangan|belum|kurang)\s+(\w+)", r"\1_\2", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    if not lower:
        return t
    return t.lower().replace(url.lower(), url).replace(mention.lower(), mention)
