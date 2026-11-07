# title: Ekstraksi terstruktur: uang, jumlah, tanggal, gejala, makanan
# when: Mengubah teks jadi tabel: apa, kapan, berapa, siapa terdampak. Bahan peta isu dan slide.
# avoid: Tanggal relatif tanpa waktu posting. Angka dari komentar bukan data resmi: jangan dijumlahkan jadi jumlah korban.
# needs: domain/mbg_lexicon
# tags: ekstraksi regex tanggal uang jumlah dateparser gejala makanan event informasi
import re
from functools import lru_cache
import pandas as pd

_UNIT = {"rb": 1e3, "ribu": 1e3, "k": 1e3, "jt": 1e6, "juta": 1e6, "m": 1e9, "miliar": 1e9, "milyar": 1e9, "t": 1e12, "triliun": 1e12}
MONEY_RE = re.compile(r"(?<![\w.])(rp\.?\s*)?(\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)\s*(rb|ribu|k|jt|juta|miliar|milyar|triliun|m|t)?(?![\w])", re.I)
QTY_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:[.,]\d{3})+|\d+|puluhan|ratusan|ribuan|belasan)\s+(siswa|murid|anak|korban|santri|guru|orang|warga|porsi|sekolah|dapur|sppg|penerima)\b", re.I)
_BULAN = "januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember|jan|feb|mar|apr|jun|jul|agu|agt|ags|sep|sept|okt|nov|des"
DATE_RE = re.compile(rf"\b(\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}|\d{{1,2}}\s+(?:{_BULAN})(?:\s+\d{{4}})?|(?:{_BULAN})\s+\d{{4}}|kemarin|kemaren|hari ini|tadi pagi|tadi siang|minggu lalu|minggu kemarin|bulan lalu|bulan kemarin|tahun lalu)\b", re.I)
_REL = {"kemarin": 1, "kemaren": 1, "hari ini": 0, "tadi pagi": 0, "tadi siang": 0, "minggu lalu": 7, "minggu kemarin": 7, "bulan lalu": 30, "bulan kemarin": 30, "tahun lalu": 365}

def parse_money(text):
    """Nilai rupiah. Angka tanpa 'Rp' dan tanpa satuan dibuang agar tahun dan jumlah orang tidak ikut."""
    out = []
    for m in MONEY_RE.finditer(str(text)):
        rp, num, unit = m.group(1), m.group(2), (m.group(3) or "").lower()
        if not rp and not unit:
            continue
        if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", num):
            val = float(re.sub(r"[.,]", "", num))
        else:
            val = float(num.replace(",", "."))
        out.append(val * _UNIT.get(unit, 1))
    return out

def parse_quantity(text):
    return [(m.group(1).lower(), m.group(2).lower()) for m in QTY_RE.finditer(str(text))]

@lru_cache(maxsize=50000)
def _resolve(span, base):
    s = span.lower()
    b = pd.Timestamp(base) if base else None
    if s in _REL:
        return (b - pd.Timedelta(days=_REL[s])).strftime("%Y-%m-%d") if b is not None else None
    import dateparser
    cfg = {"PREFER_DAY_OF_MONTH": "first", "DATE_ORDER": "DMY"}
    if b is not None:
        cfg["RELATIVE_BASE"] = b.to_pydatetime()
    d = dateparser.parse(span, languages=["id"], settings=cfg)
    return d.strftime("%Y-%m-%d") if d else None

def parse_dates(text, base=None):
    """[(teks, YYYY-MM-DD atau None)]. base = waktu posting."""
    return [(m.group(1), _resolve(m.group(1), base)) for m in DATE_RE.finditer(str(text))]

def extract_records(texts, times=None):
    """Satu baris per dokumen: uang, jumlah, tanggal, makanan, gejala, insiden, entitas."""
    food, symptom, incident, ent = build_matcher({"x": MBG_FOOD}), build_matcher({"x": MBG_SYMPTOM}), build_matcher({"x": MBG_INCIDENT}), build_matcher(MBG_ENTITIES)
    bases = [None] * len(texts) if times is None else [None if t != t or t is None else pd.to_datetime(t, errors="coerce", utc=True, format="mixed").tz_localize(None).strftime("%Y-%m-%d") if not pd.isna(pd.to_datetime(t, errors="coerce", utc=True, format="mixed")) else None for t in times]
    rows = []
    for t, b in zip(texts, bases):
        t = "" if t is None or t != t else str(t)
        rows.append({"uang": parse_money(t), "jumlah": parse_quantity(t), "tanggal": [d for _, d in parse_dates(t, b) if d],
                     "makanan": sorted({x[0] for x in find_terms(t, food)}), "gejala": sorted({x[0] for x in find_terms(t, symptom)}),
                     "insiden": sorted({x[0] for x in find_terms(t, incident)}), "entitas": sorted({k for x in find_terms(t, ent) for k in x[1]})})
    return pd.DataFrame(rows)

def ner_entities(texts, model_path, batch=32):
    """NER transformer per teks. Pakai pada subset insiden, bukan semua baris."""
    import torch
    from transformers import pipeline
    ner = pipeline("token-classification", model=str(model_path), aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1, batch_size=batch)
    return [[(e["entity_group"], e["word"]) for e in r] for r in ner([str(t)[:512] for t in texts])]
