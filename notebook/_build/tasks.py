"""Daftar semua notebook: folder, slug kernel, dataset Kaggle, dan kebutuhan GPU."""

OWNER = "ivanwllm"
PACK_SLUG = "gemastik-nlp-00-setup-offline"

DS_TIKTOK = "sinryurifal/dataset-komentar-tiktok-mbg-makan-bergizi-gratis"
DS_YT = "dzikriraihan/indonesian-youtube-comments-mbg-program"
DS_YT2 = "aryachovapratama/makan-bergizi-gratis-platfrom-yt"
DS_NEEV = "neevvv/indonesia-mbg-sentiment-analysis"
DS_HANIF = "hanif281103/sentimen-publik-terhadap-makan-bergizi-gratis"
DS_ZAIDDAN = "muhammadzaiddan/data-sentimen-makanan-bergizi-gratis-di-indonesia"
DS_NEWS_TITLE = "ibamibrahim/indonesian-news-title"
DS_HOAX = "ireddragonicy/indonesian-hoax-news-dataset"
DS_INDOSUM = "linkgish/indosum"
DS_ABUSIVE = "ilhamfp31/indonesian-abusive-and-hate-speech-twitter-text"
DS_TOKPED = "farhan999/tokopedia-product-reviews"
DS_PRDECT = "jocelyndumlao/prdect-id-indonesian-emotion-classification"
DS_NER2K = "giant77/idner-news-2k"

ALL_DATASETS = [
    DS_TIKTOK, DS_YT, DS_YT2, DS_NEEV, DS_HANIF, DS_ZAIDDAN, DS_NEWS_TITLE, DS_HOAX,
    DS_INDOSUM, DS_ABUSIVE, DS_TOKPED, DS_PRDECT, DS_NER2K,
]


def _task(num, name, family, title, datasets=(), gpu=True):
    task_id = f"{num}_{name}"
    return {
        "id": task_id,
        "folder": f"{family}/{task_id}" if family else task_id,
        "slug": f"gemastik-nlp-{num}-{name.replace('_', '-')}",
        "kernel_title": f"Gemastik NLP {num} {title}",
        "datasets": list(datasets),
        "gpu": gpu,
    }


A, B, C = "a_tanpa_label", "b_klasifikasi", "c_regresi"
D, E, F = "d_label_per_token", "e_pasangan_teks", "f_generatif"

TASKS = [
    {
        **_task("00", "setup_offline", "", "setup offline", ALL_DATASETS, gpu=False),
        "internet": True,
    },
    _task("01", "topic_modeling", A, "topic modeling", [DS_YT]),
    _task("02", "tren_topik_waktu", A, "tren topik waktu", [DS_NEEV]),
    _task("03", "clustering", A, "clustering", [DS_NEWS_TITLE]),
    _task("04", "weak_supervision", A, "weak supervision", [DS_YT]),
    _task("05", "kata_kunci", A, "kata kunci", [DS_YT]),
    _task("06", "anomali_teks", A, "anomali teks", [DS_YT, DS_NEWS_TITLE]),
    _task("07", "duplikat_spam", A, "duplikat spam", [DS_TIKTOK], gpu=False),
    _task("08", "semi_supervised", B, "semi supervised"),
    _task("09", "biner", B, "biner", [DS_ABUSIVE]),
    _task("10", "multi_kelas", B, "multi kelas", [DS_YT]),
    _task("11", "domain_shift", B, "domain shift", [DS_YT]),
    _task("12", "multi_label", B, "multi label", [DS_ABUSIVE]),
    _task("13", "absa", B, "absa"),
    _task("14", "regresi_ordinal", C, "regresi ordinal", [DS_TOKPED]),
    _task("15", "ner", D, "ner"),
    _task("16", "pos_tagging", D, "pos tagging"),
    _task("17", "term_aspek", D, "term aspek"),
    _task("18", "keyphrase", D, "keyphrase"),
    _task("19", "qa_span", D, "qa span"),
    _task("20", "kemiripan", E, "kemiripan"),
    _task("21", "nli", E, "nli"),
    _task("22", "retrieval", E, "retrieval", [DS_INDOSUM]),
    _task("23", "konsistensi", E, "konsistensi", [DS_INDOSUM]),
    _task("24", "summarization", F, "summarization", [DS_INDOSUM]),
    _task("25", "terjemahan", F, "terjemahan"),
    _task("26", "normalisasi", F, "normalisasi"),
    _task("27", "augmentasi", F, "augmentasi"),
    _task("28", "llm_ekstraksi", F, "llm ekstraksi", [DS_YT]),
]
