# Indonesian NLP resources other than transformer models: lexicons, slang dictionaries, tools, external datasets

Verification method (2026-11-05). URLs were checked with HTTP requests from this machine. "200" means the URL resolved on that date. Sizes are HTTP content-length or GitHub tree values. Licences come from the GitHub API (`spdx_id`), Hugging Face tags or README text. "No licence file" means GitHub reports none, so default copyright applies; that does not mean the resource is free to reuse.

"Local test" means I ran it myself on this laptop in an isolated venv (Python 3.12.6 on Windows 10). These are my own quick measurements, not published results, and they are labelled as such.

**Not measured:** the topic-model memory benchmark (500k documents with BERTopic, UMAP or HDBSCAN) was written but never run, because the coordinator stopped it. There are no local RAM or time figures for any topic-model component. The nlp-id, BERTopic, UMAP and HDBSCAN install was also killed partway, so none of them ran here, and nlp-id speed and offline behaviour are unknown. The local timings below were taken before that interrupted install.

## Slang and normalisation

### Takeaway
Download `colloquial-indonesian-lexicon.csv` plus the two small `new_kamusalay` and `kamus_singkatan` lists, and apply them with a plain Python dict lookup. Treat normalisation as a switch to test: it changed tweets a lot, but gave no reliable gain in my classical-model test.

### Cited Findings
- **nasalsabila/kamus-alay** resolves; the CSV is 3,093,160 bytes. Local count: 15,006 rows, but only 4,331 unique slang forms and 4,410 unique slang-formal pairs; 71 slang forms map to more than one formal word. Columns: slang, formal, In-dictionary, context, category1-3. GitHub reports no licence file — [CSV](https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv), [repo](https://github.com/nasalsabila/kamus-alay), [publication linked in README](https://ieeexplore.ieee.org/abstract/document/8629151)
- **meisaputri21 abbreviation list**: `kamus_singkatan.csv`, 18,911 bytes, 1,309 lines, semicolon-separated; no licence file — [file](https://raw.githubusercontent.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset/master/kamus_singkatan.csv)
- **Ibrohim & Budi alay dictionary**: `new_kamusalay.csv`, 285,941 bytes, 15,167 lines, no header, read as latin-1. Their README states CC BY-NC-SA 4.0 — [file](https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/new_kamusalay.csv), [repo](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection)
- **IndoCollex**: the correct repo is `haryoa/indo-collex` (MIT). The dictionary TSV is 39,167 bytes, 2,623 lines; `data/full.csv` is 63,999 bytes — [dictionary](https://raw.githubusercontent.com/haryoa/indo-collex/main/dict/inforformal-formal-Indonesian-dictionary.tsv), [repo](https://github.com/haryoa/indo-collex)
- **DEAD LINK:** `github.com/haryobagus/IndoCollex` returned 404 — [URL tested](https://github.com/haryobagus/IndoCollex)
- **STIF** (`haryoa/stif-indonesia`, MIT): repo about 1.2 MB. It holds aligned informal/formal text (`data/labelled/train.inf` and `train.for`, plus dev and test) and unlabelled batches. It is style-transfer data and Moses-based code (`scripts/download_moses.sh`), not a ready lookup table — [repo](https://github.com/haryoa/stif-indonesia)
- **Local test, slang coverage:** the kamus-alay dict changed 13.1% of tokens and touched 78.9% of documents on EmoT tweets, and 9.7% of tokens and 62.1% of documents on the Ibrohim tweets, but only 0.6% of tokens and 14.5% of documents on SmSA.
- **Local test, speed:** plain dict token lookup over 500k documents took 1.53 s on repeated real text, against 16.4 s extrapolated for FlashText replacement.

### Inferences
- A one-to-one dict is enough for this file because only 71 forms are ambiguous. Use `setdefault` or review those 71 by hand.
- Kamus-alay has no stated licence and Ibrohim's list is non-commercial. That is probably fine for a competition, but check the competition rules on external data.
- MBG-specific terms (MBG, SPPG, BGN) are unlikely to be well covered by these older lists. I did not test this; build a small custom list from the data on the day.

### Gaps
- No competition rule text was available, so whether external lexicons are allowed is unknown.
- I did not count unique pairs in STIF or open the IndoCollex category breakdown.

## Sentiment and emotion lexicons

### Takeaway
InSet and EmoLex are real and downloadable, but **do not use InSet alone as a labeller**: in my test it labelled 76-91% of SmSA reviews as negative, with macro-F1 of 0.24-0.35. Use lexicons only as extra features or for exploration.

### Cited Findings
- **InSet** (`fajri91/InSet`): `positive.tsv` is 41,462 bytes and `negative.tsv` is 82,788 bytes. The README states 3,609 positive and 6,609 negative words, weights -5 to +5. GitHub reports no licence file — [repo](https://github.com/fajri91/InSet), [positive](https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv), [negative](https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv)
- **Local inspection of InSet:** 6,607 unique negative words and 1,142 words appear in both lists. Common function words carry strong negative weights: `yang` -5, `tidak` -5, `sangat` -5, `saya` -3, `tempat` -3. `enak` is in both lists (+5 and -4).
- **Local test, InSet-only rule on the SmSA training set (11,000 rows, gold 6,416 positive / 3,436 negative / 1,148 neutral):**

| Scoring rule | Accuracy | Macro-F1 | Predicted negative |
|---|---|---|---|
| Negative weight overrides positive | 0.3417 | 0.2448 | 90.8% |
| Sum both weights | 0.4338 | 0.3467 | 75.9% |
| Drop the 1,142 overlap words | 0.4234 | 0.3448 | 76.1% |

- Separately, a 2025 comparison reports the opposite skew for InSet labelling (89.66% neutral) on its own data, so the bias depends on the data and setup. I read this only as a search summary — [search result listing](https://paperguide.ai/papers/9fec43b9-63e9-4628-811a-e60bd0c353bb-automatic-indonesian-sentiment-lexicon-curation-with-sentiment-valence-tuning-for-social-media)
- **NRC EmoLex:** the zip is 25,878,449 bytes (136 files). Local check: `NRC-Emotion-Lexicon/OneFilePerLanguage/Indonesian-NRC-EmoLex.txt` has 14,155 lines, of which 6,453 have at least one emotion or polarity flag — [zip](https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip)
- EmoLex terms: free for non-commercial research and education, no redistribution. Translations come from Google Translate and the page warns some "may still be incorrect" — [NRC page](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
- The `nrc.canada.ca` page for the same lexicons returned 403 to my request, so use the saifmohammad.com URLs — [URL tested](https://nrc.canada.ca/en/research-development/products-services/technical-advisory-services/sentiment-emotion-lexicons)
- **HoASA is a labelled hotel aspect-sentiment dataset, not a lexicon** (see the datasets section) — [IndoNLU paper](https://arxiv.org/pdf/2009.05387)
- **Barasa** (Indonesian SentiWordNet mapping, MIT per the search summary) resolves; the SEACrowd entry is a loader script only — [repo](https://github.com/neocl/barasa), [loader](https://huggingface.co/datasets/SEACrowd/barasa)
- **SentiStrengthID** (`agusmakmun`, GPL-3.0, last push 2016) and `ramaprakoso/analisis-sentimen` (no licence file) both resolve; I did not inspect their contents — [SentiStrengthID](https://github.com/agusmakmun/SentiStrengthID), [analisis-sentimen](https://github.com/ramaprakoso/analisis-sentimen)
- `louisowen6/NLP_bahasa_resources` (MIT) is an index of further links, including sentiment, slang and swear-word sections; last updated 15 March 2022 per its README — [repo](https://github.com/louisowen6/NLP_bahasa_resources)
- **Local test, InSet as TF-IDF features (4 aggregate scores added):** SmSA +0.0053 macro-F1 (8/10 folds better); EmoT +0.0002; Ibrohim HS -0.0002.

### Inferences
- InSet's negative list behaves like a list with noisy, corpus-derived weights. If used, remove function words first. I did not test a cleaned version.
- Lexicon features helped only slightly and only on the sentiment task. Their more practical uses are exploration and seeding weak labels for human review.

### Gaps
- **Vania lexicon:** two searches found nothing confirming this resource or a download URL. Do not list it as verified. The paper title I tried came from my memory and could not be confirmed.
- **SentiWords ID:** I found no Indonesian resource with this name. Barasa is the closest match.
- SentIL (Wijayanti & Arisal, 2021) appeared in search, but I found no download URL — [listing](https://www.x-mol.com/paper/1366900216915652608)
- I did not run an EmoLex ablation.

## Stopwords, stemmers, tokenisers

### Takeaway
The three common stopword lists are effectively the same list, and it contains negations. Removing it clearly hurt in my test, so skip it for classifiers and use it only for topic words and word clouds. Sastrawi is much less of a problem than feared if you stem unique words once and map.

### Cited Findings
- **Local test, the lists are the same:** NLTK `indonesian` and masdevid both have 757 unique words with zero difference; stopwords-iso has 758, with one extra word.
- **Local test, harmful entries:** the list contains `tidak`, `tak`, `bukan`, `jangan`, `belum`, `kurang`, `sangat`, `baik`, `tapi`, `tetapi`, `ingin`, `mau`, `percuma`.
- **URLs:** [NLTK stopwords.zip, 37,733 bytes](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip); [masdevid list, 6,446 bytes, no licence file](https://raw.githubusercontent.com/masdevid/ID-Stopwords/master/id.stopwords.02.01.2016.txt); [stopwords-iso, 6,444 bytes, MIT](https://raw.githubusercontent.com/stopwords-iso/stopwords-id/master/stopwords-id.txt); [spaCy Indonesian stop words, 6,507 bytes](https://raw.githubusercontent.com/explosion/spaCy/master/spacy/lang/id/stop_words.py); [Tala thesis PDF, 460,552 bytes](https://www.illc.uva.nl/Research/Publications/Reports/MoL-2003-02.text.pdf); [ranks.nl page](https://www.ranks.nl/stopwords/indonesian)
- **NLTK offline:** unzip so that `<dir>/corpora/stopwords/indonesian` exists, then set `nltk.data.path`. I read the file directly from the zip, but did not run `nltk.corpus.stopwords` itself offline.
- **PySastrawi** 1.2.1 (MIT, pure-Python wheel 0.2 MB) — [PyPI](https://pypi.org/project/PySastrawi/), [repo](https://github.com/har07/PySastrawi)
- **Local test, Sastrawi speed on real tweets (about 21 tokens each, single core), latest repeat runs:** about 1,060-1,080 tweets/s uncached (about 0.13 h for 500k); about 2,000 tweets/s with `create_stemmer()`'s built-in cache (about 0.07 h); stemming a 36,346-word vocabulary took about 4.2 s; mapping 500k documents through the resulting dict took 1.97 s.
- **Local test, variance warning:** one run of each version was about 4x slower (about 215-233 tweets/s uncached, about 0.6 h for 500k). This appeared on whichever version ran first after other work, so it looks like machine state rather than version. Versions 1.2.0 and 1.2.1 were equal in repeat runs. Plan on the slow figure as the worst case.
- **Caching approach:** `factory.create_stemmer()` returns a `CachedStemmer` (local inspection). The fastest route is `vocab = set(tokens)`, then `m = {w: stemmer.stem(w) for w in vocab}`, then map tokens through `m`. The mapping is what makes re-runs cheap. A real 500k corpus will have a bigger vocabulary than my 36k test; 8,000 words/s gives the scale.
- **spaCy:** `compatibility.json` lists no `id_core` pipeline (0 matches), so spaCy offers rule-based Indonesian language data but no trained pipeline — [compatibility file](https://raw.githubusercontent.com/explosion/spacy-models/master/compatibility.json)
- **Stanza:** the Indonesian package exists. `models/default.zip` is 394,024,071 bytes; the whole repo is about 1.79 GB, so take only the default zip. Apache-2.0 — [default.zip](https://huggingface.co/stanfordnlp/stanza-id/resolve/main/models/default.zip), [repo](https://huggingface.co/stanfordnlp/stanza-id)
- **nlp-id** (`kumparan/nlp-id`, MIT): wheel 8.1 MB, requires Python >=3.11,<3.14, so 3.12 is allowed — [repo](https://github.com/kumparan/nlp-id), [PyPI](https://pypi.org/project/nlp-id/)
- **IndoNLP/NusaCrowd:** `IndoNLP/nusa-crowd` resolves (Apache-2.0, last push 2024); the active successor is `SEACrowd/seacrowd-datahub` (last push 2026-09-24). These are catalogues and loaders, not bundled data — [nusa-crowd](https://github.com/IndoNLP/nusa-crowd), [seacrowd](https://github.com/SEACrowd/seacrowd-datahub)
- `indoNLP` on PyPI: version 0.3.4, MIT, last upload 2022-10-15; not tested — [PyPI](https://pypi.org/project/indoNLP/)

### Inferences
- Stanza is too slow to be a sensible default for 500k rows on CPU, though I did not measure it. It suits a small sample only.
- If the student needs a filtered stoplist, removing negations and intensifiers recovers part of the loss (see the preprocessing section).

### Gaps
- nlp-id speed and whether it needs anything beyond its wheel: **not measured**.
- Stanza throughput: **not measured**.
- Whether masdevid and NLTK derive from Tala: their identical content suggests a common source, but I did not compare them against the thesis appendix.

## Labelled external datasets and offline caching

### Takeaway
Hugging Face loading scripts no longer work: `datasets` 5.1.0 failed on all three script datasets I tried. Use GitHub raw files or repos that already hold parquet, then save local copies and test reloading with the network off.

### Cited Findings
- **Local test, script failure:** `load_dataset` on `indonlp/NusaX-senti`, `indonlp/indonlu` and `afaji/indonli` raised `RuntimeError: Dataset scripts are no longer supported`. So `trust_remote_code` is not a route on the current version.
- **`convert_to_parquet`:** the `datasets-cli` documentation now lists only `env`, `test` and `delete_from_hub`, so do not rely on this command, even though an old Hub error message still recommends it — [CLI docs](https://huggingface.co/docs/datasets/main/en/cli)
- **Hub parquet API:** `https://huggingface.co/api/datasets/<id>/parquet` lists files for NusaX-senti, but it returned an error for `indonlp/indonlu`, `afaji/indonli` and the SEACrowd entries tried. Passing such an API URL to `load_dataset('parquet', data_files=...)` failed with FileNotFoundError in my test — [example](https://huggingface.co/api/datasets/indonlp/NusaX-senti/parquet)
- **Tested route that works:** I ran `load_dataset('mteb/NusaX-senti', 'ind')`, then `split.to_parquet(path)`, then set `HF_HUB_OFFLINE=1` and ran `load_dataset('parquet', data_files=path)`. It reloaded 11,000 SmSA rows from `will702/fixed_indonlu`. Simple alternative: `pandas.read_parquet`/`read_csv` on files already on disk.
- **Windows note (local):** `huggingface_hub` warned that symlinks are unavailable, so its cache works in a degraded mode that may use more disk. Saving plain parquet files avoids depending on that cache.
- **SEACrowd warning:** `SEACrowd/smsa`, `emot`, `casa`, `hoasa`, `indolem_sentiment`, `id_multilabel_hs`, `indonli`, `nusax_senti` and others exist, but contain only loader scripts. Do not use them as the offline route — [example](https://huggingface.co/datasets/SEACrowd/smsa)

**Datasets checklist** (row counts are train/validation/test unless noted)

| Dataset | Rows | Exact URL (status 200) | Size | Licence | Notes |
|---|---|---|---|---|---|
| SmSA sentiment | 11,000 / 1,260 / 500 | [train TSV](https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa/train_preprocess.tsv); [parquet mirror](https://huggingface.co/datasets/will702/fixed_indonlu/resolve/main/data/smsa/train.parquet) | 2.19 MB train TSV | Conflict: GitHub Apache-2.0, README badge and HF tag MIT | Colloquial reviews, 3 classes. Also `valid_preprocess.tsv`, `test_preprocess.tsv` |
| EmoT emotion | 3,521 / 440 / 440 in mirror; paper says 442 test | [train CSV](https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/emot_emotion-twitter/train_preprocess.csv) | 0.67 MB | As above | Tweets, 5 classes; closest match to social media |
| CASA | 810 / 90 / 180 | [train CSV](https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/casa_absa-prosa/train_preprocess.csv) | 0.11 MB | As above | Automobile aspects |
| HoASA | 2,283 / 285 / 286 | [train CSV](https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/hoasa_absa-airy/train_preprocess.csv) | 0.38 MB | As above | Hotel aspects (Airy) |
| WReTE | 300 / 50 / 100 | [train CSV](https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/wrete_entailment-ui/train_preprocess.csv) | 0.10 MB | As above | Formal; low priority |
| NusaX sentiment | 500 / 100 / 400 per language | [Indonesian](https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/indonesian/train.csv), [Javanese](https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/javanese/train.csv), [Sundanese](https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/sundanese/train.csv); [parquet](https://huggingface.co/datasets/mteb/NusaX-senti/resolve/main/ind/train-00000-of-00001.parquet) | About 84 KB each train; whole mteb parquet set 1.28 MB | Conflict: GitHub Apache-2.0, HF CC BY-SA 4.0 | Parallel sentences in local languages |
| NusaWrites | Not counted | [Javanese sentiment](https://raw.githubusercontent.com/IndoNLP/nusa-writes/main/data/nusa_kalimat-senti-jav-train.csv), [Sundanese sentiment](https://raw.githubusercontent.com/IndoNLP/nusa-writes/main/data/nusa_kalimat-senti-sun-train.csv), [Javanese emotion](https://raw.githubusercontent.com/IndoNLP/nusa-writes/main/data/nusa_alinea-emot-jav-train.csv) | 489 KB, 499 KB, 1.96 MB; repo about 168 MB, so fetch single files | Apache-2.0 (GitHub) | Local-language writing |
| IndoLEM sentiment | 3,638 / 399 / 1,011 per fold | [train0](https://raw.githubusercontent.com/indolem/indolem/main/sentiment/data/train0.csv), [test0](https://raw.githubusercontent.com/indolem/indolem/main/sentiment/data/test0.csv) | 430 KB, 121 KB | No licence file | Binary sentiment. Paper: [arXiv](https://arxiv.org/pdf/2011.00677) |
| Ibrohim & Budi hate speech | 13,169 tweets | [re_dataset.csv](https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/re_dataset.csv), [abusive.csv](https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/abusive.csv) | 1.86 MB, 975 B | CC BY-NC-SA 4.0 | Multi-label; read as latin-1 |
| Alfina hate speech | Not counted | [txt](https://raw.githubusercontent.com/ialfina/id-hatespeech-detection/master/IDHSD_RIO_unbalanced_713_2017.txt) | 77 KB | No licence file | Small |
| Saputri tweet emotion | 4,401 (anger 1,101, happy 1,017, sadness 997, fear 649, love 637) | [CSV](https://raw.githubusercontent.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset/master/Twitter_Emotion_Dataset.csv) | 841 KB | No licence file | **Same class counts as EmoT pooled (4,401): do not count these as two datasets** |
| IndoNLI | Not counted | [train.jsonl](https://raw.githubusercontent.com/ir-nlp-csui/indonli/main/data/indonli/train.jsonl) | 3.79 MB | CC BY-SA 4.0 on HF; no licence file on GitHub | Entailment; useful for zero-shot only |
| MAFINDO hoax (BRIN) | Not counted | [HF page](https://huggingface.co/datasets/nlp-brin-id/fakenews-mafindo) | 19.5 MB | MIT tag | **Gated (manual approval): request access early**; README calls it beta, imbalanced |
| Indonesian hoax news | Not counted | [HF repo](https://huggingface.co/datasets/Rifky/indonesian-hoax-news) | Not recorded | No tag | Parquet; provenance unchecked |

- **IndoHoaxDetector is a model repo**, not a dataset: `theonegareth/IndoHoaxDetector`, 597 MB, MIT — [repo](https://huggingface.co/theonegareth/IndoHoaxDetector)
- **NusaBERT is a model**, not a dataset: `LazarusNLP/NusaBERT-base`, 445 MB, Apache-2.0 — [repo](https://huggingface.co/LazarusNLP/NusaBERT-base)
- Hugging Face searches for "makan bergizi", "mbg indonesia" and "makan gratis" returned no datasets — [API query](https://huggingface.co/api/datasets?search=makan+bergizi&limit=10)
- **Mirror caution:** `will702/fixed_indonlu` is a third-party mirror with no provenance check beyond matching row counts (6.55 MB, MIT tag). The EmoT test split differs from the paper (440 vs 442). Prefer GitHub raw as the authoritative copy and the mirror as a convenience — [mirror](https://huggingface.co/datasets/will702/fixed_indonlu), [paper](https://arxiv.org/pdf/2009.05387)
- **Do not assume test labels are usable:** both `test_preprocess.tsv` and `test_preprocess_masked_label.tsv` exist in IndoNLU. I did not open the former to confirm it holds real labels.

### Inferences
- With task unknown, the highest-value small set is EmoT, SmSA, Ibrohim HS, IndoLEM and NusaX, all small enough to fetch in one go.
- Domain shift is a real limit. SmSA is reviews and Ibrohim is older tweets, so external models may transfer poorly to MBG discourse. No study I found measures transfer to MBG.

### Gaps
- No evidence was collected on whether bootstrapping from these datasets improves performance on a new domain.
- `Rifky/indonesian-hoax-news` label quality and origin: not checked.

## Other tools

### Takeaway
Every tool with a timing below installed and ran on Windows with Python 3.12; sentencepiece and wordcloud were only checked for wheels. Two traps: **do not install `fasttext-wheel` alongside `fast-langdetect`**, and langdetect is about 100x slower. Prefer `fast-langdetect`, which bundles lid.176.ftz.

### Cited Findings
Local timings use repeated real tweets on one core, so treat them as rough. "x10" means extrapolated from 50k.

| Tool | Version, licence | Local result | Source |
|---|---|---|---|
| Plain dict | stdlib | Token lookup, 500k docs: 1.53 s | Local test |
| Regex | stdlib | 13 keywords, search 500k docs: 3.56 s | Local test |
| FlashText | 2.7, MIT, last upload 2018 | 4,331-key replace: 1.64 s per 50k (16.4 s x10) | [repo](https://github.com/vi3k6i5/flashtext) |
| pyahocorasick | 2.3.1, BSD-3 | 4,331-key scan: 0.77 s per 50k (7.7 s x10). Matches substrings, so needs your own word-boundary check | [repo](https://github.com/WojciechMula/pyahocorasick) |
| RapidFuzz | 3.14.6, MIT | `extractOne` against 4,331 keys: about 3,100 queries/s | [repo](https://github.com/rapidfuzz/RapidFuzz) |
| datasketch | 2.0.0, MIT | 64-perm MinHash + LSH insert: 5,000 docs in 1.36 s (about 2.3 min per 500k, linear extrapolation; RAM not measured) | [repo](https://github.com/ekzhu/datasketch) |
| emoji | 2.16.0 | `demojize`: 5.45 s per 50k (about 55 s x10) | [repo](https://github.com/carpedm20/emoji) |
| fast-langdetect | 1.1.0, MIT | About 38,000 docs/s; ships `resources/lid.176.ftz` (938,013 bytes) | [PyPI](https://pypi.org/project/fast-langdetect/) |
| langdetect | 1.0.9, last upload 2021 | About 329 docs/s | [repo](https://github.com/Mimino666/langdetect) |
| sentencepiece | 0.2.2, Apache-2.0 | cp312 win_amd64 wheel exists (1.2 MB); not run | [repo](https://github.com/google/sentencepiece) |
| wordcloud | 1.9.6, MIT | cp312 Windows wheel exists; not run | [repo](https://github.com/amueller/word_cloud) |

- **fastText trap (local):** with `fasttext-wheel` 0.9.2 and NumPy 2.5.4 installed, `import fasttext` worked but `fast_langdetect.detect` refused to run: its error says the deprecated package's `predict()` fails under NumPy 2. After uninstalling `fasttext-wheel` and force-reinstalling `fasttext-predict` it worked.
- **Language ID quality (local, tiny sample of 6 sentences; indicative only):** both detectors handled Indonesian and colloquial Indonesian. Both **failed on Javanese and Sundanese**: fast-langdetect `lite` gave `en`/`pl` for a Javanese sentence and `en`/`id` for a Sundanese one; langdetect gave `et`/`so` and `id`. Neither is reliable for short local-language text.
- **fastText files:** [lid.176.ftz, 938,013 bytes](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz); [lid.176.bin, 131,266,198 bytes](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin); [cc.id.300.bin.gz, 4,507,049,071 bytes](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.bin.gz). The last is 4.5 GB compressed, risky on 8 GB RAM.
- **Emoji sentiment:** [Emoji_Sentiment_Data_v1.0.csv, 78,880 bytes](https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1048/Emoji_Sentiment_Data_v1.0.csv), CC BY-SA 4.0 — [CLARIN record](https://www.clarin.si/repository/xmlui/handle/11356/1048). `emosent-py` 0.1.7 is on PyPI, untested — [PyPI](https://pypi.org/project/emosent-py/)
- **Emoji evidence:** a search summary reports +0.002 to +0.015 macro-F1 from emoji normalisation with IndoBERTweet. I did not open the paper or confirm its identity, so treat it as weak.
- **Fonts:** Indonesian uses plain Latin script, so no special font is needed for word text. The Noto Color Emoji raw URL I tried **returned 404** — [URL tested](https://raw.githubusercontent.com/googlefonts/noto-emoji/main/fonts/NotoColorEmoji.ttf). [Google Fonts Noto Sans folder](https://github.com/google/fonts/tree/main/ofl/notosans) resolves.
- **Pip offline:** `pip download` wheels the day before, then `pip install --no-index --find-links`. I did not test this end to end. PyPI shows wheels for pyahocorasick, sentencepiece, gensim and hdbscan on cp312 win_amd64.

### Inferences
- For whole-token slang or keyword work, a dict or regex is faster and simpler than FlashText. FlashText is useful for multi-word phrases; pyahocorasick is faster for counting, but needs boundary handling.
- For Javanese and Sundanese, a marker-word list built from NusaX and NusaWrites is likely better than language ID on short texts. This is untested.

### Gaps
- MinHash RAM at 500k, sentencepiece, wordcloud and `emosent-py`: **not measured**.
- FlashText has no separate licence check beyond GitHub's MIT, and its last PyPI release is from 2018, though it installed and ran on Python 3.12.

## Preprocessing evidence

### Takeaway
Skip stopword removal for classifiers: it was the only step with a clear, consistent effect in my test, and that effect was negative. Stemming is about neutral for classical models. For transformers, the published evidence favours light cleaning, except that slang normalisation has some support, in small and low-rigour studies.

### Cited Findings

**Local ablation (my own quick test, not a published result).** TF-IDF (1-2 grams, `min_df=2`) with LinearSVC, 5-fold CV repeated with 2 seeds, pooled splits, macro-F1. "Raw" already means lowercase with letters and digits only, so punctuation and emoji are stripped. "Better" counts folds beating raw, out of 10; folds are not independent, so this is not a significance test.

| Configuration | SmSA (n=12,760) | EmoT (n=4,401) | Ibrohim HS binary (n=13,169) |
|---|---|---|---|
| Raw | 0.8567 | 0.6729 | 0.8521 |
| + slang norm | -0.0001 (5/10) | +0.0010 (6/10) | -0.0033 (1/10) |
| + stopword removal | **-0.0493 (0/10)** | **-0.0176 (0/10)** | **-0.0124 (0/10)** |
| + stemming | +0.0010 (7/10) | +0.0052 (8/10) | -0.0010 (5/10) |
| + all three | -0.0478 (0/10) | -0.0136 (1/10) | -0.0133 (1/10) |
| Word + char TF-IDF, no prep | +0.0026 (6/10) | **+0.0173 (9/10)** | **+0.0100 (10/10)** |
| Word TF-IDF + InSet features | +0.0053 (8/10) | +0.0002 (6/10) | -0.0002 (6/10) |

- **Local follow-up on stoplist (SmSA train only, 5 folds, one seed):** raw 0.8503; full list 0.802; list minus 28 negation, intensifier and modal words 0.8347. So negations explain part of the loss, not all of it.
- **Limits of the local test:** no transformer was run, none of the data is about MBG, and the slang dict was applied without disambiguation.

**Published, classical models**
- Pradana & Hayaty (2019), SVM with TF-IDF: accuracy "did not improve and tends to decrease" with stemming or stopword removal, and the effect was not significant. The article page gives no per-scenario numbers; a search summary reports 81.44% with stemming alone against 81.06% with no preprocessing, which I could not confirm on the page — [Kinetik](https://kinetik.umm.ac.id/index.php/kinetik/article/view/912)
- Hidayatullah et al. (2016): stemming did not significantly raise accuracy with SVM or Naive Bayes on Indonesian tweets; a companion 2015 paper reports SVM with stemming 0.67% and 1.34% higher. Read via search summary only — [TELKOMNIKA](https://telkomnika.uad.ac.id/index.php/TELKOMNIKA/article/view/3113)
- Ibrohim & Budi (2019) applied normalisation, stemming and stopword removal as a fixed pipeline, so that paper is not an ablation; best accuracy reported was 66.12% in one scenario — [ACL Anthology](https://aclanthology.org/W19-3506.pdf)

**Published, transformers**
- Khairani, Mutiawani & Ahmadian (JTIIK 2024), Instagram news comments, 4 emotions: models **without** stopword removal and stemming did better, at 92.54% for IndoBERTweet and 88.81% for IndoBERT. Abstract-level only; I could not open the full text (403) — [JTIIK](https://jtiik.ub.ac.id/index.php/jtiik/article/view/8315)
- Izzati, Afendi & Masjkur (IPB, 2026), 5,403 Aceh flood tweets: IndoBERT was best with **minimal** preprocessing (macro-F1 0.8428) and declined as steps were added; IndoBERTweet was best under a more complex scenario (0.8435) and more stable. The steps per scenario were not on the page — [IPB](https://repository.ipb.ac.id/handle/123456789/178331)
- Budianoor et al. (JCTA 2026), 3,500 Instagram culinary comments: slang normalisation was the most useful single step, +0.0609 macro-F1 over no preprocessing; full pipeline macro-F1 0.8465. No separate stemming or stopword numbers on the page — [JCTA](https://publikasi.dinus.ac.id/jcta/article/view/15980)
- A Hasanuddin University study on 865 tweets reports +3.47% IndoBERT accuracy from normalisation. Read via search summary only; the page failed to load — [Unhas](https://scholar.unhas.ac.id/en/publications/text-normalization-for-indonesian-slang-words-in-sentiment-analys/)
- IndoBERTweet was trained with lowercasing, mentions and URLs converted to `@USER` and `HTTPURL`, and emoticons translated to text with the `emoji` package — [Koto et al. 2021](https://arxiv.org/pdf/2109.04607)
- In that paper's results table, IndoBERTweet (1M steps) scored 86.2 on IndoLEM sentiment, 90.4 on SmSA and 76.0 on EmoT, against 84.1, 87.9-88.7 and 71.0-73.3 for the two IndoBERT models. Values come from extracted PDF text, so check the table before quoting them — [Koto et al. 2021](https://arxiv.org/pdf/2109.04607)

### Inferences
- **For transformers, skip:** stemming, stopword removal, punctuation stripping. **Do:** the preprocessing the model was trained with. For IndoBERTweet that means lowercase, `@USER`, `HTTPURL`, and emoji to text.
- **Test, don't assume:** slang normalisation. Published gains come from small datasets, while my classical test showed about zero despite changing 10-13% of tweet tokens. Local-versus-published differences may reflect model type, data size or publication bias.
- **For classical baselines:** add character n-grams before spending time on normalisation. They were the best no-preprocessing change on both tweet datasets, and handle spelling variation directly.
- Use stemming and stopwords for **topic words and word clouds**, where readability matters more than accuracy. This is judgement, not a measured result.

### Gaps
- I found no rigorous, peer-reviewed ablation at large scale on Indonesian social media with significance testing. Most published evidence is small, single-split and from lower-tier venues.
- No local transformer ablation was run.

## Topic modelling toolkit

### Takeaway
On this laptop, precompute embeddings once, reduce them with IncrementalPCA, and cluster with MiniBatchKMeans. RAPIDS/cuML is not an option on Windows 10. **All local memory and time figures here are not measured**: the benchmark was stopped before it ran.

### Cited Findings
- **cuML/RAPIDS:** Windows support is Windows 11 through WSL2 only, and needs an NVIDIA GPU (compute capability 7.0+ recommended with 16 GB+ GPU RAM) — [RAPIDS install guide](https://docs.nvidia.com/datascience/install/)
- **BERTopic memory advice:** set `low_memory=True`, keep `calculate_probabilities=False`, raise `min_df` in `CountVectorizer` ("can significantly reduce memory"), consider online modelling. "The processing of millions of documents is quite computationally expensive and sufficient RAM is necessary" — [BERTopic FAQ](https://maartengr.github.io/BERTopic/faq.html)
- **Probabilities:** the parameter applies only to HDBSCAN or cuML HDBSCAN; other clusterers should use `.approximate_distribution` — [BERTopic FAQ](https://maartengr.github.io/BERTopic/faq.html)
- **Precompute embeddings:** compute once and pass them in, to avoid recalculating while iterating — [Best practices](https://maartengr.github.io/BERTopic/getting_started/best_practices/best_practices.html)
- **Online mode:** `partial_fit` with `IncrementalPCA(n_components=5)`, `MiniBatchKMeans(n_clusters=50)` and `OnlineCountVectorizer(decay=.01)` on chunks. You cannot use `.partial_fit` after `.fit`; `.topics_` holds only the latest batch unless you accumulate it; hierarchical topics depend on that — [Online docs](https://maartengr.github.io/BERTopic/getting_started/online/online.html)
- **Quality warning:** a third-party post reports users seeing subpar results with MiniBatchKMeans. Weak source, but test it — [devgem post](https://devgem.vercel.app/posts/performing-scalable-topic-modelling-with-bertopic-on-large-datasets)
- **HDBSCAN scale:** the docs tested up to 200,000 points in 10 dimensions and extrapolate about 500,000 points in 5 minutes on a modelled n² curve, while warning runtime curves "upward precipitously". They give no HDBSCAN memory figures — [HDBSCAN docs](https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html)
- **UMAP:** try `low_memory=True` for out-of-memory during nearest-neighbour search; PCA first suits very large data; `NUMBA_NUM_THREADS` limits threads; install `umap-learn`, not `umap`; a local `umap.py` will shadow the module — [UMAP FAQ](https://umap-learn.readthedocs.io/en/latest/faq.html)
- **Slow import:** BERTopic's FAQ notes slow UMAP imports with no current fix — [BERTopic FAQ](https://maartengr.github.io/BERTopic/faq.html)
- **OCTIS is a dependency trap:** requirements pin `scikit-learn==1.1.0`, `numpy<2.0`, `scipy<1.13`. Its PyPI release dates to 2024-07-22 — [requirements](https://raw.githubusercontent.com/MIND-Lab/OCTIS/master/requirements.txt). I did not try installing it, but scikit-learn 1.1.0 predates Python 3.12, so expect trouble.
- **Versions:** BERTopic 0.17.4 (MIT), hdbscan 0.8.44 with a cp312 win_amd64 wheel, umap-learn 0.5.12, gensim 4.4.0 (LGPL-2.1, cp312 win_amd64 wheel), Top2Vec 1.0.36 (last release 2024-11-14) — [BERTopic](https://pypi.org/project/bertopic/), [hdbscan](https://pypi.org/project/hdbscan/), [umap-learn](https://pypi.org/project/umap-learn/), [gensim](https://pypi.org/project/gensim/), [Top2Vec](https://pypi.org/project/top2vec/)
- **Dependency download (pip dry-run on this machine):** BERTopic pulls torch 2.15.0 (126.7 MB wheel), transformers, plotly, numba, llvmlite (41.9 MB), scipy (36.5 MB). The dry-run resolved NumPy 2.4.4 and scikit-learn 1.8.0; download everything the day before.

### Inferences
- **Arithmetic, not a measurement:** 500k x 384 float32 embeddings is 0.77 GB raw (500,000 x 384 x 4 bytes), so memmap them to disk. UMAP or HDBSCAN on all 500k within 8 GB is the risky path.
- **Suggested order, untested:** deduplicate, embed once and save, IncrementalPCA, then MiniBatchKMeans for the whole set. Run UMAP plus HDBSCAN only on a sample (for instance 20-50k) to discover topics, then assign the rest. `CountVectorizer(min_df=...)` high, no probabilities.
- **Gensim LDA** with `LdaMulticore` streams from disk and needs no embeddings, so it is a sensible low-RAM fallback. This is from general knowledge; I did not source or test it here.
- **UMAP offline pitfall:** numba compiles on first call, so run one small fit the day before to confirm the environment works, and expect the first call to be slow. General knowledge, not measured here.

### Gaps
- **Not measured:** peak RSS or time for IncrementalPCA, MiniBatchKMeans, UMAP, HDBSCAN or BERTopic on this laptop. The script exists but never ran.
- No published benchmark at 500k documents on 8 GB RAM was found. The official docs give no such figure.
- Top2Vec's and gensim's RAM at 500k: no source found. Topic quality of the MiniBatchKMeans route on Indonesian text: unknown.
