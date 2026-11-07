# Pretrained Indonesian NLP models for the offline MBG final (state at 5 Nov 2026)

How to read these notes:
- **Repo metadata** (existence, licence field, downloads, last modified, gating, file sizes) was queried live from `https://huggingface.co/api/models/{id}?blobs=true` on 2026-11-05. Every ID in the tables returned 200 unless marked otherwise. Pattern for checking any row: [API example](https://huggingface.co/api/models/indolem/indobertweet-base-uncased?blobs=true).
- **"Weights MB"** is one copy of the weights (safetensors if present, otherwise the `.bin`). Whole repos can be far larger because they also hold TF, Flax and ONNX copies. Example: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` has 1,112 MB of weights in a 10,851 MB repo, so download with `allow_patterns`, not a blind snapshot.
- **"4 GB?"** is my estimate, not a measurement. Y = runs on the full 500k at a usable batch size. S = small batch or a sample only. N = not realistic for 500k within 5 hours.
- **Licence "none"** means the repo declares no licence in its metadata. That does not block offline use technically, but check the competition rules.
- **Scores are not comparable across rows** unless the dataset and split match. Most scores are from the card authors' own validation sets.

## Ranked per-task recommendation table

### Takeaway
No ready-made model has been evaluated on MBG text. Scores below come from other domains (mostly SmSA, EmoT, NERGrit, IndoToxic2024), so check any ready-made model on a hand-labelled MBG sample before relying on it. Hoax and aspect-based sentiment have no ready model worth trusting.

### Cited Findings

| Task | Rank | Repo ID | Weights MB | Licence | Published score (dataset, split) | 4 GB? |
|---|---|---|---|---|---|---|
| Fine-tune base, social media | 1 | `indolem/indobertweet-base-uncased` | 445 | apache-2.0 | Avg 86.1 to 86.5 across 7 Twitter datasets, vs IndoBERT 83.4, mBERT 79.6 ([card](https://huggingface.co/indolem/indobertweet-base-uncased/raw/main/README.md)) | Y |
| Fine-tune base, code-mixed/regional | 2 | `LazarusNLP/NusaBERT-base` | 443 | apache-2.0 | NusaX mean macro-F1 79.8 vs IndoBERT base 78.5; IndoNLU classification mean 84.28 vs 85.41 ([paper](https://ar5iv.labs.arxiv.org/html/2403.01817)) | Y |
| Fine-tune base, general | 3 | `indobenchmark/indobert-base-p1` | 498 | mit | SmSA 87.73, EmoT 75.48, CASA 93.23 ([IndoNLU](https://ar5iv.labs.arxiv.org/html/2009.05387)) | Y |
| Fine-tune base, multilingual | 4 | `FacebookAI/xlm-roberta-base` | 1,116 | mit | SmSA 91.39, EmoT 71.15, CASA 91.71 ([IndoNLU](https://ar5iv.labs.arxiv.org/html/2009.05387)) | Y |
| Fine-tune base, smallest disk | 5 | `indobenchmark/indobert-lite-base-p1` | 47 | mit | SmSA 90.85, EmoT 73.88, NERP 46.58 ([IndoNLU](https://ar5iv.labs.arxiv.org/html/2009.05387)) | Y |
| Sentiment, ready | 1 | `w11wo/indonesian-roberta-base-sentiment-classifier` | 499 | mit | Test F1-macro 91.02, accuracy 93.2 (SmSA) ([card](https://huggingface.co/w11wo/indonesian-roberta-base-sentiment-classifier/raw/main/README.md)) | Y |
| Sentiment, ready | 2 | `crypter70/IndoBERT-Sentiment-Analysis` | 498 | mit | Eval accuracy 0.9452, F1 0.9451 (IndoNLU; split not stated) ([card](https://huggingface.co/crypter70/IndoBERT-Sentiment-Analysis/raw/main/README.md)) | Y |
| Sentiment, ready | 3 | `ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa` | 443 | mit | Validation accuracy 0.9373 (IndoNLU SmSA) ([card](https://huggingface.co/ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa/raw/main/README.md)) | Y |
| Sentiment, ready | Not ranked | `mdhugol/indonesia-bert-sentiment-classification` | 498 | none | No score found on the card ([card](https://huggingface.co/mdhugol/indonesia-bert-sentiment-classification/raw/main/README.md)) | Y |
| Emotion, ready | 1 | `StevenLimcorn/indonesian-roberta-base-emotion-classifier` | 499 | mit | Epoch 7 accuracy 0.725, F1 0.7336 on the validation log (EmoT, 5 classes) ([card](https://huggingface.co/StevenLimcorn/indonesian-roberta-base-emotion-classifier/raw/main/README.md)) | Y |
| Emotion, ready | 2 | `yogie27/IndoRoBERTa-Base-Emotion-Indonesian-Social-Media` | 514 | none | F1 83%, accuracy 82.7% (dataset not identified in what I read) ([card](https://huggingface.co/yogie27/IndoRoBERTa-Base-Emotion-Indonesian-Social-Media/raw/main/README.md)) | Y |
| Emotion, Plutchik 8 labels | 3 | `Aardiiiiy/EmoSense-ID-Indonesian-Emotion-Classifier` | 443 | mit | Score table exists but I did not extract it ([card](https://huggingface.co/Aardiiiiy/EmoSense-ID-Indonesian-Emotion-Classifier)) | Y |
| Hate speech/toxicity | 1 | `Exqrch/IndoBERTweet-HateSpeech` | 442 | cc-by-sa-4.0 | Accuracy 0.89, macro-F1 0.78, 10-fold CV (IndoToxic2024) ([card](https://huggingface.co/Exqrch/IndoBERTweet-HateSpeech/raw/main/README.md)) | Y |
| Profanity | 1 | `Exqrch/IndoBERTweet-Profanity` | 442 | cc-by-sa-4.0 | Accuracy 0.81, macro-F1 0.70 ([card](https://huggingface.co/Exqrch/IndoBERTweet-Profanity)) | Y |
| Hate speech | 2 | `OwLim/indonesian-roberta-hate-speech-detection` | 499 | mit | Accuracy 0.826, F1 0.8256 (dataset "None") ([card](https://huggingface.co/OwLim/indonesian-roberta-hate-speech-detection/raw/main/README.md)) | Y |
| NER | 1 | `cahya/NusaBert-ner-v1.3` | 644 | mit | F1 0.8171, 19 entity types (NERGrit) ([card](https://huggingface.co/cahya/NusaBert-ner-v1.3/raw/main/README.md)) | Y |
| NER | 2 | `w11wo/indonesian-roberta-base-nerp-tagger` | 496 | mit | Eval F1 0.8105 (IndoNLU NERP) ([card](https://huggingface.co/w11wo/indonesian-roberta-base-nerp-tagger/raw/main/README.md)) | Y |
| NER | 3 | `cahya/bert-base-indonesian-NER` | 443 | mit | No score: the card is only 70 characters, despite 138k downloads ([card](https://huggingface.co/cahya/bert-base-indonesian-NER/raw/main/README.md)) | Y |
| POS | 1 | `w11wo/indonesian-roberta-base-posp-tagger` | 496 | mit | Eval F1 0.9625 (IndoNLU POSP) ([card](https://huggingface.co/w11wo/indonesian-roberta-base-posp-tagger/raw/main/README.md)) | Y |
| Zero-shot/NLI | 1 | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | 558 | mit | No Indonesian score found. Training data includes the 26-language NLI set ([card](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7/raw/main/README.md)) | S |
| Zero-shot/NLI, Indonesian | 2 | `LazarusNLP/indobert-lite-base-p1-indonli-multilingual-nli-distil-mdeberta` | 47 | mit | IndoNLI accuracy: test_lay 74.69, test_expert 65.55 ([card](https://huggingface.co/LazarusNLP/indobert-lite-base-p1-indonli-multilingual-nli-distil-mdeberta/raw/main/README.md)) | S |
| NLI, Indonesian | 3 | `w11wo/indonesian-roberta-base-indonli` | 499 | mit | test_expert accuracy 0.607, F1 macro 0.601 ([card](https://huggingface.co/w11wo/indonesian-roberta-base-indonli)) | S |
| NLI, smallest with zero-shot tag | 4 | `ilos-vigil/bigbird-small-indonesian-nli` | 122 | mit | test_expert accuracy 0.539, F1 macro 0.53 ([card](https://huggingface.co/ilos-vigil/bigbird-small-indonesian-nli)) | S |
| Summarization | 1 | `csebuetnlp/mT5_multilingual_XLSum` | 2,330 | none in metadata (verify) | Indonesian ROUGE-1/2/L 37.00 / 17.02 / 30.76 (XL-Sum) ([card](https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum/raw/main/README.md)) | S |
| Summarization | 2 | `LazarusNLP/IndoNanoT5-base-IndoSum` | 990 | apache-2.0 | Card is self-contradictory: headline Rouge1 0.7529, training log about 0.26 ([card](https://huggingface.co/LazarusNLP/IndoNanoT5-base-IndoSum/raw/main/README.md)) | S |
| Summarization | 3 | `cahya/t5-base-indonesian-summarization-cased` | 892 | none | No score on the card, trained on Liputan6 ([card](https://huggingface.co/cahya/t5-base-indonesian-summarization-cased/raw/main/README.md)) | S |
| Summarization, smallest | 4 | `panggi/t5-small-indonesian-summarization-cased` | 242 | none | No score found ([card](https://huggingface.co/panggi/t5-small-indonesian-summarization-cased/raw/main/README.md)) | S |
| QA, Indonesian | 1 | `Rifky/Indobert-QA` | 440 | apache-2.0 | EM 51.61, F1 69.09 (translated SQuAD 2.0) ([card](https://huggingface.co/Rifky/Indobert-QA/raw/main/README.md)) | S |
| QA, multilingual | 2 | `timpal0l/mdeberta-v3-base-squad2` | 1,113 | mit | Exact 80.88, F1 84.01 on SQuAD 2 (not an Indonesian score) ([card](https://huggingface.co/timpal0l/mdeberta-v3-base-squad2/raw/main/README.md)) | S |
| Paraphrase | 1 | `Wikidepia/IndoT5-base-paraphrase` | 990 | none | No score; trained on translated PAWS ([card](https://huggingface.co/Wikidepia/IndoT5-base-paraphrase/raw/main/README.md)) | S |
| Embedding, default | 1 | `intfloat/multilingual-e5-small` | 471 | mit | 384-dim class; MIRACL nDCG@10 74.80, SmSA acc 83.6 ([LazarusNLP](https://raw.githubusercontent.com/LazarusNLP/indonesian-sentence-embeddings/main/README.md)) | Y |
| Embedding, Indonesian-tuned | 2 | `LazarusNLP/all-indo-e5-small-v4` | 471 | none | MIRACL 73.23, IndoNLI test_lay AP 75.05 ([LazarusNLP](https://raw.githubusercontent.com/LazarusNLP/indonesian-sentence-embeddings/main/README.md)) | Y |
| Embedding, similarity | 3 | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 1,112 | apache-2.0 | STS-B 83.83 (best of table), MIRACL 57.97 ([LazarusNLP](https://raw.githubusercontent.com/LazarusNLP/indonesian-sentence-embeddings/main/README.md)) | Y |
| Hoax | None recommended | see hoax section below | | | | |
| Aspect-based sentiment | None ready | see ABSA section below | | | | |
| Keyword extraction | None ready | see keyword section below | | | | |

### Inferences
- For social media text, start with IndoBERTweet for fine-tuning and `w11wo/indonesian-roberta-base-sentiment-classifier` for a no-training baseline.
- Treat high scores on cards with an unnamed dataset as uninformative, for example `fathan/indo_sentiment_indobertweet` (F1 0.9646, "unknown dataset") ([card](https://huggingface.co/fathan/indo_sentiment_indobertweet/raw/main/README.md)) and `AptaArkana/indonesian_toxic_classification` (F1 0.9963, "None dataset") ([card](https://huggingface.co/AptaArkana/indonesian_toxic_classification/raw/main/README.md)).

### Gaps
- No model has a published score on MBG data. I found no MBG-specific checkpoint, though I did not search for one exhaustively.
- `Aardiiiiy/*` score tables are HTML in the card and were not extracted.

## IDs from the brief that need correcting

### Takeaway
Several IDs in the brief are not exact, and six candidates I tried do not resolve. Use the exact IDs in these notes.

### Cited Findings
- `cahya/roberta-base-indonesian` is not the exact form. The real repos are `cahya/roberta-base-indonesian-522M` (507 MB, mit) and `cahya/roberta-base-indonesian-1.5G` (504 MB, no licence) — [API](https://huggingface.co/api/models/cahya/roberta-base-indonesian-522M).
- `google/embeddinggemma` is really `google/embeddinggemma-300m`, which is **gated** (`gated: manual`, licence `gemma`) — [API](https://huggingface.co/api/models/google/embeddinggemma-300m).
- The canonical ID for XLM-R is `FacebookAI/xlm-roberta-base` — [API](https://huggingface.co/api/models/FacebookAI/xlm-roberta-base).
- These returned HTTP 401 to an anonymous API call, which Hugging Face returns for repos that are missing or private. Treat them as unavailable:
  - `w11wo/indonesian-roberta-base-bapos-tagger`
  - `w11wo/indonesian-roberta-base-prdner-tagger`
  - `ayameRushia/indobert-base-p1-finetuned-indonlu-smsa`
  - `Exqrch/IndoBERTweet-Toxicity`
  - `qanastek/pos-indonesian`
  - `Wikidepia/IndoT5-small-paraphrase`
- `faldi05/indonesian-hoax-detection-bert` exists but holds no weights (repo size about 0 MB, 0 downloads) — [API](https://huggingface.co/api/models/faldi05/indonesian-hoax-detection-bert).

### Inferences
- A wrong ID cannot be fixed on the day because the room is offline. Test-load every model with `HF_HUB_OFFLINE=1` the day before.

### Gaps
- 401 cannot distinguish "deleted" from "private". Either way, they cannot be downloaded anonymously.

## Base encoders: best evidence on Indonesian, especially social media

### Takeaway
IndoBERTweet has the strongest evidence for social media. NusaBERT adds value on code-mixed and regional-language text. IndoBERT-large and XLM-R-large score highest on IndoNLU, but their size makes them a poor fit for 4 GB VRAM and 500k rows.

### Cited Findings
- **Twitter benchmark, 7 datasets** — [IndoBERTweet card](https://huggingface.co/indolem/indobertweet-base-uncased/raw/main/README.md):

| Model | IndoLEM | SmSA | EmoT | HS1 | HS2 | NER Formal | NER Informal | Avg |
|---|---|---|---|---|---|---|---|---|
| mBERT | 76.6 | 84.7 | 67.5 | 85.1 | 75.1 | 85.2 | 83.2 | 79.6 |
| malayBERT | 82.0 | 84.1 | 74.2 | 85.0 | 81.9 | 81.9 | 81.3 | 81.5 |
| IndoBERT (Willie) | 84.1 | 88.7 | 73.3 | 86.8 | 80.4 | 86.3 | 84.3 | 83.4 |
| IndoBERT (Koto) | 84.1 | 87.9 | 71.0 | 86.4 | 79.3 | 88.0 | 86.9 | 83.4 |
| IndoBERTweet (1M steps, scratch) | 86.2 | 90.4 | 76.0 | 88.8 | 87.5 | 88.1 | 85.4 | 86.1 |
| IndoBERT + vocab adaptation + 200k | 86.6 | 92.7 | 79.0 | 88.4 | 84.0 | 87.7 | 86.9 | 86.5 |

- Metrics for that table: IndoLEM, HS1 and HS2 use F1 of the positive class; SmSA and EmoT use macro-F1; NER uses entity F1. IndoLEM, HS1 and HS2 use 5-fold CV. Training data is 26M tweets crawled Dec 2019 to Dec 2020 — [paper](https://ar5iv.labs.arxiv.org/html/2109.04607).
- **Preprocessing IndoBERTweet expects:** lowercase, user mentions to `@USER`, URLs to `HTTPURL`, emoji translated to text — [paper](https://ar5iv.labs.arxiv.org/html/2109.04607).
- **IndoNLU macro-F1, best configuration** — [paper](https://ar5iv.labs.arxiv.org/html/2009.05387):

| Model | Params | EmoT | SmSA | CASA | HoASA | WReTE | POSP | NERGrit | NERP | KEPS |
|---|---|---|---|---|---|---|---|---|---|---|
| IndoBERT base p1 | 124.5M | 75.48 | 87.73 | 93.23 | 92.07 | 78.55 | 95.26 | 69.87 | 75.52 | 70.36 |
| IndoBERT base p2 | 124.5M | 76.28 | 87.66 | 93.24 | 92.70 | 78.68 | 95.23 | 67.42 | 75.68 | 69.17 |
| IndoBERT large p1 | 335.2M | 77.08 | 92.72 | 95.69 | 93.75 | 82.91 | 95.71 | 77.60 | 79.25 | 71.18 |
| IndoBERT large p2 | 335.2M | 79.47 | 92.03 | 94.94 | 93.38 | 80.30 | 95.34 | 76.63 | 77.99 | 71.27 |
| IndoBERT-lite base p1 | 11.7M | 73.88 | 90.85 | 89.68 | 88.07 | 82.17 | 91.40 | 66.62 | 46.58 | 69.02 |
| mBERT | 167.4M | 67.30 | 84.14 | 72.23 | 84.63 | 84.40 | 91.85 | 75.02 | 69.27 | 64.31 |
| XLM-R base | 278.7M | 71.15 | 91.39 | 91.71 | 91.57 | 79.95 | 95.16 | 79.09 | 75.03 | 68.82 |
| XLM-R large | 561.0M | 78.51 | 92.35 | 92.40 | 94.27 | 83.82 | 92.73 | 78.26 | 78.52 | 70.88 |

- **NusaBERT vs IndoBERT, macro-F1** — [paper](https://ar5iv.labs.arxiv.org/html/2403.01817):

| Model | IndoNLU classification mean | IndoNLU labelling mean | NusaX `ind` | NusaX mean | NusaTranslation sentiment |
|---|---|---|---|---|---|
| IndoBERT base | 85.41 | 77.47 | 90.0 | 78.5 | 75.24 |
| NusaBERT base | 84.28 | 79.86 | 89.7 | 79.8 | 77.07 |
| IndoBERT large | 88.43 | 81.21 | 90.2 | 80.0 | 77.40 |
| NusaBERT large | 86.84 | 84.09 | 93.3 | 82.6 | 79.54 |

- NusaX `jav` and `sun`, relevant if comments mix in Javanese or Sundanese: NusaBERT base scores 84.1 and 85.2, IndoBERT base 81.7 and 77.5 — [paper](https://ar5iv.labs.arxiv.org/html/2403.01817).
- **IndoNLI accuracy:** XLM-R 82.3 / 70.3 (test_lay / test_expert), IndoBERT-large 77.1 / 61.5, IndoBERT-lite 74.1 / 58.9, mBERT 72.5 / 57.3 — [paper](https://ar5iv.labs.arxiv.org/html/2110.14566).
- **Small recent studies, low weight:**
  - A 2026 Mercu Buana thesis on political tweets reports IndoBERTweet significantly beating IndoBERT-base and XLM-R, with macro-F1 0.7586 and 0.7637 across its two phases — [repository](https://repository.mercubuana.ac.id/103413/).
  - A fuel-subsidy study reports NusaBERT accuracy 96.3% vs IndoBERT 92.5% — [JAIC](https://jurnal.polibatam.ac.id/index.php/JAIC/article/view/12964).
  - Both were read only via search summaries, not the full text.
- `flax-community/indonesian-roberta-base` (499 MB, mit): its SmSA fine-tune reached test F1-macro 91.02 — [w11wo card](https://huggingface.co/w11wo/indonesian-roberta-base-sentiment-classifier/raw/main/README.md). A different fine-tune, on `cahya/roberta-base-indonesian-1.5G`, reached validation accuracy 0.9262 — [card](https://huggingface.co/ayameRushia/roberta-base-indonesian-1.5G-sentiment-analysis-smsa/raw/main/README.md).
- **Architecture check:** `indobert-lite-base-p1` is ALBERT with hidden size 768 and 12 layers, the same as `indobert-base-p1` — [lite config](https://huggingface.co/indobenchmark/indobert-lite-base-p1/raw/main/config.json), [base config](https://huggingface.co/indobenchmark/indobert-base-p1/raw/main/config.json).
- `microsoft/mdeberta-v3-base` has 1,333 MB of weights, mit — [API](https://huggingface.co/api/models/microsoft/mdeberta-v3-base).
- Newer option: `aisingapore/SEA-LION-ModernBERT-300M` (1,250 MB, mit, Aug 2026) — [API](https://huggingface.co/api/models/aisingapore/SEA-LION-ModernBERT-300M). I found no Indonesian classification benchmark for the fine-tuning use case.

### Inferences
- **Ranking for this competition:** IndoBERTweet, then NusaBERT-base, then IndoBERT-base-p1, then XLM-R base, then IndoBERT-lite.
- **Quick decision rule:** tweets or comments, use IndoBERTweet. Lots of Javanese or Sundanese mixing, try NusaBERT-base. News articles, IndoBERT-base-p1 or NusaBERT-base are both reasonable.
- **IndoBERT-lite saves disk and VRAM, not time.** ALBERT shares parameters between layers, but with the same hidden size and depth it still does about the same computation per token. Do not choose it to go faster.
- **mDeBERTa:** the weak spot is evidence, not a known poor result. I would not spend the 5 hours on it.
- **Do not rely on the max-score column alone.** IndoBERT-large wins many IndoNLU columns, but at 335M parameters it is much slower and needs a very small batch on 4 GB.

### Gaps
- I could not confirm which IndoBERTweet variant (1M scratch or vocab-adapted 200k) is the published HF checkpoint. The card table lists both.
- IndoLEM's own benchmark paper was not fetched separately. IndoLEM-BERT results here come from the IndoBERTweet table.
- I found no head-to-head of mDeBERTa on Indonesian sentiment.

## Ready-to-use details: hate speech, hoax, ABSA, keywords

### Takeaway
Hate speech has a usable family of models. Do not use any ready-made hoax model as a truth detector. ABSA and keyword extraction have no ready Indonesian model I would trust, so plan to use embeddings, zero-shot, or fine-tuning.

### Cited Findings
- **Exqrch toxicity family:** one binary model per label, 442 MB each, cc-by-sa-4.0, IndoBERTweet-based. Labels found: HateSpeech, Profanity, Insult, IncitingViolence, IdentityAttack, SexuallyExplicit — [author listing](https://huggingface.co/api/models?author=Exqrch&limit=50). Published scores: IdentityAttack 0.89 / 0.78, SexuallyExplicit 0.91 / 0.80, Profanity 0.81 / 0.70 (accuracy / macro-F1) — [IdentityAttack](https://huggingface.co/Exqrch/IndoBERTweet-IdentityAttack), [SexuallyExplicit](https://huggingface.co/Exqrch/IndoBERTweet-SexuallyExplicit), [Profanity](https://huggingface.co/Exqrch/IndoBERTweet-Profanity).
- IndoToxic2024 has 43,692 entries, annotated by 19 people, focused on texts targeting vulnerable groups — [arXiv](https://arxiv.org/abs/2406.19349). The arXiv listing, as reported by my search tool, says the work was expanded into IndoDiscourse and IndoToxic is no longer maintained — [arXiv](https://arxiv.org/abs/2406.19349?context=cs).
- `Exqrch/IndoDiscourse-ToxicityClassifier` and `Exqrch/IndoDiscourse-PolarizationClassifier` exist (442 MB, no declared licence), but their cards are unfilled templates and they have only single-digit downloads — [card](https://huggingface.co/Exqrch/IndoDiscourse-ToxicityClassifier/raw/main/README.md). The polarization model could be relevant to political MBG discourse, but it has no published score on its card.
- **Hoax, the strongest warning found:** the card for `theonegareth/IndoHoaxDetector` reports 99.71% accuracy on its own split, but ROC-AUC **0.50 (chance)** on an independent human-labelled benchmark. It says labels come from the source, so the model largely learns publisher style, and it says it should not be applied to social-media posts — [card](https://huggingface.co/theonegareth/IndoHoaxDetector/raw/main/README.md).
- Other hoax repos:
  - `AptaArkana/indonesian_hoax_classification_indobertweet_base_uncased`: F1 0.9577, dataset "None", apache-2.0 — [card](https://huggingface.co/AptaArkana/indonesian_hoax_classification_indobertweet_base_uncased/raw/main/README.md).
  - `Rifky/indobert-hoax-classification`: mit, 498 MB, sparse card — [API](https://huggingface.co/api/models/Rifky/indobert-hoax-classification).
  - `rizquuula/IndoBERT-Summarized-HoaxDetection`: apache-2.0, 498 MB — [API](https://huggingface.co/api/models/rizquuula/IndoBERT-Summarized-HoaxDetection).
- **ABSA:** IndoNLU includes CASA and HoASA benchmarks, with IndoBERT base p1 at 93.23 and 92.07 — [IndoNLU](https://ar5iv.labs.arxiv.org/html/2009.05387). The `indonlp/indonlu` dataset is ungated — [API](https://huggingface.co/api/datasets/indonlp/indonlu). The only Indonesian ABSA repos my search found are `Hamusssss12/spotify-absa-*`, whose v2 README is just 24 characters — [card](https://huggingface.co/Hamusssss12/spotify-absa-indonesian-v2/raw/main/README.md). Note that CASA covers car reviews and HoASA hotel reviews, which is my recollection of the paper rather than something extracted here.
- **Keyword extraction:** IndoNLU KEPS peaks at about 71 F1 — [IndoNLU](https://ar5iv.labs.arxiv.org/html/2009.05387). I found no downloadable KEPS checkpoint.
- **POS on slang:** POSP is reported by IndoNLU as a benchmark, but I found no POS tagger evaluated on informal text.

### Inferences
- **Hoax:** if a task asks for this, say plainly in the report that a pretrained model detects source style, and do not present its output as fact-checking.
- **ABSA plan:** define MBG aspects (food quality, poisoning, budget, distribution, policy) and label with zero-shot on a sample, then train a classifier on that weak labelling. Or find aspect sentences with embeddings and run sentiment on them.
- **Keywords plan:** class-based TF-IDF on clusters, or KeyBERT-style cosine ranking on e5-small. Install the package ahead of time if you use it. These are my suggestions, not benchmarked.
- For the toxicity family, download only HateSpeech and Profanity by default; all six cost about 2.6 GB.

### Gaps
- I read the Exqrch HateSpeech score from its card, but the card does not state which split or threshold suits new data.
- Whether CC BY-SA or any licence here conflicts with competition rules cannot be determined without the rules.

## Sentence embeddings for clustering, search, topic modelling

### Takeaway
Use `intfloat/multilingual-e5-small` as the default: good Indonesian retrieval, small, fast, and well supported for ONNX. Larger models score a little higher, but BGE-M3 and e5-large are impractical for 500k rows on a GTX 1650. EmbeddingGemma scores well on Indonesian but is gated and uses a restrictive licence, so accept it the day before if you want it.

### Cited Findings
- **LazarusNLP benchmark:** STS-B and SemRel use Spearman; MIRACL and TyDiQA use nDCG@10; EmoT, SmSA and Intent use linear-SVC accuracy on frozen embeddings; IndoNLI uses AP. STS-B is machine-translated — [README](https://raw.githubusercontent.com/LazarusNLP/indonesian-sentence-embeddings/main/README.md):

| Model | Params | STS-B | SemRel test | MIRACL | TyDiQA | EmoT | SmSA | Intent | IndoNLI lay |
|---|---|---|---|---|---|---|---|---|---|
| multilingual-e5-small | 118M | 78.89 | 49.60 | 74.80 | 95.39 | 64.54 | 83.6 | 64.16 | 63.97 |
| multilingual-e5-base | 278M | 79.72 | 45.04 | 75.16 | 96.69 | 68.63 | 89.4 | 66.63 | 60.25 |
| multilingual-e5-large | 560M | 79.44 | 45.04 | 76.16 | 97.44 | 74.77 | 90.0 | 70.04 | 61.39 |
| paraphrase-multilingual-mpnet-base-v2 | 125M | 83.83 | 37.51 | 57.97 | 89.44 | 63.18 | 89.6 | 65.43 | 74.87 |
| distiluse-base-multilingual-cased-v2 | 134M | 75.08 | 49.35 | 48.79 | 81.56 | 63.63 | 78.8 | 55.99 | 58.48 |
| all-Indo-e5 small-v4 | 118M | 79.85 | 42.90 | 73.23 | 96.43 | 56.94 | 82.6 | 61.38 | 75.05 |
| all-NusaBERT base-v4 | 111M | 77.65 | 41.55 | 71.24 | 95.73 | 53.18 | 84.2 | 62.10 | 73.07 |
| all-NusaBERT large-v4 | 337M | 79.23 | 42.25 | 71.99 | 96.56 | 63.18 | 84.8 | 61.41 | 73.26 |
| all-IndoBERT base-v2 | 125M | 76.03 | 40.42 | 70.16 | 92.31 | 58.86 | 83.4 | 58.31 | 71.36 |
| all-IndoBERT base-v4 | 125M | 75.99 | 38.05 | 70.97 | 93.91 | 61.36 | 82.4 | 57.80 | 70.99 |

- **SEA-BED Indonesian column (June 2026 preprint):** `harrier-oss-v1-0.6b` 0.809, SEA-Embedding-E5-Large-600M 0.808, `embeddinggemma-300m` 0.802, Qwen3-Embedding-8B 0.797, `multilingual-e5-large-instruct` 0.795, SEA-Embedding-ModernBERT-300M 0.787, `bge-m3` dense 0.781 — [arXiv 2606.03027](https://arxiv.org/html/2606.03027v1). The paper reports that its gains are mainly on lower-resource SEA languages, not Indonesian.
- **Conflict on BGE-M3:** a 2026 Indonesian institutional-document study says M3-Embedding (All) beat Sentence-BERT and multilingual E5 on nDCG@10 and Recall@100 — [JPRT](https://journals.usm.ac.id/index.php/jprt/article/view/14355). SEA-BED ranks `bge-m3` dense below e5-large-instruct on Indonesian. They use different tasks, and "M3 All" combines dense, sparse and multi-vector retrieval, so they do not directly contradict each other. I could not extract the JPRT exact scores.
- **Repo facts:**

| Repo ID | Dim | Weights MB | Licence | Notes |
|---|---|---|---|---|
| `intfloat/multilingual-e5-small` | 384 | 471 | mit | ONNX, O4 (235 MB), `qint8_avx512_vnni` (118 MB) and OpenVINO in repo |
| `intfloat/multilingual-e5-base` | 768 | 1,112 | mit | 278M params |
| `intfloat/multilingual-e5-large` | 1024 | 2,240 | mit | 560M params |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | 471 | apache-2.0 | ONNX `model_quint8_avx2.onnx` (118 MB), OpenVINO int8 (119 MB) |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | 1,112 | apache-2.0 | Repo total 10.8 GB |
| `BAAI/bge-m3` | 1024 | 2,271 | mit | Repo total 4.6 GB |
| `LazarusNLP/all-indo-e5-small-v4` | 384 | 471 | none | Full-precision ONNX only (470 MB) |
| `LazarusNLP/all-indobert-base-v2` | 768 | 498 | none | |
| `LazarusNLP/all-nusabert-base-v4` | 768 | 443 | none | |
| `firqaaa/indo-sentence-bert-base` | 768 | 498 | apache-2.0 | 31.7k downloads, no independent benchmark found |
| `denaya/indoSBERT-large` | 256 stated | 1,342 | none | Trained on machine-translated STS; card lists 1024 pooling and 256 output |
| `google/embeddinggemma-300m` | 768 (from memory, unverified) | 1,230 | gemma | **Gated**: `gated: manual`, raw card returned 401 |
| `microsoft/harrier-oss-v1-270m` | card has table, not extracted | 536 | mit | Last-token pooling |
| `microsoft/harrier-oss-v1-0.6b` | card has table, not extracted | 1,192 | mit | SEA-BED Indonesian 0.809 |
| `aisingapore/SEA-LION-E5-Embedding-600M` | not extracted | 2,240 | mit | Name differs from the paper's; mapping is likely, unconfirmed |
| `aisingapore/SEA-LION-ModernBERT-Embedding-300M` | not extracted | 1,247 | mit | Uses a Gemma 3-derived tokenizer |
| `jinaai/jina-embeddings-v3` | | 1,145 | **cc-by-nc-4.0** | Non-commercial licence; check rules |

Sources: [e5-small config](https://huggingface.co/intfloat/multilingual-e5-small/raw/main/config.json), [all-indo-e5 card](https://huggingface.co/LazarusNLP/all-indo-e5-small-v4/raw/main/README.md), [firqaaa card](https://huggingface.co/firqaaa/indo-sentence-bert-base/raw/main/README.md), [denaya card](https://huggingface.co/denaya/indoSBERT-large/raw/main/README.md), [SEA-LION E5 API](https://huggingface.co/api/models/aisingapore/SEA-LION-E5-Embedding-600M), [Harrier 270m API](https://huggingface.co/api/models/microsoft/harrier-oss-v1-270m), [Gemma API](https://huggingface.co/api/models/google/embeddinggemma-300m), [Jina API](https://huggingface.co/api/models/jinaai/jina-embeddings-v3). Dimensions for e5-base, e5-large, MiniLM, mpnet and bge-m3 are from general knowledge and not rechecked in this session.

### Inferences
- **Clustering and topic modelling on all 500k:** `intfloat/multilingual-e5-small`, with `LazarusNLP/all-indo-e5-small-v4` as an optional second opinion.
- **Frozen embeddings plus linear classifier:** if labels are provided and fine-tuning is too slow, `multilingual-e5-base` scores 68.63 on EmoT and 89.4 on SmSA vs 64.54 and 83.6 for small, at about 3 to 4 times the run time.
- **Sentence similarity or paraphrase pairs:** mpnet leads STS-B at 83.83, but only 57.97 on MIRACL. Do not use it for search.
- **Memory:** 500k × 384 dim × 4 bytes is about 0.77 GB, and 768 dim about 1.5 GB. On 8 GB RAM, write to `.npy` and memory-map it. HDBSCAN on all 500k is likely too heavy: fit UMAP and clustering on a sample of 30k to 50k, or use MiniBatchKMeans, then assign the remainder.
- **Prefix:** E5 models are documented to use `query:` and `passage:` prefixes (from memory of the card, not rechecked).

### Gaps
- `paraphrase-multilingual-MiniLM-L12-v2` is not in the LazarusNLP table, so I found no direct Indonesian score for it.
- `bge-m3`, `denaya/indoSBERT-large` and `firqaaa/indo-sentence-bert-base` are also not in that table.
- The Harrier models are newer and score well, but I did not confirm whether they count as non-LLM. Last-token pooling suggests a decoder-style architecture. I also have no 4 GB throughput evidence for them.
- No benchmark uses colloquial MBG-style text. MIRACL and TyDiQA are Wikipedia-style retrieval.

## Classic vector models: fastText and word2vec

### Takeaway
The official `.bin` is too large for 8 GB RAM. Use the `.vec` with a word limit, or the 14 MB compressed model, and treat word vectors as a fast baseline only.

### Cited Findings
- `cc.id.300.bin.gz` is **4.507 GB** compressed — [URL (HEAD request, 200)](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.bin.gz).
- `cc.id.300.vec.gz` is **1.227 GB** compressed — [URL (HEAD request, 200)](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.vec.gz).
- `wiki.id.zip` is 3.244 GB — [URL (HEAD request, 200)](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.id.zip).
- HF mirror: `facebook/fasttext-id-vectors` has `model.bin` at **7,237 MB** uncompressed, cc-by-sa-3.0 — [API](https://huggingface.co/api/models/facebook/fasttext-id-vectors).
- Compressed alternative: `bstds/ft_id_small_300` has `cc.id.300.x.bin` at about 14 MB, built with compress-fasttext, with no declared licence — [API](https://huggingface.co/api/models/bstds/ft_id_small_300).
- Word2vec: `deryrahman/word2vec-bahasa-indonesia` on GitHub has 28 stars, MIT, last pushed Dec 2019 — [GitHub API](https://api.github.com/repos/deryrahman/word2vec-bahasa-indonesia). `galuhsahid/indonesian-word2vec` returned 404 — [GitHub API](https://api.github.com/repos/galuhsahid/indonesian-word2vec).
- CBoW on IndoNLI reached only 50.1 / 36.9 accuracy vs 74 to 82 / 59 to 70 for transformers — [IndoNLI](https://ar5iv.labs.arxiv.org/html/2110.14566).

### Inferences
- A 7.2 GB `.bin` will not load safely on 8 GB RAM with Windows and Python running.
- **Recommendation:** `cc.id.300.vec.gz` with `gensim` `limit=500000` (about 600 MB of vectors by arithmetic: 500k × 300 × 4 bytes), or `bstds/ft_id_small_300` if out-of-vocabulary slang matters and disk is tight. The `.vec` loses subword handling, which matters for slang spelling.
- Averaging word vectors will run far faster than any transformer, but e5-small at 10 to 20 minutes (estimate) makes this a backup, not a default.

### Gaps
- I did not measure load time or RAM for any of these files.
- I did not test whether `bstds/ft_id_small_300` retains usable quality; no evaluation found.

## Throughput on GTX 1650 4 GB, i7-9750H, 8 GB RAM

### Takeaway
No published benchmark exists for this hardware, so all numbers below are **estimates**. Short texts are feasible for e5-small (10 to 20 minutes) and BERT-class models (35 to 70 minutes), but not for BGE-M3 or zero-shot over all rows. **Do not assume fp16 works on this GPU; test it for NaN first, and plan on fp32.**

### Cited Findings
- I found no measured sentence-transformers throughput for a GTX 1650. The one result for that GPU reports LLM-style tokens per second, which does not apply — [willitrunai](https://willitrunai.com/de/can-run/all-minilm-l6-v2-on-gtx-1650-4gb).
- **fp16 risk on the GTX 16 series:** PyTorch issue title "FP16 inference with Cuda 11.1 returns NaN on Nvidia GTX 1660" — [PyTorch #58123](https://github.com/pytorch/pytorch/issues/58123). Many user reports of black or NaN outputs in half precision on GTX 1650 and 1660 come from image generation, which is a different workload — [example](https://www.runaihome.com/blog/comfyui-black-image-output-fix-2026/). The root cause is not confirmed by a vendor source I found, and I saw no report specific to BERT on a 1650.
- Sentence Transformers reports float16 at a median 2.92× over FP32 on an RTX 3090, 3.87× with Flash Attention and unpadding. Its CPU recommendation is OpenVINO INT8 if a small accuracy loss is acceptable. These are measured on i7-13700K and RTX 3090, not this hardware — [sbert efficiency](https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html).
- BGE-M3 "Q4" was excluded from that study because its embeddings failed an agreement check against FP32 — [sbert efficiency](https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html).
- Architecture facts used for scaling: e5-small has hidden size 384 and 12 layers; IndoBERTweet, NusaBERT, IndoBERT, Indonesian RoBERTa, and mDeBERTa have hidden size 768 and 12 layers — configs for [e5-small](https://huggingface.co/intfloat/multilingual-e5-small/raw/main/config.json), [IndoBERTweet](https://huggingface.co/indolem/indobertweet-base-uncased/raw/main/config.json), [NusaBERT](https://huggingface.co/LazarusNLP/NusaBERT-base/raw/main/config.json), [mDeBERTa](https://huggingface.co/microsoft/mdeberta-v3-base/raw/main/config.json).

### Inferences
**All values below are my estimates.** Method: a GTX 1650 is about 3 TFLOPS (from memory), and BERT-base is roughly 20 to 25 GFLOP per 128-token sequence, so about 100 to 130 sequences per second at best. Cost scales linearly with tokens and roughly with hidden size squared times layers. I widened the ranges for tokenisation, laptop thermals, and real-world losses. Assumes short texts sorted by length, max_len 64, and fp32.

| Model class | Examples | Batch | Texts/s | 500k single pass |
|---|---|---|---|---|
| H384, L12 | e5-small, MiniLM-L12, all-indo-e5 | 128 to 256 | 400 to 900 | **10 to 20 min** |
| H768, L12 | IndoBERTweet, IndoBERT, NusaBERT, RoBERTa, e5-base, mpnet, XLM-R base | 64 | 120 to 250 | **35 to 70 min** |
| ALBERT lite | IndoBERT-lite | 64+ | 100 to 250 | 35 to 80 min |
| DeBERTa | mDeBERTa | 32 to 48 | 60 to 130 | 1 to 2.3 h |
| H1024, L24 | bge-m3, e5-large, IndoBERT-large | 16 to 32 | 30 to 70 | **2 to 4.6 h (avoid)** |
| Zero-shot, 4 labels | mDeBERTa NLI | 32 | 60 to 130 pairs | **4.3 to 9 h (2M pairs); sample 20k to 50k** |
| T5 generation | IndoT5, cahya T5 | 8 | 2 to 8 docs | **Days; sample only** |

Fine-tuning one epoch on 50k rows, fp32, estimated:

| Model | max_len | Batch | Samples/s | Epoch |
|---|---|---|---|---|
| IndoBERTweet, IndoBERT, NusaBERT | 64 | 16 to 32 | 35 to 70 | **12 to 24 min** |
| Same | 128 | 8 to 16 | 18 to 35 | **24 to 46 min** |
| XLM-R base | 64 | 16 to 24 | 30 to 60 | 14 to 28 min |
| mDeBERTa base | 64 | 8 to 16 | 18 to 40 | 21 to 46 min |

CPU only, i7-9750H with 6 cores, also estimated and low confidence:

| Option | Texts/s | 500k |
|---|---|---|
| MiniLM-L12 `model_quint8_avx2.onnx` or OpenVINO int8 | 100 to 300 | 30 to 80 min |
| BERT-class int8 ONNX | 30 to 70 | 2 to 4.6 h |

- **Quantisation file matters:** the i7-9750H supports AVX2 but not AVX-512 (from memory). So `model_quint8_avx2.onnx` in the MiniLM repo fits this CPU, while e5-small only ships `qint8_avx512_vnni`. Export an AVX2 int8 version of e5-small the day before if you want CPU fallback, and check embedding agreement after.
- **Practical plan for fp16:** run 1,000 rows under autocast and check for NaN and agreement with fp32. If clean, use it and rerun your timing. Even if it works, expect a smaller gain than on an RTX 3090.
- **Time budget:** the sample calibration below takes under 5 minutes and replaces all of these estimates with real ones. Do it the day before: encode 2,000 rows from a similar corpus per model at the planned batch size, and record texts per second and peak `nvidia-smi` memory.
- **Other hardware cautions:** laptop GPUs throttle under sustained load, so use the low end of each range. Free VRAM is typically below 4 GB because of the display, so start with smaller batches.

### Gaps
- No value in the tables above is measured. Throughput on real MBG text depends on actual token length, which is unknown.
- I did not confirm fp16 behaviour on a GTX 1650 for transformer text models.

## Best 3 to 5 models for the most coverage per GB

### Takeaway
Five models totalling about 2.6 GB of weights cover clustering, search, fine-tuning, sentiment, zero-shot weak labelling, and NER.

### Cited Findings
| # | Repo ID | Weights MB | Covers | Evidence |
|---|---|---|---|---|
| 1 | `intfloat/multilingual-e5-small` | 471 | Clustering, topic modelling, search, dedup, linear classifier, keywords | [LazarusNLP](https://raw.githubusercontent.com/LazarusNLP/indonesian-sentence-embeddings/main/README.md) |
| 2 | `indolem/indobertweet-base-uncased` | 445 | Any fine-tuning on labelled social media text | [card](https://huggingface.co/indolem/indobertweet-base-uncased/raw/main/README.md) |
| 3 | `w11wo/indonesian-roberta-base-sentiment-classifier` | 499 | Ready sentiment | [card](https://huggingface.co/w11wo/indonesian-roberta-base-sentiment-classifier/raw/main/README.md) |
| 4 | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | 558 | Zero-shot weak labels on a sample, ABSA by aspect | [card](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7/raw/main/README.md) |
| 5 | `cahya/NusaBert-ner-v1.3` | 644 | NER (19 entity types) | [card](https://huggingface.co/cahya/NusaBert-ner-v1.3/raw/main/README.md) |

Total 2,617 MB, ungated and with declared licences (mit or apache-2.0), per the [HF API](https://huggingface.co/api/models/intfloat/multilingual-e5-small).

Optional additions, in priority order:
- `Exqrch/IndoBERTweet-HateSpeech` (442 MB) for toxicity.
- `StevenLimcorn/indonesian-roberta-base-emotion-classifier` (499 MB) for emotion.
- `LazarusNLP/NusaBERT-base` (443 MB) for code-mixed fine-tuning.
- `w11wo/indonesian-roberta-base-posp-tagger` (496 MB) for POS.
- `LazarusNLP/indobert-lite-base-p1-indonli-multilingual-nli-distil-mdeberta` (47 MB) as an Indonesian NLI second opinion.
- `panggi/t5-small-indonesian-summarization-cased` (242 MB), or `cahya/t5-base-indonesian-summarization-cased` (892 MB), for summaries of samples.
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` ONNX AVX2 file (118 MB) as CPU fallback.
- `bstds/ft_id_small_300` (14 MB) or `cc.id.300.vec.gz` (1.227 GB) as a word-vector baseline.

Training data worth saving the day before (all ungated): [`indonlp/indonlu`](https://huggingface.co/api/datasets/indonlp/indonlu), [`indonli`](https://huggingface.co/api/datasets/indonli), [`indonlp/NusaX-senti`](https://huggingface.co/api/datasets/indonlp/NusaX-senti), [`Exqrch/IndoToxic2024`](https://huggingface.co/api/datasets/Exqrch/IndoToxic2024). Check whether rules allow external data.

### Inferences
- **Why not more?** The task is unknown, and most likely outcomes (topic discovery, sentiment, classification with labels) are handled by models 1 to 3 alone. Model 4 gives you labels when none are provided. Model 5 is the one to drop if disk is tight, because NER is only needed if the task asks for entities.
- **Avoid in the kit:** `BAAI/bge-m3` (2,271 MB, too slow), `google/embeddinggemma-300m` (gated, Gemma licence), `jinaai/jina-embeddings-v3` and `tabularisai/multilingual-sentiment-analysis` (both cc-by-nc-4.0 per [Jina API](https://huggingface.co/api/models/jinaai/jina-embeddings-v3) and [tabularisai API](https://huggingface.co/api/models/tabularisai/multilingual-sentiment-analysis)), and `Babelscape/wikineural-multilingual-ner` (cc-by-nc-sa-4.0 per [API](https://huggingface.co/api/models/Babelscape/wikineural-multilingual-ner)).
- **Before the day:** download with `allow_patterns` for one weights format plus tokenizer and config files. Then disconnect from the network and load every model with `HF_HUB_OFFLINE=1`. Also confirm tokenizer dependencies are installed, because XLM-R and DeBERTa tokenizers commonly need `sentencepiece` and `protobuf` (from memory).

### Gaps
- Coverage assumes the task is one of those listed. The real task is unknown.
- The 2,617 MB counts weights only. Tokenizers and configs add tens of MB per model.
- Competition rules on licences and external data were not available.
