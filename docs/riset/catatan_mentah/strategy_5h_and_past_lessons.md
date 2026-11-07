# Practical 5-hour offline strategy for 500k-row Indonesian NLP on MBG, unknown task (portfolio lessons + outside evidence, as of November 2026)

**How to read these notes.** In every section, "Portfolio evidence" comes only from Ivan's local skill files, "Outside evidence" comes from external sources, and "Inferences" is my judgement. Grades are the skill's own: V verified, S strong, W weak, F failed under stated conditions. Any arithmetic I did is labelled as arithmetic, and it is not a measurement. No throughput on the GTX 1650 has been measured; the plan treats that as the first thing to measure. Sources marked "[snippet]" were seen only in search-result excerpts, so check them before quoting.

**Assumed setup:** GTX 1650 4 GB VRAM, 8 GB RAM, 6 cores, no internet, 5 hours, one laptop. Whether Ivan is on a team is unknown, so both cases are covered.

## Part A: Which portfolio lessons transfer, and how?

### Takeaway
The closest projects are IF26 (Indonesian text, macro-F1), MT26 (Indonesian slang, TF-IDF + LightGBM) and CDW2 (sparse text-like n-grams, macro-F1 decision layer), plus the Compfest25 BERTopic row. They transfer strongly on rules, duplicates, the decision layer, noise floor and process. They transfer weakly on model choice, because **the portfolio holds no clean measurement of a correctly fine-tuned Indonesian transformer, and no NLP project near 500k rows**.

### Cited Findings

**Portfolio evidence: closest cases**
- IF26: Indonesian news headline-vs-body, 14,397 train / 3,603 test, macro-F1, ~90/10, pretrained banned. No punctuation in the corpus. Bag-of-words nearly useless because the task is relational: TF-IDF+LR AUC 0.547, macro-F1 0.513 vs 0.474 for all-genuine. Final TRACE OOF 0.9578 — [CS-IF26][csif]
- IF26 v2–v6 used IndoBERT/XLM-R and were disqualifiable; five submissions were lost. The file gives "0.88169 with rules" for the best raw score, without a like-for-like comparison, so it is not usable as a BERT-vs-TF-IDF measurement — [CS-IF26][csif]
- MT26: 14,573 Indonesian slang chat messages, >50% exact duplicates (template + slot), ~300-word vocabulary. Multilingual sentence embeddings 0.626 and MLP 0.639 lost to LambdaRank 0.664 — [CS-MT26][csmt]
- CDW2: 19,475 rows, macro-F1. LightGBM on sparse TF-IDF beat 10 linear/NB models by 0.0153 at baseline; a linear model on the same TF-IDF added nothing as a blend member (agreement 79–80%, honest −0.0016) — [CS-CDW2][cscdw]
- Compfest25 topic camp: LDA (4 topics, coherence 0.568) vs BERTopic (coherence 0.657, diversity 0.801, silhouette −0.005); topics were treated as broad themes, not separated clusters — [early-and-misc][csearly]
- Granite sentiment: LLM zero-shot sentiment and complaint tagging for exploratory insights; no score recorded in the row I read — [nlp.md][nlp], [early-and-misc][csearly]
- Inventory D points to these for threshold sensitivity, hidden groups, label noise and rule bans — [project-inventory][inv]
- A fine-tuning warning: SD26 fine-tuning (→ ~0.98) lost to frozen probes (0.9909), but the ledger attributes this to a **broken recipe** (fold leak, no augmentation, 100× LR random head), and a correct recipe is "untested". This is images, so it is not a verdict on BERT — [failed-experiments][fail]

**Portfolio evidence: transfer arguments.** Format: applies because; differs in; expect; test by.

| Lesson (grade) | Evidence traced | Transfer argument |
|---|---|---|
| **G-06** read rules first (V) | IF26 lost 5 submissions to a pretrained ban ([CS-IF26][csif]) | Applies because the plan depends on pretrained models; differs in that this is a one-day offline final with no second chance; expect a ban to remove the whole IndoBERT branch; test by asking organisers the question list in Part B before the day. |
| **G-07** recover hidden groups (V) | CDW1, SC26, SD26 ([LL][ll]) | Applies because MBG social/news data probably has copypasta, accounts, sources; differs in that the groups are text clusters, not physical events; expect a random-CV score above group-CV; test by comparing both on the same baseline. |
| **G-08** mimic how test was drawn; group CV can be wrong (V) | IF26 binomial check proved random rows; group CV would have removed real regimes ([CS-IF26][csif]) | Applies because duplicates may straddle train/test; differs in scale (500k makes a duplicate split check fast and tight); expect either result; test by the binomial split of pairs seen exactly twice, then choose. |
| **G-09** dedup before folds and validate it (V) | DQ/SD: removed +0.0036 inflation, fixed degenerate folds ([successful][succ]) | Applies because retweets/copypasta are near-duplicates; differs in that text needs masking and near-dup hashing, not pHash; expect inflation that depends on the duplicate share; test by size-of-cluster histogram and score before/after. |
| **G-41** measure ambiguity model-free (V) | CDW2: 2.69% byte-identical conflicts, ambiguous population ~10–20% ([CS-CDW2][cscdw]) | Applies because sentiment/stance labels are subjective; differs in that conflict comes from annotators, not compiler identity; expect a nonzero ceiling; test by counting conflicting labels in exact and near-dup clusters. |
| **G-03** decision layer often beats the model (V) | CDW2 class multipliers, part of +0.0112 honest and LB +0.007 ([successful][succ]) | Applies if metric is macro-F1 or similar; differs in class count/imbalance (unknown); expect a gain at near-zero compute, on a saved holdout; test nested against argmax. If the metric is AUC or accuracy, this lesson mostly does not apply. |
| **G-14** fit thresholds on the same score distribution (V) | IF26 threshold on evaluated rows inflated 0.575 → 0.917 with 34 negatives; DQ refit mismatch ([CS-IF26][csif], [failed][fail]) | Applies because a full refit is unaffordable anyway; differs in that holdouts are larger, so optimism should be smaller for common classes but not for rare ones; expect a smaller but real leak on rare classes; test by fit-half/score-half. Do not refit after tuning thresholds. |
| **G-15** noise floor first (V) | CDW2 public LB ±0.009; MT26 column order alone moved NDCG 0.0018 ([CS-CDW2][cscdw], [CS-MT26][csmt]) | Applies because few experiments fit in 5 hours; differs in that a big holdout shrinks sampling noise, leaving seed noise and any small public split; expect most late moves to be noise; test with `scripts/noise_floor.py` cluster-bootstrapped by dup group. |
| **G-16/G-17** nest; repeated selection overfits (V) | CDW1 best nested 0.1754 had the worst LB 0.2195 ([failed][fail]) | Applies to every tuned blend/threshold; differs in that 5 hours allows few selection rounds, so risk is lower; expect small optimism; test by keeping a final untouched slice. |
| **G-23** public LB is a noisy tie-breaker (V) | CDW2, IC ([LL][ll]) | Applies if a leaderboard exists; differs because split size is unknown; expect gross-failure detection only; test by computing SE from the public row count. |
| **G-28** learn lexical maps, audit dictionaries (V) | MT26 dictionary: `github`, `nlp` had 0 hits, `git` matched only "gitu"; users write "escuel" = SQL ([CS-MT26][csmt]) | Applies directly to any keyword labelling function for MBG; differs in that domain terms may be more stable than chat slang; expect some seed words to miss or misfire; test by hit count and manual precision on a random 30 for each rule. |
| **G-20** circular validation (V) | SC26 gap 0.011 → 0.073 ([failed][fail]) | Applies when validating weak labels with weak labels; differs in being text rules rather than pseudo-Unknown; expect an optimistic score; test against a hand-labelled gold set never used for rules. |
| **G-49** never tune on hand-labelled test (V) | SS, SD ([LL][ll]) | Applies strongly: hand-labelling test rows to pick thresholds is tempting; differs in nothing; expect a compliance risk; test by keeping gold from train/unlabelled pool only and reading the rules. |
| **G-29** mirror the generation process (V) | IF26 mutation census → OOF 0.9578 vs bag-of-words 0.51 ([successful][succ]) | Applies only if organisers synthesised labels/pairs (rule-labelled or lexicon-labelled); differs in that real-opinion labels have no generator; expect a large win if labels came from rules, none otherwise; test by training a shallow model on keywords to see if labels are recoverable almost perfectly. |
| **G-24** natural units (V) | CDW2 +0.0192 local / +0.015 LB ([successful][succ]) | Applies mildly: units here are words + char n-grams + masked tokens; differs because text is already tokenised; expect a smaller Δ; test word-only vs word+char. |
| **L-MT-06** calibrate representation knobs by CV (V) | Recency decay +0.009, intuition lost ([CS-MT26][csmt]) | Applies to ngram width and min_df; differs because the task may lack a temporal per-user view; expect small gains; test one or two knobs only. |
| **L-MT-12** template text leaves no room for pretrained (V) | 0.626 vs 0.664 ([CS-MT26][csmt]) | Applies if EDA shows a tiny vocabulary or heavy templating; differs if MBG text is real free text; expect BERT's edge to vanish on templated data; test by vocabulary size and duplicate share, then a pilot. |
| **G-34** diversity from representation (V) | CDW2 structure member with 63.8% agreement earned weight ([CS-CDW2][cscdw]) | Applies to TF-IDF + BERT since they read different views; differs in having real pretraining; expect a blend gain only if error overlap is limited; test by agreement and honest blend Δ. |
| **G-35** parameter-free averages (V) | MT26 equal rank 0.66217 > hill-climb 0.66069 > best single 0.66020 ([CS-MT26][csmt]) | Applies when members are near-tied; differs because BERT and TF-IDF may not be near-tied; expect equal weight to be enough, unless one member is clearly weak (misc: weak member dragged 0.903 → 0.894, [failed][fail]); test equal vs drop-weak. |
| **G-36** seed bagging (V) | CDW2 +0.0032 blend ([successful][succ]) | Applies to LightGBM/linear cheaply; differs for BERT where one more seed costs an hour or more; expect it for CPU members only; test by pairwise seed delta. |
| **G-39** self-training under shift (V), conditional | IC +0.0068/+0.0059; SC +0.030 ([successful][succ]). **Against:** CDW2 pseudo-labels flat or negative (−0.0022 to +0.0000), ambiguity-dominated ([CS-CDW2][cscdw]) | Applies only if test use is allowed and shift exists; differs because 500k rows is rarely label-starved unless labels are few; expect value mostly in the few-labels branch; test against a gold set with an ignore band. |
| **G-40** clean labels with audit; check direction (V) | **Backfires:** SD26 dropping 542 suspects removed 186 hard-but-correct images; DQ −0.0024 to −0.0028 ([failed][fail]) | Applies to subjective NLP labels; differs in cost of checking, since reading a tweet takes seconds; expect that blind removal hurts; test by hand-confirming a sample of flagged rows, and do not remove without it. |
| **G-27 / crippled dry runs** (V) | CDW1 dry runs reversed conclusions 4/4 times ([failed][fail]) | Applies directly to a BERT pilot on a small subsample; differs because time forces some proxy; expect a misleading pilot occasionally; test by using the pilot only for go/no-go on GPU time, and decide final selection on the full run. |
| **G-42** one change per run (V) | ARA −0.08, IC −0.026 ([LL][ll]) | Applies more under time pressure; differs because each change must also be cheap; expect fewer, cleaner runs; test by config diff logged for each run. |
| **G-44** save OOF, test, ids (V) | 25-minute GPU question → 99-second offline ([successful][succ]) | Applies directly since GPU time is the scarcest resource; differs in nothing; expect post-processing without retrain; test by a kit unit test. |
| **G-47** reproducibility, train/serve parity (V) | IF26 class weight drift cost 6 LB points ([CS-IF26][csif]) | Applies since code is prewritten; differs because deterministic exactness matters less than parity unless organisers rerun code; expect avoiding silent bugs; test by assert parity + rerun of a submission on a small sample. |
| **G-50** diagnose infra first (V) | Silent CPU fallback cost 4.5 h ([failed][fail]) | Applies strongly on a laptop; differs from Kaggle in driver/thermals; expect at least one environment surprise; test with a real CUDA op gate at startup, Wi-Fi off. |
| **L-IF-12** tune on log loss, threshold on smoothed curve (S) | Threshold 0.41, Lipton check ([CS-IF26][csif]) | Applies for F1 binary tasks; differs in larger validation size; expect stable thresholds; test by fit-half/score-half. |
| **L-IF-15** "more correct" can regress (V) | Linguistically correct extractors regressed 4× ([CS-IF26][csif]) | Applies to stemming and slang normalisation; differs for BERT, which should see natural text; expect no automatic gain; test one toggle at most. |
| **G-48** verify LLM and stale claims (V) | IF26 most cited numbers stale ([CS-IF26][csif]) | Applies to these notes too; differs in nothing; expect some advice not fitting the laptop; test everything in rehearsal. |
| **FIT26 / binary relevance** (F for chains) | Chains 0.412 vs BR 0.436 on 300 rows ([failed][fail]) | Applies if multi-label; differs because 500k rows may support label dependence; expect BR to be enough as a start; test chains only with spare time. |

**Portfolio evidence: EDA and validation frameworks that transfer directly**
- Every EDA finding ends in a decision, including exact and near-duplicates, conflicting labels, train-test overlap, how test was sampled, adversarial AUC, and length distributions train vs test — [eda-checklist][eda]
- Validation table: i.i.d. → StratifiedKFold; repeated entities not crossing → group; test random but entities repeat → random folds + fold-aware memory; red flag when random CV ≫ grouped CV — [validation-strategy][val]
- Priors on effect sizes: decision layer ≥ validation fixes > adaptation > new-representation members > label cleaning > targeted features > seed bagging > tuning (usually ≤ 0.002) > class weighting (often null once thresholds tuned) — [SKILL §7][skill]
- IF26: `class_weights=[1,1]` + tuned threshold beat `[4,1]` by +0.004 to +0.011 across seeds — [CS-IF26][csif]

### Inferences
- **Do not transfer** IF26's from-scratch hand-crafted route (gazetteers, ELECTRA-style detector, generator artefacts) unless pretrained models are banned or the day's task is relational. It took a long project, and the 5-hour format cannot repeat it. Its *questions* do transfer: is there a generator, are labels rule-made?
- **Conflict to resolve on the day:** G-09 (dedup before folds) and G-08 (group CV may remove real test regimes) can pull apart. My judgement: report both random and group scores from the first baseline (G-19 ladder), and let the test-sampling check decide which one selects models.
- MT26's embedding loss should not be read as "embeddings are weak on Indonesian". Its cause was the template corpus. Use vocabulary size and duplicate share to decide if it applies.

### Gaps
- No portfolio measurement for: BERT vs TF-IDF on free Indonesian text, BERT learning curves, GTX 1650 throughput, or NLP above ~20k rows. Fine-tune recipe quality is therefore unproven until rehearsal.
- I read case studies IF26, MT26, CDW2 in full and only the Compfest/Granite rows of the misc file.

## Part B1: What can 5 hours achieve on 500k rows with a weak GPU, comparing TF-IDF + linear, BERT on a subsample, and embeddings + LightGBM?

### Takeaway
Treat TF-IDF word + char with a linear head as the guaranteed submission, and fine-tune an Indonesian BERT on a subsample as the upside, if throughput and rules allow. Outside evidence on Indonesian text shows the BERT advantage ranges from about +12 macro-F1 points to negative. It depends on the dataset, labels and imbalance, so measure it rather than assume it.

### Cited Findings

**Outside evidence: Indonesian and MBG**

| Dataset | Classical | Transformer | Gap and caveat | Source |
|---|---|---|---|---|
| NusaX sentiment, Indonesian row, macro-F1 | SVM 78.7, LR 74.7 | IndoBERT base 90.0, large 90.2, NusaBERT large 93.3 | ~+11 for base. Feature type and sizes not confirmed from the extracted text | [NusaBERT][nusabert] |
| NusaTranslation **average over regional languages**, not Indonesian | Sentiment SVM 76.04; emotion LR 56.18 | Sentiment IndoBERT base 75.24; emotion IndoBERT base 54.50, NusaBERT large 61.40 | About zero or negative for base; the setting is regional, so low transfer here | [NusaBERT][nusabert] |
| SmSA 11,000 train, macro-F1 | fastText-indo4b 82.13 (not TF-IDF) | IndoBERT base 87.73, large 92.72 | +5.6 / +10.6 | [IndoNLU][indonlu] |
| EmoT 3,521 train, macro-F1 | fastText-indo4b 69.23 | base 75.48, large 77.08 | +6.3 / +7.9 | [IndoNLU][indonlu] |
| Product reviews, macro-F1 [snippet] | LinearSVC 0.5510 | IndoBERT 0.5088 | −4.2; imbalance cited | [benchmark review][moon] |
| Sentiment with neutral class, macro-F1 [snippet] | SVM 71.68 | IndoBERT 83.77 | +12.1; the search summary did not tie this figure to one URL and the page timed out, so the link is unconfirmed: verify before quoting | [possible source, JAIC][jaic1] |
| MBG tweets, 54,105, **accuracy** | SVM 83.39, RF 70.28 | IndoBERT 87.60 | +4.2; label source not stated, no macro-F1 or split | [JATI][jati] |
| MBG TikTok, accuracy [snippet] | tuned SVM 0.73 | IndoBERT 0.78 | +5 | [BINUS][binus] |
| MBG X, manual labels, thesis | n/a | IndoBERT accuracy 78.00, F1 77.88; negative 38.33 / neutral 34.24 / positive 27.43 | Dataset size not stated | [UMN thesis][umn] |

- IndoBERT base is ~124.5M parameters, trained on Indo4B; lite and large variants exist — [IndoBERT card][indobert]

**Outside evidence: other**
- IMDb English, 25k train: BERT 0.9387 vs LinearSVC 0.8989, Ridge 0.8990, LR 0.8949. The authors attribute part of the gap to difficulty in making traditional methods work, and the paper reports no learning curve — [González-Carvajal & Garrido-Merchán][gonz]
- SetFit paper: FineTune averaged 43.0 at 8 per class, 69.7 at 64 per class, and 84.8 on the full data, showing fine-tuning still climbing well past 64 per class (average across small benchmarks) — [SetFit][setfit]
- Dodge et al.: seed alone can produce substantially different fine-tuning results; init and data order contribute about equally, and on small data many runs diverge early — [Dodge et al.][dodge]
- `HashingVectorizer` is low-memory and has no vocabulary, supports `char_wb`, and lacks IDF and inverse transform; collisions are rarely an issue with adequate `n_features` — [scikit-learn][hash]
- Kaggle Jigsaw 2018 write-up: NB-SVM popular as a base model — [Sijun He][sijun] [snippet]. Jigsaw 2022 write-up: TF-IDF + linear widely used, but mostly used to overfit a small public LB — [Montgomerie][montg] [snippet]

**Portfolio evidence**
- CDW2: LightGBM on sparse TF-IDF beat linear at 19.5k rows ([CS-CDW2][cscdw]); MT26: learning curve 4× data → +0.005, attributed to information limit, not sample size ([CS-MT26][csmt]).

### Inferences

**Comparison (judgement):**

| Route | Strength | Risk on this laptop | Role |
|---|---|---|---|
| TF-IDF word (1–2) + char_wb (3–5) + LR/SGD/LinearSVC | Minutes, all 500k rows, robust to slang spelling via char n-grams | RAM at high nnz; OvR parallel jobs copy memory | **Guaranteed baseline, also a blend member** |
| IndoBERT base/lite fine-tuned on a subsample | Largest upside on free-text opinion | Throughput, 4 GB VRAM, fp16 risk, seed variance | **Main upside; one recipe, one or two seeds** |
| Multilingual embeddings + LR/LightGBM | Cheap after one embedding pass; reusable for topics, near-dup, kNN, SetFit-like heads | Likely below fine-tuning with lots of labels; MT26 lost on templates | **Shared asset; essential for few/no-label branches** |
| LightGBM on sparse TF-IDF | Beat linear in CDW2 | Memory and time at 500k with huge dimensionality | **Subsample or SVD only; try if linear flattens** |
| Five-fold BERT | Honest and strong | Not affordable | **Refuse** |

**Compute budget: arithmetic to compare against measured throughput (not measurements):**
- Fine-tune in 90 minutes: 50k × 1 epoch needs 9.3 rows/s; 100k × 1 needs 18.5 rows/s; 100k × 2 or 200k × 1 needs 37 rows/s.
- Embed or infer 500k rows: in 60 minutes needs 139 rows/s; in 30 minutes needs 278 rows/s.
- Zero-shot 500k × 8 labels = 4M NLI passes; in 60 minutes needs ~1,111 pairs/s. Assume this is not feasible on all rows until measured.
- Embedding RAM: 500k × 384-dim float32 = 768 MB; float16 = 384 MB; 768-dim float32 = 1.54 GB. Hence memmap.
- Sparse RAM example: 500k rows × 300 nnz × ~8 bytes ≈ 1.2 GB, before Python overhead. The nnz figure is an assumption; measure it.

**Picking subsample size (judgement):** run a TF-IDF learning curve at 5k, 20k, 80k, 320k on the same holdout. If the curve flattens early, more rows will not rescue BERT either, and label quality is the problem. Start the fine-tune size from measured throughput, not from a slogan. A rule of thumb to test in rehearsal: 30k–100k rows is likely the realistic range. Keep all rare-class rows and dedup first. If class proportions change, remember G-14: fit the decision layer on holdout rows from the original distribution.

**Machine-specific risk: half precision.** Community reports link GTX 16-series cards, including the 1650, with NaNs in fp16, mainly from Stable Diffusion; root cause is unconfirmed — [PyTorch #58123][pt58123], [HF discussion][hfblack]. Test `torch.autocast` on this exact card before relying on it. Compare loss curves against fp32 on a short run. If fp16 fails, budget fp32 with a smaller batch.

### Gaps
- **No source found** giving a BERT learning-curve plateau on Indonesian text. I found a citation for a low-shot comparison at 100–1,000 labels per class, seen only second-hand. The "where does BERT plateau" answer must come from rehearsal on Ivan's own data. A TF-IDF gap at 500k Indonesian labelled rows was not found in any source.
- No sourced throughput for IndoBERT or MiniLM on a GTX 1650. One A100 vendor figure (~3,000 sentences/s) is not applicable ([Mixpeek][mix]).
- MBG studies are mostly small venues with limited method detail, so read them as priors on what is plausible for the topic, not as benchmarks.

## Part B2: What to do if labels are few or absent?

### Takeaway
Hand-label a small gold set first, because every weak method needs an independent check. Then combine keyword rules and SetFit-style heads, use zero-shot only on samples, and scale out with TF-IDF or an embedding head trained on the confident rows.

### Cited Findings

**Outside evidence**
- SetFit: at 8 labels per class MPNet 62.3 vs FineTune 43.0; at 64 per class 75.3 vs 69.7. Pair count = 2R|C| with default R = 20, so it grows with classes. Training about 30 s on a 16 GB GPU, on small benchmarks. Multilingual runs used `paraphrase-multilingual-mpnet-base-v2`, 8 per class — [SetFit][setfit]
- Snorkel: experts built models 2.8× faster than with seven hours of hand labelling, with predictive performance up 45.5% on average; within 3.60% of models on large hand-curated sets on average. Contexts include VA and FDA collaborations, not Indonesian short text — [Snorkel][snorkel]
- mDeBERTa zero-shot: XNLI average accuracy 0.808, ~0.3B parameters, **does not support fp16**, and Indonesian is not mentioned on the card, so it relies on cross-lingual transfer — [model card][mdeb]
- IndoNLI: ~18k pairs. XLM-R reached 82.3% (lay test) and 70.3% (expert test), vs human 89.1% on expert — [IndoNLI][indonli] [snippet]. These are NLI scores, not zero-shot classification accuracy on MBG labels.
- cleanlab: needs out-of-sample `pred_probs` and integer labels; it shortlists candidates for you to decide on; tutorial accuracy 0.87 → 0.89, with no guarantee — [cleanlab text][clean]. Multi-label is supported with `Datalab(task="multilabel")` or lower-level functions, again needing out-of-sample probabilities — [cleanlab multilabel][cleanml]
- BERTopic: precompute embeddings; set `random_state` for UMAP; control topic count with `min_cluster_size` — [best practices][bt1]. For memory: `low_memory=True`, `calculate_probabilities=False`, raise `min_df`, use online modelling in chunks; k-Means has no outliers; `reduce_outliers` reassigns -1; `paraphrase-multilingual-MiniLM-L12-v2` is recommended for multilingual and mpnet for higher quality at more compute — [FAQ][btfaq]. Remove stop words after clustering, not before embedding; Model2Vec is a CPU-friendly option — [tips][bt2]
- `paraphrase-multilingual-MiniLM-L12-v2`: 384-dim, max 128 tokens, ~0.1B parameters; card tags 50 languages without a list — [card][minilm]

**Portfolio evidence**
- G-28: audit dictionaries with hit counts; G-20: circular validation; G-21: synthetic evaluation overstated real detection (AUC 0.97 vs real 0.63 on FIT26) — [LL][ll], [inventory][inv]
- G-39 vs F-CDW2-a: self-training gained under shift, flat under ambiguity — [successful][succ], [CS-CDW2][cscdw]
- G-40: suspects need manual confirmation; blind removal hurt — [failed][fail]

### Inferences

**Recommended few-label ladder (judgement):**
1. **Gold set first:** random sample, ideally 150–300 rows, kept out of all rule writing. At an assumed 10 s/row, 200 rows takes ~33 minutes solo. Use the organisers' labels if any exist.
2. **Keyword rules:** seed ideas such as keracunan, menu, gizi, anggaran, distribusi, dapur, guru. These are **untested hypotheses**, so each rule needs hit count and spot-check precision (G-28). Snorkel's label model is optional: with few rules, a simple weighted vote may be enough, and it is one less dependency.
3. **SetFit-style or simple head:** on sample with a multilingual model. Compare a logistic regression on frozen embeddings first, since it costs seconds after embedding. Use SetFit only if it beats the plain head on gold.
4. **Zero-shot on a sample:** a few thousand rows to seed or to compare. Write hypotheses in Indonesian and evaluate on gold. Do not run it on everything.
5. **Self-train once or twice:** high-confidence rows with an ignore band, scale via TF-IDF + linear across 500k rows, stop when gold stops improving. Check rule permission if test data is used.
6. **Active learning (low-tech):** label a mix of most-uncertain rows and random rows; uncertain rows go to training, random rows protect the gold's honesty.
7. **cleanlab:** only on labelled data, after a first out-of-fold model. Use it to **prioritise reading**, not to auto-delete.

**BERTopic at this scale (judgement):** dedup and fit on a sample of 50k–100k with precomputed embeddings; assign the remainder via nearest topic or transform. If UMAP + HDBSCAN times out or hits RAM, switch to MiniBatchKMeans + c-TF-IDF, which the docs' k-Means option supports in spirit. Use an Indonesian stop-word list in `CountVectorizer`. Keep a fixed random state. Report topics as themes, given the Compfest silhouette of −0.005.

### Gaps
- I did not retrieve a primary source on active learning for this exact setting, or any measurement of Snorkel or SetFit on Indonesian. The ladder order is judgement and needs rehearsal.
- UMAP + HDBSCAN runtime and memory at 50k–500k on 8 GB CPU RAM were not sourced. Measure in rehearsal.
- IndoNLI/XLM-R numbers are NLI scores, so zero-shot quality on MBG labels is unknown.

## Part B3: How to handle imbalance, thresholds, near-duplicates and the noise floor?

### Takeaway
Use grouped validation, determined by a test-sampling check, then a nested, cheap decision layer, with acceptance bars measured from a noise floor. These usually return more per minute than model changes in the portfolio.

### Cited Findings

**Outside evidence**
- Mu et al. analysed 20 social media datasets: removing training posts identical to test posts lowered F1 in 14 of 19. Twitter16 fell 62.6 → 51.9 without duplicates, and 43.0 without near-duplicates. Model rankings changed in 17 of 19. Exact duplicates ranged ~0.1%–8.1%; retained share after near-dup removal could fall to 71.2%. Procedure: mask @USER and URLs, remove exact duplicates, exclude conflicting duplicates unless re-annotated, check near-duplicates with Levenshtein + cosine, dedup train against test — [Mu et al.][mu]
- Lipton et al.: for calibrated probabilities the F1-optimal threshold is half the best achievable F1; an uninformative classifier maximises F1 by predicting all positive, a warning for rare labels — [Lipton et al.][lipton]

**Portfolio evidence**
- CDW2: per-class multipliers by coordinate ascent on macro-F1, fit on 4 folds, scored on 1, over 5 split seeds; argmax under-predicted O2 by 22% — [CS-CDW2][cscdw]
- CDW2: stacking lost 5/5 by 0.008, temperature scaling redundant — [CS-CDW2][cscdw]
- IF26: threshold on evaluated rows inflated 0.575 → 0.917 with 34 negatives; undersampling worsened results monotonically — [CS-IF26][csif]
- IC: small validation subsets had SE 2–2.5× larger, and rankings reversed on LB — [failed][fail]
- `scripts/noise_floor.py` does bootstrap, cluster bootstrap, and paired P(B better) — [SKILL][skill]

### Inferences

**Procedure (judgement, tied to G-07/08/09/14/15):**
1. Normalise copy: lowercase, mask URLs, mentions, numbers optional; hash for exact groups; MinHash/SimHash LSH for near-dup groups. Check cluster sizes for a giant-cluster bug (portfolio's SD26 needed a guard).
2. Check test sampling with the binomial test on pairs seen exactly twice. **Mostly random** → random folds, but report group score too. **Pure clusters not crossing** → group folds.
3. Decide about duplicates in training. Dedup speeds fine-tuning by itself. If test contains the same copypasta, an exact-match lookup may be a legitimate regime, as in IF26 (check rules).
4. Imbalance: start with no class weights, plus a decision layer. Weighted loss only if a rare class is starved in pilot.
5. **Multi-class macro-F1:** per-class multipliers, coordinate ascent, nested. **Binary:** threshold on a smoothed OOF curve, sanity check against F1*/2. **Multi-label:** per-label thresholds, shrunk toward a global threshold for rare labels, because few positives make a free threshold unreliable. Binary relevance first.
6. **Blend vs decision order:** blend first, then fit multipliers on the blended holdout scores, then apply to the same blended models on test. No refit.

**Noise floor example (arithmetic only):** accuracy-like SE for p = 0.85: n = 100k gives 0.0011, n = 20k gives 0.0025. A rare class with 1,000 rows and recall near 0.6 gives SE ≈ 0.0155 on that recall, so a macro score can be much noisier than overall accuracy. Compute the real one via cluster bootstrap on the actual metric. Acceptance bar: 2 SE, plus a seed-delta check where seeds exist.

### Gaps
- No outside source quantifying retweet leakage in Indonesian data specifically. Mu et al. is the best general evidence, and its retweet relevance is my inference.
- Portfolio threshold and multiplier gains come from smaller datasets; size of gain on 500k unknown.

## Part B4: What is the fine-tuning recipe (model order, LLRD, max_len, epochs, ensembling)?

### Takeaway
Run one disciplined recipe: an Indonesian model first, short length from the histogram, 1–2 epochs with fractional-epoch validation, checkpoint and resume, and a rule-based stop. Blend with TF-IDF on the holdout. LLRD and hyperparameters are defaults, not experiments.

### Cited Findings

**Outside evidence**
- Sun et al. settings: LR 2e-5, batch 24, length 512, up to 4 epochs with best-on-validation; decay factors 1.00–0.85 tested, ξ = 0.95 best at 5.40% error on IMDb; head+tail 128+382 got 5.42% vs head-only 5.63% and tail-only 5.44%; the last BERT layer feature is best — [Sun et al.][sun]
- A later study on shortening found a different winner than head+tail in some datasets, attributing it to document type or length — [Truncation vs summarisation][trunc] [snippet]
- HF course: mixed precision, gradient accumulation, dynamic padding are standard — [HF course][hfc]
- Dodge et al.: stopping unpromising runs early is recommended, with seed variance substantial — [Dodge et al.][dodge]
- NusaBERT base and IndoBERT base were close on the Indonesian NusaX row (89.7 vs 90.0), with large variants higher in that table — [NusaBERT][nusabert]
- IndoNLU: large variant beat base on all four datasets read, e.g. SmSA 92.72 vs 87.73 — [IndoNLU][indonlu]

**Portfolio evidence**
- Tuning usually ≤ 0.002; CDW2 eight directions all lost and tuning was dropped — [SKILL §7][skill], [CS-CDW2][cscdw]
- G-33: pretrained helps when pretraining aligns with target — [LL][ll]
- G-34/G-35 for blending; weak members can hurt — [LL][ll], [failed][fail]

### Inferences

**Model order (judgement, all pre-download candidates to verify in rehearsal):**
1. `indobenchmark/indobert-base-p1`: default.
2. `indobenchmark/indobert-lite-base-p1`: fallback if throughput or memory fails.
3. `indolem/indobert-base-uncased` and a NusaBERT base checkpoint: alternatives if code-mix or regional languages dominate. Check names and offline loading first.
4. `xlm-roberta-base`: multilingual fallback only. Large IndoBERT variants look stronger in table but VRAM is the issue; test only if rehearsal shows it fits.
5. Embeddings: `paraphrase-multilingual-MiniLM-L12-v2` first, mpnet only on samples.

**Defaults (judgement):**
- **max_len:** from token-length histogram on 20k samples. Cover about 95% of rows; for short posts probably 64–128. Long-text fallback: head+tail, but only one trial. Every doubling in length costs VRAM and time.
- **Batching:** dynamic padding + length bucketing, gradient accumulation to reach effective ~16–32, LR 2e-5, linear warmup ~5–10%.
- **LLRD:** 0.95 as a set-and-forget default. The IMDb difference is tiny, so never spend a run on it.
- **Validation:** every 0.1–0.25 epoch on a fixed 5k–10k stratified slice from the group-safe holdout. Stop after 2–3 non-improving checks. Save best and latest; save every N minutes.
- **Epochs:** 1 for ≥ ~50k rows, up to 2 if the second finishes by the gate. Time-budget: compute `max_steps` from measured rows/s and the remaining time, leaving inference time.
- **Kill rule:** if at 25% of budget the validation macro-F1 is below the TF-IDF model trained on the *same* rows and not rising, reduce rows or kill. Note the CDW1 warning: treat a pilot as a resource gate, not a scientific verdict.
- **Pooling:** use the model's standard classification head on CLS or simple mean pooling. No custom head.
- **Windows:** `num_workers=0–2` at first; avoid surprises. Keep CPU jobs below 4 threads while the GPU trains. RAM check before each job.

**Ensembling:**
- Binary, AUC or ranking: equal rank average TF-IDF + BERT, comparing against BERT only.
- Multi-class macro-F1: **average probabilities** (or log-probs) instead of ranks, since per-class multipliers need a meaningful scale. Rank averaging per class is possible but needs testing.
- Accept the blend only if holdout Δ clears the bar, and drop weak members. No stacking unless the metric is log-loss.

### Gaps
- Layer-wise decay evidence comes from one English dataset. No source found for an optimal number of epochs on 100k+ Indonesian short text; 1–2 epochs is judgement.
- No source confirmed rank averaging in the Jigsaw winners; the only direct evidence for equal-weight rank averaging is MT26.
- Model-card availability for alternate models, including exact NusaBERT names, was not confirmed in this research beyond search results.

## Part B5: How to win if judged on insight and presentation?

### Takeaway
Indonesian finals have shifted format and sometimes ask for demos or platforms, so prepare a chart and story pipeline as seriously as the classifier. I found no official rubric, so ask the organisers. The winning shape is likely a decision story: what is happening, why, for whom, and what to do, each claim traced to data.

### Cited Findings

**Outside evidence**
- Gemastik XIV data mining, ITB team: prepared for text classification, and the final case was a QA task they had not prioritised. A large internal-vs-submission gap was fixed about 10–30 minutes before the deadline. Note that participants could use internet resources in that event, unlike here — [ITB][itbgem]
- Satria Data 2025, ITB 2nd place: emotion pipeline on Instagram video; semifinal topic clustering and visualisation with BERTopic + HDBSCAN plus an interactive web platform; final onsite at UII — [STEI ITB][stei] [snippet, page later returned HTTP 500]
- Satria Data 2020 finalists presented model and program demonstration — [ITS][its] [snippet]
- UMN thesis on MBG: manually labelled IndoBERT sentiment + BERTopic, 14 main topics — [UMN][umn]. UPN Jatim thesis: tweets 6 Jan–6 Oct 2025, stance + issues via IndoBERT and BERTopic — [UPN][upn] [snippet]. Neither source is a competition; they show that a sentiment-plus-topic approach to MBG is already common, so plain sentiment plus topics may not differentiate.

**Portfolio evidence**
- DataQuest final deck "Juara 1"; IF26 offered operating points (precision ≥ 98% auto-flag, 95%-recall review queue) for deployment, grade S — [inventory][inv], [CS-IF26][csif]
- BERTopic silhouette near zero means topics are themes, not hard clusters — [nlp.md][nlp]

### Inferences

**Analysis stack (judgement):** dedup → topic themes → time series by week → aspect × sentiment heatmap → event alignment → segment comparison if source, location or account fields exist → quote evidence per finding → recommendations with action owner, metric, and risk.

**Storyline, 8–10 slides (judgement; SCQR, as the `sleek-deck` skill describes):**
1. One-sentence answer.
2. Data and method transparency, including dedup (raw → unique → bots/copypasta share).
3. Volume timeline with verified events annotated from data, or marked as hypothesis.
4. Theme map: top topics by share, growth, and negativity.
5. Aspect × sentiment heatmap.
6. Deep dive 1: largest gap or spike, with quotes.
7. Deep dive 2: segment or geography.
8. Validation: gold-set accuracy, topic-stability check.
9. Recommendations: 3 prioritised actions, each tied to evidence, metric and owner.
10. Limits: social media is not the population, no causal claims from co-movement.

**Differentiators (judgement):** separate real volume from copypasta; show a gold-validated classifier score; show stability across seeds; quantify "if X addressed, Y share of complaints affected" without claiming causality. Rehearse 3 likely judge questions: labels? bias? what would you do differently?

**Charts:** one message per chart; a title stating the finding; line for timelines, ranked bars for themes, heatmap for aspects. Prepare matplotlib templates in advance; they must run offline.

**Hybrid events:** if both a submission score and a presentation count, set the freeze so the deck gets at least 45 minutes, and write the model explanation, validation scheme and error analysis into the slides.

### Gaps
- No official scoring rubric for the MBG final was found; the name of the competition was not given. The Satria Data and Gemastik details are past events and may not reflect this event's rules.
- I found no direct evidence on what wins Indonesian presentation-judged data finals. Recommendations above are judgement.

## Part B6: What must the pre-built code kit contain?

### Takeaway
Build parameterised scripts that run offline from a single config, write outputs that can be re-used, and pass a Wi-Fi-off test. The kit's purpose is to remove thinking time and environment risk on the day.

### Cited Findings

**Portfolio evidence**
- G-44 (save OOF + test + ids), G-47 (shared config, parity), G-50 (GPU gate), L-MT-13 (bundle + hash check), and C. team artefacts (`YYYYMMDD-HHMMSS-author-slug` experiment packages, `build_notebook.py`, registries) — [LL][ll], [CS-MT26][csmt], [inventory][inv]
- Infrastructure failures: silent CPU fallback, deadlocks in DataLoader workers, stale outputs — [failed][fail]

**Outside evidence**
- BERTopic: precompute and reuse embeddings, save with safetensors — [best practices][bt1]
- cleanlab: requires out-of-sample probabilities, which the kit's OOF format supplies — [cleanlab][clean]

### Inferences

All tools below are recommended by judgement.

| # | Script | What it does | Pre-test |
|---|---|---|---|
| 0 | `00_env_check.py` | Prints GPU, CUDA op timing, fp16 NaN check, free RAM/disk, offline load of every model; `HF_HUB_OFFLINE=1` | Wi-Fi off |
| 1 | `config.yaml` + `common.py` | Paths, column names, metric, seeds, group file, run logger; one source of truth | Used by all scripts |
| 2 | `01_load_parquet.py` | Chunked read to parquet, id and duplicate asserts, memory report | CSV, JSONL, Excel variants |
| 3 | `02_fast_eda.py` / notebook | Row counts, null map, label census, token-length histogram, language/slang probes, time span, top ngrams, source/user counts, train-test differences; outputs PNGs + short text | 500k-row scale test |
| 4 | `03_dedup_groups.py` | Mask, exact hash, near-dup LSH, cluster-size report, conflicting labels, train-test overlap, binomial split check; writes `groups.parquet` | Synthetic copypasta test |
| 5 | `04_metric.py` | Macro-F1, micro, QWK, AUC, MAE/RMSE, NDCG, custom; unit tests vs sklearn | Pass tests |
| 6 | `05_tfidf_linear.py` | Word + char features, hashing fallback, LR/SGD/LinearSVC, learning-curve flag, K-fold OOF, test, ids, top terms | Runs under 8 GB; records minutes |
| 7 | `06_finetune.py` | Time-budgeted HF or plain PyTorch loop, dynamic padding, length bucket, accumulation, fp16 toggle, LLRD, multi-class/multi-label/regression heads, **resume from checkpoint**, val every N, early stop, predict | Kill and resume test; batch-size finder |
| 8 | `07_embed_memmap.py` | Batch embeddings to float16 or float32 memmap, resume via done index | Interrupt test; alignment assert |
| 9 | `08_head_on_embed.py` | LR/LightGBM head and kNN on embeddings, with folds | Runs on sample |
| 10 | `09_weak_labels.py` | Keyword rules with hit-rate and precision template, weighted vote, gold-set eval | Word-boundary test |
| 11 | `10_setfit_zeroshot.py` | SetFit fit; zero-shot with hypothesis template, fp32 for mDeBERTa, sample mode | Offline load check |
| 12 | `11_topic.py` | BERTopic on embeddings, sample-then-assign; stop words; k-Means fallback; save topic table, over time | 50k/100k timing |
| 13 | `12_decision_layer.py` | Binary threshold, multi-class multipliers, multi-label thresholds; nested report; loads saved preds | Unit test on synthetic |
| 14 | `13_blend.py` | Equal mean, rank mean, agreement, holdout Δ, noise-floor bootstrap (copy `noise_floor.py`, keep the original unchanged) | Runs on stored preds |
| 15 | `14_error_audit.py` | Random error sample, confident-wrong list, cleanlab shortlist, segments | Readable output |
| 16 | `15_validate_submission.py` | Columns, row count, id set and order, dtype, no NaN, allowed labels, class distribution compared to train, encoding/line endings, SHA-256, copies to `submissions/` | Sample submission format variations |
| 17 | `16_charts.py` | Timeline, ranked bars, aspect heatmap, sentiment by topic, confusion matrix, learning curve templates with Indonesian-friendly labels | Offline fonts OK |
| 18 | `slides_template.pptx` | Skeleton from B5 with empty charts and notes; `pptx` export if Python-based | Opens offline |
| 19 | `RUNBOOK.md` | Decision tree, gates, commands to paste, error fixes | Print a copy |

**Offline package (judgement):** Python env (conda-pack or `pip download` wheelhouse), torch build matching the driver, GPU driver installer, Indonesian stop words, HF cache, VS Code/Jupyter extensions, fonts, slide tool, offline docs for sklearn/transformers/BERTopic, USB backup, charger/extension cord if allowed. Also pre-cache only licensed resources and declare them if organisers require.

### Gaps
- Exact package versions compatible with this driver and laptop can only be pinned by testing here, and I did not test any.
- Stop-word lists and slang dictionaries were not reviewed; apply the G-28 hit-rate check before use.

## Part B7: What is the 5-hour timeline, decision tree, freeze and rehearsal plan?

### Takeaway
Submit a validated baseline by about minute 60–70, decide branch at minute 40, allow a single main GPU bet until about minute 205, freeze modelling at minute 255, and stop submitting by minute 290. Rehearse at full length on public Indonesian data at least twice.

### Cited Findings

**Portfolio evidence**
- Freeze a reproducible pipeline and select two finals: one principled and one robust hedge; stop after 3–5 attempts within noise — [SKILL §9][skill]
- Problem-card items (unit, metric, decision layer, rules, compute) — [SKILL §1][skill]

**Outside evidence**
- The ITB Gemastik team discovered a big internal-vs-submission gap on first submission, and solved it just before the deadline, which supports early submission and a task-shift plan — [ITB][itbgem]
- IndoNLU sets for rehearsal: EmoT 3,521, SmSA 11,000, CASA 810, HoASA 2,283 train — [IndoNLU][indonlu]. Liputan6: 215,827 Indonesian document-summary pairs, useful for scale; access terms not confirmed — [Liputan6][lip]

### Inferences

**Master timeline (T = minutes elapsed; judgement, adjust after rehearsal):**

| T | Action | Output or gate |
|---|---|---|
| 0–10 | Read everything twice; problem card; ask rule questions; start env check; back up raw data read-only | Written metric, submission format, deadline, limit |
| 10–25 | Load to parquet; fast EDA; metric replica | Label census, length histogram |
| 25–40 | Dedup, groups, test-sampling check, folds/holdout | **Gate G1 at T40: choose branch** |
| 40–70 | TF-IDF + linear (or branch baseline), validator, **first submission by T70** | **G2:** local-vs-submission gap acceptable? If not, debug format/parity first, before anything else |
| 70–90 | Noise floor, decision layer, TF-IDF learning curve, error audit (read 50 random errors) | Acceptance bar; hedge submission |
| 90–105 | BERT throughput test; pilot on same rows as a TF-IDF comparison | **G3 at T105:** go, shrink, or skip BERT |
| 105–205 | Main GPU run (≤100 min with checkpoints); CPU work kept under ~3–4 threads: one extra TF-IDF view, audit, slide draft | **G4 at T150:** projected finish on time and validation trend OK? Else cut steps and predict from best checkpoint |
| 205–235 | Test inference, holdout blend | **G5 at T235:** blend beats best by bar? Else keep best single |
| 235–255 | Nested decision layer; candidate A (principled) and B (hedge) | Validated files and hashes |
| **255** | **Freeze modelling.** No new models | |
| 255–280 | Reproduce on a small sample; write method notes; finish deck | Notes, code, charts |
| 280–290 | Final upload; verify receipt; screenshot | **Do not wait for the last minutes** |
| 290–300 | Buffer only | No code changes |

**Variants:** presentation-heavy → freeze analysis at T210 and slides T210–280; no labels → replace BERT block with embedding + topic + gold validation; limited submissions → use local holdout, submit only at T70, ~T205, and final.

**Branch tree (thresholds are judgement):**
- First: **Are pretrained models allowed?** No → TF-IDF, LightGBM, word2vec, IF26 feature questions; skip BERT.
- **A. Supervised, many labels** (say ≥ 20k usable after dedup, classes reasonably populated): TF-IDF on all → decision layer → IndoBERT on subsample → blend. If the pilot is flat: LightGBM/SVD, extra char view, label audit, seed averaging.
- **B. Few labels** (~50–5k): gold split → TF-IDF + embedding LR → SetFit → rules/zero-shot sample → one or two self-training rounds → TF-IDF on confident rows. Validate only on gold.
- **C. No labels, score from insight or a hidden metric:** dedup → embedding → topic themes → weak sentiment/aspect models on gold → timeline → recommendations. If a hidden label score exists (e.g. clustering vs hidden labels), submit a robust baseline early.
- **D. Non-classification:**
  - Ordinal/regression: TF-IDF Ridge/LightGBM, BERT regression head, cut points (G-03, QWK note).
  - Ranking/retrieval: TF-IDF/BM25 + embedding similarity, rank average per query, per-group metric (G-02, L-MT-09).
  - NER/aspect extraction: pretrained or token classification if labels exist; regex and gazetteer backstop.
  - Pairs/consistency: relation features, retrieval margin, cross-encoder if allowed; BOW is not enough (IF26).
  - QA/summarisation: extractive baseline first (TF-IDF sentence retrieval, lead-sentence, TextRank-like), then pretrained if pre-downloaded. Generation on 4 GB will likely be limited, so do not plan on it. Ask whether LLMs are allowed.

**Safe-submission rules:**
- Validate each file via `15_validate_submission.py`; name `T{minute}_{model}_{localscore}_{sha8}.csv`; keep a last-known-good; log every upload.
- First submission early; check that local-vs-platform is in a plausible range; compute SE for the public score.
- Do not select on tiny public moves; two finals if allowed (principled + hedge).
- Never overwrite; never edit manually in Excel, which can mangle ids and encoding.

**Rehearsal plan (judgement; start now, dates relative to the event):**

| When | Drill | Pass criterion |
|---|---|---|
| Asap | Env build; `00_env_check`; run all scripts on small data; fp16 vs fp32 test | All pass with Wi-Fi off |
| D-7 to D-5 | Scale test: 500k synthetic rows (repeat Liputan6 or other Indonesian texts with perturbation) for RAM, TF-IDF time, embedding rate, inference rate. Use it only for timings, never scores | Timings table filled; no OOM |
| D-6 | Timed 5-hour A drill: SmSA or EmoT with an ad hoc metric and format | Baseline by T70; hit freeze |
| D-4 | Timed few-label drill: hide most labels; 200 gold; ladder; compare to full-label score | Know which steps pay |
| D-3 | Timed no-label drill: any Indonesian text → topics, timeline, slides | Finish a 10-slide deck |
| D-2 | Surprise D drill: pair/regression/ranking or QA task picked by a friend, or randomly | Have a working baseline in 60–90 min |
| D-1 | Freeze kit; print runbook; battery, charger, disk space, USB; no new features; sleep | Checklist signed off |

Also, in one rehearsal, deliberately kill training and resume; run the machine plugged in for hours to observe thermal throttling; disable sleep and Windows Update for the day if rules permit.

**If alone (judgement):** Ivan's sequence is the master timeline. Make the kit do the coding, hold 10-minute breaks around long GPU blocks, and use GPU blocks for the deck or error audit. Keep only one experiment on the GPU at a time.

**If in a team (judgement):**
- 2 people: **Modeller** owns GPU, validation, decision layer. **Analyst** owns EDA, gold labels, error audit, charts, deck, submission log.
- 3 people: add a **Data/QA** role: dedup, kit operator, weak rules, submission validator, reproducing runs; or split insight and deck.
- Everyone has a clone of the kit on their laptop if allowed; one person controls submission. Sync every 30 minutes, with pre-agreed gates. One shared config and one file naming rule. A second person must verify before final upload.

**Rule checks to ask organisers before the day (send in advance, ideally get answers in writing):**
1. Are pretrained models allowed, including IndoBERT, sentence-transformers and local LLMs? Do they need to be declared or public before a date?
2. May external data, lexicons, stop-word lists, gazetteers, or previously downloaded corpora be used?
3. Own laptop allowed? Any provided machine? Admin rights? Can USB drives/external disks be used? When can files be transferred?
4. Internet: fully off, or allowed only for submission? How is submission done? Any AI assistants or LLM services allowed?
5. May test data be used in training (self-training, TF-IDF fit on test, dedup)? Is hand-labelling allowed, including test rows?
6. Submission limit per day/total? Public/private split? Which submission counts? Tie-break rule?
7. Is code submission, reproducibility or rerun required? Runtime limit? Which format?
8. Judging: leaderboard vs presentation weights, and rubric? Presentation time, language (Indonesian/English), and when slides are due?
9. Is the data format and field list known ahead of time (language, source, label type)? Are test labels hidden? Is there a topic hint?
10. Power sockets, extension cords, table space, wifi for submission, phones, printed notes, external GPU, team-members' number of laptops, breaks and food?

### Gaps
- Event rules, scoring and date are not known. All timings above are planning assumptions and should be revised after the rehearsal measurements.
- No public source was found with a validated minute-by-minute plan for this format; the timeline is judgement built from the portfolio's process lessons.
- Access terms for Liputan6 and other rehearsal sets were not confirmed.

[skill]: skills/ml-problem-solving/SKILL.md
[nlp]: skills/ml-problem-solving/domains/nlp.md
[ll]: skills/ml-problem-solving/knowledge/lessons-learned.md
[inv]: skills/ml-problem-solving/case-studies/project-inventory.md
[fail]: skills/ml-problem-solving/knowledge/failed-experiments.md
[succ]: skills/ml-problem-solving/knowledge/successful-experiments.md
[eda]: skills/ml-problem-solving/frameworks/eda-checklist.md
[val]: skills/ml-problem-solving/principles/validation-strategy.md
[csif]: skills/ml-problem-solving/case-studies/ifest26-forged-headline.md
[csmt]: skills/ml-problem-solving/case-studies/minetoday26-ranking.md
[cscdw]: skills/ml-problem-solving/case-studies/cdw26-binary-optlevel.md
[csearly]: skills/ml-problem-solving/case-studies/early-and-misc-projects.md
[nusabert]: https://ar5iv.labs.arxiv.org/html/2403.01817
[indonlu]: https://ar5iv.labs.arxiv.org/html/2009.05387
[moon]: https://www.themoonlight.io/review/benchmarking-logistic-regression-svm-naive-bayes-and-indobert-fine-tuning-for-sentiment-analysis-on-indonesian-product-reviews
[jaic1]: https://jurnal.polibatam.ac.id/index.php/JAIC/article/view/12610
[jati]: https://ejournal.itn.ac.id/jati/article/view/15611
[binus]: https://journal.binus.ac.id/index.php/ijcshai/article/view/15184
[umn]: https://kc.umn.ac.id/id/eprint/47931/
[upn]: https://repository.upnjatim.ac.id/52506/
[indobert]: https://huggingface.co/indobenchmark/indobert-base-p1
[gonz]: https://ar5iv.labs.arxiv.org/html/2005.13012
[setfit]: https://ar5iv.labs.arxiv.org/html/2209.11055
[dodge]: https://arxiv.org/abs/2002.06305
[hash]: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html
[sijun]: https://sijunhe.github.io/2018/05/01/kaggle-toxic-comment.html
[montg]: https://amontgomerie.github.io/2022/02/08/jigsaw-toxic-severity-competition.html
[pt58123]: https://github.com/pytorch/pytorch/issues/58123
[hfblack]: https://huggingface.co/runwayml/stable-diffusion-v1-5/discussions/12
[mix]: https://mixpeek.com/model/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
[snorkel]: https://arxiv.org/abs/1711.10160
[mdeb]: https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
[indonli]: https://arxiv.org/abs/2110.14566
[clean]: https://docs.cleanlab.ai/stable/tutorials/clean_learning/text.html
[cleanml]: https://docs.cleanlab.ai/stable/tutorials/multilabel_classification.html
[bt1]: https://maartengr.github.io/BERTopic/getting_started/best_practices/best_practices.html
[btfaq]: https://maartengr.github.io/BERTopic/faq.html
[bt2]: https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html
[minilm]: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
[mu]: https://ar5iv.labs.arxiv.org/html/2410.03545
[lipton]: https://arxiv.org/abs/1402.1892
[sun]: https://ar5iv.labs.arxiv.org/html/1905.05583
[trunc]: https://arxiv.org/pdf/2403.12799
[hfc]: https://huggingface.co/learn/llm-course/chapter3/6
[itbgem]: https://itb.ac.id/news/persistence-become-the-success-key-for-team-nasigoreng-to-won-bronze-medal-at-gemastik-xiv/58223
[stei]: https://stei.itb.ac.id/en/?p=26866
[its]: https://www.its.ac.id/news/en/its-team-successfully-wins-3rd-place-in-satria-data-2020/
[lip]: https://arxiv.org/abs/2011.00679
