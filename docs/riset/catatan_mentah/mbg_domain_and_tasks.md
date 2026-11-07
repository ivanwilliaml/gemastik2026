# MBG as an NLP domain, and which competition tasks are plausible (as of 5 Nov 2026)

Evidence labels used below: "Cited Findings" are sourced facts. "Own analysis" means I computed it from a public file. "Inferences" and all probabilities are my judgement, not facts.

## Which competition is this? ("temasek", 5 hours, offline, download the day before)

### Takeaway
"Temasek" is very probably a speech-to-text error for **GEMASTIK**. The official GEMASTIK XIX 2026 Data Mining final matches the user's description: a 5-hour case, a separate workstation set-up session the day before, and an offline presentation. The final runs 10-13 Nov 2026, which is days away. No source confirms the MBG theme or the 500k rows, because the guide says the case is only released at the final.

### Cited Findings
- **Schedule, Kompetisi III Penambangan Data:** workstation set-up on 10 Nov 2026 12:00-18:00 WIB, case solving on 11 Nov 13:00-18:00 WIB, solution presentations on 12 Nov 08:00-17:00 WIB (20 teams), closing and winners on the night of 13 Nov. — [Panduan GEMASTIK XIX 2026, Bab II](https://kemahasiswaan.unand.ac.id/images/documents/Panduan-GEMASTIK-2026.pdf)
- **Venue:** Universitas Bina Nusantara (BINUS), with the final on 10-13 Nov 2026. — [UNIMMA summary of the guide](https://kemahasiswaan.unimma.ac.id/pedoman/panduan-gemastik-2026/)
- **Final rules:** teams receive a dataset with a short description, install their environment on devices provided by the committee, get 5 hours to build a model and use it for evaluation, document the work as a PPT, and present offline to the jury. Tools, libraries and frameworks are allowed. — [Panduan 2026, Bagian C, pp. 25-26](https://kemahasiswaan.unand.ac.id/images/documents/Panduan-GEMASTIK-2026.pdf)
- **Case release:** "Kasus soal akan diberikan pada saat final". Deliverables are a zip/rar with model-development documentation, source code and the team PPT. — [Panduan 2026](https://kemahasiswaan.unand.ac.id/images/documents/Panduan-GEMASTIK-2026.pdf)
- **Final scoring:** preliminary score 20%, **performance score (leaderboard) 30%**, method appropriateness 25%, clarity and completeness of presentation 25%. — [Panduan 2026, p. 27](https://kemahasiswaan.unand.ac.id/images/documents/Panduan-GEMASTIK-2026.pdf)
- **The format is stable:** the 2024 edition used the same three stages (set-up, 13:00-18:00 case solving, next-day presentation), the same 5 hours and the same weights. Judges do not score teams from their own campus. — [GEMASTIK XVII slides, secondary upload](https://www.slideserve.com/caringal/gemastik-xvii-data-mining-competition-2024). 2025 also had a set-up day then a case day, offline at Telkom University. — [search-result summary of the 2025 guide](https://kemdiktisaintek.go.id/wp-content/uploads/2025/07/Panduan-GEMASTIK-2025-ver2.pdf)
- **Theme:** the 2026 preliminary theme is "Penambangan Data untuk Solusi yang Berdampak, Inklusif, dan Berkelanjutan". The guide names no MBG theme. — [Panduan 2026](https://kemahasiswaan.unand.ac.id/images/documents/Panduan-GEMASTIK-2026.pdf)
- **Past finals, the most useful precedents:**
  - GEMASTIK XIV (2021): the preliminary was classification of Indonesian vaccine tweets (positive/negative and subjective/objective). **The final case then changed to question answering**, which the bronze-medal team had not prepared for. They also saw a large gap between internal test accuracy and the official output, and fixed it 10-30 minutes before the deadline. Internet was allowed that year. — [ITB](https://itb.ac.id/news/persistence-become-the-success-key-for-team-nasigoreng-to-won-bronze-medal-at-gemastik-xiv/58223)
  - GEMASTIK 2018: a 6-hour task using BMKG weather data; the IPB team spent the first three hours on cleaning because the data was "dirty". The top five presented. — [IPB](https://www.ipb.ac.id/?p=54885)
- **Other events checked:**
  - ACTION 2026 (UNESA): data-mining subtheme on public complaints, final case 14 Nov, presentations 15-16 Nov; the finalist announcement is only on 7 Nov. — [ACTION](https://action-unesa.com/id)
  - Data Slayer 4.0: grand final 19 Dec 2026. — [Data Slayer](https://data-slayer.com/)
  - Satria Data 2026: offline final at Undip on 25-29 Oct 2026, already past. — [RRI](https://rri.co.id/tarakan/iptek/2385935/siapkan-talenta-data-masa-depan-kemdikti-saintek-luncurkan-satria-data-2026)

### Inferences
- "Gemastik" and "temasek" sound close, and the 5-hour and set-up-day details match. I put about 90% on GEMASTIK, as a judgement. The user should confirm in one word.
- A 30% leaderboard weight means the case almost certainly has a machine-scorable target. A purely unsupervised "insights only" case is therefore unlikely.
- Method appropriateness plus presentation (50%) outweigh the leaderboard (30%). A defensible method and clear MBG insight can beat a small score gain, and the deck must be planned from hour one.
- The preliminary 20% is already fixed, so only 80% remains in play.
- Committee-provided devices mean hardware, GPU and package availability are unknown until 10 Nov. The set-up session is the only reliable moment to test the stack and load model weights.
- The 2021 question-answering surprise is the key risk: prepare a generic pipeline, not only a sentiment one.

### Gaps
- My searches found no competition literally named Temasek, and no 2026 Indonesian data-mining final publicly themed on MBG.
- The guide does not state "no internet" or rule on external data, pretrained weights or LLM use during the final. These must be asked at the set-up session or technical meeting on 10 Nov.
- I could not find how the user learned of MBG and 500k rows. If it came from the committee, ask whether any schema or label information was also given.
- No official archive of past final cases exists that I could find. The 2024 technical guide PDF was unreachable, and no 2022-2025 final case descriptions turned up.

## Published NLP work on MBG

### Takeaway
The literature is dominated by 3-class sentiment and BERTopic on small samples. IndoBERT-family models report about 70-89% accuracy. Class balance swings from strongly positive to strongly negative across studies, so do not assume a prior. I found stance work, one unread ABSA thesis, and no multi-label aspect paper.

### Cited Findings
| Study | Data | Labels / task | Result |
|---|---|---|---|
| [UPN Jatim thesis](https://repository.upnjatim.ac.id/52506/) | X, 6 Jan-6 Oct 2025 | Stance: pro / neutral / contra; BERTopic | IndoBERT acc 0.8910, macro-F1 0.8703 (normalised text, 70:30, class weights); 12 topics, coherence 0.6367 |
| [JSAI 2026](https://jurnal.umb.ac.id/index.php/JSAI/article/download/10555/6282/46478) | 5,509 YouTube comments | Stance; BERTopic vs LDA | IndoBERT-base-p1 acc 70.00%, macro-F1 0.7002; 19 topics, C_v 0.4918 vs LDA 0.4786; oppositional 50.0% |
| [JAIC](https://jurnal.polibatam.ac.id/index.php/JAIC/article/download/9754/3007/33149) | 19,843 YouTube comments, 71 videos, Nov 2024-Apr 2025 | BERTopic (384-dim embeddings, UMAP, HDBSCAN) | 10 topics; coherence 0.46 vs LDA 0.39; diversity 76% vs 71% |
| [UMN thesis](https://kc.umn.ac.id/id/eprint/47931/) | X | 3-class; BERTopic | IndoBERT acc 78.00%, F1 77.88%; neg 38.33 / neu 34.24 / pos 27.43; 14 topics |
| [JATI SNA](https://ejournal.itn.ac.id/jati/article/view/16948) | X, de-duplicated; test N=142 | 3-class; mention-network centrality | Weighted F1 0.7413; pos 41.9 / neu 30.1 / neg 28.0 |
| [JATI](https://ejournal.itn.ac.id/jati/article/download/15611/8415) (search snippet only) | X | 3-class; IndoBERT vs SVM vs RF | IndoBERT acc 87.60%; pos 43.9 / neu 41.8 / neg 14.3 |
| [METIK](https://journal.universitasmulia.ac.id/index.php/metik/article/download/1403/617/5812) (search snippet only; full text returned 403) | 126,365 tweets | IndoBERTweet | Acc 85.50%; negative 60.6% |
| [UMN thesis](https://kc.umn.ac.id/id/eprint/48247/) (snippet only) | 9,000 YouTube comments | IndoBERT | Acc 0.8507 |
| [IJID](https://ejournal.uin-suka.ac.id/saintek/ijid/article/view/5211) | YouTube | LDA | 3 topics: price and distribution worries, benefits for children, emotional and political reactions |
| [Mendeley / BERT Instagram](https://data.mendeley.com/datasets/gfsn8x62dk/1) | 14,051 Instagram comments, South Sumatra | Sentiment; BERT | Scores not on the data page |

- **Ready-made taxonomy.** The 19 JSAI topics are: facilities, sustainability evaluation, side-dish and milk substitution, equity of coverage, officials' potential corruption, a "tuntutan" (demands) topic, kitchen and yayasan management, poisoning victims, school and kitchen infrastructure, campaign-programme accusations, criticism of the President, expansion to regions and villages, student poisoning incidents, fear of toxic food, calls to stop the programme, budget waste, state budget, corruption of education funds, plus irrelevant comments. Opposition reached 85.0% on calls to stop, 86.1% on officials' corruption and 69.0% on the state budget; kitchen and yayasan management was 46.4% neutral. Its abstract also reports opposition of 91.1% on food safety. — [JSAI](https://jurnal.umb.ac.id/index.php/JSAI/article/download/10555/6282/46478)
- **Other topics found:** campaign promises and political figures, misuse of budget, benefits for children and schools, urban-rural and eastern Indonesia equity, economic impact, military oversight ("militer", "tentara"). — [JAIC](https://jurnal.polibatam.ac.id/index.php/JAIC/article/download/9754/3007/33149)
- **Aspects in other studies:** negative sentiment centres on food safety and implementation quality ([Univ. Jember](https://repository.unej.ac.id/handle/123456789/13952)), and on food safety and budget transparency ([SVM study on X](https://jurnal.polgan.ac.id/index.php/jmp/article/view/16343)). Online news is critical on execution and distribution ([IPB thesis](https://repository.ipb.ac.id/jspui/handle/123456789/165583?mode=full)).
- **Monitoring firms and institutes:**
  - Drone Emprit, 20 Oct 2024-20 Oct 2025: 50% positive, 36% negative, 14% neutral.
  - INDEF on X and TikTok: negative rose from 85% to 97% during September 2025, with 444,900 conversations crawled 1-27 Sep 2025.
  - INDEF/Continuum, Oct 2026: 83% negative on about 173k posts across X, TikTok, YouTube and Instagram.
  - All are compiled with source links in [this GitHub CSV](https://raw.githubusercontent.com/andikapratamazaid/analisis-program-mbg/main/data/sentimen_medsos_studi.csv). The underlying citations include low-tier sites, so treat them as secondary.

### Inferences
- A realistic baseline expectation is macro-F1 of 0.70-0.87 on clean labels. Many reported scores come from small test sets (N=142 in one case) with single random splits, so they likely overstate performance on a 500k social dataset.
- The positive-versus-negative contradiction across studies follows from different platforms, periods and labellers. Class priors in the user's dataset will depend on the sampling windows, especially September 2025.
- The JSAI paper's argument that sentiment does not separate operational criticism from rejection of the programme is good jury material.

### Gaps
- The [UII ABSA thesis](https://dspace.uii.ac.id/123456789/66313) returned 403, so its aspect list is unknown. I found no published multi-label aspect paper on MBG.
- Search results mention a 13,538-document TikTok LDA vs BERTopic study, and theses with 13,142, 5,516, 1,000 and 813 tweets. I could not tie these to verified URLs, so they are not tabulated.
- I found no published MBG work on hoax detection, bot detection, NER, summarisation or engagement regression. Absence from my search is not proof of absence.

## Public MBG datasets to download the day before

### Takeaway
No public dataset near 500k rows exists that I could find. The public total is roughly 100k rows and mostly auto-labelled. They are best used as a domain corpus (embeddings, TF-IDF, lexicon, sanity tests), not as gold labels. **Check whether external data is allowed before using any of it for training.**

### Cited Findings
| Dataset | Content and rows | Labels | Licence |
|---|---|---|---|
| [Kaggle: TikTok MBG comments](https://www.kaggle.com/datasets/sinryurifal/dataset-komentar-tiktok-mbg-makan-bergizi-gratis) | 48.4 MB, raw comments and replies; row count not shown in metadata I read; updated 26 May 2026 | None | CC BY-NC-SA 4.0 |
| [GitHub: TikTok, MBG and KDMP](https://github.com/fidopecado/Indonesian-Tiktok-Comments-Sentiment-Analysis) (`MBG_PROCESSED.csv`) | 57,362 rows, 33,897 unique comment IDs (own analysis) | Auto: several HF models plus majority vote | None stated |
| [GitHub: YouTube sentiment](https://github.com/briannaufal211/mbg-youtube-sentiment-analysis) | 12,000 comments; 11,557 cleaned | AI-assisted, human-reviewed; not independent gold | None stated |
| [Mendeley: Instagram](https://data.mendeley.com/datasets/gfsn8x62dk/1) | 14,051 comments from 93 posts, 15 Feb-12 Oct 2025 | Not stated on page | CC BY-SA 4.0 |
| [Kaggle: YouTube, one video](https://www.kaggle.com/datasets/dzikriraihan/indonesian-youtube-comments-mbg-program) | 5,035 comments, 4.6 MB, scraped 30 Jun 2026 | Auto: `w11wo/indonesian-roberta-base-sentiment-classifier` | CC0 |
| [Kaggle: YouTube](https://www.kaggle.com/datasets/muhammadzaiddan/data-sentimen-makanan-bergizi-gratis-di-indonesia) | About 1,380 rows, 1.07 MB (author, published_at, like_count, text) | None listed | CC BY-SA 4.0 |
| [Kaggle: Twitter](https://www.kaggle.com/datasets/hanif281103/sentimen-publik-terhadap-makan-bergizi-gratis) | 1.33 MB; rows not stated; June 2025 | Sentiment per subtitle; not inspected | Apache 2.0 |
| [HF: X sentiment 2025](https://huggingface.co/datasets/tasyapndya/free_nutritious_meals_sentiment_x_id_2025) | About 1,102 rows, Jan-Mar 2025 | positif / negatif / netral, manual per card | Not stated |
| [GitHub: structured facts](https://github.com/andikapratamazaid/analisis-program-mbg) | Incidents, SPPG counts, closures, surveys, each with source URL; through 29 Oct 2026 | n/a | None stated in page text |

- The sentiment models used in the GitHub TikTok repo are named by the handles taufiqdp, Agufsamudra, Mdhugol and Aardiiiiy. — [repo](https://github.com/fidopecado/Indonesian-Tiktok-Comments-Sentiment-Analysis)
- **Own analysis of `MBG_PROCESSED.csv`, directly useful as a stand-in:**
  - 40.9% of rows repeat an existing `comment_id`; a further 6.5% of the unique-ID rows repeat text.
  - 9 videos; one video supplies 23,981 of 33,897 unique comments (71%).
  - By month, 20,783 unique comments are from September 2025 and 7,361 from May 2026, with very little in December-January.
  - Ensemble labels on unique IDs: negatif 62.2%, positif 23.8%, netral 6.9%, empty 7.1%.
  - Median comment length is 42 characters, computed on all rows.
  - Emoji-only rows and "[Sticker]" rows are among the most frequent exact texts; 273 unique-ID rows have an empty comment.

### Inferences
- Priority downloads if allowed: the TikTok CSV and the YouTube sets as a dev corpus, the 1,102 manual rows as a small sanity set, the facts repo for slides. Use the facts repo's numbers only after opening their source links.
- If the organisers' labels were machine-generated (see next section), the public HF sentiment models become very valuable: their predictions may partly reproduce the hidden labelling. Resolve exact HF model IDs for the handles above on the download day, and save the weights locally.
- CC BY-NC-SA and "licence not stated" sets should not be redistributed in deliverables. Cite the source in the deck if used.

### Gaps
- Kaggle file-level row counts for the TikTok and Twitter sets were not visible without opening the files.
- I did not check Satu Data directly; no MBG text dataset surfaced there through search.

## Domain facts: timeline, institutions, spikes and hoaxes

### Takeaway
The discourse has clear regime changes: launch (Jan 2025), the poisoning peak (Sep-Oct 2025), the corruption and leadership shock (Jun-Jul 2026), and a second poisoning wave with budget debate (Aug-Oct 2026). A model trained on one window will drift. **Any lexicon that treats Dadan Hindayana as current head is out of date: the head is Sudaryono.**

### Cited Findings
**Timeline for spike features**
- 6 Jan 2025: launch with 190 SPPG and 570,000 recipients. — [compiled CSV citing BGN](https://raw.githubusercontent.com/andikapratamazaid/analisis-program-mbg/main/data/perkembangan_program.csv)
- 29 Apr 2025 Kota Bandung (342); 7 May 2025 Kota Bogor (223; Salmonella and E. coli); 11 Aug Sragen (365); 27 Aug Lebong (427; S. aureus in meatballs); 17 Sep Garut (569) and Banggai Kepulauan (339); 22 Sep Bandung Barat (1,333, other counts differ). — [compiled incident CSV](https://raw.githubusercontent.com/andikapratamazaid/analisis-program-mbg/main/data/insiden_terbesar.csv)
- 24 Sep 2025: Bandung Barat declares KLB; the BGN count at that time was 70 cases and 5,914 recipients for Jan-Sep 2025, with JPPI around 6,452. — [INP Polri](https://inp.polri.go.id/artikel/mass-poisoning-in-west-bandung-local-administration-declares-extraordinary-event); [Republika](https://en.republika.co.id/berita/t377zb415/nearly-6000-students-experience-food-poisoning-during-the-free-nutritious-meal)
- Oct 2025: incident peak at 85 cases, then 40 in Nov, 12 in Dec and 10 in Jan 2026. — [Tirto](https://tirto.id/bgn-ungkap-jumlah-keracunan-mbg-turun-hanya-10-kasus-di-januari-hpxM)
- 5 Jan 2026 Grobogan (803). — [detikHealth](https://health.detik.com/berita-detikhealth/d-8306014/bgn-investigasi-803-orang-dilaporkan-keracunan-mbg-di-grobogan)
- 2 Jun 2026: Prabowo removes Dadan Hindayana and two deputies, Kejagung searches the BGN office that night. 3 Jun: Dadan, Sony Sonjaya and Lodewyk Pusung named suspects and Dadan detained, over alleged sale of SPPG points, verification interference and mark-ups. These remain allegations. — [Investortrust](https://www.investortrust.id/national/105105/kantor-bgn-digeledah-kejagung-seusai-dadan-hindayana-dicopot-prabowo); [Jawa Pos](https://jawapos.com/kasuistika/2606030202/mantan-kepala-bgn-dadan-hindayana-resmi-jadi-tersangka-dan-ditahan-kejagung-belum-24-jam-usai-dicopot-prabowo)
- Jun-Jul 2026: Nanik S. Deyang leads BGN, then resigns on health grounds. 22 Jul: Sudaryono is sworn in. — [Bisnis](https://kabar24.bisnis.com/read/20260722/15/1990142/resmi-dilantik-jadi-kepala-bgn-ini-langkah-awal-sudaryono); [afu.id](https://afu.id/nasional/warisan-di-balik-kursi-bgn-untuk-sudaryono). Kompas's 100-days piece describes him as replacing Dadan directly, omitting Nanik. — [Kompas](https://nasional.kompas.com/read/2026/10/30/08004121/100-hari-sudaryono-pimpin-bgn-bongkar-yayasan-calo-hingga-janji-transformasi)
- 6 Jul 2026: Banggar estimates the 2027 budget at Rp174T against Rp268T in 2026. — [Bisnis](https://ekonomi.bisnis.com/read/20260706/10/1985932/banggar-dpr-estimasi-anggaran-mbg-2027-turun-35-jadi-rp174-triliun). BGN cites an indicative Rp270.2T, not final. — [Bisnis](https://ekonomi.bisnis.com/read/20260721/12/1989901/bgn-sebut-pagu-anggaran-2027-rp270-triliun-belum-final). Mid-Aug: government sets Rp240T provisionally for 70 million recipients. — [Katadata](https://katadata.co.id/berita/nasional/6a7f145ed65f1/pemerintah-siapkan-anggaran-mbg-2027-rp-240-triliun-bidik-70-juta-penerima)
- 31 Jul 2026 Semarang (707); 1 Aug Jayapura (527); 1 Sep Sidoarjo (730 in one compilation; 512 santri in BGN's hearing figure); 2 Sep Rembang (777); 21 Oct Cirebon (423); 26 Oct Kab. Bogor (481, while BGN says only 13 remained under observation). — [compiled incident CSV](https://raw.githubusercontent.com/andikapratamazaid/analisis-program-mbg/main/data/insiden_terbesar.csv)
- 3 Sep 2026 hearing: 5,482 victims in 71 incidents from 62 SPPG since 22 Jul; over 80% of causes are SOP violations, mainly kitchen capacity mismatched with recipients; 27,022 active kitchens. — [Katadata](https://katadata.co.id/berita/nasional/6a9a5b8dc9d81/5482-korban-keracunan-mbg-sejak-akhir-juli-2026-bgn-soroti-pelanggaran-sop)
- 29 Oct 2026, 100-day report: Rp44T efficiency claim, Rp229T real need against Rp268T allocation (the article's figures do not reconcile), SPPG points allegedly sold for Rp300-350 million, "yayasan calo" allegedly taking Rp500-1,500 per ompreng, "MBG 2.0". — [Kompas](https://nasional.kompas.com/read/2026/10/30/08004121/100-hari-sudaryono-pimpin-bgn-bongkar-yayasan-calo-hingga-janji-transformasi)
- Latest scale: 26,723 SPPG and 58 million recipients on 29 Oct 2026, after a correction of about 6 million duplicate records. — [compiled CSV](https://raw.githubusercontent.com/andikapratamazaid/analisis-program-mbg/main/data/perkembangan_program.csv)

**Contested numbers: never merge these into one figure**
- BGN 11,640 victims (Nov 2025); Kemenkes 50,059 in 560 events (2 Sep 2026); JPPI 43,838 (early Sep 2026); KPAI 12,658 children in 2025. — [repo README](https://github.com/andikapratamazaid/analisis-program-mbg)
- SPPG enforcement: 8,182 ever suspended (May 2026), 833 permanently suspended (Jul), 1,288 closed (Oct 2026), 353 kitchen heads removed, Rp311.246 billion returned. — [repo README](https://github.com/andikapratamazaid/analisis-program-mbg)
- Survey satisfaction fell: LSI 89.6% approval early on, Indikator 72.8% satisfied (Jan 2026); Oct 2026 Poltracking 47.4% satisfied vs 49.5% not, IPO 36% vs 64%, Tempo Data Science 48% vs 51%. The questions and samples differ, so this is not one trend. — [repo README](https://github.com/andikapratamazaid/analisis-program-mbg)

**Hoaxes and disputed claims (usable for a misinformation task)**
- BGN-Kemkomdigi cooperation on hoaxes and an early detection system. — [Antara](https://megapolitan.antaranews.com/berita/505966/bgn-gandeng-kemkomdigi-perangi-hoaks-program-mbg)
- Denied claims: SPPG fund disbursement stopped ([Jawa Pos](https://jawapos.com/nasional/2606040257/isu-penyaluran-dana-sppg-dihentikan-dipastikan-hoaks-bgn-tegaskan-layanan-tetap-berjalan-normal)); student death in Aceh ([Antara](https://www.antaranews.com/berita/5447710/hoaks-bgn-informasi-siswa-meninggal-karena-mbg-di-aceh-tidak-benar)); Rp5 million incentive for positive content ([Investortrust](https://investortrust.id/financial/83609/soal-intensif-rp-5-juta-untuk-konten-positif-mbg-bgn-beri-klarifikasi)); profit share to the President ([Warta Ekonomi](https://wartaekonomi.co.id/read617009/prabowo-dituduh-ikut-ambil-untung-mbg-bos-bgn-bilang-begini)); money moved to parents' accounts ([IUWASH page](https://iuwashtangguh.or.id/berita-edukasi/2115363682/viral-dana-mbg-akan-ditransfer-ke-rekening-orang-tua-hoaks-atau-fakta-ini-penjelasannya/)); kitchens stopping when funds run short ([JPNN](https://www.jpnn.com/news/mbg-disasar-hoaks-lagi-bgn-sampaikan-klarifikasi-soal-sppg-tetap-beroperasi)).
- These denials come from an interested party (BGN) in most cases. Note that the "money to parents" source is a low-tier page.

### Inferences
**Keyword lexicon per aspect** (tags: [D] seen in the public TikTok CSV by my count, [P] from papers or news above, [I] my own suggestion, unverified). Match on normalised lower-case text. Treat this as a starting point for weak labelling and validate on a sample.

- **keamanan_pangan:** keracunan[D], racun[D], basi[D], bau[D], mual, muntah, diare, pusing, korban[D], klb[P], puskesmas[P], sop[P], ulat[I], belatung[I], berjamur[I], e coli[P], salmonella[P], hiu[D, meaning unverified], "tidak layak"/"layak"[D]
- **kualitas_menu:** menu[D], menunya[D], burger[D], ayam[D], nasi[D], sayur[D], susu[D], kotak[D], roti[D], telur/telor[D], tempe[D], tahu[D], ikan[D], kacang[D], mie[D], gula[D], buah[D], pisang[D], jeruk[D], enak[D], rasa[D], ompreng[P], ultra processed/upf[I], porsi[I]
- **gizi_kesehatan:** gizi[D], gizinya[D], bergizi[D], ahli gizi[D], dokter[D], sehat[D], stunting[I], protein[I], kalori[I], akg[I], obesitas[I]
- **anggaran:** anggaran[D], dana[D], uang[D], buang[D], apbn[P], rapbn[P], pajak[P], triliun[P], pagu[P], efisiensi[P], pemborosan[P], pendidikan[D], refocusing[P], "mending"[D, signals budget alternatives]
- **korupsi_tata_kelola:** korupsi[D], koruptor[P], ladang[D], lahan[D], ajang[D], berjamaah[D], proyek[D], pejabat[D], nepotisme[P], kolusi[P], markup/mark up[P], tersangka[P], kejagung[P], jual beli titik[P], titik sppg[P], yayasan[P], calo[P], untung[D], transparan[P]
- **implementasi_dapur:** sppg[D], dapur[D], kepala sppg[P], mitra[P], distribusi[P], terlambat[I], kapasitas[P], suspend[P], ditutup/tutup[D], investigasi[P], tepat waktu[I], sanitasi[P], slhs[I], relawan[I]
- **pendidikan_guru:** guru[D], gaji[D], honorer[D], murid[D], siswa[D], santri[P], sekolah[D], kantin[D], bawa pulang[D-adjacent], pip[I]
- **ekonomi_umkm:** umkm[D], lapangan kerja/lapangan[D], kerja[D], pekerja[D], pedagang[D], jualan[D], kantin[D], catering[D], harga[D], naik[D], petani[I], peternak[I], susu lokal[I]
- **politik:** prabowo[D], wowo[D], gibran[P], janji kampanye[P], dpr[D], tni[D], polri[D], polisi[D], tentara[P], militer[P], purnawirawan[D], purbaya[P], zulhas[P], sudaryono[P], nanik[P], dadan[P], ombudsman[P], jppi[P], kpai[P], icw[I], cisdi[I]
- **penolakan_dukungan (stance cues):** stop[D], tutup[D], bubar[D], bubarkan[P], setop, hentikan, evaluasi[D], ganti[D], mending[D] for opposition; bagus[D], setuju[D], lanjut[I], terima kasih/makasih[I], alhamdulillah[I], semoga[P], program bagus[D-adjacent] for support
- **media/hoaks/buzzer:** hoax[D], hoaks[P], buzzer[D], influencer[D], narasi[D], digoreng[D], komdigi[P], klarifikasi[P], fitnah[I], bot[I], pendukung[D], fanatik[D]
- **Spelling normalisation seen in the data [D]:** yg, gk, gw, bgt, tp, klo, udh, dpt, jg, jd, bnyk, bnyak, karna, krn, utk, tdk, jgn, org, dr, anak2, daerah2, dkorup, pejbat, jdi, lgi, bnr, byar.
- **Javanese and Sundanese starter list:** ndk, madak, joss are [D]; ora, ndak, mboten, wes, sing, teu, henteu, atuh, euy are [I].

**Features suggested by the facts** [I]: period flags (pre-Sep 2025; Sep-Oct 2025 crisis; Jun-Jul 2026 scandal; Aug-Oct 2026 second wave), head-of-BGN era, presence of a place name from incident lists, days since the nearest large incident, and a "contains number + victims" pattern. Check for leakage if the test set is time-split.

**Jury-ready insights** [I]: sentiment can be separated into "rejects the programme" versus "wants it fixed"; poisoning figures must be presented per agency; survey decline and online negativity are not the same measure; online data over-represents viral incident days (one video held 71% of unique comments in the public TikTok set).

### Gaps
- The 2025 budget of Rp71T and the Rp335T 2026 figure from my background knowledge were not verified in this research. Only the Rp268T pagu and the Rp229T BGN figure were sourced. Reconcile before putting it on a slide.
- The specific events behind the "burger" (1,039 unique comments in Sep 2025), "ahli gizi" and "hiu" spikes in the TikTok data were not traced. I found no source tying them to a dated event, so check them on data before narrating them.
- ICW, CISDI, ultra-processed food, milk, the pork-oil and ompreng controversies returned no verifying results in my one targeted search.
- I found no Drone Emprit or Komdigi report specific to MBG buzzers.

## Why 500k rows? Likely sources, schema and quality problems

### Takeaway
500k is plausible as a merged multi-platform crawl over 2025-2026. Expect TikTok, X, YouTube and news comments with heavy duplication, burstiness and short noisy text. **Human labelling of 500k rows is implausible, so either labels are sparse or they are machine-generated.** This is the most important structural guess.

### Cited Findings
- **Volume benchmarks:** INDEF crawled 444,900 conversations in one month (1-27 Sep 2025) from X and TikTok. — [compiled CSV](https://raw.githubusercontent.com/andikapratamazaid/analisis-program-mbg/main/data/sentimen_medsos_studi.csv). One paper reports 126,365 tweets. — [METIK, snippet](https://journal.universitasmulia.ac.id/index.php/metik/article/download/1403/617/5812)
- **TikTok schema:** video_id, comment_id, parent_comment_id, level, username, nickname, comment, create_time. Raw noise is listed as slang, abbreviations, emoji, mentions, special characters, inconsistent capitals, typos. — [Kaggle](https://www.kaggle.com/datasets/sinryurifal/dataset-komentar-tiktok-mbg-makan-bergizi-gratis)
- **YouTube schema:** author, published_at, like_count, text. — [Kaggle](https://www.kaggle.com/datasets/muhammadzaiddan/data-sentimen-makanan-bergizi-gratis-di-indonesia)
- **Code-mixing and informal language are documented.** — [Kaggle YouTube set](https://www.kaggle.com/datasets/dzikriraihan/indonesian-youtube-comments-mbg-program); noisy informal text is also noted by [IJID](https://ejournal.uin-suka.ac.id/saintek/ijid/article/view/5211)
- **Coordination evidence is weak so far.** A 12,000-comment YouTube study found 11,205 unique commenters, 12 with at least two signal families, 0 strict multi-signal candidates, and 4 repeated-text clusters, and explicitly says this is not a bot count. — [GitHub](https://github.com/briannaufal211/mbg-youtube-sentiment-analysis)
- **Own analysis:** duplicates by ID at 40.9% and a single-video concentration of 71%, as listed in the datasets section.

### Inferences
- Likely source mix, as a judgement: X/Twitter and TikTok most likely; YouTube likely; Instagram and news comments possible; synthetic augmentation or LLM-labelled data possible because 500k is a round number.
- Expected traps: exact and near duplicates across train and test (leakage or leaderboard inflation), retweets and copy-paste templates, emoji-only rows, off-topic "MBG" meanings (for example the abbreviation in other contexts, or gambling spam), Javanese and Sundanese mixing, sarcasm that flips sentiment, and label noise.
- Group-aware validation by `video_id`, author, or near-duplicate cluster is the safest default. A time-based holdout is worth adding as a second check. This addresses the kind of internal-versus-official gap reported at GEMASTIK XIV.
- With 5 hours and unknown hardware, full transformer fine-tuning on 500k rows is unlikely to be feasible. Plan TF-IDF first, then a subsample or a frozen-embedding approach.

### Gaps
- The actual schema, sources and label mechanism remain unknown until the case is released.
- I found no public evidence of the buzzer share in MBG discourse.

## Likely task types, metrics, and what to prepare

### Takeaway
My top guesses are 3-class sentiment or stance scored with macro-F1, then aspect classification. Prepare one fast, generic text pipeline that can switch among single-label, multi-label and semi-supervised modes, and keep a fallback for a surprise task. All probabilities below are **labelled judgement**, not data.

### Cited Findings
- Leaderboard is 30% of the GEMASTIK final; method and presentation are 50%. — [Panduan 2026](https://kemahasiswaan.unand.ac.id/images/documents/Panduan-GEMASTIK-2026.pdf)
- GEMASTIK has moved from text classification to QA between rounds. — [ITB](https://itb.ac.id/news/persistence-become-the-success-key-for-team-nasigoreng-to-won-bronze-medal-at-gemastik-xiv/58223)
- Text is not the only data type named: structured data, images, video and text are all in scope, so the "NLP" premise rests on the user's information. — [Panduan 2026](https://kemahasiswaan.unand.ac.id/images/documents/Panduan-GEMASTIK-2026.pdf)
- Macro-F1 is already the reporting norm in stance work on MBG. — [UPN Jatim](https://repository.upnjatim.ac.id/52506/); [JSAI](https://jurnal.umb.ac.id/index.php/JSAI/article/download/10555/6282/46478)
- The standard ABSA framing is aspect-category detection followed by polarity per aspect. — [SemEval-2014 Task 4](https://alt.qcri.org/semeval2014/task4/)
- **Other Indonesian finals:** Satria Data's Big Data Challenge is the nearest analogue, with online preliminaries and an offline final in 2026. — [RRI](https://rri.co.id/tarakan/iptek/2385935/siapkan-talenta-data-masa-depan-kemdikti-saintek-luncurkan-satria-data-2026). ACTION 2026 uses a final case plus presentation. — [ACTION](https://action-unesa.com/id)

### Inferences
**Primary scored task (judgement; sums to 100%)**

| Rank | Task | P | Likely metric | Prepare |
|---|---|---|---|---|
| 1 | Sentiment or stance, 3-class | 35% | Macro-F1 (accuracy possible) | TF-IDF word + char with LinearSVC or LogReg; IndoBERT/IndoBERTweet weights offline; class weights; threshold tuning; pretrained sentiment models as features |
| 2 | Aspect or topic classification, multi-class or multi-label | 20% | Macro-F1 or micro-F1, subset accuracy, Hamming loss | One-vs-rest; per-label thresholds; the lexicon above as features and weak labels; the JSAI taxonomy for naming |
| 3 | ABSA or joint aspect + polarity | 10% | Macro-F1 over pairs or per-aspect average | Per-aspect models; or input "[aspect] [SEP] text"; sentence splitting |
| 4 | Binary detection: hoax, relevance, spam, buzzer/bot | 10% | F1 of positive class, PR-AUC, ROC-AUC | Hoax claim list above; author and burst features (posting rate, near-duplicate clusters, account age if given); be careful not to call people bots without proof |
| 5 | Surprise extraction or generation: QA, NER, summarisation, keyphrase | 10% | Exact match/F1, entity F1, ROUGE | Save an Indonesian QA or NER model if allowed; gazetteer of institutions and places; extractive fallback (TextRank, TF-IDF sentences) |
| 6 | Emotion or hate/toxicity | 5% | Macro-F1 | Same pipeline as 1; pretrained Indonesian emotion or hate models if allowed; profanity list |
| 7 | Engagement regression (likes, replies) | 5% | RMSE or RMSLE, MAE | log1p target; text + time + source features; LightGBM on TF-IDF SVD |
| 8 | Unsupervised only, topic or clustering with report | 5% | Silhouette or coherence if scored; else jury | BERTopic or NMF on a subsample; prior results to compare: C_v of about 0.46-0.64 |

**Modifiers that can apply on top of any task (judgement, not exclusive)**
- Sparse labels, with under 10% of rows labelled: 35%. Prepare pseudo-labelling with a high confidence cut-off, label spreading on embeddings, and weak labels from the lexicon.
- Machine-generated or noisy labels: 40%. Prepare confident-learning-style filtering and compare against public sentiment models.
- Time-based or source-based test split: 30%. Prepare adversarial validation and avoid over-relying on time features.
- A required topic or spike-analysis section in the deck, regardless of scored task: 70%. Prepare a template chart set: monthly volume and sentiment with event annotations, topic-by-sentiment heatmap, top words per stance, and a short policy recommendation.

**Plan for 5 hours (suggestion):** hour 0-1: profile, dedupe, baseline, first submission. Hour 1-3: improve with the best validated approach. Hour 3-4: ensemble and thresholds, freeze. Hour 4-5: charts, insights and deck skeleton; do not start new models. Deck: problem and domain, data issues, validation design, model comparison, error analysis, MBG insights, recommendations.

**Things to do at set-up on 10 Nov:** ask about internet, external data, pretrained weights, LLM use, GPU, submission limits, deadline for the PPT and whether it may be edited overnight before 12 Nov. Install and test the whole stack on the committee device, including loading each model with `local_files_only=True` or the equivalent. Bring wheels and weights on removable media if allowed.

### Gaps
- I could not verify judging specifics for Find IT, Hology, COMPFEST, ICAIC or general Datathon events. Searches returned nothing usable, so no claims are made about them.
- I found no public record of the metric used in any past GEMASTIK final. The leaderboard metric here is inference only.
- No GEMASTIK source confirms the choice of NLP or MBG for 2026; both rest on the user's information.
