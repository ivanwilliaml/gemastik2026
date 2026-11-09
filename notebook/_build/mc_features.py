"""Sel feature engineering bersama (NB1, NB4, NB5): ±75 fitur (paket awal 37) handcrafted dalam enam kelompok + TF-IDF-SVD.

Dasar: LFTK (Lee & Lee 2023), literatur klasifikasi medsos, dan catatan riset proyek. Fitur gaya penulisan dihitung dari
teks ASLI (huruf besar, tanda baca, huruf berulang tidak boleh hilang); fitur leksikon dihitung dari teks NORMALISASI.
"""

TIER1 = ["char_count", "word_count", "avg_word_length", "unique_word_ratio", "uppercase_ratio", "all_caps_word_ratio", "elongated_word_count",
         "max_char_repetition", "exclamation_count", "question_count", "punctuation_ratio", "emoji_count", "unique_emoji_count", "laughter_count",
         "negation_count", "intensifier_count", "slang_count", "has_url", "has_mention", "has_hashtag"]


def features(book, num, frag_model=None):
    book.md(f"""## {num}. Feature engineering: fitur handcrafted

Tujuan bukan mengekstrak sebanyak mungkin fitur, tetapi fitur yang **menyimpan informasi yang belum dipelajari representasi teks utama**. Enam kelompok:

| Kelompok | Contoh | Dihitung dari |
|---|---|---|
| `structural` | panjang karakter dan kata, rasio kata unik, jumlah kalimat, URL, mention, hashtag, rasio digit | teks asli |
| `stylistic` | huruf kapital, huruf berulang (`baguuus`), tanda baca, panjang deret tanda baca | teks asli |
| `expressive` | emoji, emoticon, tawa (`wkwk`) | teks asli |
| `lexical` | slang, singkatan, stopword, kata langka, OOV, campur bahasa Inggris, kata kasar | normalisasi |
| `lexicon` | negasi dan cakupannya, kata positif/negatif, emosi, intensifier | normalisasi |
| `discourse` | kata ganti orang, penanda kontras, sebab, kondisi, modal | normalisasi |

Sengaja **tidak** dipakai: fitur POS (butuh tagger bahasa Indonesia yang akurat di teks informal), serta `Confidence Score` dan `Commenter ID` (bocor label / bisa dihafal). Statistik korpus (frekuensi kata, OOV) dihitung dari **train saja**, dan untuk baris train dokumen itu sendiri tidak dihitung (leave-one-out), supaya distribusinya sama dengan validation/holdout.""")
    book.config("Kamus leksikon kecil dan bisa diperluas; kosakata seed ini disusun untuk komentar MBG. Perluas dari kata khas per kelas di EDA 5.3. Semua kata dalam bentuk setelah normalisasi (huruf kecil, slang dibakukan).", r'''
        import string
        LEX = dict(
            positif  = "bagus baik mantap keren hebat suka senang sukses puas enak lezat sehat bergizi bermanfaat membantu terbaik top mantul sip sempurna cocok adil lancar bersih peduli makasih terimakasih syukur alhamdulillah dukung setuju",
            negatif  = "buruk jelek parah gagal korupsi bohong tipu rugi mahal basi keracunan racun sakit kecewa sampah zonk bahaya salah bau mentah cacing ulat tolak batal boros pemborosan mubazir merugikan menyusahkan miris prihatin lemah gagal bubarkan hapus",
            intens   = "sangat banget parah super amat sekali terlalu paling bener banyak",
            dimin    = "agak sedikit kurang lumayan cukup",
            kasar    = "anjing anjir anjay bangsat goblok tolol bodoh bego kampret brengsek sialan babi tai dungu idiot",
            marah    = "marah kesal benci muak geram emosi murka sebel jengkel",
            senang   = "senang bahagia gembira bangga lega",
            sedih    = "sedih menangis nangis pilu miris nelangsa",
            takut    = "takut khawatir cemas ngeri",
            jijik    = "jijik najis mual",
            kaget    = "kaget terkejut heran astaga",
            p1       = "aku saya gue gua kami kita",
            p2       = "kamu lu lo anda kalian engkau",
            p3       = "dia mereka beliau",
            kontras  = "tapi namun tetapi sedangkan padahal",
            sebab    = "karena sebab gara",
            simpul   = "jadi kesimpulannya intinya",
            kondisi  = "jika kalau apabila bila",
            tekanan  = "bahkan justru malah",
            modal    = "mungkin harus bisa seharusnya semoga",
            singkat  = "yg dgn utk krn tdk blm sdh udh bgt tp gk jgn dr sm pd lg org tmn bs skrg gmn klo kalo jg dll dsb spt kpd bgmn",
            inggris  = "the is are and you this that very good bad not no yes thanks thank please love hate fake real people government free food program"
        )
        LEXS = {k: set(v.split()) for k, v in LEX.items()}
        RARE_MAX = 2                                   # kata dengan frekuensi train <= RARE_MAX dianggap langka
        SVD_DIM = 64                                   # dimensi TF-IDF-SVD (sinyal leksikal terkompresi)
        SVD_TFIDF = dict(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)
    ''')
    book.md("**Kelompok `structural`, `stylistic`, `expressive`**: dihitung dari teks asli. Setiap fitur satu baris; jumlah dan intensitas dipisah (bukan hanya flag 0/1).")
    book.code(r'''
        raw = data[TEXT_COL].astype(str); PUN = "[" + re.escape(string.punctuation) + "]+"
        rtok = raw.str.casefold().str.findall(r"[^\W\d_]+"); wc = raw.str.split().str.len().clip(lower=1); cc = raw.str.len().clip(lower=1)
        letters = raw.str.count(r"[A-Za-z]").clip(lower=1)
        F = pd.DataFrame({
            # structural
            "char_count": raw.str.len(), "char_count_no_space": raw.str.replace(r"\s", "", regex=True).str.len(), "word_count": wc,
            "unique_word_count": rtok.map(lambda q: len(set(q))), "avg_word_length": rtok.map(lambda q: np.mean([len(w) for w in q]) if q else 0),
            "max_word_length": rtok.map(lambda q: max(map(len, q), default=0)), "sentence_count": raw.str.count(r"[.!?]+(?:\s|$)").clip(lower=1),
            "has_url": raw.str.contains(URL).astype(int), "has_mention": raw.str.contains(r"@\w+").astype(int), "has_hashtag": raw.str.contains(r"#\w+").astype(int),
            "digit_ratio": raw.str.count(r"\d") / cc,
            # stylistic
            "uppercase_count": raw.str.count(r"[A-Z]"), "uppercase_ratio": raw.str.count(r"[A-Z]") / letters, "all_caps_word_count": raw.str.count(r"\b[A-Z]{2,}\b"),
            "is_all_caps": raw.map(lambda t: int(t.isupper())), "max_char_repetition": raw.map(lambda t: max((len(m[0]) for m in re.finditer(r"(.)\1*", t)), default=0)),
            "repeated_char_count": raw.str.count(r"(.)\1{2,}"), "consecutive_repetition_count": raw.str.count(r"(.)\1+"),
            "elongated_word_count": raw.str.count(r"\b\w*([^\W\d_])\1{2,}\w*\b"),
            "exclamation_count": raw.str.count("!"), "question_count": raw.str.count(r"\?"), "ellipsis_count": raw.str.count(r"\.{2,}"),
            "comma_count": raw.str.count(","), "period_count": raw.str.count(r"\."), "punctuation_count": raw.map(lambda t: sum(c in string.punctuation for c in t)),
            "max_punctuation_sequence": raw.map(lambda t: max((len(m[0]) for m in re.finditer(PUN, t)), default=0)),
            # expressive
            "emoji_count": raw.map(emoji.emoji_count), "unique_emoji_count": raw.map(lambda t: len({e["emoji"] for e in emoji.emoji_list(t)})),
            "emoticon_count": raw.str.count(r"(?<!\w)(?:[:;=8][\-^']?[)(/DPp]+|<3)(?!\w)"),
            "laughter_count": raw.str.count(r"\b(?:(?:wk)+w*k*|(?:ha){2,}h?|(?:he){2,}h?|(?:hi){2,}h?)\b", flags=re.I)}, index=data.index)
        F["all_caps_word_ratio"] = F.all_caps_word_count / wc; F["elongated_word_ratio"] = F.elongated_word_count / wc
        F["punctuation_ratio"] = F.punctuation_count / cc; F["avg_sentence_length"] = wc / F.sentence_count; F["avg_chars_per_sentence"] = cc / F.sentence_count
        F["unique_word_ratio"] = F.unique_word_count / wc
        print("fitur dari teks asli:", F.shape[1])
    ''')
    book.md("**Kelompok `lexical`, `lexicon`, `discourse`**: dihitung dari teks normalisasi, supaya `gak`, `ga`, `nggak` terbaca sebagai satu negasi. Frekuensi kata dan OOV dipelajari dari train saja. `negation_before_*` menghitung kata evaluatif yang berada dalam tiga token setelah negasi, jadi `tidak bagus` berbeda dari `bagus`.")
    book.code(r'''
        ntok = data.normalized.str.findall(r"[^\W\d_]+"); freq = Counter(w for q in ntok[train.index] for w in q); is_tr = data.index.isin(train.index)
        # frekuensi korpus TANPA menghitung dokumen itu sendiri untuk baris train; kalau tidak, OOV train selalu 0 dan nilainya bergeser di validation/holdout
        adj = pd.Series([[freq[w] - (c[w] if tr else 0) for w in q] for q, tr, c in zip(ntok, is_tr, ntok.map(Counter))], index=data.index)
        cnt = lambda toks, S: toks.map(lambda q: sum(w in S for w in q)); mean_of = lambda toks, f: toks.map(lambda q: np.mean([f(w) for w in q]) if q else 0.0)
        near_neg = lambda q, S: sum(any(x in NEGATIONS for x in q[max(0, i - 3):i]) for i, w in enumerate(q) if w in S)
        L = LEXS
        G = pd.DataFrame({
            # lexical
            "stopword_ratio": mean_of(ntok, lambda w: w in STOPWORDS), "avg_token_frequency": adj.map(lambda a_: np.mean(np.log1p(a_)) if a_ else 0.0),
            "rare_word_ratio": adj.map(lambda a_: np.mean([x <= RARE_MAX for x in a_]) if a_ else 0.0), "oov_ratio": adj.map(lambda a_: np.mean([x <= 0 for x in a_]) if a_ else 0.0),
            "slang_count": cnt(rtok, set(SLANG)), "abbreviation_count": cnt(rtok, L["singkat"]), "profanity_count": cnt(ntok, L["kasar"]),
            "code_switch_ratio": mean_of(ntok, lambda w: w in L["inggris"]),
            # lexicon
            "negation_count": cnt(ntok, NEGATIONS), "negation_before_positive": ntok.map(lambda q: near_neg(q, L["positif"])),
            "negation_before_negative": ntok.map(lambda q: near_neg(q, L["negatif"])),
            "negation_scope_length": ntok.map(lambda q: next((min(len(q) - i - 1, 5) for i, w in enumerate(q) if w in NEGATIONS), 0)),
            "positive_word_count": cnt(ntok, L["positif"]), "negative_word_count": cnt(ntok, L["negatif"]),
            "intensifier_count": cnt(ntok, L["intens"]), "diminisher_count": cnt(ntok, L["dimin"]),
            "anger_word_count": cnt(ntok, L["marah"]), "joy_word_count": cnt(ntok, L["senang"]), "sadness_word_count": cnt(ntok, L["sedih"]),
            "fear_word_count": cnt(ntok, L["takut"]), "disgust_word_count": cnt(ntok, L["jijik"]), "surprise_word_count": cnt(ntok, L["kaget"]),
            # discourse
            "first_person_count": cnt(ntok, L["p1"]), "second_person_count": cnt(ntok, L["p2"]), "third_person_count": cnt(ntok, L["p3"]),
            "contrastive_count": cnt(ntok, L["kontras"]), "causal_count": cnt(ntok, L["sebab"]), "conclusion_count": cnt(ntok, L["simpul"]),
            "conditional_count": cnt(ntok, L["kondisi"]), "emphasis_marker_count": cnt(ntok, L["tekanan"]), "modal_count": cnt(ntok, L["modal"])}, index=data.index)
        nwords = ntok.str.len().clip(lower=1)
        G["has_negation"] = (G.negation_count > 0).astype(int); G["negation_ratio"] = G.negation_count / nwords; G["double_negation_flag"] = (G.negation_count >= 2).astype(int)
        G["positive_word_ratio"] = G.positive_word_count / nwords; G["negative_word_ratio"] = G.negative_word_count / nwords
        G["sentiment_balance"] = G.positive_word_count - G.negative_word_count; G["intensifier_ratio"] = G.intensifier_count / nwords
        G["personal_pronoun_ratio"] = (G.first_person_count + G.second_person_count + G.third_person_count) / nwords
        F = F.join(G).fillna(0); print("total fitur:", F.shape[1])
    ''')
    if frag_model:
        book.md("**Kelompok `fragmentasi` (khusus model BERT).** Seberapa banyak sebuah kata dipecah tokenizer. Kata informal (`menggibahhhh`) pecah jadi banyak subword; ini penanda variasi ejaan dan ketidakcocokan vocabulary yang tidak terlihat oleh BERT sendiri.")
        book.config("Tokenizer yang dipakai menghitung fragmentasi.", f'''
            FRAG_MODEL = "{frag_model}"
        ''')
        book.code(r'''
            from transformers import AutoTokenizer
            tkf = AutoTokenizer.from_pretrained(FRAG_MODEL); sub = data.safe.map(tkf.tokenize)
            F["subword_count"] = sub.str.len(); F["subword_per_word"] = F.subword_count / F.word_count
            F["avg_subword_per_token"] = sub.map(lambda q: np.mean([len(w) for w in q]) if q else 0.0)
            F["subword_fragmentation_ratio"] = sub.map(lambda q: np.mean([w.startswith("##") for w in q]) if q else 0.0)
            print("fitur setelah fragmentasi:", F.shape[1])
        ''')
    book.md("""**Kelompok dan paket fitur.** `TIER1` (20 fitur) = diimplementasikan dahulu; `TIER2` = ditambahkan setelah baseline. Fitur bertipe hitungan sangat miring ke kanan, jadi diubah `sign(x)·log1p(|x|)` lalu distandarkan; scaler **hanya dipelajari dari train**, dan nilai dipotong ke ±5. TF-IDF-SVD menyediakan sinyal leksikal terkompresi sebagai pembanding fitur handcrafted.""")
    book.code(r'''
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import StandardScaler
        GROUP = {"structural": ["char_count", "char_count_no_space", "word_count", "unique_word_count", "avg_word_length", "max_word_length", "sentence_count", "avg_sentence_length",
                                "avg_chars_per_sentence", "unique_word_ratio", "has_url", "has_mention", "has_hashtag", "digit_ratio"],
                 "stylistic": ["uppercase_count", "uppercase_ratio", "all_caps_word_count", "all_caps_word_ratio", "is_all_caps", "max_char_repetition", "repeated_char_count",
                               "consecutive_repetition_count", "elongated_word_count", "elongated_word_ratio", "exclamation_count", "question_count", "ellipsis_count",
                               "comma_count", "period_count", "punctuation_count", "punctuation_ratio", "max_punctuation_sequence"],
                 "expressive": ["emoji_count", "unique_emoji_count", "emoticon_count", "laughter_count"],
                 "lexical": ["stopword_ratio", "avg_token_frequency", "rare_word_ratio", "oov_ratio", "slang_count", "abbreviation_count", "profanity_count", "code_switch_ratio"],
                 "lexicon": ["negation_count", "has_negation", "negation_ratio", "double_negation_flag", "negation_before_positive", "negation_before_negative", "negation_scope_length",
                             "positive_word_count", "negative_word_count", "positive_word_ratio", "negative_word_ratio", "sentiment_balance", "intensifier_count", "diminisher_count",
                             "intensifier_ratio", "anger_word_count", "joy_word_count", "sadness_word_count", "fear_word_count", "disgust_word_count", "surprise_word_count"],
                 "discourse": ["first_person_count", "second_person_count", "third_person_count", "personal_pronoun_ratio", "contrastive_count", "causal_count", "conclusion_count",
                               "conditional_count", "emphasis_marker_count", "modal_count"]}
        if "subword_count" in F: GROUP["fragmentasi"] = ["subword_count", "subword_per_word", "avg_subword_per_token", "subword_fragmentation_ratio"]
        TIER1 = ''' + repr(TIER1) + r'''
        TIER2 = ["positive_word_count", "negative_word_count", "anger_word_count", "joy_word_count", "sadness_word_count", "fear_word_count", "disgust_word_count", "surprise_word_count",
                 "profanity_count", "abbreviation_count", "code_switch_ratio", "stopword_ratio", "first_person_count", "second_person_count", "contrastive_count", "digit_ratio", "oov_ratio"]
        assert set(sum(GROUP.values(), [])) <= set(F.columns) and set(TIER1 + TIER2) <= set(F.columns), "Nama fitur tidak cocok"
        T = np.sign(F) * np.log1p(np.abs(F)); sc_f = StandardScaler().fit(T.loc[train.index])
        Zf = pd.DataFrame(np.clip(sc_f.transform(T), -5, 5), index=F.index, columns=F.columns)
        tf_svd = TfidfVectorizer(**SVD_TFIDF).fit(train.normalized); svd = TruncatedSVD(SVD_DIM, random_state=SEED).fit(tf_svd.transform(train.normalized))
        S_ = svd.transform(tf_svd.transform(data.normalized)); sc_s = StandardScaler().fit(S_[data.index.get_indexer(train.index)])
        Zsvd = pd.DataFrame(np.clip(sc_s.transform(S_), -5, 5), index=data.index, columns=[f"svd_{i}" for i in range(SVD_DIM)])
        print({g: len(c) for g, c in GROUP.items()}, "| SVD:", Zsvd.shape[1], "| varians SVD terjelaskan:", round(svd.explained_variance_ratio_.sum(), 3))
    ''')
    book.md("**Diagnosis fitur (train).** Rata-rata fitur per kelas menunjukkan fitur mana yang membedakan kelas; mutual information mengukur sinyal terhadap label; pasangan berkorelasi tinggi adalah redundansi yang bisa dibuang. Fitur konstan tidak berguna.")
    book.code(r'''
        from sklearn.feature_selection import mutual_info_classif
        Ftr = F.loc[train.index]
        show("Rata-rata fitur Tier 1 per kelas · train", Ftr[TIER1].groupby(train[LABEL_COL]).mean().T)
        mi = pd.Series(mutual_info_classif(Zf.loc[train.index], y_train, random_state=SEED), index=F.columns).sort_values(ascending=False)
        show("Mutual information dengan label · 20 teratas", mi.head(TOP + 5).rename("MI").to_frame().assign(kelompok=[next(g for g, c in GROUP.items() if f in c) for f in mi.head(TOP + 5).index]))
        show("MI rata-rata per kelompok", pd.Series({g: mi[c].mean() for g, c in GROUP.items()}).rename("MI_rata2").to_frame())
        cor = Ftr.corr(numeric_only=True).abs().where(np.triu(np.ones((F.shape[1],) * 2), 1).astype(bool)).stack().sort_values(ascending=False)
        show("Pasangan fitur paling berkorelasi (|r|) · kandidat dibuang", cor.head(TOP).rename("|r|").to_frame())
        print("fitur konstan di train:", Ftr.columns[Ftr.nunique() <= 1].tolist(), "| pasangan |r| > 0,9:", int((cor > .9).sum()))
    ''')
    return book
