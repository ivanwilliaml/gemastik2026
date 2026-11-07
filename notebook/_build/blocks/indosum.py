# %% [markdown]
# **Pembaca IndoSum.** Tiap artikel berupa paragraf, kalimat, dan token. `load_indosum` meratakannya menjadi
# daftar kalimat, ringkasan, dan penanda kalimat mana yang masuk ringkasan.

# %%
def load_indosum(file_name: str, n_rows: int) -> pd.DataFrame:
    frame = pd.read_json(find_file(file_name), lines=True).head(n_rows)

    def flatten(paragraphs) -> list[str]:
        return [" ".join(tokens) for paragraph in paragraphs for tokens in paragraph]

    frame["sentences"] = frame["paragraphs"].map(flatten)
    frame["article"] = frame["sentences"].map(" ".join)
    frame["summary_text"] = frame["summary"].map(lambda s: " ".join(" ".join(t) for t in s))
    frame["first_summary"] = frame["summary"].map(lambda s: " ".join(s[0]) if len(s) else "")
    frame["labels"] = frame["gold_labels"].map(lambda g: [flag for paragraph in g for flag in paragraph])
    return frame[["category", "sentences", "article", "summary_text", "first_summary", "labels"]]
