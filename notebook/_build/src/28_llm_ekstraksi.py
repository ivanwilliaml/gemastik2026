# %% [markdown]
# # 28 LLM Lokal untuk Ekstraksi Terstruktur
#
# Memakai LLM kecil yang jalan offline untuk mengubah teks menjadi data terstruktur. Contoh: sentimen SmSA untuk audit,
# lalu ekstraksi aspek dari komentar MBG.

# %% include toc

# %% [markdown]
# ## 1. Kartu Masalah
#
# | Bagian | Isi |
# |---|---|
# | **Task** | Ekstraksi terstruktur dengan LLM kecil lokal |
# | **Unit prediksi** | Satu komentar mendapat JSON berisi sentimen dan aspek |
# | **Metrik** | Macro-F1 sentimen pada label manusia, dan tingkat lolos skema |
# | **Masalah terbesar** | LLM kecil tidak selalu patuh: hanya 90% jawaban lolos skema pada percobaan pertama, dan aspeknya bias ke `anggaran` (59,7%). |
# | **Data contoh** | SmSA untuk audit, 1.000 komentar MBG untuk demo |

# %% [markdown]
# ## 2. Kapan Pakai Notebook Ini
#
# **Pakai kalau** kategorinya banyak atau sering berubah, dan datanya tanpa label, serta aturan membolehkan membawa model
# sebesar 3 GB. Untuk sentimen biasa, folder `04` dan `10` biasanya lebih cepat dan lebih stabil.
#
# **Yang perlu diganti kalau datanya beda:**
# - `SYSTEM_PROMPT`, `EXAMPLES`, dan `ALLOWED_VALUES`.
# - `parse_output`, kalau bentuk JSON-nya berubah.

# %% [markdown]
# ## 3. Config

# %% include setup

# %% [markdown]
# **Pengaturan utama.** Semua yang perlu diubah ada di sel ini.

# %%
MODE = "cepat"  # "cepat" (<30 menit) atau "penuh" (<2 jam)
USE_PRETRAINED = True  # False: hanya kata kunci, karena LLM butuh model dari luar
SEED = 42
ALLOWED_VALUES = {"sentimen": ["negatif", "netral", "positif"],
                  "aspek": ["keracunan", "korupsi", "anggaran", "gizi", "dapur", "lainnya"]}
SENTIMENT_TO_CLASS = {"negatif": "negative", "netral": "neutral", "positif": "positive"}
CLASSES = ["negative", "neutral", "positive"]
MAX_NEW_TOKENS = 48

SETTINGS = {"cepat": {"n_audit": 200, "n_mbg": 100}, "penuh": {"n_audit": 1000, "n_mbg": 1000}}[MODE]

seed_everything(SEED)
print(SETTINGS)

# %% include text_utils

# %% include plot_confusion

# %% [markdown]
# ## 4. EDA
#
# Muat data audit berlabel manusia, dan komentar MBG tanpa label.

# %%
import json as json_lib  # noqa: E402

from sklearn.metrics import f1_score  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

path = EXTRA_DATA_DIR / "indonlu/dataset/smsa_doc-sentiment-prosa/test_preprocess.tsv"
audit = pd.read_csv(path, sep="\t", header=None, names=["text", "label"]).sample(frac=1, random_state=SEED)
audit = audit.head(SETTINGS["n_audit"]).reset_index(drop=True)
mbg = pd.read_csv(find_file("comment_data.csv"))["Comment"].dropna().astype(str).map(clean_text)
mbg = mbg[mbg.str.split().str.len().between(5, 80)].sample(n=SETTINGS["n_mbg"], random_state=SEED).reset_index(drop=True)
print(f"audit={len(audit)} | komentar MBG={len(mbg)}")
print("proporsi kelas audit:", audit["label"].value_counts(normalize=True).round(3).to_dict())
pd.DataFrame({"audit": text_profile(audit["text"]), "mbg": text_profile(mbg)})

# %% [markdown]
# **Temuan dan keputusan**
#
# | Temuan | Keputusan |
# |---|---|
# | Data audit netral 17,6% | Prompt berisi aturan dan beberapa contoh |
# | LLM kadang menulis di luar JSON | Validasi skema dan ulangi jawaban yang gagal |
# | Komentar panjang | Teks dipotong supaya prompt pendek |

# %% [markdown]
# ## 5. Preprocessing
#
# Teks dipotong supaya prompt pendek. Prompt berisi aturan dan beberapa contoh, karena LLM kecil jauh lebih patuh kalau
# diberi contoh. Jawaban wajib berupa JSON, supaya bisa diperiksa otomatis.

# %%
SYSTEM_PROMPT = (
    "Anda mengekstrak informasi dari komentar berbahasa Indonesia. Jawab HANYA dengan satu objek JSON "
    'berisi kunci "sentimen" dan "aspek". Nilai sentimen: negatif, netral, positif. '
    "Nilai aspek: keracunan, korupsi, anggaran, gizi, dapur, lainnya."
)
EXAMPLES = [
    ("anak saya sakit perut setelah makan menu sekolah", {"sentimen": "negatif", "aspek": "keracunan"}),
    ("programnya bagus, gizi anak jadi lebih terjaga", {"sentimen": "positif", "aspek": "gizi"}),
    ("pembagian makan dimulai pukul sepuluh di aula", {"sentimen": "netral", "aspek": "lainnya"}),
]


def build_messages(text: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example, answer in EXAMPLES:
        messages += [{"role": "user", "content": example}, {"role": "assistant", "content": json_lib.dumps(answer)}]
    return messages + [{"role": "user", "content": " ".join(text.split()[:100])}]


print(build_messages("contoh")[-1])

# %% [markdown]
# ## 6. Validasi
#
# Ada dua ukuran: **tingkat lolos skema** dan **macro-F1 sentimen** pada label manusia. Metrik dan pembaca JSON diuji dulu.
# LLM hanya untuk ekstraksi. Keputusan akhir tetap melewati pemeriksaan skema yang deterministik.

# %%
JSON_RE = re.compile(r"\{.*?\}", re.S)


def parse_output(raw: str) -> dict | None:
    """Ambil objek JSON pertama. Kembalikan None kalau tidak sesuai skema."""
    match = JSON_RE.search(raw)
    if not match:
        return None
    try:
        record_json = json_lib.loads(match.group())
    except json_lib.JSONDecodeError:
        return None
    if not all(record_json.get(key) in allowed for key, allowed in ALLOWED_VALUES.items()):
        return None
    return {key: record_json[key] for key in ALLOWED_VALUES}


assert parse_output('x {"sentimen":"positif","aspek":"gizi"} y') == {"sentimen": "positif", "aspek": "gizi"}
assert parse_output('{"sentimen":"bagus","aspek":"gizi"}') is None
assert parse_output("bukan json") is None


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


results: list[dict] = []


def record(name: str, predictions: list[str], valid_share: float) -> None:
    results.append({"metode": name, "macro_f1_audit": round(macro_f1(audit["label"], predictions), 4),
                    "lolos_skema": round(valid_share, 3)})
    print(results[-1])


# %% [markdown]
# ## 7. Baseline
#
# Kata kunci sebagai pembanding, dan sebagai cadangan kalau jawaban LLM tidak lolos skema.

# %%
NEGATIVE_WORDS = "buruk jelek kecewa parah kotor mahal lambat kasar rusak marah gagal tidak enak payah korupsi".split()
POSITIVE_WORDS = "bagus enak mantap puas suka nyaman ramah keren senang hebat lezat bersih baik sukses".split()
NEG_RE = re.compile(r"\b(" + "|".join(NEGATIVE_WORDS) + r")\b")
POS_RE = re.compile(r"\b(" + "|".join(POSITIVE_WORDS) + r")\b")


def keyword_sentiment(text: str) -> str:
    negative, positive = len(NEG_RE.findall(text.lower())), len(POS_RE.findall(text.lower()))
    return "negatif" if negative > positive else ("positif" if positive > negative else "netral")


keyword_pred = audit["text"].map(lambda t: SENTIMENT_TO_CLASS[keyword_sentiment(t)]).tolist()
record("kata_kunci", keyword_pred, 1.0)

# %% [markdown]
# ## 8. Model Utama
#
# Qwen2.5 1.5B berjalan lokal dengan batch. `do_sample=False` membuat hasilnya bisa diulang.

# %%
def llm_generate(tokenizer, model, texts: list[str], batch_size: int = 16) -> list[str]:
    prompts = [tokenizer.apply_chat_template(build_messages(t), tokenize=False, add_generation_prompt=True) for t in texts]
    outputs = []
    for start in range(0, len(prompts), batch_size):
        tokens = tokenizer(prompts[start : start + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.inference_mode():
            generated = model.generate(**tokens, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        outputs += tokenizer.batch_decode(generated[:, tokens["input_ids"].shape[1] :], skip_special_tokens=True)
    return outputs


raw_audit, raw_mbg = None, None
if USE_PRETRAINED:
    llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "llm", padding_side="left")
    llm_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR / "llm", torch_dtype=torch.float16).to(DEVICE).eval()
    with stage("llm_audit"):
        raw_audit = llm_generate(llm_tokenizer, llm_model, audit["text"].tolist())
    print("contoh jawaban:", raw_audit[0][:120])

# %% [markdown]
# ## 9. Post-processing
#
# Jawaban yang tidak lolos skema dicoba sekali lagi dengan sisa kalimat yang lebih pendek. Kalau masih gagal, diganti kata
# kunci. Baris cadangan ditandai, supaya bisa dihitung berapa banyak yang benar-benar dari LLM.

# %%
def resolve(raws: list[str] | None, texts: list[str]) -> tuple[list[dict], np.ndarray]:
    """Validasi, coba ulang yang gagal, lalu pakai cadangan."""
    if raws is None:
        return [{"sentimen": keyword_sentiment(t), "aspek": "lainnya"} for t in texts], np.zeros(len(texts), dtype=bool)
    parsed = [parse_output(r) for r in raws]
    failed = [i for i, p in enumerate(parsed) if p is None]
    if failed:
        retried = llm_generate(llm_tokenizer, llm_model, [" ".join(texts[i].split()[:40]) for i in failed])
        for i, raw in zip(failed, retried):
            parsed[i] = parse_output(raw)
    from_llm = np.array([p is not None for p in parsed])
    return [p or {"sentimen": keyword_sentiment(t), "aspek": "lainnya"} for p, t in zip(parsed, texts)], from_llm


with stage("resolve_audit"):
    first_pass_ok = 0.0 if raw_audit is None else float(np.mean([parse_output(r) is not None for r in raw_audit]))
    resolved_audit, from_llm = resolve(raw_audit, audit["text"].tolist())
final_pred = [SENTIMENT_TO_CLASS[r["sentimen"]] for r in resolved_audit]
record("llm_plus_pengaman", final_pred, float(from_llm.mean()))
print(f"lolos skema tanpa ulang: {first_pass_ok:.1%}")

# %% [markdown]
# ## 10. Post-analysis
#
# **Tabel perbandingan.**

# %%
result_table = pd.DataFrame(results)
result_table.to_csv(RESULT_DIR / "perbandingan.csv", index=False)
result_table

# %% [markdown]
# **Confusion matrix audit.**

# %%
plot_confusion(audit["label"], final_pred, CLASSES, "Confusion matrix audit")

# %% [markdown]
# **Demo MBG.** Tidak ada label, jadi hasilnya perlu dibaca. Sebaran aspek menunjukkan isu yang paling sering dibahas.

# %%
if USE_PRETRAINED:
    with stage("llm_mbg"):
        raw_mbg = llm_generate(llm_tokenizer, llm_model, mbg.tolist())
resolved_mbg, mbg_from_llm = resolve(raw_mbg, mbg.tolist())
mbg_table = pd.DataFrame(resolved_mbg).assign(komentar=mbg.str[:90])
print("aspek:", mbg_table["aspek"].value_counts(normalize=True).round(3).to_dict())
print("sentimen:", mbg_table["sentimen"].value_counts(normalize=True).round(3).to_dict())
mbg_table.to_csv(RESULT_DIR / "ekstraksi_mbg.csv", index=False)
mbg_table.head(6)

# %% [markdown]
# ## 11. Keluaran
#
# File untuk dikumpulkan dan ringkasan run.

# %%
pd.DataFrame({"id": audit.index, "label": final_pred}).to_csv(RESULT_DIR / "submission.csv", index=False)
run_summary("llm_ekstraksi", len(audit) + len(mbg),
            {"lolos_skema_tanpa_ulang": round(first_pass_ok, 3),
             **{r["metode"]: r["macro_f1_audit"] for r in results}},
            needs_gpu=USE_PRETRAINED)

# %% [markdown]
# **Poin untuk presentasi**
#
# - LLM + pengaman macro-F1 0,7264, sedangkan kata kunci 0,6101.
# - Mengulang jawaban hanya menaikkan lolos skema dari 90,0% ke 90,6%. Sisanya tetap perlu nilai cadangan.
# - Aspek pada data MBG tidak diaudit, dan contoh menunjukkan banyak salah, misalnya komentar harapan dianggap negatif. Jangan jadikan ini angka resmi.
# - 1.000 komentar selesai dalam 97 detik. Cocok untuk rangkuman cepat, bukan pengganti model utama.
