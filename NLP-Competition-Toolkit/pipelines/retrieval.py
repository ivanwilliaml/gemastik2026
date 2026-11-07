# title: Retrieval dan ranking
# for: ir doc_sim ranking
# %% stage: setup | Setup & konfigurasi
# note: Satu-satunya sel yang wajib diubah. Jalankan paling awal.
# uses: optimization/gpu_check, optimization/seed_everything
import re, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
MODELS = Path(os.environ.get("COMP_ROOT", r"C:\gemastik")) / "models"
SEED = 42
seed_everything(SEED)
env_report(MODELS)

DOCS, QUERIES = "docs.csv", "queries.csv"       # kolom: doc_id,text dan query_id,text,relevant_id
K = 10

# %% stage: load | Muat data
# note: Baca file, otomatis mencoba encoding, simpan parquet.
# uses: data_loading/load_table
docs, q = read_any(DOCS), read_any(QUERIES)
dt, qt = docs.text.fillna("").astype(str).tolist(), q.text.fillna("").astype(str).tolist()
doc_ids = docs.doc_id.to_numpy()
print(docs.shape, q.shape)

# %% stage: prep | Preprocessing
# note: Query dan dokumen harus sama perlakuannya.
# uses: preprocessing/preprocess_all
dt, qt = preprocess_corpus(dt, "tfidf"), preprocess_corpus(qt, "tfidf")

# %% stage: model | Retrieval
# note: BM25, TF-IDF, lalu digabung RRF. E5 bisa ditambahkan sebagai ranking ketiga.
# uses: modeling/retrieval_bm25, modeling/retrieval_dense, feature_engineering/make_vectorizer
bm, _ = BM25().fit(dt).search(qt, K * 5)
vec = make_vectorizer("tfidf_word", min_df=1)
Dm = vec.fit_transform(dt)
tf, _ = topk_sparse(vec.transform(qt), Dm, K * 5)
fused = rrf_merge([bm, tf], top=K * 5)

# %% stage: eval | Evaluasi
# note: Bila ada relevant_id.
# uses: evaluation/metrics_retrieval
gold = [{x} for x in q.relevant_id] if "relevant_id" in q else None
if gold:
    for n, r in (("bm25", bm), ("tfidf", tf), ("rrf", fused)):
        print(n, retrieval_report([doc_ids[x].tolist() for x in r], gold, K))

# %% stage: post | Re-ranking
# note: LGBMRanker dengan fitur peringkat, bila ada relevansi.
# uses: modeling/ranker_lgbm, evaluation/metrics_retrieval
final = fused[:, :K]
if gold:
    rows = []
    for i, c in enumerate(fused):
        for rk, d in enumerate(c):
            rows.append((i, int(d), rk, int(np.where(bm[i] == d)[0][0]) if d in bm[i] else 99, int(np.where(tf[i] == d)[0][0]) if d in tf[i] else 99, int(doc_ids[d] in gold[i])))
    r = pd.DataFrame(rows, columns=["q", "d", "rrf", "bm", "tf", "rel"])
    fit = r.q % 2 == 0                       # latih pada query genap, nilai pada ganjil
    rk = fit_ranker(r.loc[fit, ["rrf", "bm", "tf"]], r.rel[fit], r.q[fit], 100)
    r["s"] = rk.predict(r[["rrf", "bm", "tf"]])
    rr = r.sort_values(["q", "s"], ascending=[True, False]).groupby("q").d.apply(list)
    odd = [i for i in range(len(q)) if i % 2]
    print("ranker, query ganjil", retrieval_report([doc_ids[rr[i]].tolist() for i in odd], [gold[i] for i in odd], K))

# %% stage: out | Output
# note: Satu baris per query.
# uses: 
q["pred"] = [" ".join(map(str, doc_ids[x])) for x in final]
q[["query_id", "pred"]].to_csv("submission.csv", index=False)
print(q[["query_id", "pred"]].head(3).to_string())
