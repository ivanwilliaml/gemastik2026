# title: Embedding SentenceTransformer dengan cache
# when: Dedup, klaster, retrieval, atau anggota ensemble. Encode sekali, lalu pakai berulang.
# avoid: Encode ulang tiap eksperimen. Tanpa prefiks pada model E5.
# tags: sbert e5 embedding cache prefix
import os
import numpy as np

def encode_st(model_path, texts, cache=None, prefix="", batch_size=128, device=None, max_len=None):
    """Hasil L2-normalized. E5: prefix='passage: ' untuk dokumen, 'query: ' untuk query."""
    # Cache hanya valid untuk urutan teks, model, prefix, dan panjang yang sama.
    import hashlib, json
    signature = hashlib.sha256(json.dumps([str(model_path), list(map(str,texts)), prefix, max_len], ensure_ascii=False).encode()).hexdigest()
    meta = str(cache) + '.sha256' if cache else None
    if cache and os.path.exists(cache) and os.path.exists(meta):
        if open(meta, encoding='utf-8').read().strip() == signature:
            return np.load(cache, mmap_mode="r")
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(str(model_path), device=device, local_files_only=True)
    if max_len:
        m.max_seq_length = max_len
    E = m.encode([prefix + str(t) for t in texts], batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    if cache:
        np.save(cache, E)
        with open(meta, 'w', encoding='utf-8') as file: file.write(signature)
    return E
