# Notebook modular

Pilih satu, ubah **CFG** (file, kolom, grup), lalu jalankan di Kaggle.
Untuk mengubah komponen gunakan [Rakit kode](../../index.html#/compose), bukan
menyalin seluruh notebook kasus lain. Fungsi sudah tertanam, tidak perlu import toolkit.

| Notebook | Pilihan default |
|---|---|
| [classification](classification_modular.ipynb) | Dummy, word TF-IDF, word+char TF-IDF, Word2Vec, IndoBERTweet |
| [multilabel](multilabel_modular.ipynb) | Dummy, dua TF-IDF, E5, IndoBERTweet; unknown masking dan threshold per label |
| [similarity](similarity_modular.ipynb) | Mean, dua TF-IDF Ridge, E5 Ridge, paired IndoBERT; skor kontinu simetris |
| [nli](nli_modular.ipynb) | Dummy, TF-IDF, E5, pretrained mDeBERTa NLI, fine-tuned IndoBERT; tiga kelas |
| [retrieval](retrieval_modular.ipynb) | TF-IDF, BM25, chunked E5, RRF, cross-encoder reranker |
| [topic](topic_modular.ipynb) | NMF, LDA, TF-IDF/SVD KMeans; pilih k di validation |
| [clustering](clustering_modular.ipynb) | NMF, LDA, TF-IDF/SVD KMeans; diagnostik silhouette dan contoh cluster |
| [temporal](temporal_modular.ipynb) | Tiga model topik beku; proporsi waktu dan sentimen lexicon; pretrained opsional |

Semua notebook ini template **belum dieksekusi pada data Anda**. Output eksperimen
sumber berada di `../../../notebook/`, tetap dipertahankan. Perakit menyimpan
jejak sumber, bukan mengimpor angka hasil sumber. Jangan memilih model dari test.

`--no-install` mematikan sel pemasangan paket untuk bundle offline.
