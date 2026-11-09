# title: ABSA: input aspek [SEP] teks
# when: Fine-tune satu model untuk semua aspek.
# avoid: Membuang pasangan aspek yang tidak muncul.
# tags: absa pair transformer
import numpy as np

def absa_long(df, text_col, aspects):
    """Lebarkan ke satu baris per (teks, aspek). Latih dengan texts=aspek, texts2=teks."""
    out = df.melt(id_vars=[text_col], value_vars=list(aspects), var_name="aspect", value_name="label")
    return out.rename(columns={text_col: "text"})

def absa_wide(long_df, preds, row_ids):
    """Kembalikan tabel baris x aspek dari prediksi format panjang."""
    d = long_df.assign(pred=preds, row=row_ids)
    return d.pivot(index="row", columns="aspect", values="pred")
