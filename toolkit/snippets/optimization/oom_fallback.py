# title: Tangkap OOM dan kecilkan batch
# when: VRAM 8 GB. Atau auto_find_batch_size=True di TrainingArguments.
# avoid: Terus mengecil tanpa mencatat batch final.
# tags: oom cuda batch vram
import gc
import torch

def with_oom_retry(fn, batch=64, min_batch=2):
    """fn(batch) dijalankan ulang dengan batch separuh saat OOM. Kembalikan (hasil, batch final)."""
    while batch >= min_batch:
        try:
            return fn(batch), batch
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            print("OOM, batch", batch, "->", batch // 2)
            batch //= 2
    raise RuntimeError("OOM pada batch minimum")
