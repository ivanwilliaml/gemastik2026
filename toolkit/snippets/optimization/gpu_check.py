# title: Cek lingkungan, GPU, model lokal, mode offline
# when: Sel pertama. Set env offline sebelum import transformers.
# avoid: n_jobs=-1 di Windows pada matriks besar. Kode paralel harus di dalam if __name__ == '__main__'.
# tags: gpu cuda offline versi vram bf16
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
from pathlib import Path
import importlib.metadata as md

def env_report(models_dir=None):
    import torch
    for p in ("torch", "transformers", "scikit-learn", "pandas", "numpy", "sentence-transformers", "lightgbm"):
        try:
            print(p, md.version(p))
        except md.PackageNotFoundError:
            print(p, "TIDAK ADA")
    gpu = torch.cuda.is_available()
    print("cuda", gpu, torch.cuda.get_device_name(0) if gpu else "")
    if gpu:
        print("VRAM GB", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "| bf16", torch.cuda.is_bf16_supported())
    if models_dir and Path(models_dir).exists():
        print("model lokal:", sorted(p.name for p in Path(models_dir).iterdir() if p.is_dir()))
    return gpu
