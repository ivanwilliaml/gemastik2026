# title: Loop PyTorch: AMP, akumulasi, clipping, LLRD
# when: Butuh kontrol di luar Trainer. Layerwise LR decay 0,95 bila ada waktu.
# avoid: Lupa zero_grad tiap langkah akumulasi.
# tags: amp bf16 fp16 akumulasi clipping llrd
import torch

def llrd_groups(model, lr=2e-5, decay=.95, wd=.01):
    """Parameter encoder bawah mendapat LR lebih kecil. Untuk model gaya BERT."""
    layers = [n for n, _ in model.named_parameters() if ".layer." in n]
    depth = 1 + max([int(n.split(".layer.")[1].split(".")[0]) for n in layers], default=0)
    groups = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        d = int(n.split(".layer.")[1].split(".")[0]) + 1 if ".layer." in n else (0 if "embeddings" in n else depth + 1)
        groups.append({"params": [p], "lr": lr * decay ** (depth + 1 - d), "weight_decay": 0.0 if "LayerNorm" in n or n.endswith("bias") else wd})
    return groups

def train_loop(model, loader, optimizer, scheduler=None, accum=1, clip=1.0, device=None, precision="auto"):
    """loader menghasilkan dict tensor berisi labels. Satu epoch. Kembalikan rata-rata loss."""
    device = device or next(model.parameters()).device
    cuda = str(device).startswith("cuda")
    dtype = None
    if precision == "auto" and cuda:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    elif precision == "fp16" and cuda:
        dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    model.train()
    total = 0.0
    optimizer.zero_grad(set_to_none=True)
    for i, b in enumerate(loader):
        b = {k: v.to(device) for k, v in b.items()}
        with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            loss = model(**b).loss
        scaler.scale(loss / accum).backward()
        total += float(loss)
        if (i + 1) % accum == 0 or i + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
    return total / max(len(loader), 1)
