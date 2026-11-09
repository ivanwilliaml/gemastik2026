def e5_encode(texts, cfg, role='query'):
    """E5: masked mean pooling, L2 normalisasi, prefix query/passage yang eksplisit."""
    require_kaggle(gpu=True)
    import gc
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    assert role in {'query','passage'}
    repo = cfg.get('e5_model','intfloat/multilingual-e5-base')
    tok = AutoTokenizer.from_pretrained(repo,local_files_only=cfg.get('offline',False))
    model = AutoModel.from_pretrained(repo,local_files_only=cfg.get('offline',False)).cuda().eval()
    rows = []
    texts = list(texts)
    with torch.inference_mode():
        for start in range(0,len(texts),cfg.get('embed_batch',32)):
            batch = [role+': '+str(t) for t in texts[start:start+cfg.get('embed_batch',32)]]
            enc = tok(batch,padding=True,truncation=True,max_length=cfg.get('max_len',256),return_tensors='pt').to('cuda')
            hidden = model(**enc).last_hidden_state.float()
            pooled = (hidden*enc.attention_mask.unsqueeze(-1)).sum(1)/enc.attention_mask.sum(1,keepdim=True).clamp(min=1)
            rows.append(F.normalize(pooled,p=2,dim=1).cpu().numpy())
    result = np.vstack(rows) if rows else np.empty((0,model.config.hidden_size),dtype='float32')
    del model
    gc.collect();torch.cuda.empty_cache()
    return result


ENCODERS['e5'] = e5_encode
