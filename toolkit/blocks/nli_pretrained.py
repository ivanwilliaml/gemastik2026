@register('nli_pretrained')
def fit_nli_pretrained(data, cfg):
    require_kaggle(gpu=True)
    import gc
    import torch
    from scipy.special import softmax
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    repo = cfg.get('nli_model','MoritzLaurer/mDeBERTa-v3-base-mnli-xnli')
    tok = AutoTokenizer.from_pretrained(repo,local_files_only=cfg.get('offline',False))
    model = AutoModelForSequenceClassification.from_pretrained(repo,local_files_only=cfg.get('offline',False)).cuda().eval()
    mapping = {str(label).casefold():int(i) for i,label in model.config.id2label.items()}
    assert set(data['labels']).issubset(mapping), f'Mapping label model perlu eksplisit: {mapping}'
    order = [mapping[label] for label in data['labels']]
    results = []
    with torch.inference_mode():
        for split in ['val','test']:
            rows = []
            for start in range(0,len(data[split]),32):
                f = data[split].iloc[start:start+32]
                # Premise -> hypothesis. Tidak ada swap pada NLI.
                enc = tok(f.a.tolist(),f.b.tolist(),padding=True,truncation=True,max_length=cfg['max_len'],return_tensors='pt').to('cuda')
                rows.append(softmax(model(**enc).logits.float().cpu().numpy()[:,order],axis=1))
            results.append(np.vstack(rows))
    del model
    gc.collect();torch.cuda.empty_cache()
    return tuple(results)
