PRETRAINED_MODELS = {
    'indobertweet':'indolem/indobertweet-base-uncased',
    'indobert_p1':'indobenchmark/indobert-base-p1',
    'xlmr':'FacebookAI/xlm-roberta-base',
    'indolem':'indolem/indobert-base-uncased',
    'nusabert':'LazarusNLP/NusaBERT-base'
}


def fit_pretrained(data, cfg, name):
    require_kaggle(gpu=True)
    import gc
    import torch
    from scipy.special import expit, softmax
    from datasets import Dataset, Value, Sequence
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, set_seed
    set_seed(cfg['seed'])
    repo = cfg.get('model_paths',{}).get(name,PRETRAINED_MODELS[name])
    offline = cfg.get('offline',False)
    tok = AutoTokenizer.from_pretrained(repo,local_files_only=offline)
    task = cfg['task'];multi=task=='multilabel';reg=task=='similarity'
    problem = 'multi_label_classification' if multi else 'regression' if reg else 'single_label_classification'
    model = AutoModelForSequenceClassification.from_pretrained(repo,local_files_only=offline,
        num_labels=1 if reg else len(data['labels']),problem_type=problem,ignore_mismatched_sizes=True)
    if not reg:
        model.config.id2label = dict(enumerate(data['labels']))
        model.config.label2id = {name:i for i,name in enumerate(data['labels'])}
    def input_text(series):
        return series.map(lambda t:preprocess_text(t,'indobertweet')).tolist() if name=='indobertweet' else series.tolist()
    def encoded(frame, labels=True):
        a = input_text(frame.a)
        payload = tok(a,input_text(frame.b) if 'b' in frame else None,truncation=True,max_length=cfg['max_len'])
        if labels:
            y = target_array(frame,cfg)
            payload['labels'] = np.where(np.isfinite(y),y,-1).astype('float32') if multi else y.astype('float32' if reg else 'int64')
        ds = Dataset.from_dict(payload)
        if labels and reg:
            ds = ds.cast_column('labels',Value('float32'))
        if labels and multi:
            ds = ds.cast_column('labels',Sequence(Value('float32')))
        return ds
    y = target_array(data['train'],cfg)
    pos_weight = np.sqrt(np.clip(np.sum(y==0,axis=0)/np.maximum(np.sum(y==1,axis=0),1),1,20)) if multi else None
    class MaskedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop('labels').float();mask=labels>=0
            output = model(**inputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output.logits.float(),labels.clamp(0,1),
                pos_weight=torch.as_tensor(pos_weight,device=labels.device,dtype=torch.float32),reduction='none')
            loss = (loss*mask).sum()/mask.sum().clamp(min=1)
            return (loss,output) if return_outputs else loss
    def probability(z):
        return z.ravel() if reg else expit(z) if multi else softmax(z,axis=1)
    def metric(prediction):
        score = task_metrics(prediction.label_ids,probability(prediction.predictions),task)
        return {'selection':-score['RMSE'] if reg else score['macro_f1']}
    out = Path(cfg['out'])/('model_'+name)
    args = TrainingArguments(output_dir=str(out),num_train_epochs=cfg['epochs'],learning_rate=cfg['learning_rate'],
        per_device_train_batch_size=cfg['batch_size'],per_device_eval_batch_size=cfg['batch_size']*2,
        gradient_accumulation_steps=cfg.get('accumulation',1),fp16=True,weight_decay=.01,
        eval_strategy='epoch',save_strategy='epoch',save_only_model=True,save_total_limit=1,
        load_best_model_at_end=True,metric_for_best_model='selection',greater_is_better=True,
        logging_strategy='epoch',disable_tqdm=True,report_to='none',seed=cfg['seed'])
    trainer_class = MaskedTrainer if multi else Trainer
    fit_frame = data['train']
    if reg:
        swapped = fit_frame.copy()
        swapped['a'],swapped['b'] = fit_frame.b,fit_frame.a
        fit_frame = pd.concat([fit_frame,swapped],ignore_index=True)
    trainer = trainer_class(model=model,args=args,train_dataset=encoded(fit_frame),eval_dataset=encoded(data['val']),
        data_collator=DataCollatorWithPadding(tok),compute_metrics=metric)
    if multi:
        trainer.model_accepts_loss_kwargs = False
    trainer.train()
    results = []
    for split in ['val','test']:
        frame = data[split]
        scores = probability(trainer.predict(encoded(frame,False)).predictions)
        if reg:
            swapped = frame.copy()
            swapped['a'],swapped['b'] = frame.b,frame.a
            scores = (scores+probability(trainer.predict(encoded(swapped,False)).predictions))/2
        results.append(scores)
    results = tuple(results)
    if reg:
        results = tuple(np.clip(p,*cfg['score_range']) for p in results)
    trainer.save_model(str(out));tok.save_pretrained(str(out))
    (out/'model_provenance.json').write_text(json.dumps({'repo':repo,'commit':getattr(model.config,'_commit_hash',None),'labels':data['labels']},indent=2))
    del trainer,model
    gc.collect();torch.cuda.empty_cache()
    return results


for _model_name in PRETRAINED_MODELS:
    MODEL_RUNNERS[_model_name] = lambda data,cfg,name=_model_name:fit_pretrained(data,cfg,name)
