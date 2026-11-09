# title: SetFit dengan pembanding LR
# when: Label sekitar 50-5.000. Pakai hanya bila mengalahkan LR di atas embedding.
# avoid: Label banyak: lambat dan tak perlu.
# tags: setfit few_shot
import numpy as np

def fit_setfit(model_path, texts, y, batch=16, num_iterations=10, epochs=1, seed=42):
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments
    m = SetFitModel.from_pretrained(str(model_path), local_files_only=True)
    ds = Dataset.from_dict({"text": [str(t) for t in texts], "label": [int(v) for v in y]})
    args = TrainingArguments(batch_size=batch, num_epochs=epochs, num_iterations=num_iterations, seed=seed, report_to="none", show_progress_bar=False)
    Trainer(model=m, args=args, train_dataset=ds).train()
    return m

def predict_setfit(m, texts):
    return np.asarray(m.predict_proba([str(t) for t in texts]).cpu())
