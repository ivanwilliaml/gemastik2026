# title: QA span: prediksi dengan stride dan fine-tune
# when: Model span seperti Rifky/Indobert-QA, atau latih sendiri.
# avoid: Argmax start dan end terpisah.
# tags: qa span stride decode indobert_qa
import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, DefaultDataCollator, Trainer, TrainingArguments

@torch.inference_mode()
def qa_predict(model, tok, questions, contexts, max_len=384, stride=128, max_answer=30, batch=16):
    """Kembalikan (jawaban, skor terbaik dikurangi skor CLS). Skor rendah berarti mungkin tak terjawab."""
    dev = next(model.parameters()).device
    model.eval()
    e = tok(list(questions), list(contexts), truncation="only_second", max_length=max_len, stride=stride, return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length")
    n = len(e["input_ids"])
    start, end = [], []
    for i in range(0, n, batch):
        b = {k: torch.tensor(e[k][i:i + batch], device=dev) for k in tok.model_input_names if k in e}
        o = model(**b)
        start.append(o.start_logits.float().cpu().numpy()); end.append(o.end_logits.float().cpu().numpy())
    start, end = np.vstack(start), np.vstack(end)
    best = [("", -1e9, 0.0)] * len(questions)
    best = [list(x) for x in best]
    for f in range(n):
        q = e["overflow_to_sample_mapping"][f]
        off, types = e["offset_mapping"][f], e.sequence_ids(f)
        cls = start[f, 0] + end[f, 0]
        for s in np.argsort(start[f])[-20:]:
            if types[s] != 1:
                continue
            for t in np.argsort(end[f])[-20:]:
                if t < s or t - s >= max_answer or types[t] != 1:
                    continue
                sc = start[f, s] + end[f, t]
                if sc > best[q][1]:
                    best[q] = [contexts[q][off[s][0]:off[t][1]], float(sc), float(sc - cls)]
    return [b[0] for b in best], np.array([b[2] for b in best])

def train_qa(model_path, questions, contexts, answers, starts, max_len=384, stride=128, batch=8, epochs=1, lr=3e-5, out="run_qa", seed=42):
    """answers: teks jawaban. starts: posisi karakter awal."""
    tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    m = AutoModelForQuestionAnswering.from_pretrained(str(model_path), local_files_only=True)
    e = tok(list(questions), list(contexts), truncation="only_second", max_length=max_len, stride=stride, return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length")
    data = []
    for f in range(len(e["input_ids"])):
        q = e["overflow_to_sample_mapping"][f]
        off, types = e["offset_mapping"][f], e.sequence_ids(f)
        a, b = starts[q], starts[q] + len(answers[q])
        idx = [i for i, t in enumerate(types) if t == 1]
        s, t = 0, 0
        if idx and off[idx[0]][0] <= a and off[idx[-1]][1] >= b:
            s = next(i for i in idx if off[i][1] > a)
            t = next(i for i in reversed(idx) if off[i][0] < b)
        data.append({k: e[k][f] for k in tok.model_input_names if k in e} | {"start_positions": s, "end_positions": t})
    args = TrainingArguments(output_dir=out, per_device_train_batch_size=batch, num_train_epochs=epochs, learning_rate=lr, warmup_steps=.1, save_strategy="no",
                             dataloader_num_workers=0, report_to="none", seed=seed, disable_tqdm=True, bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    tr = Trainer(model=m, args=args, train_dataset=data, processing_class=tok, data_collator=DefaultDataCollator())
    tr.train()
    return tr.model, tok
