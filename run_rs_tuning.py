import torch
import os
import optuna
import gc
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer

from src.config import MODEL_NAME, DEVICE, EPOCHS, DATA_DIR, MAX_LEN, BATCH_SIZE, MODEL_DIR
from src.utils import set_seed
from src.data_loader import tokenizer, load_and_process_data
from src.rs_model import RationaleSupervisedRoberta
from src.evaluation import get_preds

# CONFIG
TUNE_SEED = 7
FIXED_LAMBDA = 0.1 

class RationaleDataset(Dataset):
    def __init__(self, texts, labels, rationales):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.encodings = tokenizer(self.texts, truncation=True, padding="max_length", max_length=MAX_LEN)
        self.rationales = [str(r) if pd.notna(r) else "" for r in rationales.tolist()]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['rationale'] = self.rationales[idx]
        return item
    def __len__(self): return len(self.labels)

teacher_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
teacher_model.eval()

def train_for_trial(trial, df_train_rs, df_val, df_test, df_golden):
    set_seed(TUNE_SEED)
    
    lr = trial.suggest_float("learning_rate", 5e-6, 3e-5, log=True)
    dropout_prob = trial.suggest_float("dropout_prob", 0.1, 0.25, step=0.05)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.15, step=0.05)
    
    g = torch.Generator()
    g.manual_seed(TUNE_SEED)
    train_ds = RationaleDataset(df_train_rs["text"], df_train_rs["label"], df_train_rs["rationale"])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = RationaleSupervisedRoberta(MODEL_NAME, config, dropout_prob=dropout_prob).to(DEVICE)
    
    # LLRD 
    optimizer_grouped_parameters = [
        {'params': model.roberta.embeddings.parameters(), 'lr': lr * 0.1},
        {'params': model.roberta.encoder.layer[:6].parameters(), 'lr': lr * 0.2},
        {'params': model.roberta.encoder.layer[6:10].parameters(), 'lr': lr * 0.5},
        {'params': model.roberta.encoder.layer[10:].parameters(), 'lr': lr},
        {'params': model.sentiment_head.parameters(), 'lr': lr * 2},
        {'params': model.rationale_head.parameters(), 'lr': lr * 2}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * warmup_ratio), num_training_steps=total_steps)
    
    criterion_sent = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_rat = torch.nn.CosineEmbeddingLoss() 

    best_val_f1 = -1.0
    best_state_dict = None

    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(train_loader, desc=f"Trial {trial.number} | Ep {epoch+1}", leave=False):
            optimizer.zero_grad()
            
            with torch.no_grad():
                target_rat = teacher_model.encode(batch['rationale'], convert_to_tensor=True, device='cpu').to(DEVICE).detach().clone()
            
            outputs, pred_rat = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            
            l_sent = criterion_sent(outputs.logits, batch['labels'].to(DEVICE).long())
            l_rat = criterion_rat(pred_rat, target_rat, torch.ones(pred_rat.size(0), device=DEVICE))
            
            loss = l_sent + (FIXED_LAMBDA * l_rat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        val_preds = get_preds(model, df_val['text'], is_mtl=False)
        val_f1 = f1_score(df_val['label'], val_preds, average='macro')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    if best_state_dict is None:
        raise optuna.TrialPruned()

    model.load_state_dict(best_state_dict)
    model.eval()
    
    test_preds = get_preds(model, df_test['text'], is_mtl=False)
    trial.set_user_attr("test_f1", f1_score(df_test['label'], test_preds, average='macro'))
    
    gold_preds = get_preds(model, df_golden['text'], is_mtl=False)
    trial.set_user_attr("gold_f1", f1_score(df_golden['label'], gold_preds, average='macro'))

    del model, optimizer, scheduler, best_state_dict
    torch.cuda.empty_cache()
    gc.collect()

    return best_val_f1

if __name__ == "__main__":
    dataframes = list(load_and_process_data(save_to_processed=False))
    for i in range(len(dataframes)):
        dataframes[i] = dataframes[i].dropna(subset=['label'])
        dataframes[i]['label'] = dataframes[i]['label'].astype(int)
    
    df_train, df_val, df_test, _, df_golden = dataframes
    
    proc_path = os.path.join(DATA_DIR, "processed/")
    df_train_rs = pd.read_csv(os.path.join(proc_path, "sentiment_train_with_rationales.csv")).dropna(subset=['label'])
    df_train_rs['label'] = df_train_rs['label'].astype(int)

    study = optuna.create_study(
        direction="maximize", 
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1)
    )
    
    study.optimize(lambda trial: train_for_trial(trial, df_train_rs, df_val, df_test, df_golden), n_trials=15)
    print(f"Best Val F1: {study.best_value:.4f} | Best Params: {study.best_params}")