import torch
import os
import optuna
import gc
import pandas as pd
import numpy as np
import itertools
from tqdm.auto import tqdm
from transformers import AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer

from src.config import (MODEL_NAME, DEVICE, EPOCHS, DATA_DIR, MAX_LEN, 
                        BATCH_SIZE, MODEL_DIR)
from src.utils import set_seed
from src.data_loader import tokenizer, load_and_process_data
from src.combined_model import CombinedRoberta
from src.evaluation import get_preds

TUNE_SEED = 7
FIXED_LAMBDA = 0.2 

# DATASET
class RationaleDataset(Dataset):
    def __init__(self, texts, labels, embeddings):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.encodings = tokenizer(self.texts, truncation=True, padding="max_length", max_length=MAX_LEN)
        # Store the pre-computed embeddings as a tensor
        self.embeddings = embeddings 

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['target_rat'] = self.embeddings[idx] # Grab the specific vector
        return item
    def __len__(self): return len(self.labels)

def train_for_trial(trial, sent_data, sarc_data, df_val, df_test, df_golden):
    set_seed(TUNE_SEED)
    
    lr = trial.suggest_float("learning_rate", 1e-5, 4e-5, log=True)
    dropout_prob = trial.suggest_float("dropout_prob", 0.1, 0.2, step=0.05)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.1, 0.2, step=0.05)
    
    g = torch.Generator().manual_seed(TUNE_SEED)
    # sent_data and sarc_data are tuples of (texts, labels, pre_encoded_embeddings)
    train_sent_loader = DataLoader(RationaleDataset(*sent_data), batch_size=BATCH_SIZE, shuffle=True, generator=g)
    train_sarc_loader = DataLoader(RationaleDataset(*sarc_data), batch_size=BATCH_SIZE, shuffle=True, generator=g)
    sarc_iterator = itertools.cycle(train_sarc_loader)

    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = CombinedRoberta(MODEL_NAME, config, dropout_prob=dropout_prob).to(DEVICE)
    
    opt_params = [
        {'params': model.roberta.embeddings.parameters(), 'lr': lr * 0.1},
        {'params': model.roberta.encoder.layer[:6].parameters(), 'lr': lr * 0.2},
        {'params': model.roberta.encoder.layer[6:10].parameters(), 'lr': lr * 0.5},
        {'params': model.roberta.encoder.layer[10:].parameters(), 'lr': lr},
        {'params': model.sentiment_head.parameters(), 'lr': lr * 2.0},
        {'params': model.sarcasm_head.parameters(), 'lr': lr * 2.0},
        {'params': model.rationale_head.parameters(), 'lr': lr * 2.0},
    ]
    optimizer = AdamW(opt_params, weight_decay=0.01)
    
    total_steps = len(train_sent_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * warmup_ratio), num_training_steps=total_steps)
    
    loss_fn_class = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_fn_rat = torch.nn.CosineEmbeddingLoss() 

    best_val_f1 = -1.0
    best_state_dict = None

    for epoch in range(EPOCHS):
        model.train()
        for batch_sent in tqdm(train_sent_loader, desc=f"Trial {trial.number} | Ep {epoch+1}", leave=False):
            optimizer.zero_grad()
            
            # Sentiment Pass 
            out_sent, pred_rat_sent = model(batch_sent['input_ids'].to(DEVICE), batch_sent['attention_mask'].to(DEVICE), task='sentiment')
            l_sent_class = loss_fn_class(out_sent.logits, batch_sent['labels'].to(DEVICE).long())
            l_sent_rat = loss_fn_rat(pred_rat_sent, batch_sent['target_rat'].to(DEVICE), torch.ones(pred_rat_sent.size(0), device=DEVICE))
            
            # Sarcasm Pass 
            batch_sarc = next(sarc_iterator)
            out_sarc, pred_rat_sarc = model(batch_sarc['input_ids'].to(DEVICE), batch_sarc['attention_mask'].to(DEVICE), task='sarcasm')
            l_sarc_class = loss_fn_class(out_sarc.logits, batch_sarc['labels'].to(DEVICE).long())
            l_sarc_rat = loss_fn_rat(pred_rat_sarc, batch_sarc['target_rat'].to(DEVICE), torch.ones(pred_rat_sarc.size(0), device=DEVICE))
            
            # Loss
            loss = l_sent_class + FIXED_LAMBDA * (l_sarc_class + l_sent_rat + l_sarc_rat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        # Validation
        val_preds = get_preds(model, df_val['text'], is_mtl=True)
        val_f1 = f1_score(df_val['label'], val_preds, average='macro')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    if best_state_dict is None: return 0.0
    model.load_state_dict(best_state_dict)
    
    # User attributes for final reporting
    test_preds = get_preds(model, df_test['text'], is_mtl=True)
    trial.set_user_attr("test_f1", f1_score(df_test['label'], test_preds, average='macro'))
    
    gold_preds = get_preds(model, df_golden['text'], is_mtl=True)
    trial.set_user_attr("gold_f1", f1_score(df_golden['label'], gold_preds, average='macro'))

    del model; torch.cuda.empty_cache(); gc.collect()
    return best_val_f1

if __name__ == "__main__":
    dataframes = list(load_and_process_data(save_to_processed=False))
    df_train, df_val, df_test, _, df_golden = dataframes
    
    proc_path = os.path.join(DATA_DIR, "processed/")
    df_sent_train = pd.read_csv(os.path.join(proc_path, "sentiment_train_with_rationales.csv")).dropna(subset=['label'])
    df_sarc_train = pd.read_csv(os.path.join(proc_path, "sarcasm_train_with_rationales.csv")).dropna(subset=['label'])
    
    teacher = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE) # Use GPU for this one-time step
    
    sent_rat_embeds = teacher.encode(df_sent_train['rationale'].fillna("").tolist(), convert_to_tensor=True, show_progress_bar=True).cpu()
    sarc_rat_embeds = teacher.encode(df_sarc_train['rationale'].fillna("").tolist(), convert_to_tensor=True, show_progress_bar=True).cpu()
    
    sent_data = (df_sent_train["text"], df_sent_train["label"].astype(int), sent_rat_embeds)
    sarc_data = (df_sarc_train["text"], df_sarc_train["label"].astype(int), sarc_rat_embeds)
    
    del teacher; torch.cuda.empty_cache() # Free teacher memory

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: train_for_trial(t, sent_data, sarc_data, df_val, df_test, df_golden), n_trials=15)
    
    print(f"\nOptimization Finished. Best Params: {study.best_params}")