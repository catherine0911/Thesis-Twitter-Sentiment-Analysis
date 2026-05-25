import os
import numpy as np
import torch
import torch.nn as nn
import gc
from tqdm.auto import tqdm
from transformers import AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score

from src.config import MODEL_NAME, DEVICE, SEEDS, MODEL_DIR
from src.utils import set_seed
from src.data_loader import load_and_process_data, get_dataloaders
from src.evaluation import get_preds
from src.baseline_model import RobertaBaseline

from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput

def train_baseline(seed, dataframes):
    set_seed(seed)
    df_train, df_val, df_test, _, df_golden = dataframes

    LEARNING_RATE = 2e-5    
    DROPOUT       = 0.1
    WARMUP        = 0.1
    EPOCHS        = 3
    WEIGHT_DECAY  = 0.01
    LABEL_SMOOTH  = 0.1
    MAX_GRAD_NORM = 1.0

    save_dir = os.path.join(MODEL_DIR, 'baseline_models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'baseline_seed{seed}.pt')

    config = AutoConfig.from_pretrained(MODEL_NAME)
    model  = RobertaBaseline(MODEL_NAME, config, DROPOUT).to(DEVICE)

    # LLRD:use smaller learning rates for lower RoBERTa layers and a larger rate for the classifier
    opt_params = [
        {'params': model.roberta.embeddings.parameters(),          'lr': LEARNING_RATE * 0.1},
        {'params': model.roberta.encoder.layer[:6].parameters(),   'lr': LEARNING_RATE * 0.2},
        {'params': model.roberta.encoder.layer[6:10].parameters(), 'lr': LEARNING_RATE * 0.5},
        {'params': model.roberta.encoder.layer[10:].parameters(),  'lr': LEARNING_RATE},
        {'params': model.classifier.parameters(),                  'lr': LEARNING_RATE * 2.0},
    ]
    optimizer = AdamW(opt_params, weight_decay=WEIGHT_DECAY)

    train_loader, val_loader, test_loader, _ = get_dataloaders(df_train, df_val, df_test, df_train, seed) 

    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP),
        num_training_steps=total_steps
    )
    loss_fn     = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    best_val_f1 = -1.0
    # Save the checkpoint with the best validation macro-F1
    # If the checkpoint already exists, skip training and load it directly
    if os.path.exists(save_path):
        print(f'  Found saved model at {save_path}, skipping training.')
        model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    else:
        for epoch in range(EPOCHS):
            model.train()
            for batch in tqdm(train_loader, desc=f'Seed {seed} | Ep {epoch+1}', leave=False):
                optimizer.zero_grad()
                out  = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
                loss = loss_fn(out.logits, batch['labels'].to(DEVICE))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()

            # Validation
            model.eval()
            v_preds, v_labels = [], []
            with torch.no_grad():
                for b in val_loader:
                    out = model(b['input_ids'].to(DEVICE), b['attention_mask'].to(DEVICE))
                    v_preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
                    v_labels.extend(b['labels'].numpy())
            val_f1 = f1_score(v_labels, v_preds, average='macro')
            print(f'  Seed {seed} | Epoch {epoch+1} | Val F1: {val_f1:.4f}')
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), save_path)

        model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))

    # Evaluation
    model.eval()
    t_preds = get_preds(model, df_test['text'],   is_mtl=False)
    g_preds = get_preds(model, df_golden['text'], is_mtl=False)

    t_f1  = f1_score(df_test['label'],   t_preds, average='macro')
    t_acc = accuracy_score(df_test['label'],   t_preds)
    g_f1  = f1_score(df_golden['label'], g_preds, average='macro')
    g_acc = accuracy_score(df_golden['label'], g_preds)

    print(f'  Seed {seed} | Test F1: {t_f1:.4f} | Gold F1: {g_f1:.4f}')
    del model; torch.cuda.empty_cache(); gc.collect()
    return (t_f1, t_acc), (g_f1, g_acc)


if __name__ == '__main__':
    dataframes = load_and_process_data()
    test_f1s, gold_f1s = [], []

    for seed in SEEDS:
        (t_f1, t_acc), (g_f1, g_acc) = train_baseline(seed, dataframes)
        test_f1s.append(t_f1); gold_f1s.append(g_f1)

    print('\n' + '='*60)
    print('BASELINE SUMMARY')
    print('='*60)
    print(f'Test F1: {np.mean(test_f1s):.4f} (±{np.std(test_f1s, ddof=1):.4f})')
    print(f'Gold F1: {np.mean(gold_f1s):.4f} (±{np.std(gold_f1s, ddof=1):.4f})')