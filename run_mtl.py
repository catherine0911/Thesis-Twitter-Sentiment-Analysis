import torch
import os
import torch.nn as nn
import itertools
import numpy as np
import gc
from tqdm.auto import tqdm
from transformers import AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score

from src.config import MODEL_NAME, DEVICE, EPOCHS, SEEDS, LAMBDAS, MODEL_DIR
from src.utils import set_seed
from src.data_loader import load_and_process_data, get_dataloaders
from src.mtl_model import RobertaMTL
from src.evaluation import get_preds

# CONFIG
search_lambda_mode = False # Set to True to search lambda, False to final evaluation with selected lambda

MODEL_DIR = "models/mtl_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Final selected lambda for final evaluation mode
FINAL_LAMBDA = 0.2

CONFIGS = {
    "Default": {
        "learning_rate": 2e-5,
        "dropout": 0.1,
        "warmup": 0.1,
        "prefix": "mtl_"
    },
    "Tuned": {
        "learning_rate": 8.461228954824954e-06,
        "dropout": 0.15000000000000002,
        "warmup": 0.05,
        "prefix": "tuned_mtl_"
    }
}


def train_single_seed_mtl(seed, lambda_val, dataframes, config_name, config_values):
    set_seed(seed)

    df_train, df_val, df_test, df_sarc, df_golden = dataframes

    learning_rate = config_values["learning_rate"]
    dropout = config_values["dropout"]
    warmup = config_values["warmup"]
    prefix = config_values["prefix"]

    model_save_path = os.path.join(
        MODEL_DIR,
        f"{prefix}seed{seed}_lam{lambda_val}.pt"
    )

    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model = RobertaMTL(MODEL_NAME, model_config, dropout_prob=dropout).to(DEVICE)

    train_sent, val_sent, test_sent, train_sarc = get_dataloaders(
        df_train, df_val, df_test, df_sarc, seed
    )

    if os.path.exists(model_save_path):
        print(f"Model found at {model_save_path}. Skipping training...")
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE, weights_only=True))

    else:
        optimizer_grouped_parameters = [
            {'params': model.roberta.embeddings.parameters(), 'lr': learning_rate * 0.1},
            {'params': model.roberta.encoder.layer[:6].parameters(), 'lr': learning_rate * 0.2},
            {'params': model.roberta.encoder.layer[6:10].parameters(), 'lr': learning_rate * 0.5},
            {'params': model.roberta.encoder.layer[10:].parameters(), 'lr': learning_rate},
            {'params': model.sentiment_head.parameters(), 'lr': learning_rate * 2.0},
            {'params': model.sarcasm_head.parameters(), 'lr': learning_rate * 2.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, weight_decay=0.01)

        total_steps = len(train_sent) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup),
            num_training_steps=total_steps
        )

        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        best_val_f1 = -1.0
        sarc_iterator = itertools.cycle(train_sarc)

        for epoch in range(EPOCHS):
            model.train()

            for batch_sent in tqdm(
                train_sent,
                desc=f"Seed {seed} | Lam {lambda_val} | Ep {epoch + 1}",
                leave=False
            ):
                optimizer.zero_grad()

                # Sentiment task
                out_sent = model(
                    batch_sent['input_ids'].to(DEVICE),
                    batch_sent['attention_mask'].to(DEVICE),
                    task='sentiment'
                )
                loss_sent = loss_fn(out_sent.logits, batch_sent['labels'].to(DEVICE))

                # Sarcasm auxiliary task
                batch_sarc = next(sarc_iterator)
                out_sarc = model(
                    batch_sarc['input_ids'].to(DEVICE),
                    batch_sarc['attention_mask'].to(DEVICE),
                    task='sarcasm'
                )
                loss_sarc = loss_fn(out_sarc.logits, batch_sarc['labels'].to(DEVICE))

                total_loss = loss_sent + (lambda_val * loss_sarc)
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            # Validation checkpoint selection
            model.eval()
            val_preds, val_labels = [], []

            with torch.no_grad():
                for batch in val_sent:
                    out = model(
                        batch['input_ids'].to(DEVICE),
                        batch['attention_mask'].to(DEVICE),
                        task='sentiment'
                    )
                    val_preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
                    val_labels.extend(batch['labels'].numpy())

            val_f1 = f1_score(val_labels, val_preds, average='macro')

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), model_save_path)

        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE, weights_only=True))

    # Evaluation after loading best validation checkpoint
    model.eval()

    # Validation evaluation
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_sent:
            out = model(
                batch['input_ids'].to(DEVICE),
                batch['attention_mask'].to(DEVICE),
                task='sentiment'
            )
            val_preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            val_labels.extend(batch['labels'].numpy())

    v_f1 = f1_score(val_labels, val_preds, average='macro')
    v_acc = accuracy_score(val_labels, val_preds)

    # Golden Set evaluation
    golden_preds = get_preds(model, df_golden['text'], is_mtl=True)
    g_f1 = f1_score(df_golden['label'], golden_preds, average='macro')
    g_acc = accuracy_score(df_golden['label'], golden_preds)

    # Test Set evaluation only in final mode
    if search_lambda_mode:
        t_f1, t_acc = np.nan, np.nan
    else:
        test_preds, test_labels = [], []
        with torch.no_grad():
            for batch in test_sent:
                out = model(
                    batch['input_ids'].to(DEVICE),
                    batch['attention_mask'].to(DEVICE),
                    task='sentiment'
                )
                test_preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
                test_labels.extend(batch['labels'].numpy())

        t_f1 = f1_score(test_labels, test_preds, average='macro')
        t_acc = accuracy_score(test_labels, test_preds)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return (v_f1, v_acc), (t_f1, t_acc), (g_f1, g_acc)


def run_lambda_search(dataframes):
    print("Running MTL Lambda Search with DEFAULT hyperparameters")

    active_lambdas = LAMBDAS
    active_seeds = SEEDS
    config_name = "Default"
    config_values = CONFIGS["Default"]

    results = {
        lam: {
            'val_f1': [], 'val_acc': [],
            'gold_f1': [], 'gold_acc': [],
            'score': []
        }
        for lam in active_lambdas
    }

    for lam in active_lambdas:
        print(f"\nRunning MTL | Lambda: {lam} | Lambda Search\n{'=' * 50}")

        for seed in active_seeds:
            (v_f1, v_acc), (_, _), (g_f1, g_acc) = train_single_seed_mtl(
                seed, lam, dataframes, config_name, config_values
            )

            score = (0.4 * v_f1) + (0.6 * g_f1)

            results[lam]['val_f1'].append(v_f1)
            results[lam]['val_acc'].append(v_acc)
            results[lam]['gold_f1'].append(g_f1)
            results[lam]['gold_acc'].append(g_acc)
            results[lam]['score'].append(score)

            # Per-seed print: validation only, as requested
            print(f"Seed {seed:3} | Val F1: {v_f1:.4f}")

    print("\n" + "=" * 80)
    print("FINAL MTL LAMBDA SEARCH SUMMARY")
    print("=" * 80)

    lambda_scores = {}

    for lam in active_lambdas:
        res = results[lam]

        avg_vf1, std_vf1 = np.mean(res['val_f1']), np.std(res['val_f1'])
        avg_gf1, std_gf1 = np.mean(res['gold_f1']), np.std(res['gold_f1'])
        avg_score, std_score = np.mean(res['score']), np.std(res['score'])

        lambda_scores[lam] = avg_score

        print(
            f"Lambda: {lam} | "
            f"Val F1: {avg_vf1:.4f} (±{std_vf1:.4f}) | "
            f"Gold F1: {avg_gf1:.4f} (±{std_gf1:.4f}) | "
            f"Score: {avg_score:.4f} (±{std_score:.4f})"
        )

    best_lam = max(lambda_scores, key=lambda_scores.get)
    print(
        f"\nSuggested Lambda: {best_lam} "
        f"(0.4 * Val F1 + 0.6 * Gold F1 = {lambda_scores[best_lam]:.4f})"
    )


def run_final_default_vs_tuned(dataframes):
    active_lambdas = [FINAL_LAMBDA]
    active_seeds = SEEDS

    all_results = {}

    for config_name, config_values in CONFIGS.items():
        print(
            f"\nRunning MTL | Lambda: {FINAL_LAMBDA} | {config_name} Hyperparameter\n"
            f"{'=' * 50}"
        )

        results = {
            'val_f1': [], 'val_acc': [],
            'test_f1': [], 'test_acc': [],
            'gold_f1': [], 'gold_acc': []
        }

        for seed in active_seeds:
            (v_f1, v_acc), (t_f1, t_acc), (g_f1, g_acc) = train_single_seed_mtl(
                seed, FINAL_LAMBDA, dataframes, config_name, config_values
            )

            results['val_f1'].append(v_f1)
            results['val_acc'].append(v_acc)
            results['test_f1'].append(t_f1)
            results['test_acc'].append(t_acc)
            results['gold_f1'].append(g_f1)
            results['gold_acc'].append(g_acc)

            print(
                f"Seed {seed:3} | "
                f"Val F1: {v_f1:.4f} | "
                f"Test F1: {t_f1:.4f} | "
                f"Gold F1: {g_f1:.4f}"
            )

        all_results[config_name] = results

    print("\n" + "=" * 80)
    print("FINAL MTL REPORT SUMMARY")
    print("=" * 80)

    for config_name, res in all_results.items():
        print(
            f"{config_name}: "
            f"Val F1: {np.mean(res['val_f1']):.4f} (±{np.std(res['val_f1']):.4f}) | "
            f"Test F1: {np.mean(res['test_f1']):.4f} (±{np.std(res['test_f1']):.4f}) | "
            f"Gold F1: {np.mean(res['gold_f1']):.4f} (±{np.std(res['gold_f1']):.4f}) | "
            f"Val Acc: {np.mean(res['val_acc']):.4f} (±{np.std(res['val_acc']):.4f}) | "
            f"Test Acc: {np.mean(res['test_acc']):.4f} (±{np.std(res['test_acc']):.4f}) | "
            f"Gold Acc: {np.mean(res['gold_acc']):.4f} (±{np.std(res['gold_acc']):.4f})"
        )


if __name__ == "__main__":
    dataframes = load_and_process_data()

    if search_lambda_mode:
        run_lambda_search(dataframes)
    else:
        run_final_default_vs_tuned(dataframes)