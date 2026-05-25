import torch
import os
import torch.nn as nn
import numpy as np
import gc
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader

from src.config import (
    MODEL_NAME, DEVICE, EPOCHS, SEEDS, LAMBDAS,
    MAX_LEN, DATA_DIR, BATCH_SIZE
)
from src.utils import set_seed
from src.rs_model import RationaleSupervisedRoberta
from src.data_loader import load_and_process_data, tokenizer
from src.evaluation import get_preds


# CONFIGURATION
search_lambda_mode = False   # Set to True to search lambda, False to final evaluation with selected lambda
refined_model = False        # Set to True to run refined RS (test from pilot experiments)

MODEL_DIR = "models/rs_models"
os.makedirs(MODEL_DIR, exist_ok=True)

final_lambdas = {"Default": [0.1], "Tuned": [0.1], "Refined": [0.1, 0.15, 0.2]}
CONFIGS = {
    "Default": {
        "learning_rate": 2e-5,
        "dropout": 0.1,
        "warmup": 0.1,
        "prefix": "rs_",
        "model_class": RationaleSupervisedRoberta
    },
    "Tuned": {
        "learning_rate": 7.902120558097247e-06,
        "dropout": 0.2,
        "warmup": 0.15,
        "prefix": "tuned_rs_",
        "model_class": RationaleSupervisedRoberta
    }
}

if refined_model:
    from src.rs_model_refined import RationaleSupervisedRoberta as RefinedRS

    CONFIGS["Refined"] = {
        "learning_rate": 2e-5,
        "dropout": 0.1,
        "warmup": 0.1,
        "prefix": "refined_rs_",
        "model_class": RefinedRS
    }


class RationaleDataset(Dataset):
    def __init__(self, texts, labels, rationales):
        self.texts = texts.tolist() if hasattr(texts, "tolist") else list(texts)
        self.labels = labels.tolist() if hasattr(labels, "tolist") else list(labels)

        rationales_list = rationales.tolist() if hasattr(rationales, "tolist") else list(rationales)

        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )

        self.rationales = [
            str(r) if pd.notna(r) else ""
            for r in rationales_list
        ]

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        item["rationale"] = self.rationales[idx]
        return item

    def __len__(self):
        return len(self.labels)


def train_single_seed_rs(seed, lambda_val, dataframes, config_name, config_values):
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)

    df_train, df_val, df_test, _, df_golden = dataframes

    learning_rate = config_values["learning_rate"]
    dropout = config_values["dropout"]
    warmup = config_values["warmup"]
    prefix = config_values["prefix"]
    model_class = config_values["model_class"]

    df_train_rs = pd.read_csv(
        os.path.join(DATA_DIR, "processed/sentiment_train_with_rationales.csv")
    ).dropna(subset=["label"])

    df_val_rs = pd.read_csv(
        os.path.join(DATA_DIR, "processed/sentiment_validation_with_rationales.csv")
    ).dropna(subset=["label"])

    df_train_rs["label"] = df_train_rs["label"].astype(int)
    df_val_rs["label"] = df_val_rs["label"].astype(int)

    train_loader = DataLoader(
        RationaleDataset(
            df_train_rs["text"],
            df_train_rs["label"],
            df_train_rs["rationale"]
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g
    )

    val_loader = DataLoader(
        RationaleDataset(
            df_val_rs["text"],
            df_val_rs["label"],
            df_val_rs["rationale"]
        ),
        batch_size=BATCH_SIZE
    )

    model_save_path = os.path.join(
        MODEL_DIR,
        f"{prefix}seed{seed}_lam{lambda_val}.pt"
    )

    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = model_class(
        MODEL_NAME,
        config,
        dropout_prob=dropout
    ).to(DEVICE)

    teacher_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    teacher_model.eval()

    if os.path.exists(model_save_path):
        print(f"Model found at {model_save_path}. Skipping training...")
        model.load_state_dict(
            torch.load(model_save_path, map_location=DEVICE, weights_only=True)
        )

    else:
        opt_params = [
            {"params": model.roberta.embeddings.parameters(), "lr": learning_rate * 0.1},
            {"params": model.roberta.encoder.layer[:6].parameters(), "lr": learning_rate * 0.2},
            {"params": model.roberta.encoder.layer[6:10].parameters(), "lr": learning_rate * 0.5},
            {"params": model.roberta.encoder.layer[10:].parameters(), "lr": learning_rate},
            {"params": model.sentiment_head.parameters(), "lr": learning_rate * 2.0},
            {"params": model.rationale_head.parameters(), "lr": learning_rate * 2.0},
        ]

        optimizer = AdamW(opt_params, weight_decay=0.01)

        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup),
            num_training_steps=total_steps
        )

        loss_fn_sent = nn.CrossEntropyLoss(label_smoothing=0.1)
        loss_fn_rat = nn.CosineEmbeddingLoss()

        best_val_f1 = -1.0

        for epoch in range(EPOCHS):
            model.train()

            for batch in tqdm(
                train_loader,
                desc=f"{config_name} | Seed {seed} | Lam {lambda_val} | Ep {epoch + 1}",
                leave=False
            ):
                optimizer.zero_grad()

                with torch.no_grad():
                    target_rat = teacher_model.encode(
                        batch["rationale"],
                        convert_to_tensor=True,
                        device="cpu"
                    ).to(DEVICE).detach().clone()

                out, pred_rat = model(
                    batch["input_ids"].to(DEVICE),
                    batch["attention_mask"].to(DEVICE)
                )

                l_sent = loss_fn_sent(
                    out.logits,
                    batch["labels"].to(DEVICE).long()
                )

                l_rat = loss_fn_rat(
                    pred_rat,
                    target_rat,
                    torch.ones(
                        pred_rat.size(0),
                        device=DEVICE,
                        dtype=torch.float
                    )
                )

                total_loss = l_sent + (lambda_val * l_rat)
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            model.eval()
            v_preds, v_labels = [], []

            with torch.no_grad():
                for batch in val_loader:
                    out, _ = model(
                        batch["input_ids"].to(DEVICE),
                        batch["attention_mask"].to(DEVICE)
                    )

                    v_preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
                    v_labels.extend(batch["labels"].long().numpy())

            v_f1 = f1_score(v_labels, v_preds, average="macro")

            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                torch.save(model.state_dict(), model_save_path)

        model.load_state_dict(
            torch.load(model_save_path, map_location=DEVICE, weights_only=True)
        )

    model.eval()

    v_preds, v_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            out, _ = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE)
            )

            v_preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            v_labels.extend(batch["labels"].long().numpy())

    v_f1 = f1_score(v_labels, v_preds, average="macro")
    v_acc = accuracy_score(v_labels, v_preds)

    g_preds = get_preds(model, df_golden["text"], is_mtl=False)
    g_f1 = f1_score(df_golden["label"], g_preds, average="macro")
    g_acc = accuracy_score(df_golden["label"], g_preds)

    if search_lambda_mode:
        t_f1, t_acc = np.nan, np.nan
    else:
        t_preds = get_preds(model, df_test["text"], is_mtl=False)
        t_f1 = f1_score(df_test["label"], t_preds, average="macro")
        t_acc = accuracy_score(df_test["label"], t_preds)

    del model, teacher_model
    torch.cuda.empty_cache()
    gc.collect()

    return (v_f1, v_acc), (t_f1, t_acc), (g_f1, g_acc)


def run_lambda_search(dataframes):
    print("Running RS Lambda Search with DEFAULT hyperparameters")

    active_lambdas = LAMBDAS
    active_seeds = SEEDS
    config_name = "Default"
    config_values = CONFIGS["Default"]

    results = {
        lam: {
            "val_f1": [], "val_acc": [],
            "gold_f1": [], "gold_acc": [],
            "score": []
        }
        for lam in active_lambdas
    }

    for lam in active_lambdas:
        print(f"\nRunning RS | Lambda: {lam} | Lambda Search\n{'=' * 50}")

        for seed in active_seeds:
            (v_f1, v_acc), (_, _), (g_f1, g_acc) = train_single_seed_rs(
                seed,
                lam,
                dataframes,
                config_name,
                config_values
            )

            score = (0.4 * v_f1) + (0.6 * g_f1)

            results[lam]["val_f1"].append(v_f1)
            results[lam]["val_acc"].append(v_acc)
            results[lam]["gold_f1"].append(g_f1)
            results[lam]["gold_acc"].append(g_acc)
            results[lam]["score"].append(score)

            print(f"Seed {seed:3} | Val F1: {v_f1:.4f}")

    print("\n" + "=" * 80)
    print("FINAL RS LAMBDA SEARCH SUMMARY")
    print("=" * 80)

    lambda_scores = {}

    for lam in active_lambdas:
        res = results[lam]

        avg_vf1, std_vf1 = np.mean(res["val_f1"]), np.std(res["val_f1"], ddof=1)
        avg_gf1, std_gf1 = np.mean(res["gold_f1"]), np.std(res["gold_f1"], ddof=1)
        avg_score, std_score = np.mean(res["score"]), np.std(res["score"], ddof=1)
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
    active_seeds = SEEDS
    all_results = {}

    final_lambdas = {
        "Default": [0.1],
        "Tuned": [0.1],
        "Refined": [0.1]
    }

    for config_name, config_values in CONFIGS.items():
        lambdas_to_run = final_lambdas[config_name]

        for lambda_val in lambdas_to_run:
            print(
                f"\nRunning RS | Lambda: {lambda_val} | {config_name} Hyperparameter\n"
                f"{'=' * 50}"
            )

            results = {
                "val_f1": [], "val_acc": [],
                "test_f1": [], "test_acc": [],
                "gold_f1": [], "gold_acc": []
            }

            for seed in active_seeds:
                (v_f1, v_acc), (t_f1, t_acc), (g_f1, g_acc) = train_single_seed_rs(
                    seed,
                    lambda_val,
                    dataframes,
                    config_name,
                    config_values
                )

                results["val_f1"].append(v_f1)
                results["val_acc"].append(v_acc)
                results["test_f1"].append(t_f1)
                results["test_acc"].append(t_acc)
                results["gold_f1"].append(g_f1)
                results["gold_acc"].append(g_acc)

                print(
                    f"Seed {seed:3} | "
                    f"Val F1: {v_f1:.4f} | "
                    f"Test F1: {t_f1:.4f} | "
                    f"Gold F1: {g_f1:.4f}"
                )

            all_results[(config_name, lambda_val)] = results

    print("\n" + "=" * 80)
    print("FINAL RS REPORT SUMMARY")
    print("=" * 80)

    for (config_name, lambda_val), res in all_results.items():
        print(
            f"{config_name} | Lambda: {lambda_val}: "
            f"Val F1: {np.mean(res['val_f1']):.4f} (±{np.std(res['val_f1'], ddof=1):.4f}) | "
            f"Test F1: {np.mean(res['test_f1']):.4f} (±{np.std(res['test_f1'], ddof=1):.4f}) | "
            f"Gold F1: {np.mean(res['gold_f1']):.4f} (±{np.std(res['gold_f1'], ddof=1):.4f}) | "
            f"Val Acc: {np.mean(res['val_acc']):.4f} (±{np.std(res['val_acc'], ddof=1):.4f}) | "
            f"Test Acc: {np.mean(res['test_acc']):.4f} (±{np.std(res['test_acc'], ddof=1):.4f}) | "
            f"Gold Acc: {np.mean(res['gold_acc']):.4f} (±{np.std(res['gold_acc'], ddof=1):.4f})"
        )


if __name__ == "__main__":
    dataframes = load_and_process_data()

    if search_lambda_mode:
        run_lambda_search(dataframes)
    else:
        run_final_default_vs_tuned(dataframes)