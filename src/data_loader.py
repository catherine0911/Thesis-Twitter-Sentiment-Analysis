import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import Dataset as HFDataset
from src.config import MODEL_NAME, MAX_LEN, DATA_DIR, BATCH_SIZE
from src.preprocessing import preprocess_sarcasm

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding="max_length", max_length=MAX_LEN)
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_golden_set(filepath):
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    data = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                data.append({"text": parts[2], "label": label_map[parts[1].lower()]})
    return pd.DataFrame(data)

def load_and_process_data(save_to_processed=True):
    """
    1. Loads from data/raw/
    2. Removes Leakage
    3. Preprocesses Sarcasm
    4. Saves to data/processed/
    """
    raw_path = os.path.join(DATA_DIR, "raw/")
    proc_path = os.path.join(DATA_DIR, "processed/")

    # Load raw data
    df_sent_train = pd.read_csv(os.path.join(raw_path, "sentiment_train.csv"))
    df_sent_val = pd.read_csv(os.path.join(raw_path, "sentiment_validation.csv"))
    df_sent_test = pd.read_csv(os.path.join(raw_path, "sentiment_test.csv"))
    # Load Golden dataset
    df_golden = load_golden_set(os.path.join(raw_path, "twitter-2014sarcasm-A.txt"))
    # Load iSarcasm data
    df_isarc_train = pd.read_csv(os.path.join(raw_path, "iSarcasm_Train.csv"))
    df_isarc_train = df_isarc_train[["tweet", "sarcastic"]].rename(
        columns={"tweet": "text", "sarcastic": "label"}
    )
    df_isarc_test = pd.read_csv(os.path.join(raw_path, "iSarcasm_Test.csv"))
    df_isarc_test = df_isarc_test[["text", "sarcastic"]].rename(
        columns={"sarcastic": "label"}
    )

    # Combine train and test set of iSarcasm (no need to evaluate on test set)
    df_sarc = pd.concat([df_isarc_train, df_isarc_test], ignore_index=True)

    # Remove nulls and duplicates
    df_sarc = df_sarc.dropna(subset=["text"]).drop_duplicates(subset=["text"])
    df_sarc["text"] = df_sarc["text"].astype(str)

    # Remove data leakage (remove tweets of golden set which also appear in sentiment training data)
    df_golden = load_golden_set(os.path.join(raw_path, "twitter-2014sarcasm-A.txt"))
    leaked_texts = set(df_golden["text"].tolist())
    df_sent_train = df_sent_train[~df_sent_train["text"].isin(leaked_texts)]

    print(f"Final Sarcasm Task Size: {len(df_sarc)}")
    print(f"Final Sentiment Task Size: {len(df_sent_train)}")
    # Preprocess sarcasm data
    df_sarc['text'] = df_sarc['text'].astype(str).apply(preprocess_sarcasm)

    # Save to Processed folder
    if save_to_processed:
        if not os.path.exists(proc_path): os.makedirs(proc_path)
        df_sent_train.to_csv(f"{proc_path}sentiment_train_clean.csv", index=False)
        df_sent_val.to_csv(f"{proc_path}sentiment_val_clean.csv", index=False)
        df_sent_test.to_csv(f"{proc_path}sentiment_test_clean.csv", index=False)
        df_sarc.to_csv(f"{proc_path}sarcasm_train_clean.csv", index=False)
        df_golden.to_csv(f"{proc_path}golden_set_clean.csv", index=False)
        print(f"Processed files saved to {proc_path}")

    return df_sent_train, df_sent_val, df_sent_test, df_sarc, df_golden

def get_hf_datasets(df_train, df_val, df_test):
    """Returns Hugging Face datasets for Baseline Trainer with extra columns removed."""
    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

    datasets = []
    for df in [df_train, df_val, df_test]:
        # create and tokenize
        ds = HFDataset.from_pandas(df).map(tokenize_fn, batched=True).rename_column("label", "labels")
        
        # Keep only columns that the model needs
        keep_cols = ["input_ids", "attention_mask", "labels"]
        cols_to_remove = [col for col in ds.column_names if col not in keep_cols]
        ds = ds.remove_columns(cols_to_remove)
        
        datasets.append(ds)
    return datasets

def get_dataloaders(df_train, df_val, df_test, df_sarc, seed):
    """Returns PyTorch DataLoaders"""
    g = torch.Generator()
    g.manual_seed(seed)

    train_sent_ds = TextDataset(df_train["text"], df_train["label"])
    val_sent_ds = TextDataset(df_val["text"], df_val["label"])
    test_sent_ds = TextDataset(df_test["text"], df_test["label"])
    train_sarc_ds = TextDataset(df_sarc["text"], df_sarc["label"])

    train_sent_loader = DataLoader(train_sent_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_sent_loader = DataLoader(val_sent_ds, batch_size=BATCH_SIZE)
    test_sent_loader = DataLoader(test_sent_ds, batch_size=BATCH_SIZE)
    train_sarc_loader = DataLoader(train_sarc_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)

    return train_sent_loader, val_sent_loader, test_sent_loader, train_sarc_loader