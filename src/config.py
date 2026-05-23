import torch
import os

# Hyperparameters
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5

# Experiment Settings
SEEDS = [7, 42, 100]        # 3 seeds for averaging
LAMBDAS = [0.1, 0.15, 0.2, 0.25, 0.3]   # Sarcasm/Rationale loss weights
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_DIR = "data" 
MODEL_DIR = "models/"
OUTPUT_DIR = "outputs/"

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)