import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

# --- Configuration ---

MODEL_CHECKPOINT = "distilbert-base-uncased"
DATASET_NAME = "squad_v2" 

# --- Load Dataset ---
print(f"[INFO] Loading the {DATASET_NAME} dataset...")

raw_datasets = load_dataset(DATASET_NAME)

# --- Load Tokenizer and Model ---
print(f"[INFO] Loading tokenizer and model for '{MODEL_CHECKPOINT}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"[INFO] Model loaded on: {device}")