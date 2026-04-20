"""
Knowledge Distillation with hf_distiller (Python Script)

This script demonstrates:
1. Loading a teacher model from Hugging Face Hub
2. Creating a smaller student model
3. Preparing a toy dataset
4. Training the student using knowledge distillation

Run:
    pip install -r requirements.txt
    python distill_demo.py
"""

import sys
import os
from transformers import AutoTokenizer, TrainingArguments
from datasets import Dataset
from transformers_distillation.models import load_teacher, load_student
from transformers_distillation import DistillTrainer

# -------------------------------------------------------------------------
# Step 1 — Ensure src/ is in Python path
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Step 2 — Select teacher model
# -------------------------------------------------------------------------
MODEL_NAME = "google-bert/bert-base-uncased"

# -------------------------------------------------------------------------
# Step 3 — Load Teacher & Tokenizer
# -------------------------------------------------------------------------
teacher = load_teacher(model_name_or_path=MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------------------------------------------------------------
# Step 4 — Create Student model (smaller)
# -------------------------------------------------------------------------
student = load_student(
    model_name_or_path=MODEL_NAME,
    from_scratch=True,
    n_layers=4,
    n_heads=4,
    n_embd=256,
    is_pretrained=False
)

# -------------------------------------------------------------------------
# Step 5 — Prepare Dataset
# -------------------------------------------------------------------------
texts = [
    "Hello world!",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming industries.",
    "Once upon a time, there was a curious developer.",
    "PyTorch makes deep learning both fun and powerful."
]
dataset = Dataset.from_dict({"text": texts})

def tokenize(batch):
    return tokenizer(batch["text"], max_length=128, padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])

# -------------------------------------------------------------------------
# Step 6 — Training Arguments
# -------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./student-llm",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    save_total_limit=5,
    report_to="none",
    lr_scheduler_type="cosine",
    warmup_steps=500,
)

# -------------------------------------------------------------------------
# Step 7 — Initialize Distillation Trainer
# -------------------------------------------------------------------------
trainer = DistillTrainer(
    teacher_model=teacher,
    student_model=student,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    training_args=training_args,
    kd_alpha=0.5,
    temperature=2.0,
    is_pretrained=False
)

# -------------------------------------------------------------------------
# Step 8 — Train
# -------------------------------------------------------------------------
trainer.train()

# -------------------------------------------------------------------------
<<<<<<< HEAD
# Optional: Evaluate Student (Requires Eval Dataset)
# -------------------------------------------------------------------------
# results = trainer.evaluate()
# print("Evaluation results:", results)
=======
# Optional: Evaluate Student
# -------------------------------------------------------------------------
results = trainer.evaluate()
print("Evaluation results:", results)
>>>>>>> origin/main
