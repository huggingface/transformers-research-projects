# 🧪 HF Distiller — Knowledge Distillation for Hugging Face Models

![HF Banner](https://huggingface.co/front/assets/huggingface_logo.svg)

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/huggingface-Dhiraj309-orange)](https://huggingface.co/Dhiraj309)

**HF Distiller** is an **open-source toolkit** for performing **knowledge distillation** on Hugging Face Transformers models. It allows developers to **train smaller, faster student models** from large pre-trained teacher models while maintaining high performance.

---

## 📖 Overview

Knowledge Distillation (KD) compresses a large model into a smaller one by transferring the “knowledge” learned by the teacher to the student. HF Distiller wraps around Hugging Face’s `Trainer` to make KD **accessible, modular, and intuitive**.

**Key Features:**

* ✅ Load any teacher model from Hugging Face Hub
* ✅ Create smaller student models from scratch
* ✅ Supports Hugging Face tokenizers
* ✅ Seamless integration with the `datasets` library
* ✅ Transparent logging and checkpointing
* ✅ Fully compatible with PyTorch and Transformers

---

## 🖼 Architecture

```text
           ┌────────────────────────┐
           │      Teacher Model      │  Pretrained Hugging Face LM
           └────────────┬───────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │ Knowledge Distillation  │  Transfer teacher knowledge + KD loss
           └────────────┬───────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │      Student Model      │  Smaller, efficient model trained from scratch
           └────────────────────────┘
```

---

## ⚡ Installation

```bash
#Install transformers_distilattion (Recommended)
pip install --no-deps git+https://github.com/Dhiraj309/transformers_distillation.git

#OR

# Clone repository
git clone https://github.com/Dhiraj309/transformers_distillation.git
cd transformers_distillation.git

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃 Quick Start

```python
from transformers_distillation.models import load_teacher, load_student
from transformers_distillation.trainer import DistillTrainer
from transformers import AutoTokenizer, TrainingArguments
from datasets import Dataset

# Example dataset
dataset = Dataset.from_dict({"text": ["Hello world!", "AI is amazing."]})

# Load teacher
teacher = load_teacher("google-bert/bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

# Create student model
student = load_student(
    model_name_or_path="google-bert/bert-base-uncased",
    from_scratch=True,
    n_layers=4,
    n_heads=4,
    n_embd=256,
    is_pretrained=False
)

# Tokenize
def tokenize(batch):
    return tokenizer(batch["text"], max_length=128, padding=True, truncation=True)

tokenized = dataset.map(tokenize, remove_columns=["text"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./student-llm",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=2e-4,
    report_to="none"
)

# Train student with KD
trainer = DistillTrainer(
    teacher_model=teacher,
    student_model=student,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    training_args=training_args,
    kd_alpha=0.5,
    temperature=2.0
)
trainer.train()
```

---

## 📂 Project Status

| Stage                | Status         |
| -------------------- | -------------- |
| Core Development     | ✅ Complete     |
| Documentation        | ✅ Complete     |
| Community Feedback   | 🚧 In Progress |
| Tutorials & Examples | 🚧 In Progress |

---

## 🤝 Collaboration

We welcome contributions from the community, including:

* Pull requests for new KD strategies
* Bug reports and feature requests
* Tutorials and example scripts
* Optimization for faster student training

🔗 GitHub: [Dhiraj309](https://github.com/Dhiraj309)
🔗 Hugging Face: [dignity045](https://huggingface.co/dignity045)

---

## 📜 License

Released under the **MIT License** — free to use, modify, and distribute. See [LICENSE](LICENSE) for full terms.

