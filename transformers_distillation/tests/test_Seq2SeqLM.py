import sys
import os
import pytest
from transformers import AutoTokenizer, TrainingArguments
from datasets import Dataset
from transformers_distillation.models import load_teacher, load_student
from transformers_distillation import DistillTrainer

# MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
model_names =[
        "google/flan-t5-small",
        "google-t5/t5-small"
]

@pytest.mark.parametrize("model_name", model_names)
def test_distillation_runs(model_name):
    print(F"\nThe {model_name} Is Currently Being Tested")
    teacher = load_teacher(model_name_or_path=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    student = load_student(
        model_name_or_path=model_name,
        from_scratch=True,
        n_layers=4,
        n_heads=4,
        n_embd=256,
        is_pretrained=False
    )

    sources = [
        "Translate English to French: Hello world!",
        "Translate English to French: The quick brown fox jumps over the lazy dog.",
        "Translate English to French: Artificial intelligence is transforming industries.",
        "Translate English to French: Once upon a time, there was a curious developer.",
        "Translate English to French: PyTorch makes deep learning both fun and powerful."
    ]
    
    targets = [
        "Bonjour le monde!",
        "Le renard brun rapide saute par-dessus le chien paresseux.",
        "L'intelligence artificielle transforme les industries.",
        "Il était une fois un développeur curieux.",
        "PyTorch rend l'apprentissage profond à la fois amusant et puissant."
    ]
    
    dataset = Dataset.from_dict({"source": sources, "target": targets})
    
    def tokenize(batch):
        # Tokenize encoder inputs
        model_inputs = tokenizer(
            batch["source"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )
    
        # Tokenize decoder targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target"],
                max_length=128,
                truncation=True,
                padding="max_length"
            )["input_ids"]
    
        model_inputs["labels"] = labels
        return model_inputs

    
    tokenized_dataset = dataset.map(tokenize, remove_columns=["source", "target"])
    eval_dataset = tokenized_dataset.select(range(1))

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

    trainer.train()

    results = trainer.evaluate(eval_dataset = eval_dataset)
    print("Evaluation results:", results)
