from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from .utils import TaskType, detect_task_type

try:
    from transformers.integrations.accelerate import AcceleratorConfig
except ImportError:
    AcceleratorConfig = None  # Older Transformers versions won't have this


class DistillationTrainer(Trainer):
    def __init__(
        self,
        model, 
        args: TrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        teacher_model=None,
        is_pretrained=False,
        kd_alpha=0.5,
        temperature=2.0,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            **kwargs
        )

        self.teacher_model = teacher_model
        self.kd_alpha = kd_alpha
        self.temperature = temperature

        # Detect task type
        self.task_type = detect_task_type(model.name_or_path if is_pretrained else model)

        # Setup teacher model
        if self.teacher_model is not None:
            self.teacher_model.to(self.model.device)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

    def shift_tokens_right(self, input_ids, pad_token_id, decoder_start_token_id):
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[:, 1:] = input_ids[:, :-1].clone()
        shifted[:, 0] = decoder_start_token_id
        shifted.masked_fill_(shifted == -100, pad_token_id)
        return shifted

    def prepare_labels(self, inputs):
        """
        Prepare labels depending on task type.
        Ensures causal LM and seq2seq LM have properly shifted labels.
        """
        if "labels" not in inputs:
            inputs["labels"] = inputs["input_ids"].clone()

        if self.task_type == TaskType.CAUSAL_LM:
            # Optionally shift labels for causal LM if model requires it
            if getattr(self.model.config, "use_cache", False):
                inputs["labels"] = inputs["labels"].clone()
            return inputs["labels"]

        elif self.task_type == TaskType.MLM:
            # Labels for MLM should already have -100 for masked tokens
            return inputs["labels"]

        elif self.task_type == TaskType.SEQ2SEQ_LM:
            if "decoder_input_ids" not in inputs:
                inputs["decoder_input_ids"] = self.shift_tokens_right(
                    inputs["labels"],
                    self.model.config.pad_token_id,
                    self.model.config.decoder_start_token_id
                )
            return inputs["labels"]

        else:
            # Fallback
            return inputs["labels"]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = self.prepare_labels(inputs)
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )

        # Knowledge Distillation loss
        if self.teacher_model is not None and model.training:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits

            kd_loss = F.kl_div(
                input=F.log_softmax(student_logits / self.temperature, dim=-1),
                target=F.softmax(teacher_logits / self.temperature, dim=-1),
                reduction="batchmean"
            ) * (self.temperature ** 2)

            loss = self.kd_alpha * kd_loss + (1.0 - self.kd_alpha) * lm_loss
        else:
            loss = lm_loss

        return (loss, student_outputs) if return_outputs else loss


def DistillTrainer(
    teacher_model,
    student_model,
    train_dataset,
    tokenizer,
    training_args: TrainingArguments,
    is_pretrained=False,
    kd_alpha=0.5,
    temperature=2.0
):
    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        kd_alpha=kd_alpha,
        temperature=temperature
    )
    return trainer
