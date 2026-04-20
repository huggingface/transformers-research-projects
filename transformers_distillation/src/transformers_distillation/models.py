from typing import Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig
)
from .utils import detect_task_type, TaskType


def _freez_eval(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def load_teacher(model_name_or_path: str, quant_config: Optional[object] = None, device_map: str = "auto"):

    #DETECTING TASK AUTOMATICALLY FOR LM
    task = detect_task_type(model_name_or_path)
    common_kargs = {}
    if quant_config is not None:
        common_kargs["quantization_config"] = quant_config
        common_kargs["device_map"] = device_map

    #CausalLM Model
    if task == TaskType.CAUSAL_LM:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **common_kargs)
    
    #Seq2SeqLM Model
    elif task ==TaskType.SEQ2SEQ_LM:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **common_kargs)

    #MLM Model
    elif task == TaskType.MLM:
        model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, **common_kargs)

    #Fallback For CausalLM
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **common_kargs)

    #Freezing Model To Make The Teacher Model Training False
    return _freez_eval(model)

def load_student(
        model_name_or_path: str,
        is_pretrained: bool = False,
        n_layers: int = None,
        n_heads: int = None,
        num_key_value_heads: int = None,  # If None, will match n_heads
        n_embd: int = None,
        from_scratch: bool = True,
        explicit_task: Optional[TaskType] = None
):
    # Detect Task Or Take Explicit Task
    task = explicit_task or detect_task_type(model_name_or_path)

    if from_scratch:
        cfg = AutoConfig.from_pretrained(model_name_or_path)

        # NUM LAYERS
        if hasattr(cfg, "n_layers") and n_layers is not None:
            cfg.n_layers = n_layers
        if hasattr(cfg, "num_hidden_layers") and n_layers is not None:
            cfg.num_hidden_layers = n_layers

        # NUM HEADS
        if hasattr(cfg, "n_heads") and n_heads is not None:
            cfg.n_heads = n_heads
        if hasattr(cfg, "num_attention_heads") and n_heads is not None:
            cfg.num_attention_heads = n_heads

        # FIX: Ensure num_key_value_heads matches attention heads if not explicitly set
        if hasattr(cfg, "num_key_value_heads"):
            cfg.num_key_value_heads = (
                num_key_value_heads if num_key_value_heads is not None
                else getattr(cfg, "num_attention_heads", n_heads or cfg.num_key_value_heads)
            )

        # HIDDEN SIZE
        if hasattr(cfg, "n_embd") and n_embd is not None:
            cfg.n_embd = n_embd
        if hasattr(cfg, "hidden_dim") and n_embd is not None:
            cfg.hidden_dim = n_embd

        if task == TaskType.CAUSAL_LM:
            return AutoModelForCausalLM.from_config(cfg)
        if task == TaskType.SEQ2SEQ_LM:
            return AutoModelForSeq2SeqLM.from_config(cfg)
        if task == TaskType.MLM:
            return AutoModelForMaskedLM.from_config(cfg)
        return AutoModelForCausalLM.from_config(cfg)

    else:
        if task == TaskType.CAUSAL_LM:
            return AutoModelForCausalLM.from_pretrained(model_name_or_path)
        if task == TaskType.SEQ2SEQ_LM:
            return AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        if task == TaskType.MLM:
            return AutoModelForMaskedLM.from_pretrained(model_name_or_path)