from enum import Enum
from transformers import AutoConfig, PreTrainedModel

class TaskType(str, Enum):
    CAUSAL_LM = "causal_lm"
    SEQ2SEQ_LM = "seq2seq_lm"
    MLM = "mlm"

def detect_task_type(model_or_path) -> TaskType:
    # If it's already a model, use its config
    if isinstance(model_or_path, PreTrainedModel):
        cfg = model_or_path.config
    else:
        cfg = AutoConfig.from_pretrained(model_or_path)

    archs = (cfg.architectures or [])
    model_type = getattr(cfg, "model_type", "").lower()

    if any("ForCausalLM" in a for a in archs) or model_type in {"gpt2", "llama", "mistral", "gpt_neo", "phi"}:
        return TaskType.CAUSAL_LM
    
    if any("ForConditionalGeneration" in a for a in archs) or model_type in {"t5", "flan-t5", "ul", "mt5", "mbart"}:
        return TaskType.SEQ2SEQ_LM
    
    if any("ForMaskedLM" in a for a in archs) or model_type in {"bert", "roberta", "albert", "electra"}:
        return TaskType.MLM
    
    return TaskType.CAUSAL_LM
