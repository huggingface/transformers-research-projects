import torch
import time

from transformers import LlamaConfig, AutoTokenizer
from transformers import LlamaForCausalLM

K_BITS = 2
V_BITS = 2
GROUP_SIZE = 64
RESIDUAL_LENGTH = 64
PATH_TO_YOUR_SAVE_DIR = './cached_models'

model_name_or_path = 'meta-llama/Llama-2-7b-hf'

config = LlamaConfig.from_pretrained(model_name_or_path)
config.k_bits = K_BITS  # 2/4 bit support for KV Cache
config.v_bits = V_BITS
config.group_size = GROUP_SIZE
config.residual_length = RESIDUAL_LENGTH
config.use_flash = True

config.method = 'squat_post'
config.subspace_dim = 20
config.squat_lambda = 0.001
config.quant_group_size = 64
config.shared_svd = 'true'
config.power_method = True
config.fill_key_quant_with_first_col = 'false'

# Load model
# from models.llama_kivi import LlamaForCausalLM_KIVI
# model = LlamaForCausalLM.from_pretrained(
#     pretrained_model_name_or_path=model_name_or_path,
#     config=config,
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=True,
#     device_map="auto",
# )
from models.llama_squat import LlamaForCausalLM_SQuat
model = LlamaForCausalLM_SQuat.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path,
    config=config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

## Llama-2 tokenizer!
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    use_fast=False,
    trust_remote_code=True,
    tokenizer_type='llama'
)

model.cuda().eval()

# Parameters
prompt_length = 160
output_length = 338
num_repeats = 1

def test_batch_size(bsz):
    """
    Attempt to run the model with the specified batch size.
    Returns:
        success: bool
        peak_mem_gb: float (peak memory usage, in GB)
        avg_time_ms: float (time in ms, averaged if num_repeats > 1)
        tokens_per_sec: float
    """
    try:
        # Prepare data
        context = ['t,' * (prompt_length // 2) for _ in range(bsz)]
        inputs = tokenizer(context, return_tensors="pt").to('cuda')

        # We'll measure total tokens = bsz * (prompt_length + output_length)
        total_tokens = bsz * (prompt_length + output_length) * num_repeats

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_repeats):
                _ = model.generate(
                    **inputs,
                    max_new_tokens=output_length
                )
        torch.cuda.synchronize()
        end_time = time.time()

        elapsed_time_s = (end_time - start_time)  # seconds
        avg_time_ms = (elapsed_time_s / num_repeats) * 1000.0

        # tokens/sec
        tokens_per_sec = total_tokens / elapsed_time_s

        # Peak mem
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        return True, peak_mem_gb, avg_time_ms, tokens_per_sec
    
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False, None, None, None

# Simple linear iteration:

results = []

for bsz in range(32, 9999, 32):
    print(f"Trying batch size = {bsz} ...")
    success, peak_mem, time_ms, tps = test_batch_size(bsz)
    
    if not success:
        print(f"  OOM at batch size = {bsz}, stopping.")
        break
    
    results.append((bsz, peak_mem, time_ms, tps))
    print(f"  Success. "
          f"Peak mem: {peak_mem:.3f} GB, "
          f"Time: {time_ms:.2f} ms, "
          f"Tokens/s: {tps:,.2f}")

print("\nAll successful runs:")
for (bsz, mem_gb, used_ms, tokens_sec) in results:
    print(f"  Batch size = {bsz}, "
          f"Peak mem = {mem_gb:.3f} GB, "
          f"Avg time = {used_ms:.2f} ms, "
          f"Throughput = {tokens_sec:,.2f} tokens/s")
