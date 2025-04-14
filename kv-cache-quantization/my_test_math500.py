import warnings
warnings.filterwarnings("ignore")
import torch
import random
from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
from matheval import math_equal, memoized_canonical_form, extract
import os

system_prompts = {
    'math500': (
        "Solve the following math problem efficiently and clearly:\n\n"
        "- For simple problems (2 steps or fewer):\n"
        "Provide a concise solution with minimal explanation.\n\n"
        "- For complex problems (3 steps or more):\n"
        "Use this step-by-step format:\n\n"
        "## Step 1: [Concise description]\n"
        "[Brief explanation and calculations]\n\n"
        "## Step 2: [Concise description]\n"
        "[Brief explanation and calculations]\n\n"
        "...\n\n"
        "Regardless of the approach, always conclude with:\n\n"
        "Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\n"
        "Where [answer] is just the final number or expression that solves the problem."
    ),
    'math500_v2': ( # for R1 distilled
        r"Please reason step by step, and put your final answer within \boxed{}."
    )
}

@torch.no_grad()
def main():
    # For reproducibility
    random.seed(0)
    torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description='LLaMA model with SQuat')
    parser.add_argument('--dataset', type=str, default='math500', help='Dataset to use')
    parser.add_argument('--template', type=str, default='math500', help='Template to use')
    parser.add_argument('--k_bits', type=int, default=2, help='K bits (2 or 4)')
    parser.add_argument('--v_bits', type=int, default=2, help='V bits (2 or 4)') 
    parser.add_argument('--group_size', type=int, default=32, help='Group size for quantization')
    parser.add_argument('--residual_length', type=int, default=32, help='Number of recent fp16 tokens')
    parser.add_argument('--residual_length_prefill', type=int, default=32, help='Number of recent fp16 tokens')
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Path to pretrained model')
    parser.add_argument('--prompt', type=str, default="Repeat the following sentence: 'The capital of California is Beijing.'", help='Input prompt')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='Maximum number of new tokens to generate')
    parser.add_argument('--method', type=str, default='squat_pre', help='Method to use')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--subspace_dim', type=int, default=20, help='Subspace dimension')
    parser.add_argument('--squat_lambda', type=float, default=0.1, help='Lambda for Lagrangian')
    parser.add_argument('--quant_group_size', type=int, default=32, help='Quantization group size')
    parser.add_argument("--output_file", type=str, default="output.txt")
    parser.add_argument("--output_acc_file", type=str, default="output_acc.txt")
    parser.add_argument("--index_file", type=str, default="idx.txt")
    parser.add_argument("--batch_size", type=int, default=1, help='Batch size')
    parser.add_argument("--padding_side", type=str, default="left", help='Padding side')
    parser.add_argument("--shared_svd", type=str, default="false", help='Shared SVD')
    parser.add_argument("--fill_key_quant_with_first_col", type=str, default="true")
    parser.add_argument("--squat_query_subspace_prerope", type=str, default="after")
    parser.add_argument("--save_config_only", action="store_true")
    parser.add_argument("--save_config_path", type=str, default="config.json")
    parser.add_argument("--force_use_flash", action="store_true")
    parser.add_argument("--cuda_bmm_implementation", type=str, default="cos_sin", help='Cuda BMM implementation')
    args = parser.parse_args()

    if "mistral" in args.model_path.lower():
        from transformers import MistralConfig
        config = MistralConfig.from_pretrained(args.model_path)
    else:
        from transformers import LlamaConfig
        config = LlamaConfig.from_pretrained(args.model_path)
    config.shared_svd = args.shared_svd
    config.k_bits = args.k_bits
    config.v_bits = args.v_bits
    config.group_size = args.group_size
    config.residual_length = args.residual_length
    config.use_flash = True
    config.attn_implementation = "flash_attention_2"
    config.force_use_flash = args.force_use_flash
    config.method = args.method
    config.subspace_dim = args.subspace_dim
    config.squat_lambda = args.squat_lambda
    config.quant_group_size = args.quant_group_size
    config.squat_query_subspace_prerope = args.squat_query_subspace_prerope
    config.fill_key_quant_with_first_col = args.fill_key_quant_with_first_col
    config.cuda_bmm_implementation = args.cuda_bmm_implementation
    config.config_str = f"K bit: {config.k_bits}, V bits: {config.v_bits}, group_size: {config.group_size}, residual_length: {config.residual_length}, method: {config.method}, subspace_dim: {config.subspace_dim}, squat_lambda: {config.squat_lambda}, quant_group_size: {config.quant_group_size}, shared_svd: {config.shared_svd}, cuda_bmm_implementation: {config.cuda_bmm_implementation}"
    if config.force_use_flash and args.batch_size > 1:
        warnings.warn("Flash attention currently set attention_mask to None and will not work with batch size > 1")
    
    if args.save_config_only: 
        config.to_json_file(args.save_config_path)
        exit()

    args.output_file = f"{args.output_file.replace('.txt', '')}-{args.dataset}-{args.num_samples}-{args.method}_{args.k_bits}_{args.v_bits}_{args.group_size}_{args.residual_length}_{args.squat_lambda}_{args.quant_group_size}_{args.subspace_dim}_{args.shared_svd}_{args.cuda_bmm_implementation}.txt"

    if "mistral" in args.model_path.lower():
        if "kivi" in args.method:
            from models.mistral_kivi import MistralForCausalLM_KIVI
            model = MistralForCausalLM_KIVI.from_pretrained(
                pretrained_model_name_or_path=args.model_path,
                config=config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            ).cuda()
        elif "squat" in args.method:
            from models.mistral_squat import MistralForCausalLM_SQuat
            model = MistralForCausalLM_SQuat.from_pretrained(
                pretrained_model_name_or_path=args.model_path,
                config=config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            ).cuda()
        elif args.method in ["fp16", "baseline"]:
            from transformers import MistralForCausalLM
            model = MistralForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.model_path,
                config=config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).cuda()
        else:
            raise NotImplementedError(f"Method {args.method} not implemented")
    else:
        if "kivi" in args.method:
            from models.llama_kivi import LlamaForCausalLM_KIVI
            model = LlamaForCausalLM_KIVI.from_pretrained(
                pretrained_model_name_or_path=args.model_path,
                config=config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            ).cuda()
        elif "squat" in args.method:
            from models.llama_squat import LlamaForCausalLM_SQuat
            model = LlamaForCausalLM_SQuat.from_pretrained(
                pretrained_model_name_or_path=args.model_path,
                config=config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            ).cuda()
        elif args.method in ["fp16", "baseline"]:
            from transformers import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.model_path,
                config=config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).cuda()
        else:
            raise NotImplementedError(f"Method {args.method} not implemented")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        padding_side=args.padding_side,
        trust_remote_code=True)

    # if index file exists, read the indices from the file
    if os.path.exists(args.index_file):
        with open(args.index_file, "r") as f:
            selected_idx = [int(line.strip()) for line in f.readlines()]
    else:
        # selected_idx = [1, 5, 7, 9, 10, 11, 12, 14, 15, 17, 18, 19, 21, 22, 23, 24, 25, 26, 31, 32, 33, 34, 35, 36, 39, 41, 43, 46, 48, 50, 58, 60, 62, 63, 64, 66, 68, 69, 71, 72, 78, 80, 81, 82, 83, 84, 88, 89, 90, 91, 92, 94, 96, 97, 98]
        selected_idx = random.sample(range(500), args.num_samples)
        selected_idx.sort()
        with open(args.index_file, "w") as f:
            for idx in selected_idx:
                f.write(f"{idx}\n")

    print(f"Selected indices: {selected_idx}")

    if args.dataset == 'math500' or args.dataset == 'math500_v2':
        dataset = load_dataset('HuggingFaceH4/MATH-500', split="test")
        ds = dataset.select(selected_idx)
    else:
        raise NotImplementedError("Only math500 is supported for now")

    correct = []

    batch_size = args.batch_size  # Can be adjusted based on GPU memory
    correct = []
    
    for batch_start in range(0, len(ds), batch_size):
        batch_end = min(batch_start + batch_size, len(ds))
        batch_inputs = []
        batch_indices = []
        
        # Prepare batch
        for idx in range(batch_start, batch_end):
            sentence_idx = selected_idx[idx]
            input_text = ds[idx]
            # Get length of template tokens without content

            # Generate actual prompt
            messages = [{'role': 'system', 'content': system_prompts[args.dataset]}, 
                       {'role': 'user', 'content': input_text['problem']}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt = prompt.replace("Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n", "")
            batch_inputs.append(prompt)
            batch_indices.append((sentence_idx, input_text))
            
        # Tokenize batch
        if tokenizer.unk_token:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif tokenizer.eos_token:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        inputs = tokenizer(batch_inputs, return_tensors="pt", padding='longest', add_special_tokens=False).to('cuda')

        # Generate for batch
        outputs = model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        # Process batch outputs
        for i, output in enumerate(outputs):
            sentence_idx, input_text = batch_indices[i]
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            res = math_equal(memoized_canonical_form(input_text['answer']), 
                           memoized_canonical_form(extract(generated_text, 'math')), timeout=False)
            
            print("-----------------------------------------")
            print(f"Generated text: {generated_text}")
            print(f"Ground truth: {input_text['answer']}")
            print(f"Is correct: {res}")
            print(f"^sentence_idx: {sentence_idx}")
            print("-----------------------------------------")

            if args.output_file:
                with open(args.output_file, "a") as f:
                    f.write("-----------------------------------------\n")
                    f.write(f"sentence_idx: {sentence_idx}\n")
                    f.write(generated_text + "\n\n")
                    f.write(f"Ground truth: {input_text['answer']}\n")
                    f.write(f"Is correct: {res}\n\n")
                    correct.append(res)

    if args.output_file:
        with open(args.output_file, "a") as f:
            f.write("=========================================\n")
            f.write(f"Accuracy: {sum(correct) / len(correct)}\n")
    
    print(f"Accuracy: {sum(correct) / len(correct)}")
    if args.output_acc_file:
        # write config_str and accuracy to the file
        with open(args.output_acc_file, "a") as f:
            f.write("=========================================\n")
            f.write(f"Config: {config.config_str}\n")
            f.write(f"Accuracy: {sum(correct) / len(correct)}\n")


if __name__ == "__main__":
    main()
