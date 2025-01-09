#!/usr/bin/env python

import deepspeed
import torch
import argparse
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed inference for multiple prompts")
    
    # Required / commonly changed args
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="./qwen-math-finetuned-merged",
        help="Path to your merged or fine-tuned Qwen model"
    )
    parser.add_argument(
        "--prompts_file", 
        type=str, 
        default="prompts.json",
        help="Path to JSON file containing a list of plain text prompts"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="prompts_output.json",
        help="Path to JSON file where outputs will be saved"
    )
    
    # Generation hyperparameters
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--local_rank", type=int, default=0,
                    help="Local rank passed by DeepSpeed/torchrun")

    # DeepSpeed inference config
    parser.add_argument("--mp_size", type=int, default=4, help="Tensor model parallelism (number of GPUs)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16","bfloat16"], help="Inference precision")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # -----------------------
    # 1) Load Tokenizer
    # -----------------------
    print(f"[INFO] Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # -----------------------
    # 2) Load HF Model
    # -----------------------
    print(f"[INFO] Loading model from: {args.model_path}")
    
    # If you have a *single* merged checkpoint, you can load like a normal HF model:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.dtype == "float16" else torch.bfloat16
    )

    # -----------------------
    # 3) Initialize DeepSpeed Inference
    # -----------------------
    print("[INFO] Initializing DeepSpeed Inference Engine...")
    engine = deepspeed.init_inference(
        model=model,
        mp_size=args.mp_size,  
        dtype=torch.float16 if args.dtype == "float16" else torch.bfloat16,
        replace_method="auto",
        replace_with_kernel_inject=True
    )
    
    model_engine = engine.module
    model_engine.eval()

    # -----------------------
    # 4) Load Prompts (List of Strings)
    # -----------------------
    if not os.path.exists(args.prompts_file):
        raise FileNotFoundError(f"Could not find prompts file: {args.prompts_file}")
    
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
    
    # Expecting `prompts_data` to be a list of strings:
    # [ "Explain the Big-O notation of bubble sort", "What's the integral of x dx?", ... ]

    if not isinstance(prompts_data, list):
        raise ValueError("prompts.json must contain a list of plain text strings.")

    # -----------------------
    # 5) Prepare Results Array
    # -----------------------
    # We'll build up a list of { prompt, output } dictionaries.
    # If a previous run partially wrote to output_file, we can reload to resume.
    results = []
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as out_f:
            try:
                existing_data = json.load(out_f)
                if isinstance(existing_data, list):
                    results = existing_data
            except:
                pass

    # We can track which prompts we've already processed by their index
    processed_indices = set(range(len(results)))

    # -----------------------
    # 6) Inference on Each Prompt
    # -----------------------
    for idx, raw_prompt_text in enumerate(prompts_data):
        if idx < len(results):
            print(f"[INFO] Skipping prompt #{idx} because results already exist in {args.output_file}")
            continue

        # (A) Format Prompt in Qwen Conversation Style
        prompt_text = (
            f"<|im_start|>user\n{raw_prompt_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        print(f"\n[INFO] Generating output for prompt #{idx+1}/{len(prompts_data)}")
        print(f"Raw prompt: {raw_prompt_text[:80]}{'...' if len(raw_prompt_text) > 80 else ''}")

        # (B) Tokenize & Move Tensors
        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(engine.local_rank) for k, v in inputs.items()}
        
        # (C) Generate
        with torch.no_grad():
            outputs = model_engine.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True
            )
        
        # (D) Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # (E) Store result
        results.append({
            "prompt": raw_prompt_text,  # Original user prompt (plain text)
            "output": generated_text
        })
        
        # (F) Write partial results to disk
        with open(args.output_file, "w", encoding="utf-8") as out_f:
            json.dump(results, out_f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Output (truncated): {generated_text[:100]}{'...' if len(generated_text) > 100 else ''}")

    print(f"\n[INFO] All prompts processed. Final results in {args.output_file}")

if __name__ == "__main__":
    main()
