import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model():
    # Load base model and tokenizer
    base_model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=True  # Enable KV cache
    )
    
    # Load the fine-tuned LoRA weights
    model = PeftModel.from_pretrained(
        base_model,
        "./qwen-math-finetuned-final",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    model.eval()  # Ensure model is in evaluation mode
    return model, tokenizer

def generate_response(model, tokenizer, question, max_length=4096, temperature=0.7):
    # Format the input with the proper instruction template
    prompt = (
        "You are a helpful math assistant. Solve the following problem step by step, "
        "showing your work clearly and explaining each step.\n\n"
        f"Problem: {question}\n\n"
        "Solution (Output final answer in boxed form):"
    )
    
    # Add the chat template
    prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response with KV cache enabled
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.3,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    # Decode and clean up response
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract just the assistant's response
    response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
    
    return response

# Example usage
if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # Example questions
    questions = [
        "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."
     
    ]
    
    # Generate responses
    for question in questions:
        print("\nQuestion:", question)
        print("\nAnswer:", generate_response(model, tokenizer, question))
        print("-" * 50)