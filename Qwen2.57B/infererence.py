from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

merged_model = AutoModelForCausalLM.from_pretrained(
    "./qwen-math-finetuned-merged",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("./qwen-math-finetuned-merged", trust_remote_code=True)
merged_model.eval()

prompt = "<|im_start|>user\nCompute the integral of x.\n<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)

with torch.no_grad():
    outputs = merged_model.generate(
        **inputs,
        max_new_tokens=8192,
        temperature=0.7,
        top_p=0.9
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=False))
