import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
import json
import os

# ------------------------- #
# 1. DEEPSPEED CONFIG SETUP #
# ------------------------- #
# You can define this inline as a Python dictionary or load it from a JSON file.
# Below is a minimal example; adjust stage, offload, or other parameters as needed.
deepspeed_config = {
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "overlap_comm": True,
        "contiguous_gradients": True
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_batch_size": 8,             # Global batch size across all GPUs
    "steps_per_print": 2000,
    "wall_clock_breakdown": False
}

# Add a parameter at the top with your configurations
RESUME_FROM_CHECKPOINT = False  # Set this to False to ignore checkpoints

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Check if final model exists and load it instead of base model
final_model_path = "./qwen-math-finetuned-final"
if os.path.exists(final_model_path):
    print(f"Loading from previous final model: {final_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        final_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
        # IMPORTANT: Usually do not set device_map when using DeepSpeed.
        # device_map="auto",
    )
else:
    print("No final model found. Loading base model.")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
        # device_map="auto",
    )

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=1024,  # Reduced rank
    lora_alpha=4096,
    lora_dropout=0.1,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention modules
        "gate_proj", "up_proj",                  # Reduced MLP/FFN modules
        # "down_proj", "w_proj" intentionally removed
    ],
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Print parameter statistics
trainable_params = 0
all_param = 0
for _, param in model.named_parameters():
    num_params = param.numel()
    all_param += num_params
    if param.requires_grad:
        trainable_params += num_params
print(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} "
      f"|| trainable%: {100 * trainable_params / all_param:.2f}%")

# ------------------------------ #
# 2. LOAD AND PREPROCESS DATASET #
# ------------------------------ #
def load_dataset_from_jsonl(file_path):
    formatted_data = []
    with open(file_path, 'r') as f:
        for line_number, line in enumerate(f, 1):
            try:
                if not line.strip():
                    continue
                data = json.loads(line)
                messages = data.get('messages', [])
                
                conversation = ""
                for msg in messages:
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    if role == "user":
                        conversation += f"<|im_start|>user\n{content}<|im_end|>\n"
                    elif role == "assistant":
                        conversation += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                
                if conversation:
                    formatted_data.append({"text": conversation})
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {line_number}: {e}")
                print(f"Problematic line: {line[:100]}...")
                continue
            except Exception as e:
                print(f"Unexpected error at line {line_number}: {e}")
                continue
    
    if not formatted_data:
        raise ValueError("No valid data was loaded from the JSONL file")
        
    print(f"Successfully loaded {len(formatted_data)} conversations")
    return formatted_data

train_data = load_dataset_from_jsonl("training_data.jsonl")
train_dataset = Dataset.from_list(train_data)
print(f"\nTotal number of examples in dataset: {len(train_dataset)}")

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=8192,
        padding="max_length",
        return_tensors=None
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized_train = train_dataset.map(
    preprocess_function,
    remove_columns=train_dataset.column_names,
    batch_size=8,
)
print(f"Total number of examples after tokenization: {len(tokenized_train)}\n")

# -------------------------- #
# 3. TRAINING ARGS + DEEPSPEED
# -------------------------- #
training_args = TrainingArguments(
    output_dir="./qwen-math-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=1,
    # Let Transformers/Trainer know we want to use DeepSpeed
    deepspeed=deepspeed_config
)

# 4. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
)

# 5. Optionally resume from checkpoint
checkpoint_dir = "./qwen-math-finetuned"
if RESUME_FROM_CHECKPOINT and os.path.exists(checkpoint_dir):
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        print("No checkpoint found. Starting training from scratch.")
        trainer.train()
else:
    print("Starting training from scratch.")
    trainer.train()

# 6. Save final model
trainer.save_model("./qwen-math-finetuned-final")

print("Training complete!")
