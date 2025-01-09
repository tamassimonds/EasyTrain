import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import json
import os
import wandb
from datetime import datetime

# PEFT / LoRA imports
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# -----------------------
# Configuration
# -----------------------
RESUME_FROM_CHECKPOINT = False  # Set to True if you want to resume from a checkpoint
MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"  # Base Qwen model
OUTPUT_DIR = "./qwen-math-finetuned"          # Intermediate training outputs
FINAL_MODEL_PATH = "./qwen-math-finetuned-merged"  # Final, merged checkpoint

# -----------------------
# Load Base Model & Tokenizer
# -----------------------
# (1) We'll always load from the base model, ignoring any existing "final" folder.
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# -----------------------
# LoRA Configuration
# -----------------------
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # For causal language modeling tasks
    r=1024,                        # LoRA rank
    lora_alpha=4096,               # Scaling alpha
    lora_dropout=0.1,              # Dropout
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # typical attention modules
        "gate_proj", "up_proj"                  # typical MLP modules
    ],
)

# Optionally, if you plan to do 4-bit or 8-bit quantization, you could do:
# model = prepare_model_for_kbit_training(model)

model = get_peft_model(model, lora_config)

# Optional: Print out the parameter counts
trainable_params = 0
all_param = 0
for _, param in model.named_parameters():
    num_params = param.numel()
    all_param += num_params
    if param.requires_grad:
        trainable_params += num_params

print(f"Trainable params: {trainable_params:,d} || "
      f"All params: {all_param:,d} || "
      f"Trainable%: {100 * trainable_params / all_param:.2f}%")

# -----------------------
# Load & Preprocess Data
# -----------------------
def load_dataset_from_jsonl(file_path):
    """
    Loads a JSONL file where each line is a dict:
    {
      "messages": [
          {"role": "user", "content": "..."},
          {"role": "assistant", "content": "..."},
          ...
      ]
    }
    """
    formatted_data = []
    error_count = 0
    total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            try:
                line = line.strip()
                if not line:
                    continue
                    
                data = json.loads(line)
                messages = data.get('messages', [])
                
                # Combine messages with Qwen's conversation style
                conversation = ""
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        conversation += f"<|im_start|>user\n{content}<|im_end|>\n"
                    elif role == "assistant":
                        conversation += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                
                if conversation:
                    formatted_data.append({"text": conversation})
                
                if line_number % 1000 == 0:
                    print(f"Processed {line_number}/{total_lines} lines...")
                    
            except (json.JSONDecodeError, Exception) as e:
                error_count += 1
                print(f"Error at line {line_number}, skipping: {str(e)}")
                continue
    
    print(f"\nSuccessfully loaded {len(formatted_data)} conversations from {file_path}")
    print(f"Total lines in file: {total_lines}")
    print(f"Encountered and skipped {error_count} errors while loading")
    
    if not formatted_data:
        raise ValueError("No valid data was loaded from the JSONL file")
        
    return formatted_data

# Replace with your training data file path
TRAIN_JSONL = "training_data.jsonl"
train_data = load_dataset_from_jsonl(TRAIN_JSONL)
train_dataset = Dataset.from_list(train_data)
print(f"\nTotal number of examples in dataset: {len(train_dataset)}")

# Tokenization function
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=4096,
        padding="max_length"
    )
    # Make labels = input_ids
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# Map the dataset
tokenized_train = train_dataset.map(
    preprocess_function,
    remove_columns=train_dataset.column_names,
    batched=True
)

print(f"Total number of examples after tokenization: {len(tokenized_train)}\n")

# -----------------------
# Training Arguments
# -----------------------
wandb.init(
    project="qwen-math-finetuning",
    name=f"lora-train-{datetime.now().strftime('%Y%m%d-%H%M')}",
    config={
        "model_name": MODEL_NAME,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "learning_rate": 1e-5,
        "num_epochs": 3,
        "batch_size": 1,
    }
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    optim="adamw_torch",
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=True,
    seed=42,
    logging_steps=10,
    save_steps=500,
    save_total_limit=1,
    gradient_checkpointing=False,
    deepspeed="ds_config.json",
    report_to="wandb",
    logging_dir="./logs",
)

# -----------------------
# Data Collator
# -----------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

# -----------------------
# Initialize Trainer
# -----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    data_collator=data_collator
)

# -----------------------
# Resume from checkpoint or start fresh
# -----------------------
checkpoint_dir = OUTPUT_DIR
if RESUME_FROM_CHECKPOINT and os.path.exists(checkpoint_dir):
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        resume_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"Resuming training from checkpoint: {resume_checkpoint_path}")
        trainer.train(resume_from_checkpoint=resume_checkpoint_path)
    else:
        print("No checkpoint found. Starting training from scratch.")
        trainer.train()
else:
    print("Starting training from scratch.")
    trainer.train()

# -----------------------
# Merge & Save Final Model
# -----------------------
print("Saving LoRA adapter separately...")
trainer.model.save_pretrained("./qwen-math-lora-adapter")

print("Merging LoRA adapter into the base model...")
merged_model = trainer.model.merge_and_unload()

print("Saving merged model to:", FINAL_MODEL_PATH)
merged_model.save_pretrained(FINAL_MODEL_PATH)
tokenizer.save_pretrained(FINAL_MODEL_PATH)

print("Training complete! Merged model saved to", FINAL_MODEL_PATH)

wandb.finish()
