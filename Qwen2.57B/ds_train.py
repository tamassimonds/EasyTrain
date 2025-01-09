import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import json
import os

# -----------------------
# Configuration
# -----------------------
RESUME_FROM_CHECKPOINT = False  # Set this to True to resume from a previous checkpoint if available
MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
FINAL_MODEL_PATH = "./qwen-math-finetuned-final"

# -----------------------
# Load Model & Tokenizer
# -----------------------
if os.path.exists(FINAL_MODEL_PATH):
    print(f"Loading from final model: {FINAL_MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        FINAL_MODEL_PATH,
       
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
else:
    print("No final model found. Loading base model.")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
     
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

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
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            try:
                if not line.strip():
                    continue
                    
                data = json.loads(line)
                messages = data.get('messages', [])
                
                # Combine messages with Qwen/Qwen2.5's conversation style
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
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {line_number}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error at line {line_number}: {e}")
                continue
    
    if not formatted_data:
        raise ValueError("No valid data was loaded from the JSONL file")
        
    print(f"Successfully loaded {len(formatted_data)} conversations from {file_path}")
    return formatted_data

train_data = load_dataset_from_jsonl("training_data.jsonl")
train_dataset = Dataset.from_list(train_data)

print(f"\nTotal number of examples in dataset: {len(train_dataset)}")

# Tokenization
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=4096,  
        padding="max_length"
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized_train = train_dataset.map(
    preprocess_function,
    remove_columns=train_dataset.column_names,
    batched=True
)

print(f"Total number of examples after tokenization: {len(tokenized_train)}\n")

# -----------------------
# Training Arguments
# -----------------------
training_args = TrainingArguments(
    output_dir="./qwen-math-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # GPU batch size
    gradient_accumulation_steps=1,
    # Combined => 4 * 2 = 8 samples per GPU
    # If you have 2 GPUs, 8 * 4 = 32 total batch size

    learning_rate=1e-5,
    optim="adamw_torch",
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=True,  # Mixed precision training
    seed=42,
    logging_steps=10,
    save_steps=500,
    save_total_limit=1,
    gradient_checkpointing=False,  # Disable if not needed
    deepspeed="ds_config.json"  # Use DeepSpeed
)

# -----------------------
# Create Data Collator
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
# Resume or Start Fresh
# -----------------------
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

# -----------------------
# Save Final Model
# -----------------------
trainer.save_model("./qwen-math-finetuned-final")
print("Training complete! Model saved to ./qwen-math-finetuned-final")
