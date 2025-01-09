import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerState,
    TrainerControl
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
import json
import os

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
        device_map="auto",
        trust_remote_code=True
    )
else:
    print("No final model found. Loading base model.")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=1024,  # Reduced rank from 2048 to 1024
    lora_alpha=4096,  # Reduced alpha to match lower rank
    lora_dropout=0.1,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention modules
        "gate_proj", "up_proj",                  # Reduced MLP/FFN modules
        # Removed "down_proj" and "w_proj" to decrease trainable parameters
    ],
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Add these lines to print parameter statistics
trainable_params = 0
all_param = 0
for _, param in model.named_parameters():
    num_params = param.numel()
    all_param += num_params
    if param.requires_grad:
        trainable_params += num_params
print(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.2f}%")

# Load and preprocess dataset
def load_dataset_from_jsonl(file_path):
    formatted_data = []
    with open(file_path, 'r') as f:
        for line_number, line in enumerate(f, 1):
            try:
                # Skip empty lines
                if not line.strip():
                    continue
                    
                data = json.loads(line)
                messages = data.get('messages', [])
                
                # Combine messages into a single string with proper formatting
                conversation = ""
                for msg in messages:
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    if role == "user":
                        conversation += f"<|im_start|>user\n{content}<|im_end|>\n"
                    elif role == "assistant":
                        conversation += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                
                if conversation:  # Only add if there's actual content
                    formatted_data.append({"text": conversation})
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {line_number}: {e}")
                print(f"Problematic line: {line[:100]}...")  # Print first 100 chars of problematic line
                continue
            except Exception as e:
                print(f"Unexpected error at line {line_number}: {e}")
                continue
    
    if not formatted_data:
        raise ValueError("No valid data was loaded from the JSONL file")
        
    print(f"Successfully loaded {len(formatted_data)} conversations")
    return formatted_data

# Load and tokenize data
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
    # Create labels by copying input_ids
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized_train = train_dataset.map(
    preprocess_function,
    remove_columns=train_dataset.column_names,
    batch_size=8,
)
print(f"Total number of examples after tokenization: {len(tokenized_train)}\n")

# Training arguments
training_args = TrainingArguments(
    output_dir="./qwen-math-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=1
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
)

# Check for latest checkpoint
checkpoint_dir = "./qwen-math-finetuned"
if RESUME_FROM_CHECKPOINT and os.path.exists(checkpoint_dir):
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        # Start training from checkpoint
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        print("No checkpoint found. Starting training from scratch.")
        trainer.train()
else:
    print("Starting training from scratch.")
    trainer.train()

# Save the final model
trainer.save_model("./qwen-math-finetuned-final") 