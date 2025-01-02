import os
import json
import torch
import deepspeed
import wandb
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from huggingface_hub import login

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Message:
    role: str
    content: str

@dataclass
class Conversation:
    messages: List[Message]
    
    @property
    def instruction(self) -> Optional[str]:
        for msg in self.messages:
            if msg.role == "user":
                return msg.content
        return None
    
    @property
    def response(self) -> Optional[str]:
        for msg in self.messages:
            if msg.role == "assistant":
                return msg.content
        return None

class TrainingDataLoader:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        
    def load(self) -> List[Conversation]:
        conversations = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    messages = [Message(**msg) for msg in data.get('messages', [])]
                    conversations.append(Conversation(messages=messages))
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON line: {e}")
                except Exception as e:
                    logger.error(f"Error processing line: {e}")
        return conversations

    def load_instruction_pairs(self) -> List[Dict[str, str]]:
        conversations = self.load()
        pairs = []
        for conv in conversations:
            if conv.instruction and conv.response:
                pairs.append({
                    "instruction": conv.instruction,
                    "response": conv.response
                })
        return pairs

class InstructionDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.loader = TrainingDataLoader(data_path)
        self.data = self.loader.load_instruction_pairs()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        # Format: <|system|>Instruction: {instruction}<|assistant|>{response}</s>
        prompt = f"<|system|>Instruction: {item['instruction']}<|assistant|>{item['response']}</s>"
        
        encodings = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"][0],
            "attention_mask": encodings["attention_mask"][0],
            "labels": encodings["input_ids"][0].clone()
        }

@dataclass
class TrainingConfig:
    # Model configuration
    model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    max_length: int = 1024
    
    # Training configuration
    train_micro_batch_size_per_gpu: int = 4
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    
    # Logging & Saving
    logging_steps: int = 1
    save_steps: int = 500
    eval_steps: int = 500
    
    # Paths
    output_dir: str = "checkpoints"
    dataset_path: str = "training.jsonl"

    @property
    def total_train_batch_size(self) -> int:
        # This should equal micro_batch * grad_acc * world_size
        return self.train_micro_batch_size_per_gpu * self.gradient_accumulation_steps * 2  # world_size is 2

def create_ds_config(config: TrainingConfig, dataset_size: int) -> Dict:
    # Calculate total steps
    total_steps = int(config.num_train_epochs * dataset_size / config.total_train_batch_size)
    warmup_steps = int(config.warmup_ratio * total_steps)
    
    return {
        "train_micro_batch_size_per_gpu": config.train_micro_batch_size_per_gpu,
        "train_batch_size": config.total_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": config.weight_decay
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": total_steps,
                "warmup_min_lr": 0,
                "warmup_max_lr": config.learning_rate,
                "warmup_num_steps": warmup_steps
            }
        },
        "fp16": {
            "enabled": True,
            "auto_cast": True,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
            "offload_param": {
                "device": "none",
                "pin_memory": True,
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto"
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
        "monitor_config": {
            "enabled": False,
            "tag": "train"
        }
    }

def setup_wandb(config: TrainingConfig):
    try:
        wandb_api_key = os.getenv('WANDB_API_KEY')
        if not wandb_api_key:
            raise ValueError("WANDB_API_KEY environment variable not set")
            
        wandb.login(key=wandb_api_key)
        wandb.init(
            project="llama-instruction-tuning",
            config={
                "model_name": config.model_name,
                "batch_size": config.train_batch_size * config.gradient_accumulation_steps * 2,
                "learning_rate": config.learning_rate,
                "epochs": config.num_train_epochs,
            }
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        logger.warning("Continuing training without wandb logging...")
        return False

def train():
    # Initialize config
    config = TrainingConfig()
    
    # Login to Hugging Face
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")
    login(token=hf_token)
    
    # Set up distributed training
    deepspeed.init_distributed()
    
    # Set up logging and tracking
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Prepare dataset
    train_dataset = InstructionDataset(
        data_path=config.dataset_path,
        tokenizer=tokenizer,
        max_length=config.max_length
    )
    
    # Initialize DeepSpeed config with dataset size
    ds_config = create_ds_config(config, len(train_dataset))
    
    # Initialize model with DeepSpeed
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        config_params=ds_config,
        model_parameters=model.parameters(),
        training_data=train_dataset
    )
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_micro_batch_size_per_gpu,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop
    global_step = 0
    for epoch in range(config.num_train_epochs):
        model_engine.train()
        for step, batch in enumerate(train_dataloader):
            # Move tensors to device
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model_engine(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss = outputs.loss
            
            # Backward pass and optimization
            model_engine.backward(loss)
            model_engine.step()
            
            # Log progress
            if model_engine.global_rank == 0:
                logger.info(f"Epoch {epoch}, Step {global_step}: Loss {loss.item():.4f}")
            
            global_step += 1

    # Modified saving code
    if model_engine.global_rank == 0:  # Only save on the primary GPU
        logger.info("Training completed. Starting model save process...")
        final_output_dir = os.path.join(config.output_dir, "final")
        os.makedirs(final_output_dir, exist_ok=True)
        
        try:
            logger.info("Saving DeepSpeed checkpoint...")
            model_engine.save_checkpoint(final_output_dir)
            logger.info("DeepSpeed checkpoint saved successfully")
            
            logger.info("Saving tokenizer...")
            tokenizer.save_pretrained(final_output_dir)
            logger.info("Tokenizer saved successfully")
            
            # Verify the save
            if os.path.exists(os.path.join(final_output_dir, "zero_to_fp32.py")):
                logger.info(f"Model successfully saved to {final_output_dir}")
            else:
                logger.warning("Model save may be incomplete - checkpoint files not found")
                
        except Exception as e:
            logger.error(f"Error during model saving: {e}")
            raise  # Re-raise the exception to make sure the error is visible
    
    # Synchronize all processes to ensure saving is complete
    torch.distributed.barrier()
    logger.info("Training and saving completed!")

if __name__ == "__main__":
    train()