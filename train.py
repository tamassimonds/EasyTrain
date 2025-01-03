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
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
import os
import sys


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
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    max_length: int = 1024
    
    # Training configuration
    train_micro_batch_size_per_gpu: int = 4
    gradient_accumulation_steps: int = 16
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
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000
        },
        "zero_optimization": {
            "stage": 2,  # Changed to stage 2 for more stable saving
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e7,
            "allgather_bucket_size": 5e7
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "steps_per_print": 1,
        "wall_clock_breakdown": False
    }

def setup_wandb(config: TrainingConfig):
    try:
        api_key = "bdcf2519ae5a2abc6658658c17f34b1109d9a672"

            
        wandb.login(key=api_key)
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
    login(token="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")
    
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

    # Replace the saving code section in your train() function with this:

    # Synchronize before starting save process
     # First save the DeepSpeed checkpoint
    # Replace the saving section in train() with this:
    # Simple synchronization before save
    torch.distributed.barrier()
    
    if model_engine.global_rank == 0:
        logger.info("Training completed. Starting model save process...")
        final_output_dir = os.path.join(config.output_dir, "final")
        fp32_output_dir = os.path.join(config.output_dir, "fp32")
        os.makedirs(final_output_dir, exist_ok=True)
        os.makedirs(fp32_output_dir, exist_ok=True)
        
        try:
            # Save using state dict approach
            logger.info("Saving model state...")
            state_dict = model_engine.module.state_dict()
            checkpoint = {
                'epoch': config.num_train_epochs,
                'global_step': global_step,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            }
            
            torch.save(checkpoint, os.path.join(fp32_output_dir, "pytorch_model.bin"))
            logger.info("Model state saved successfully")
            
            # Save config and tokenizer
            logger.info("Saving tokenizer and config...")
            tokenizer.save_pretrained(final_output_dir)
            model.config.save_pretrained(final_output_dir)
            logger.info("Tokenizer and config saved successfully")
            
        except Exception as e:
            logger.error(f"Error during save process: {e}")
            raise

    # Wait for save to complete
    torch.distributed.barrier()
    
    # Clean up - do this on all ranks
    if torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    logger.info(f"Process {model_engine.global_rank} completed")
if __name__ == "__main__":
    train()