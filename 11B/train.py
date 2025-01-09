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
    model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    max_length: int = 1024
    
    # Training configuration
    train_micro_batch_size_per_gpu: int = 4
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.01
    
    # Logging & Saving
    logging_steps: int = 1
    save_steps: int = 500
    eval_steps: int = 500
    
    # Paths
    output_dir: str = "checkpoints"
    dataset_path: str = "training.jsonl"
    
    # Add new parameter
    load_from_checkpoint: bool = True
    checkpoint_dir: str = "checkpoints/final"

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
            "stage": 3,  # Changed to stage 3
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
    
    # Add checkpoint loading logic
    if config.load_from_checkpoint and os.path.exists(config.checkpoint_dir):
        logger.info(f"Loading model from checkpoint: {config.checkpoint_dir}")
        try:
            # Check if weights are sharded
            weights_dir = os.path.join(config.checkpoint_dir, "pytorch_model.bin")
            if os.path.isdir(weights_dir):
                # Handle sharded weights similar to inference.py
                import glob
                import shutil
                
                # Copy index file if needed
                index_src = os.path.join(weights_dir, "pytorch_model.bin.index.json")
                index_dst = os.path.join(config.checkpoint_dir, "pytorch_model.bin.index.json")
                if os.path.exists(index_src) and not os.path.exists(index_dst):
                    shutil.copy2(index_src, index_dst)
                
                # Copy shard files if needed
                for shard in glob.glob(os.path.join(weights_dir, "pytorch_model-*.bin")):
                    shard_name = os.path.basename(shard)
                    shard_dst = os.path.join(config.checkpoint_dir, shard_name)
                    if not os.path.exists(shard_dst):
                        shutil.copy2(shard, shard_dst)
                
                logger.info("Copied sharded weights to main directory")

            # Load the checkpoint
            model = AutoModelForCausalLM.from_pretrained(
                config.checkpoint_dir,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # Fix weights tying if needed
            if hasattr(model, 'tie_weights'):
                model.tie_weights()
                
            logger.info("Successfully loaded checkpoint")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise

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

    # Synchronize before save
    torch.distributed.barrier()
    
    checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # First save the DeepSpeed checkpoint (this saves sharded model state)
    logger.info("Saving DeepSpeed checkpoint...")
    model_engine.save_checkpoint(checkpoint_dir)
    
    if model_engine.global_rank == 0:
        logger.info("Training completed. Converting checkpoint to single file...")
        
        try:
            # Convert ZeRO checkpoint to single file
            single_file_path = os.path.join(config.output_dir, "pytorch_model.bin")
            
            # Use DeepSpeed's utility to convert checkpoint to single file
            logger.info("Converting ZeRO checkpoint to single file...")
            convert_zero_checkpoint_to_fp32_state_dict(
                checkpoint_dir,  # Load from DeepSpeed checkpoint dir
                single_file_path  # Save as single file
            )
            
            # Save config and tokenizer
            logger.info("Saving tokenizer and config...")
            final_output_dir = os.path.join(config.output_dir, "final")
            os.makedirs(final_output_dir, exist_ok=True)
            tokenizer.save_pretrained(final_output_dir)
            model.config.save_pretrained(final_output_dir)
            
            # Move the consolidated model file to final directory
            if os.path.exists(single_file_path):
                os.rename(
                    single_file_path,
                    os.path.join(final_output_dir, "pytorch_model.bin")
                )
            
            logger.info("Save process completed successfully")
            
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