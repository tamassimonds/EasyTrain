import os
import json
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)


class RLTrainer:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct",
        learning_rate: float = 1e-5,
        device: str = "cuda",
    ):
        """
        Initialize the RLTrainer with a LoRA-wrapped model, tokenizer, and optimizer.
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        
        # Load model with bf16 (rather than 8-bit) 
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",    # or {"": device} if you prefer
            trust_remote_code=True,
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj",
            ],
        )
        
        # Wrap the base model with LoRA adapters
        self.model = get_peft_model(self.model, lora_config)
        self.model.train()
        
        # Enable gradient checkpointing (automatically sets use_cache=False)
        self.model.gradient_checkpointing_enable()
        
        # Simple Adam optimizer over LoRA parameters
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def load_data(self, json_path: str) -> List[Dict]:
        """
        Load and preprocess the evaluation/training data from a JSON file.
        The file should contain a list of dicts like {"prompt": "...", "reward": float}.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data

    def generate_with_grad(
        self,
        prompt: str,
        max_length: int = 200,
    ) -> (str, torch.Tensor, torch.Tensor):
        """
        Generate a response by sampling tokens one-by-one in a gradient-enabled context.
        Returns:
            generated_text (str): The final decoded text from sampled tokens.
            log_probs (torch.Tensor): A tensor of shape [T], where T is the number of generated tokens.
            entropies (torch.Tensor): A tensor of shape [T], the entropy at each step.
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        log_probs = []
        entropies = []
        generated_token_ids = []
        
        current_input_ids = input_ids
        
        # Clear cache before generation to manage VRAM
        torch.cuda.empty_cache()
        
        # Use automatic mixed precision
        with torch.amp.autocast(device_type="cuda"):
            for _ in range(max_length):
                # Forward pass: get logits for the next token
                outputs = self.model(current_input_ids)
                next_token_logits = outputs.logits[:, -1, :]  # shape [batch_size, vocab_size]

                # Softmax -> categorical distribution
                probs = F.softmax(next_token_logits, dim=-1)
                dist = Categorical(probs)
                
                # Sample next token
                next_token = dist.sample()  # shape [batch_size], but here batch_size=1
                
                # Store log_prob
                log_prob = dist.log_prob(next_token)  # shape [1]
                log_probs.append(log_prob)
                
                # Store entropy of the distribution: H = -Σ p log p
                entropy = dist.entropy()  # shape [1]
                entropies.append(entropy)
                
                # Append sampled token to the sequence
                generated_token_ids.append(next_token.item())
                
                # Check for EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Update the context
                current_input_ids = torch.cat(
                    [current_input_ids, next_token.unsqueeze(0)], dim=1
                )
        
        # Decode the generated tokens to a string
        generated_text = self.tokenizer.decode(generated_token_ids)
        
        # Convert from list of scalars -> single tensor [T]
        log_probs_tensor = torch.stack(log_probs, dim=0).squeeze(-1)
        entropies_tensor = torch.stack(entropies, dim=0).squeeze(-1)
        
        return generated_text, log_probs_tensor, entropies_tensor

    def train_step(self, prompt: str, reward: float, entropy_coef: float = 0.01):
        """
        Perform a single training step using REINFORCE with an entropy bonus.
        
        REINFORCE objective (for each token):
            L = - (log_prob * R + entropy_coef * entropy).
            
        All tokens share the same reward here, but you can modify to incorporate
        more complex reward shaping if needed.
        """
        self.optimizer.zero_grad()
        
        # Generate text in a grad-enabled context; get log_probs & entropies at each token
        response, log_probs, entropies = self.generate_with_grad(prompt)
        
        # For simplicity, replicate the single scalar reward over all generated tokens
        # In practice, you might do something more advanced like discounted rewards or 
        # a separate reward signal per step.
        num_tokens = log_probs.size(0)
        rewards = torch.tensor([reward] * num_tokens, device=self.device)
        
        # REINFORCE loss: L = - E[ log_prob * R + alpha * entropy ]
        # We'll take the mean over time to avoid scale issues with variable-length generations
        policy_loss = -torch.mean(log_probs * rewards + entropy_coef * entropies)
        
        # Backward pass and step
        policy_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item(), response

    def train(self, data_path: str, num_epochs: int = 5, max_length: int = 200):
        """
        Train loop over the dataset. 
        data_path should have JSON with items: [{"prompt": "...", "reward": float}, ...].
        """
        training_data = self.load_data(data_path)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for item in training_data:
                prompt = item["prompt"]
                reward = item["reward"]
                
                loss, response = self.train_step(prompt, reward)
                epoch_loss += loss
                
                print(f"Prompt: {prompt}")
                print(f"Generated response: {response}")
                print(f"Reward: {reward}")
                print(f"Loss: {loss}\n")
            
            avg_epoch_loss = epoch_loss / len(training_data)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss}")
            
            # Optionally save checkpoint after each epoch (or every few epochs)
            if (epoch + 1) % 5 == 0:
                self.save_model(f"rl_checkpoint_epoch_{epoch+1}")

    def save_model(self, path: str):
        """
        Save the LoRA-adapted model and tokenizer to disk.
        """
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def main():
    # Initialize trainer
    trainer = RLTrainer()
    
    # Start training
    trainer.train(
        data_path="evaluation.json",  # JSON with list of {"prompt":..., "reward":...}
        num_epochs=5,
        max_length=200,
    )

if __name__ == "__main__":
    main()
