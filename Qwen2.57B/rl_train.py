import os
import json
import torch
import torch.nn.functional as F
from typing import List, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)


class OfflineRLTrainer:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct",
        learning_rate: float = 1e-5,
        device: str = "cuda",
    ):
        """
        Offline RL trainer with LoRA on Qwen. 
        We do not sample new responses; we use provided {prompt, response, reward} data.
        """
        self.device = device
        
        # 1) Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 2) Load model in bf16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # If you only have 1 GPU, it usually goes to that GPU
            trust_remote_code=True,
        )
        
        # 3) Configure LoRA
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
        
        # 4) Wrap model with LoRA adapters
        self.model = get_peft_model(self.model, lora_config)
        self.model.train()
        
        # 5) Simple Adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 6) (Optional) Move to device, if needed
        #    Usually HF’s device_map handles it, but you can do:
        # self.model.to(self.device)
        
        # 7) Debug: count trainable vs total parameters
        total_params = 0
        trainable_params = 0
        for _, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"Trainable params: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")

    def load_data(self, json_path: str) -> List[Dict]:
        """
        Expects a JSON list of { "prompt": <str>, "response": <str>, "reward": <float> }.
        """
        with open(json_path, "r") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} samples from {json_path}")
        return data

    def offline_train_step(self, prompt: str, given_response: str, reward: float):
        """
        Offline RL step: 
          1) Compute log-prob of `given_response` given `prompt`.
          2) Multiply by `reward`.
          3) Backprop with negative sign (REINFORCE).
        """
        self.optimizer.zero_grad()
        
        # 1) Tokenize the prompt and the response
        #    We'll just concatenate them: [prompt_tokens, response_tokens]
        #    Then we'll measure log-prob of each token in the response part.
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(given_response, add_special_tokens=False)
        
        # We'll create a single input = prompt + response
        # The model at position i (in the final sequence) tries to predict token i.
        input_ids = torch.tensor(prompt_ids + response_ids, device=self.device).unsqueeze(0)
        
        # 2) Forward pass
        outputs = self.model(input_ids)
        # outputs.logits: [batch=1, seq_len, vocab_size]
        logits = outputs.logits
        
        # 3) We only want to compute log-prob for the response portion
        #    Typically, the token at position i is predicted by logits at position i-1
        #    So for response token index i in range(len(response_ids)),
        #    the relevant logits are at sequence index (len(prompt_ids) + i - 1).
        #    We'll do a simple loop to gather each token's log-prob.
        
        log_probs = []
        for i, resp_token_id in enumerate(response_ids):
            # The index in the combined sequence that predicts token i in the response
            seq_index_for_this_token = len(prompt_ids) + i - 1  
            if seq_index_for_this_token < 0:
                # If i=0 but prompt is empty, handle edge case
                seq_index_for_this_token = 0

            # Probability distribution for next token
            dist_logits = logits[0, seq_index_for_this_token, :]  # shape [vocab_size]
            dist_log_probs = torch.log_softmax(dist_logits, dim=-1)
            # The correct token is resp_token_id
            log_prob_value = dist_log_probs[resp_token_id]
            log_probs.append(log_prob_value)
        
        # Convert to a single tensor
        log_probs_tensor = torch.stack(log_probs, dim=0)
        
        # Sum or mean the log-probs across all response tokens
        # Then multiply by the reward for REINFORCE
        log_prob_sum = torch.sum(log_probs_tensor)
        
        # REINFORCE typically:  loss = - (R * log_prob) [or - (R * sum of log_probs)]
        # We'll do sum of log probs * reward
        policy_loss = -log_prob_sum * reward
        
        # 4) Backprop
        policy_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item()

    def train(self, data_path: str, num_epochs: int = 3):
        """
        Main training loop for the offline dataset of {prompt, response, reward} samples.
        """
        dataset = self.load_data(data_path)
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            for i, sample in enumerate(dataset):
                prompt = sample["prompt"]
                response = sample["response"]
                reward = float(sample["reward"])
                
                loss = self.offline_train_step(prompt, response, reward)
                total_loss += loss
                
                # Debug prints
                if (i+1) % 1 == 0:
                    print(f"Epoch {epoch+1}, Sample {i+1}/{len(dataset)}")
                    print(f"  Prompt: {prompt}")
                    print(f"  Response: {response}")
                    print(f"  Reward: {reward}")
                    print(f"  Step Loss: {loss:.4f}\n")
            
            avg_loss = total_loss / len(dataset)
            print(f"[Epoch {epoch+1}] avg loss: {avg_loss:.4f}\n")

    def save_model(self, save_path: str):
        """
        Save LoRA-adapted model & tokenizer to disk.
        """
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)


def main():
    # Instantiate the trainer
    trainer = OfflineRLTrainer(
        model_name="Qwen/Qwen2.5-Math-7B-Instruct",
        learning_rate=1e-5,
        device="cuda"
    )

    # Train on your JSON: a list of { "prompt": "...", "response": "...", "reward": float }
    trainer.train(data_path="evaluation.json", num_epochs=3)

    # Optionally, save final
    trainer.save_model("qwen_offline_rl_final")

if __name__ == "__main__":
    main()
