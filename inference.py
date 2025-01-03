import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import logging
from typing import Optional, List, Dict
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(
        self,
        base_model_name: str = "meta-llama/Llama-3.2-1B-Instruct",  # Original model architecture
        checkpoint_dir: str = "checkpoints/fp32",  # Directory with our trained weights
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False,
        max_memory: Optional[Dict] = None,
    ):
        self.base_model_name = base_model_name
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.max_memory = max_memory or {0: "24GiB"}
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the model and tokenizer."""
        try:
            # Load tokenizer from base model
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load config from base model
            logger.info("Loading model configuration...")
            config = AutoConfig.from_pretrained(self.base_model_name)
            
            # Load base model with config
            logger.info(f"Loading base model architecture to {self.device}...")
            model_kwargs = {
                "config": config,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "trust_remote_code": True,
                "device_map": "auto" if self.device == "cuda" else None,
                "max_memory": self.max_memory if self.device == "cuda" else None,
            }
            
            if self.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                **model_kwargs
            )
            
            # Load our trained weights
            logger.info("Loading trained weights...")
            checkpoint = torch.load(
                os.path.join(self.checkpoint_dir, "pytorch_model.bin"),
                map_location=self.device
            )
            
            # Load just the model weights from our checkpoint
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint  # In case the whole file is the state dict
                
            # Load our trained weights
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys when loading weights: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading weights: {unexpected_keys}")
            
            if not self.load_in_8bit and self.device == "cuda":
                self.model = self.model.half()  # Convert to FP16 for faster inference
            
            self.model.eval()  # Set to evaluation mode
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def generate(
        self,
        instruction: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
    ) -> List[str]:
        """Generate response for given instruction."""
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Format input like during training
        prompt = f"<|system|>Instruction: {instruction}<|assistant|>"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Generate
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=num_return_sequences,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode outputs
            responses = []
            for output in outputs:
                response = self.tokenizer.decode(output, skip_special_tokens=True)
                # Remove the instruction part and clean up
                response = response.split("<|assistant|>")[-1].strip()
                responses.append(response)
                
            return responses[0] if num_return_sequences == 1 else responses
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise

    def __call__(self, instruction: str, **kwargs) -> str:
        """Convenience method to generate a single response."""
        return self.generate(instruction, **kwargs)

# Example usage
if __name__ == "__main__":
    # Initialize inference with the original model name
    inferencer = ModelInference(
        base_model_name="meta-llama/Llama-3.2-1B-Instruct",  # Original model architecture
        checkpoint_dir="checkpoints/fp32",  # Directory with our trained weights
        device="cuda",
        load_in_8bit=False
    )
    
    # Example instructions
    instructions = [
        "What is the capital of France?",
        "Write a short poem about winter.",
        "Explain what machine learning is to a 10 year old."
    ]
    
    # Generate responses
    for instruction in instructions:
        print(f"\nInstruction: {instruction}")
        response = inferencer.generate(
            instruction,
            temperature=0.7,
            max_new_tokens=256,
            do_sample=True
        )
        print(f"Response: {response}")