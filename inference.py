from typing import Optional, List, Dict
import logging
import os
import torch
from transformers import AutoTokenizer, MllamaForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(
        self,
        base_model_name: str = "meta-llama/Llama-3.2-11B-Instruct",
        checkpoint_dir: str = "checkpoints/final",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.base_model_name = base_model_name
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the model and tokenizer."""
        try:
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            
            # Ensure padding token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            logger.info(f"Loading model architecture to {self.device}...")
            self.model = MllamaForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            
            # Load our trained weights if they exist
            if os.path.exists(self.checkpoint_dir):
                logger.info(f"Loading trained weights from {self.checkpoint_dir}...")
                
                # Setup proper paths for sharded weights
                weights_dir = os.path.join(self.checkpoint_dir, "pytorch_model.bin")
                if os.path.isdir(weights_dir):
                    # First, copy all shard files to the parent directory
                    import glob
                    import shutil
                    
                    # Copy index file
                    index_src = os.path.join(weights_dir, "pytorch_model.bin.index.json")
                    index_dst = os.path.join(self.checkpoint_dir, "pytorch_model.bin.index.json")
                    if os.path.exists(index_src) and not os.path.exists(index_dst):
                        shutil.copy2(index_src, index_dst)
                    
                    # Copy all shard files
                    for shard in glob.glob(os.path.join(weights_dir, "pytorch_model-*.bin")):
                        shard_name = os.path.basename(shard)
                        shard_dst = os.path.join(self.checkpoint_dir, shard_name)
                        if not os.path.exists(shard_dst):
                            shutil.copy2(shard, shard_dst)
                    
                    logger.info("Copied sharded weights to main directory")
                
                # Load the model with the correct file structure
                self.model = MllamaForCausalLM.from_pretrained(
                    self.checkpoint_dir,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                )
            
            # Fix weights tying
            if hasattr(self.model, 'tie_weights'):
                self.model.tie_weights()
            
            self.model.eval()
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
    ) -> str:
        """Generate response for given instruction."""
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Process input
        inputs = self.tokenizer(
            instruction,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)
        
        # Generate
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=num_return_sequences,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Decode output
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise

    def __call__(self, instruction: str, **kwargs) -> str:
        """Convenience method to generate a single response."""
        return self.generate(instruction, **kwargs)

# Example usage
if __name__ == "__main__":
    from PIL import Image
    
    # Initialize the model
    inferencer = ModelInference(
        base_model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
        checkpoint_dir="checkpoints/final",
        device="cuda"
    )
    
    # Text-only example
    response = inferencer.generate("There exist real numbers $x$ and $y$, both greater than 1, such that $\log_x\left(y^x\right)=\log_y\left(x^{4y}\right)=10$. Find $xy$.")

    
    print(f"Response: {response}")