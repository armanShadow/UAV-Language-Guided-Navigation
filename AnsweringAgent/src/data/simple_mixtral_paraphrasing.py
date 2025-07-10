#!/usr/bin/env python3
"""
Simple Mixtral Paraphrasing - Minimal Implementation
Initialize Mixtral model and paraphrase a simple sentence.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleMixtralParaphraser:
    """Minimal Mixtral paraphrasing implementation"""
    
    def __init__(self, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        """Initialize Mixtral model for paraphrasing"""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing Mixtral on device: {self.device}")
        logger.info(f"Model: {model_name}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        logger.info("Loading Mixtral model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half precision for GPU memory efficiency
            device_map="auto",  # Automatically distribute across available GPUs
            trust_remote_code=True
        )
        
        self.model.eval()
        logger.info("Mixtral model loaded successfully!")
    
    def paraphrase(self, sentence, max_length=150, temperature=0.7, num_return_sequences=1):
        """Generate paraphrases of the input sentence"""
        
        # Create prompt for paraphrasing
        prompt = f"""[INST] Please paraphrase the following sentence while keeping the same meaning:

"{sentence}"

Provide a clear, natural paraphrase: [/INST]"""
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate paraphrase
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(inputs['input_ids'][0]) + max_length,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and extract paraphrases
        paraphrases = []
        for output in outputs:
            # Decode the full output
            full_text = self.tokenizer.decode(output, skip_special_tokens=True)
            
            # Extract just the paraphrase (after [/INST])
            if "[/INST]" in full_text:
                paraphrase = full_text.split("[/INST]")[-1].strip()
                paraphrases.append(paraphrase)
        
        return paraphrases

def main():
    """Simple test of Mixtral paraphrasing"""
    
    # Initialize the paraphraser
    logger.info("=== Initializing Simple Mixtral Paraphraser ===")
    paraphraser = SimpleMixtralParaphraser()
    
    # Test sentence (UAV navigation related)
    test_sentence = "Fly to the white building on your right."
    
    logger.info(f"\n=== Testing Paraphrasing ===")
    logger.info(f"Original: {test_sentence}")
    
    # Generate paraphrase
    logger.info("Generating paraphrase...")
    paraphrases = paraphraser.paraphrase(test_sentence)
    
    # Display results
    logger.info(f"\nResults:")
    for i, paraphrase in enumerate(paraphrases, 1):
        logger.info(f"Paraphrase {i}: {paraphrase}")
    
    logger.info("\n=== Paraphrasing Complete ===")

if __name__ == "__main__":
    main() 