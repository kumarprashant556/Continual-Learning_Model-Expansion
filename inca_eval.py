import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class RealTimeQAEvaluator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def score_choice(self, question, choice_text):
        """
        Calculates the perplexity (loss) of the choice given the question.
        Uses explicit label masking for robustness.
        """
        prompt_text = f"Question: {question}\nAnswer:"
        full_text = f"{prompt_text} {choice_text}"
        
        # 1. Tokenize full sequence
        enc_full = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        input_ids = enc_full.input_ids
        
        # 2. Tokenize prompt to find where answer starts
        enc_prompt = self.tokenizer(prompt_text, return_tensors="pt")
        prompt_len = enc_prompt.input_ids.shape[1]
        
        # 3. Create labels: -100 for prompt tokens (ignored), actual ids for answer tokens
        labels = input_ids.clone()
        if prompt_len < labels.shape[1]:
            labels[:, :prompt_len] = -100
        else:
            # Fallback: if prompt length >= full length (rare tokenizer edge case),
            # keep strictly the last token or similar.
            # Usually happens if 'choice_text' is empty or tokenizes to nothing.
            pass 

        with torch.no_grad():
            # GPT-2 / CausalLM models accept 'labels'. 
            # They automatically shift labels internally (logits[..., :-1], labels[..., 1:])
            # and compute CrossEntropyLoss only on non-(-100) indices.
            outputs = self.model(**enc_full, labels=labels)
            
            # The model returns the average loss over valid tokens.
            # This corresponds exactly to the likelihood of the answer given the prompt.
            loss = outputs.loss.item()
            
        return loss

    def evaluate_week(self, probes):
        """
        Evaluates a list of probes (questions) for a single week.
        Returns Accuracy (0.0 to 1.0).
        """
        if not probes:
            return 0.0

        correct_count = 0
        total = 0
        
        # Suppress tqdm for short lists
        iterator = probes if len(probes) < 10 else tqdm(probes, desc="  Evaluating", leave=False)

        for item in iterator:
            question = item.get('question', '')
            choices = item.get('choices', {})
            
            # Handle choices as list or dict
            if isinstance(choices, list): 
                choices = {str(i): c for i, c in enumerate(choices)}
            
            # Handle answer_key
            answer_key = str(item.get('answer_key', ''))
            if isinstance(item.get('answer'), list) and len(item['answer']) > 0:
                answer_key = str(item['answer'][0])

            if not choices or not answer_key:
                continue

            # Calculate loss for each choice
            candidate_losses = []
            keys = sorted(choices.keys()) 
            
            for key in keys:
                loss = self.score_choice(question, choices[key])
                candidate_losses.append((loss, key))
            
            # Prediction is choice with LOWEST loss
            candidate_losses.sort(key=lambda x: x[0])
            predicted_key = candidate_losses[0][1]
            
            if predicted_key == answer_key:
                correct_count += 1
            
            total += 1

        if total == 0: return 0.0
        return correct_count / total