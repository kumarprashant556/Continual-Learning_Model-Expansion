"""
QA-Specific Loss Module for Multitask Learning

Combines Language Modeling loss with explicit QA selection loss.
This teaches the model to both predict next tokens AND select correct answers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QALoss(nn.Module):
    """
    Calculates QA-specific loss by comparing perplexities of multiple choice options.
    
    Approach:
    - For each question with multiple choice options
    - Calculate perplexity of each choice given the question
    - Create ranking loss: correct choice should have lowest perplexity
    - Uses margin ranking loss to enforce ordering
    """
    
    def __init__(self, tokenizer, model, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)
    
    def score_choice(self, question, choice_text):
        """
        Calculate perplexity (negative log likelihood) for a choice.
        Lower perplexity = model thinks this choice is more likely.
        """
        prompt_text = f"Question: {question}\nAnswer:"
        full_text = f"{prompt_text} {choice_text}"
        
        # Tokenize
        enc_full = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        enc_prompt = self.tokenizer(prompt_text, return_tensors="pt")
        
        input_ids = enc_full.input_ids
        prompt_len = enc_prompt.input_ids.shape[1]
        
        # Create labels: -100 for prompt (ignored), actual IDs for answer
        labels = input_ids.clone()
        if prompt_len < labels.shape[1]:
            labels[:, :prompt_len] = -100
        
        with torch.no_grad():
            outputs = self.model(**enc_full, labels=labels)
            loss = outputs.loss.item()
        
        return loss
    
    def forward(self, probe_data, batch_size=4):
        """
        Calculate QA loss for a batch of probes.
        
        Args:
            probe_data: List of probe dictionaries with 'question', 'choices', 'answer_key'
            batch_size: Number of probes to process per batch
        
        Returns:
            qa_loss: Scalar tensor with QA ranking loss
        """
        if not probe_data or len(probe_data) == 0:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = 0.0
        loss_count = 0
        
        for probe in probe_data:
            question = probe.get('question', '')
            choices = probe.get('choices', {})
            answer_key = str(probe.get('answer_key', probe.get('answer', '0')))
            
            if not question or not choices or answer_key not in choices:
                continue
            
            # Score all choices
            choice_losses = []
            choice_keys = []
            
            for key in sorted(choices.keys()):
                loss = self.score_choice(question, choices[key])
                choice_losses.append(loss)
                choice_keys.append(key)
            
            if len(choice_losses) < 2:
                continue
            
            # Create ranking pairs: correct choice should have LOWER loss than wrong choices
            correct_idx = choice_keys.index(answer_key)
            correct_loss = torch.tensor(choice_losses[correct_idx], device=self.device)
            
            # Compare with each wrong choice
            for wrong_idx, wrong_loss in enumerate(choice_losses):
                if wrong_idx != correct_idx:
                    wrong_loss_tensor = torch.tensor(wrong_loss, device=self.device)
                    # MarginRankingLoss wants: y_true=1 when first < second
                    # We want: correct_loss < wrong_loss
                    pair_loss = self.margin_loss(
                        correct_loss.unsqueeze(0),
                        wrong_loss_tensor.unsqueeze(0),
                        torch.ones(1, device=self.device)
                    )
                    total_loss += pair_loss
                    loss_count += 1
        
        if loss_count == 0:
            return torch.tensor(0.0, device=self.device)
        
        return total_loss / loss_count


class CombinedLoss(nn.Module):
    """
    Combines Language Modeling loss with QA loss.
    
    Total Loss = α * LM_Loss + β * QA_Loss
    
    α = 0.8 (weight on language modeling - primary task)
    β = 0.2 (weight on QA selection - auxiliary task)
    """
    
    def __init__(self, alpha=0.8, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, lm_loss, qa_loss):
        """
        Combine losses with weights.
        
        Args:
            lm_loss: Language modeling loss from model
            qa_loss: QA ranking loss
        
        Returns:
            combined_loss: Weighted combination
        """
        combined = self.alpha * lm_loss + self.beta * qa_loss
        return combined
