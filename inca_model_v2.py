import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from inca_layer_manager import INCALayerManager

class INCA_GPT2(nn.Module):
    def __init__(self, model_name="distilgpt2", selector_type='cross_attn'):
        super().__init__()
        print(f"Loading INCA Base: {model_name} with {selector_type} selector")
        
        # Load pre-trained GPT2
        self.base = GPT2LMHeadModel.from_pretrained(model_name)
        self.config = self.base.config
        
        # Extract components we keep static
        self.transformer = self.base.transformer
        self.lm_head = self.base.lm_head
        
        # Initialize INCA Manager with selector type
        self.layer_manager = INCALayerManager(self.config, self.base.transformer, selector_type=selector_type)
        
        # Clear original layers to prevent double-processing/memory usage
        self.transformer.h = nn.ModuleList([]) 

    def forward(self, input_ids, attention_mask=None, labels=None):
        # 1. Embeddings (Standard GPT2)
        inputs_embeds = self.transformer.wte(input_ids)
        position_embeds = self.transformer.wpe(
            torch.arange(0, input_ids.size(-1), device=input_ids.device)
        )
        hidden_states = self.transformer.drop(inputs_embeds + position_embeds)
        
        # 2. INCA Processing (Dynamic Layers)
        # Pass through the Manager (Frozen History -> Current Layer)
        hidden_states = self.layer_manager(hidden_states, attention_mask=attention_mask)
        
        # 3. Final Norm & Head
        hidden_states = self.transformer.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift for Causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return type('ModelOutput', (object,), {'loss': loss, 'logits': logits})()

    def trigger_growth(self):
        """External API to trigger Stage 3"""
        self.layer_manager.freeze_and_grow()