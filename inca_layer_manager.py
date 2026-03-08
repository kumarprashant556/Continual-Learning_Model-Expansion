import torch
import torch.nn as nn
import copy
from inca_selectors import WeightedSumSelector, CrossAttentionSelector, GatedSelector

class INCALayerManager(nn.Module):
    """
    INCA v3 with Vertical Expansion:
    
    Architecture:
    Input -> Block_0 -> Block_1 -> ... -> Block_N (Frozen Blocks)
                 |         |               |
                 +-------> Selector Head <-+
                              |
                           Current Block
                              |
                          Output
    
    All blocks (frozen + current) output independently.
    All block outputs are collected and routed through a shared Selector Head.
    The Selector Head produces the final feature representation.
    Current block trains normally; frozen blocks are inference-only.
    """
    def __init__(self, config, base_model, selector_type='cross_attn'):
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embd
        self.selector_type = selector_type
        
        # 1. Extract block template from base model
        self.block_template = base_model.h[0] 
        
        # 2. Initialize Frozen Blocks (Memory)
        self.frozen_blocks = nn.ModuleList([])
        
        # 3. Initialize Current Trainable Block (Plasticity)
        self.current_block = copy.deepcopy(self.block_template)
        
        # 4. Initialize Shared Selector Head (Trainable, aggregates all blocks)
        print(f"[LayerManager] Initializing Shared Selector Head: {selector_type}")
        if selector_type == 'weighted': 
            self.selector_head = WeightedSumSelector(self.hidden_size)
        elif selector_type == 'gated': 
            self.selector_head = GatedSelector(self.hidden_size)
        else: 
            # Default / Primary
            self.selector_head = CrossAttentionSelector(self.hidden_size)
        
        print(f"[LayerManager] INCA v3 initialized with {len(self.frozen_blocks)} frozen blocks and 1 current block")

    def _prepare_attention_mask(self, mask, dtype, batch_size, seq_len):
        """
        Ensures attention mask is in the correct format [Batch, 1, 1, Seq_Len]
        and has the correct dtype (float with -10000.0 for masked positions).
        """
        if mask is None:
            return None
            
        # If mask is already 4D, assume it's correct
        if mask.dim() == 4:
            return mask.to(dtype=dtype)

        # Standard [Batch, Seq] mask (1 for keep, 0 for discard)
        # Expand to [Batch, 1, 1, Seq]
        extended_mask = mask[:, None, None, :]
        
        # Convert to float: 1.0 -> 0.0, 0.0 -> -10000.0
        extended_mask = extended_mask.to(dtype=dtype) 
        extended_mask = (1.0 - extended_mask) * -10000.0
        
        return extended_mask

    def forward(self, hidden_states, attention_mask=None):
        """
        Vertical Expansion Forward Pass:
        
        1. Pass input through all frozen blocks sequentially
        2. Pass input through current block
        3. Collect all block outputs
        4. Route all outputs through Selector Head
        5. Return selected representation
        """
        # Prepare attention mask
        if attention_mask is not None:
             head_mask = self._prepare_attention_mask(
                attention_mask, 
                hidden_states.dtype,
                hidden_states.size(0), 
                hidden_states.size(1)
            )
        else:
            head_mask = None
        
        # Store original input for selector routing
        original_input = hidden_states
        block_outputs = []

        # A. Process Frozen Blocks (inference only)
        if len(self.frozen_blocks) > 0:
            frozen_hidden = hidden_states
            with torch.no_grad():
                for idx, block in enumerate(self.frozen_blocks):
                    frozen_hidden = block(frozen_hidden, attention_mask=head_mask)[0]
                    block_outputs.append(frozen_hidden)
        
        # B. Process Current Trainable Block
        current_hidden = hidden_states
        current_hidden = self.current_block(current_hidden, attention_mask=head_mask)[0]
        block_outputs.append(current_hidden)
        
        # C. Route All Block Outputs Through Shared Selector Head
        # The selector head takes all block outputs and produces final representation
        if len(block_outputs) > 1:
            # Multiple blocks: use selector to aggregate
            final_output = self.selector_head(block_outputs[:-1], block_outputs[-1])
        else:
            # Only current block: use its output directly
            final_output = block_outputs[0]
        
        return final_output

    def freeze_and_grow(self):
        """
        Triggered by Plateau Detector.
        Implements Vertical Growth:
        1. Freezes current block.
        2. Adds to frozen list (grows vertically).
        3. Spawns new trainable block.
        4. Selector head remains trainable and adapts to new block configuration.
        """
        print(f"\n[INCA Manager] Triggering Vertical Growth Phase...")
        print(f"   Current Block #{len(self.frozen_blocks)} -> Freezing")
        
        # 1. Freeze Parameters of current block
        for p in self.current_block.parameters():
            p.requires_grad = False
            
        # 2. Move to Frozen List (Vertical Stack)
        self.frozen_blocks.append(self.current_block)
        
        # 3. Spawn New Trainable Block
        self.current_block = copy.deepcopy(self.block_template)
        
        # Ensure new block is trainable
        for p in self.current_block.parameters():
            p.requires_grad = True
        
        print(f"   New Block initialized. Total Blocks: {len(self.frozen_blocks)} frozen + 1 current")
        print(f"   Selector Head continues to adapt to {len(self.frozen_blocks) + 1} blocks")
        
    def get_block_count(self):
        """Returns total number of blocks (frozen + current)"""
        return len(self.frozen_blocks) + 1