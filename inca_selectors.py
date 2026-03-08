import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseSelector(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

class WeightedSumSelector(BaseSelector):
    """
    Selector A (Baseline): Learned scalar weights.
    W = softmax(learnable_params)
    Output = Sum(W_i * Block_i)
    """
    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        # We don't know N blocks yet, so we use a dynamic ParameterDict 
        # or simply append parameters as blocks grow. 
        # For simplicity in PyTorch, we often project the blocks.
        # Here we will implement a lightweight attention-like mechanism that 
        # acts as a weighted sum based on a global context vector.
        self.scorer = nn.Linear(hidden_size, 1)

    def forward(self, frozen_outputs, query):
        # frozen_outputs: list of [Batch, Seq, Hidden]
        if not frozen_outputs: return torch.zeros_like(query)
        
        # Stack: [Batch, Seq, N_Blocks, Hidden]
        stack = torch.stack(frozen_outputs, dim=2)
        
        # Calculate score for each block based on the query (current input)
        # We average query over sequence to get a "gist" of the input
        # Query: [Batch, Seq, Hidden] -> Global: [Batch, Hidden]
        global_ctx = query.mean(dim=1, keepdim=True).unsqueeze(2) 
        
        # Simple dot product similarity between Input and Block Output
        # This effectively learns which block aligns with current input
        # [Batch, Seq, N, Hidden] * [Batch, 1, 1, Hidden]
        scores = torch.matmul(stack, global_ctx.transpose(-1, -2)) # [Batch, Seq, N, 1]
        
        weights = F.softmax(scores, dim=2)
        
        # Weighted Sum
        context = torch.sum(stack * weights, dim=2)
        return context

class CrossAttentionSelector(BaseSelector):
    """
    Selector C (Primary): Multi-Head Cross Attention.
    Query = Current Input State
    Key/Value = Output of Frozen Blocks
    """
    def __init__(self, hidden_size, n_heads=4):
        super().__init__(hidden_size)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, frozen_outputs, query):
        if not frozen_outputs: return torch.zeros_like(query)

        # 1. Prepare Key/Value from Frozen Blocks
        # We treat the frozen blocks as a sequence of "memories"
        # Concatenate along sequence length or treat each block as a token?
        # Standard approach: Stack them.
        # [Batch, Seq, Hidden] * N -> [Batch, Seq * N, Hidden]
        # This allows the model to attend to specific tokens in frozen history.
        kv_tensor = torch.cat(frozen_outputs, dim=1)
        
        # 2. Cross Attention
        # Q = Current Input, K=V = Frozen History
        attn_output, _ = self.attn(query, kv_tensor, kv_tensor)
        
        return self.norm(attn_output)

class GatedSelector(BaseSelector):
    """
    Selector B (Advanced): Gated Mechanism.
    Uses a neural network to output a sigmoid gate (0-1) for each block.
    """
    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        # Gate generator: Input -> Scalar per block
        # Since number of blocks grows, we use a shared projection
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, frozen_outputs, query):
        if not frozen_outputs: return torch.zeros_like(query)
        
        context_accum = torch.zeros_like(query)
        
        # Calculate a gate for each block independently
        for block_out in frozen_outputs:
            # Gate depends on the *combination* of Input and Block Output
            # "Does this block output match what the input needs?"
            combined = query + block_out
            gate = self.gate_net(combined) # [Batch, Seq, 1]
            
            context_accum += gate * block_out
            
        return context_accum