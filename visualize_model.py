#!/usr/bin/env python3
"""
Model visualization script for INCA.
Uses torchviz to create computational graph and architecture diagrams.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torchviz import make_dot
import os

from inca_model_v2 import INCA_GPT2

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ===== 1. ARCHITECTURE VISUALIZATION =====
def visualize_architecture():
    """Generate static architecture diagram"""
    print("\n" + "="*70)
    print("INCA ARCHITECTURE DIAGRAM")
    print("="*70)
    
    architecture = """
    INCA_GPT2 Model Architecture
    ════════════════════════════════════════════════════════════
    
    Input: Token IDs [batch_size, seq_length]
           ↓
    ┌──────────────────────────────────────────────────────────┐
    │ [1] EMBEDDINGS LAYER                                     │
    ├──────────────────────────────────────────────────────────┤
    │ • Token Embedding (wte): 50257 → 768 dims               │
    │ • Position Embedding (wpe): position → 768 dims         │
    │ • Dropout: p=0.1                                         │
    │ Output: [batch, seq, 768]                               │
    └──────────────────────────────────────────────────────────┘
           ↓
    ┌──────────────────────────────────────────────────────────┐
    │ [2] INCA LAYER MANAGER (Dynamic Vertical Expansion)     │
    ├──────────────────────────────────────────────────────────┤
    │                                                           │
    │  Input: [batch, seq, 768]                               │
    │         ↓                                                │
    │         ├─→ Frozen Block 0 ──┐                          │
    │         ├─→ Frozen Block 1 ──┤→ Stack outputs           │
    │         ├─→ ... (if exists) ──┤                         │
    │         └─→ Current Block ────┘                         │
    │                ↓                                         │
    │         All Block Outputs [N, batch, seq, 768]          │
    │                ↓                                         │
    │         Selector Head (CrossAttention)                  │
    │         - Query: Current block                          │
    │         - Key/Value: Frozen blocks                      │
    │         - Multi-head: 12 heads, 64 dims/head            │
    │                ↓                                         │
    │         Final Output: [batch, seq, 768]                │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
           ↓
    ┌──────────────────────────────────────────────────────────┐
    │ [3] OUTPUT LAYER                                         │
    ├──────────────────────────────────────────────────────────┤
    │ • LayerNorm (ln_f)                                       │
    │ • Language Model Head (lm_head): 768 → 50257           │
    │ Output: [batch, seq, 50257] (vocabulary predictions)   │
    └──────────────────────────────────────────────────────────┘
           ↓
    Loss (if labels provided): CrossEntropyLoss
    ════════════════════════════════════════════════════════════
    
    Key Features:
    • Vertical Expansion: Blocks stack vertically as model grows
    • Frozen Memory: Old blocks remain inference-only
    • Selector Head: Learns to combine frozen + current knowledge
    • Continual Learning: No catastrophic forgetting
    """
    
    print(architecture)
    
    # Save to file
    with open("results/inca_v3/architecture.txt", "w") as f:
        f.write(architecture)
    print("\n✓ Saved to: results/inca_v3/architecture.txt")


# ===== 2. COMPUTATIONAL GRAPH =====
def visualize_computation_graph():
    """Generate computation graph visualization (DOT format)"""
    print("\n" + "="*70)
    print("GENERATING COMPUTATIONAL GRAPH (DOT format)")
    print("="*70)
    
    device = "cpu"
    
    # Load model
    print("\n[1] Loading model...")
    model = INCA_GPT2("distilgpt2", selector_type="cross_attn")
    model.to(device)
    model.eval()
    
    # Create sample input
    print("[2] Creating sample input...")
    batch_size = 2
    seq_length = 64
    sample_input = torch.randint(0, 50257, (batch_size, seq_length), device=device)
    
    # Forward pass to capture graph
    print("[3] Running forward pass...")
    with torch.no_grad():
        output = model(sample_input)
        logits = output.logits
    
    # Generate computational graph
    print("[4] Generating graph visualization...")
    
    # Graph shows: how tensors flow through operations
    graph = make_dot(
        logits.mean(),
        params=dict(model.named_parameters()),
        show_attrs=False,
        show_saved=False
    )
    
    # Save to DOT file (text format)
    output_path = "results/inca_v3/model_computation_graph"
    graph.save(output_path + ".dot")
    
    print(f"\n✓ Computation graph saved to: {output_path}.dot (DOT format)")
    print(f"  Note: Install graphviz (brew install graphviz) to render to PNG")
    print(f"  Or use online viewer: http://www.webgraphviz.com/")
    
    # Also save a simplified version as text
    info = f"""
COMPUTATION GRAPH INFORMATION
════════════════════════════════════════════════════════════

The computation graph (saved in DOT format) shows:
- All operations performed during forward pass
- Tensor shapes at each step
- Parameter dependencies
- Gradient flow paths

To visualize (on macOS):
  1. Install graphviz:  brew install graphviz
  2. Render to PNG:     dot -Tpng {output_path}.dot -o {output_path}.png
  3. View:              open {output_path}.png

Or use online viewer: http://www.webgraphviz.com/
  Copy-paste the contents of {output_path}.dot

Graph Statistics:
- Input size: batch=2, seq_len=64
- Hidden dim: 768
- Output size: [2, 64, 50257] (vocabulary)
- Path: Input → Embeddings → LayerManager → Output Layer → Logits

Key Nodes in Graph:
1. Input (input_ids)
2. Token Embedding (wte)
3. Position Embedding (wpe)
4. Dropout (embeddings)
5. LayerManager forward pass
6. Output LayerNorm (ln_f)
7. Language Model Head (lm_head)
8. Loss computation (if labels present)
"""
    
    with open(output_path + "_info.txt", "w") as f:
        f.write(info)
    
    print(f"✓ Instructions saved to: {output_path}_info.txt")


# ===== 3. MODEL SUMMARY =====
def print_model_summary():
    """Print model architecture and parameter counts"""
    print("\n" + "="*70)
    print("MODEL SUMMARY")
    print("="*70)
    
    device = "cpu"
    model = INCA_GPT2("distilgpt2", selector_type="cross_attn")
    model.to(device)
    
    print("\nModel Structure:")
    print("─" * 70)
    print(model)
    
    print("\n\nParameter Counts:")
    print("─" * 70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"Total Parameters:     {total_params:>15,}")
    print(f"Trainable Parameters: {trainable_params:>15,}")
    print(f"Frozen Parameters:    {frozen_params:>15,}")
    
    print("\n\nComponent Breakdown:")
    print("─" * 70)
    
    components = {
        "Embeddings (token + position)": sum(p.numel() for name, p in model.named_parameters() if "wte" in name or "wpe" in name),
        "Layer Norm (ln_f)": sum(p.numel() for name, p in model.named_parameters() if "ln_f" in name),
        "Language Model Head": sum(p.numel() for name, p in model.named_parameters() if "lm_head" in name),
        "Current Transformer Block": sum(p.numel() for name, p in model.named_parameters() if "current_block" in name),
        "Selector Head": sum(p.numel() for name, p in model.named_parameters() if "selector" in name),
    }
    
    for name, count in components.items():
        pct = (count / total_params) * 100 if total_params > 0 else 0
        print(f"{name:<35} {count:>12,} ({pct:>5.1f}%)")
    
    # Save to file
    with open("results/inca_v3/model_summary.txt", "w") as f:
        f.write("INCA Model Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(str(model) + "\n\n")
        f.write(f"Total Parameters:     {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Frozen Parameters:    {frozen_params:,}\n")
    
    print(f"\n✓ Saved to: results/inca_v3/model_summary.txt")


# ===== 4. DATA FLOW DIAGRAM =====
def print_data_flow():
    """Print detailed data flow through model"""
    print("\n" + "="*70)
    print("DATA FLOW THROUGH MODEL")
    print("="*70)
    
    flow = """
    FORWARD PASS DATA FLOW
    ════════════════════════════════════════════════════════════
    
    Input: input_ids [2, 64]      (batch_size=2, seq_len=64)
    
    Step 1: Embedding Layer
    ─────────────────────────────────────────────────────────
    input_ids          [2, 64]           → token IDs
         ↓ (wte embedding lookup)
    inputs_embeds      [2, 64, 768]      → token vectors
    
    position           [64]              → indices [0,1,...,63]
         ↓ (wpe embedding lookup)
    position_embeds    [1, 64, 768]      → position vectors
    
    inputs_embeds + position_embeds     → broadcasted addition
         ↓ (dropout)
    hidden_states      [2, 64, 768]      → combined embeddings
    
    
    Step 2: INCA Layer Manager
    ─────────────────────────────────────────────────────────
    hidden_states      [2, 64, 768]
         ├─→ Frozen Block 0  → [2, 64, 768]  ┐
         ├─→ Frozen Block 1  → [2, 64, 768]  ├→ Stack
         ├─→ ... (if any)                    │
         └─→ Current Block   → [2, 64, 768]  ┘
    
    block_outputs list of tensors [N, 2, 64, 768]
         ↓ (stack on dim 0)
    block_stack        [N, 2, 64, 768]
         ↓ (selector cross-attention)
         Query: current_output [2, 64, 768]
         Key/Value: frozen_outputs concatenated [2, 64*M, 768]
    
    final_output       [2, 64, 768]       ← selected representation
    
    
    Step 3: Output Layer
    ─────────────────────────────────────────────────────────
    final_output       [2, 64, 768]
         ↓ (LayerNorm)
    normalized         [2, 64, 768]
         ↓ (lm_head: Linear 768→50257)
    logits             [2, 64, 50257]     ← vocabulary predictions
    
    
    Step 4: Loss Computation (if labels provided)
    ─────────────────────────────────────────────────────────
    logits             [2, 64, 50257]     (all positions)
    labels             [2, 64]            (target token IDs)
         ↓ (shift for causal LM)
    shift_logits       [2, 63, 50257]     (positions 0:63)
    shift_labels       [2, 63]            (positions 1:64)
         ↓ (CrossEntropyLoss)
    loss               scalar             ← averaged across batch
    
    ════════════════════════════════════════════════════════════
    
    Memory Footprint (approximate):
    • Input:       2 × 64 × 2 bytes = 256 B
    • Embeddings:  2 × 64 × 768 × 4 = 393 KB
    • Hidden:      2 × 64 × 768 × 4 = 393 KB (per block)
    • Logits:      2 × 64 × 50257 × 4 = 25.6 MB
    • Total:       ~100-200 MB per batch
    """
    
    print(flow)
    
    with open("results/inca_v3/data_flow.txt", "w") as f:
        f.write(flow)
    
    print("\n✓ Saved to: results/inca_v3/data_flow.txt")


# ===== 5. LAYER MANAGER DETAIL =====
def print_layer_manager_detail():
    """Detailed visualization of layer manager architecture"""
    print("\n" + "="*70)
    print("LAYER MANAGER ARCHITECTURE (INCA Core)")
    print("="*70)
    
    detail = """
    INCALayerManager: Dynamic Vertical Expansion
    ════════════════════════════════════════════════════════════
    
    COMPONENTS:
    ───────────
    1. frozen_blocks: nn.ModuleList([Block_0, Block_1, ...])
       • Initialized empty at start
       • Grows as model learns (max_blocks = number of periods)
       • Always in eval mode (no gradient)
    
    2. current_block: nn.Module
       • Single trainable transformer block
       • Deepcopy of original GPT2 block
       • Gets frozen when learning plateaus
    
    3. selector_head: nn.Module (CrossAttentionSelector by default)
       • Multi-head attention with:
         - 4 attention heads
         - 768 embedding dimension
         - Layer normalization
       • Learns to combine frozen + current outputs
       • Always trainable
    
    
    FORWARD PASS ALGORITHM:
    ──────────────────────
    
    input: hidden_states [batch, seq, 768]
    
    1. Process Frozen Blocks (Inference Only)
    ────────────────────────────────────────
    block_outputs = []
    for block in frozen_blocks:
        with torch.no_grad():              # ← No gradients!
            hidden_states = block(hidden_states, attn_mask)
            block_outputs.append(hidden_states)
    
    Rationale: Frozen blocks preserve learned knowledge
              No parameters updated, pure inference
    
    
    2. Process Current Block (Trainable)
    ───────────────────────────────────
    current_hidden = hidden_states
    current_hidden = current_block(current_hidden, attn_mask)
    block_outputs.append(current_hidden)
    
    Rationale: Current block learns from new temporal data
              Gradients flow through this block
    
    
    3. Aggregate via Selector Head
    ──────────────────────────────
    if len(block_outputs) > 1:
        # Multiple blocks: learn how to combine
        frozen_outputs = block_outputs[:-1]
        current_output = block_outputs[-1]
        final = selector_head(frozen_outputs, current_output)
    else:
        # Single block: use directly
        final = block_outputs[0]
    
    Mechanism (CrossAttention):
    • Query: current_output (what does new block need?)
    • Key/Value: frozen_outputs (what knowledge do we have?)
    • Output: weighted combination capturing relevant history
    
    
    GROWTH MECHANISM (freeze_and_grow):
    ──────────────────────────────────
    When learning plateaus (detected by PlateauDetector):
    
    1. Freeze current block:
       for p in current_block.parameters():
           p.requires_grad = False
    
    2. Move to frozen list:
       frozen_blocks.append(current_block)
    
    3. Spawn new trainable block:
       current_block = deepcopy(block_template)
       for p in current_block.parameters():
           p.requires_grad = True
    
    4. Selector continues training:
       selector_head.requires_grad = True
       # Adapts to new configuration of N+1 blocks
    
    
    EXAMPLE EVOLUTION (6 Periods):
    ─────────────────────────────
    
    Period 1: [Current]
             Selector
    
    Period 2: [Frozen_0] → Selector ← [Current]
    
    Period 3: [Frozen_0][Frozen_1] → Selector ← [Current]
    
    Period 4: [Frozen_0][Frozen_1][Frozen_2] → Selector ← [Current]
    
    Period 5: [Frozen_0][Frozen_1][Frozen_2][Frozen_3] → Selector ← [Current]
    
    Period 6: [Frozen_0][Frozen_1][Frozen_2][Frozen_3][Frozen_4] → Selector ← [Current]
    
    Final: 5 frozen blocks (past knowledge) + 1 current block (latest)
           Selector learned to aggregate across 6 temporal periods!
    
    ════════════════════════════════════════════════════════════
    """
    
    print(detail)
    
    with open("results/inca_v3/layer_manager_detail.txt", "w") as f:
        f.write(detail)
    
    print("\n✓ Saved to: results/inca_v3/layer_manager_detail.txt")


# ===== MAIN =====
if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("INCA MODEL VISUALIZATION SUITE")
    print("="*70)
    
    # Ensure output directory exists
    os.makedirs("results/inca_v3", exist_ok=True)
    
    try:
        # Generate all visualizations
        visualize_architecture()
        print_model_summary()
        print_data_flow()
        print_layer_manager_detail()
        visualize_computation_graph()
        
        print("\n" + "="*70)
        print("✓ ALL VISUALIZATIONS COMPLETE")
        print("="*70)
        print("\nGenerated files:")
        print("  • results/inca_v3/architecture.txt")
        print("  • results/inca_v3/model_summary.txt")
        print("  • results/inca_v3/data_flow.txt")
        print("  • results/inca_v3/layer_manager_detail.txt")
        print("  • results/inca_v3/model_computation_graph.dot (DOT format)")
        print("  • results/inca_v3/model_computation_graph_info.txt")
        print("\nTo visualize computation graph as PNG (macOS):")
        print("  brew install graphviz")
        print("  dot -Tpng results/inca_v3/model_computation_graph.dot -o results/inca_v3/model_computation_graph.png")
        print("  open results/inca_v3/model_computation_graph.png")
        
    except Exception as e:
        print(f"\n✗ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
