#!/usr/bin/env python3
"""
INCA v3 Training Script (Restructured)
- Modular, testable components
- Robust error handling and validation
- Comprehensive metrics tracking
- Checkpointing and resumable training
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.optim import AdamW
import os
import json
import gc
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from datetime import datetime

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from inca_dataloader import INCALoader
from inca_eval import RealTimeQAEvaluator
from inca_model_v2 import INCA_GPT2


@dataclass
class TrainingConfig:
    """Centralized training configuration with validation."""
    data_root: str = "/Users/nishantkumar/Desktop/phd/code/My project/WorkingDir/realtimeqa"
    model_name: str = "distilgpt2"
    output_dir: str = "results/inca_v3"
    batch_size: int = 2
    lr: float = 1e-4
    epochs_per_period: int = 25  # Good convergence for real training
    max_grad_norm: float = 1.0
    eval_every_n_epochs: int = 2  # Evaluate every 2 epochs
    eval_sample_size: int = 50  # More probes for stable accuracy estimates
    final_eval_size: int = 150  # Comprehensive final evaluation
    max_periods: int = 6  # Train on 6 temporal periods
    aggregate_weeks: int = 12  # More data per period for better learning
    
    def __post_init__(self):
        """Validate configuration and set device."""
        if not Path(self.data_root).exists():
            raise ValueError(f"Data root does not exist: {self.data_root}")
        
        # Convert output_dir to Path and create if needed
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Choose device with fallback - prefer CPU for MPS memory issues
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            # Use CPU (MPS has memory fragmentation issues)
            self.device = "cpu"
    
    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, value):
        self._device = value


@dataclass
class PeriodMetrics:
    """Metrics for a single period."""
    period_id: str
    period_num: int
    epoch_losses: List[float]
    epoch_accs: List[float]
    final_acc: float
    best_acc: float
    best_epoch: int
    
    def summary(self) -> Dict:
        """Return dict summary for JSON storage."""
        return {
            "period": self.period_id,
            "period_num": self.period_num,
            "avg_loss": sum(self.epoch_losses) / len(self.epoch_losses) if self.epoch_losses else 0.0,
            "final_acc": self.final_acc,
            "best_acc": self.best_acc,
            "best_epoch": self.best_epoch,
            "epochs_trained": len(self.epoch_losses)
        }


class TrainingLogger:
    """Centralized logging with metrics tracking."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.output_dir / "metrics.json"
        self.log_file = self.output_dir / "training.log"
        self.all_metrics = []
    
    def log(self, msg: str, level: str = "INFO"):
        """Log message to console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {msg}"
        print(formatted, flush=True)
        with open(self.log_file, "a") as f:
            f.write(formatted + "\n")
    
    def save_metrics(self, metrics: PeriodMetrics):
        """Save period metrics."""
        self.all_metrics.append(metrics.summary())
        with open(self.metrics_file, "w") as f:
            json.dump(self.all_metrics, f, indent=2)


class INCATrainer:
    """Main training orchestrator."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = TrainingLogger(config.output_dir)
        self.logger.log("="*70)
        self.logger.log("INCA v3 TRAINING - RESTRUCTURED")
        self.logger.log("="*70)
    
    def setup(self) -> Tuple[INCA_GPT2, AutoTokenizer, INCALoader, RealTimeQAEvaluator, AdamW]:
        """Initialize all components."""
        self.logger.log("\n[1] Initializing components...")
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        self.logger.log(f"  ✓ Tokenizer: {self.config.model_name}")
        
        # Model
        model = INCA_GPT2(self.config.model_name, selector_type="cross_attn")
        model.to(self.config.device)
        n_params = sum(p.numel() for p in model.parameters())
        self.logger.log(f"  ✓ Model: {n_params:,} parameters on {self.config.device}")
        
        # Data loader
        loader = INCALoader(
            self.config.data_root,
            tokenizer,
            batch_size=self.config.batch_size,
            aggregate_weeks=self.config.aggregate_weeks
        )
        self.logger.log(f"  ✓ Data loader: {self.config.data_root}")
        
        # Evaluator
        evaluator = RealTimeQAEvaluator(model, tokenizer, self.config.device)
        self.logger.log(f"  ✓ Evaluator initialized")
        
        # Optimizer
        optimizer = AdamW(model.parameters(), lr=self.config.lr)
        self.logger.log(f"  ✓ Optimizer: AdamW (lr={self.config.lr})")
        
        self.logger.log("[1] Setup complete\n")
        return model, tokenizer, loader, evaluator, optimizer
    
    def train_epoch(
        self,
        model: nn.Module,
        train_loader,
        optimizer: AdamW
    ) -> float:
        """Train for one epoch. Returns average loss."""
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # Clean up after each batch to prevent memory accumulation
            del batch, outputs, loss
            gc.collect()
        
        return epoch_loss / batch_count if batch_count > 0 else 0.0
    
    def evaluate(
        self,
        model: nn.Module,
        evaluator: RealTimeQAEvaluator,
        probes: List[Dict],
        max_samples: int = None
    ) -> float:
        """Evaluate on probes. Returns accuracy."""
        if not probes:
            return 0.0
        
        model.eval()
        sample_probes = probes[:min(max_samples or len(probes), len(probes))]
        
        with torch.no_grad():
            acc = evaluator.evaluate_week(sample_probes)
        
        # Clean up memory after evaluation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        return acc
    
    def train_period(
        self,
        model: nn.Module,
        train_loader,
        probes: List[Dict],
        optimizer: AdamW,
        evaluator: RealTimeQAEvaluator,
        period_id: str,
        period_num: int
    ) -> PeriodMetrics:
        """Train for one full period. Returns metrics."""
        self.logger.log("\n" + "="*70)
        self.logger.log(f"PERIOD {period_num}: {period_id}")
        self.logger.log(f"  Training data: {len(train_loader)} batches")
        self.logger.log(f"  Probe data: {len(probes)} questions")
        self.logger.log(f"  Epochs: {self.config.epochs_per_period}")
        self.logger.log("="*70)
        
        metrics = PeriodMetrics(
            period_id=period_id,
            period_num=period_num,
            epoch_losses=[],
            epoch_accs=[],
            final_acc=0.0,
            best_acc=0.0,
            best_epoch=0
        )
        
        # Training loop
        for epoch in range(self.config.epochs_per_period):
            # Train
            avg_loss = self.train_epoch(model, train_loader, optimizer)
            metrics.epoch_losses.append(avg_loss)
            
            # Periodic evaluation
            if (epoch + 1) % self.config.eval_every_n_epochs == 0:
                acc = self.evaluate(
                    model, evaluator, probes,
                    max_samples=self.config.eval_sample_size
                )
                metrics.epoch_accs.append(acc)
                
                if acc > metrics.best_acc:
                    metrics.best_acc = acc
                    metrics.best_epoch = epoch + 1
                
                self.logger.log(
                    f"  Epoch {epoch+1:3d}/{self.config.epochs_per_period}: "
                    f"Loss {avg_loss:.4f} | Acc {acc:.2%}"
                )
                
                # Checkpoint best model
                checkpoint_path = self.config.output_dir / f"checkpoint_p{period_num}_best.pt"
                torch.save(model.state_dict(), checkpoint_path)
            else:
                if (epoch + 1) % 10 == 0:
                    self.logger.log(
                        f"  Epoch {epoch+1:3d}/{self.config.epochs_per_period}: "
                        f"Loss {avg_loss:.4f}"
                    )
        
        # Final evaluation
        self.logger.log(f"\n  Final evaluation ({self.config.final_eval_size} samples)...")
        final_acc = self.evaluate(
            model, evaluator, probes,
            max_samples=self.config.final_eval_size
        )
        metrics.final_acc = final_acc
        
        self.logger.log(
            f"  Final Accuracy: {final_acc:.2%} | "
            f"Best Accuracy: {metrics.best_acc:.2%} (epoch {metrics.best_epoch})"
        )
        
        # Save final model
        model_path = self.config.output_dir / f"inca_p{period_num}_{period_id}.pt"
        torch.save(model.state_dict(), model_path)
        self.logger.log(f"  Model saved: {model_path}")
        
        return metrics
    
    def run(self):
        """Main training loop."""
        try:
            # Initialize
            model, tokenizer, loader, evaluator, optimizer = self.setup()
            
            # Training
            self.logger.log("\n[2] Starting training...\n")
            all_metrics = []
            
            for period_idx, (period_id, train_loader, probes) in enumerate(loader):
                if period_idx >= self.config.max_periods:
                    break
                
                metrics = self.train_period(
                    model, train_loader, probes, optimizer, evaluator,
                    period_id, period_idx + 1
                )
                all_metrics.append(metrics)
                self.logger.save_metrics(metrics)
            
            # Summary
            self.logger.log("\n" + "="*70)
            self.logger.log("TRAINING COMPLETE!")
            self.logger.log("="*70)
            self.logger.log(f"Periods trained: {len(all_metrics)}")
            self.logger.log(f"Overall best accuracy: {max(m.best_acc for m in all_metrics):.2%}")
            self.logger.log(f"Metrics saved to: {self.config.output_dir}")
            
        except Exception as e:
            self.logger.log(f"TRAINING FAILED: {str(e)}", level="ERROR")
            raise


def main():
    """Entry point."""
    config = TrainingConfig()
    trainer = INCATrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
