import torch
import numpy as np
from collections import deque

class PlateauDetector:
    """
    INCA-2.0 Stage 1: Multi-Criterion Plateau Detection System.
    Monitors training dynamics to trigger the Freeze & Prune cycle.
    """
    def __init__(self, config):
        self.threshold = config.get("plateau_threshold", 0.75)
        self.patience = config.get("plateau_patience", 3)
        self.window_size = config.get("metric_window", 5)
        
        # Weighted Criteria from Spec (Section 1.2)
        self.weights = {
            "val_loss_stagnation": 0.30,
            "grad_norm_decay": 0.20,
            "param_movement": 0.15,
            "task_perf_sat": 0.10,
            "fisher_info": 0.10,
            "rep_drift": 0.10,
            "opt_state": 0.05
        }
        
        # History buffers
        self.history = {
            "val_loss": deque(maxlen=self.window_size),
            "grad_norms": deque(maxlen=self.window_size),
            "param_dist": deque(maxlen=self.window_size),
            "accuracies": deque(maxlen=self.window_size)
        }
        
        # Internal state
        self.prev_params = None
        self.consecutive_triggers = 0
        
    def update(self, metrics, model):
        """
        Ingest new metrics and model state to calculate plateau score.
        Args:
            metrics (dict): {'loss': float, 'accuracy': float}
            model (nn.Module): The current trainable block
        """
        # 1. Update History
        self.history['val_loss'].append(metrics['loss'])
        self.history['accuracies'].append(metrics.get('accuracy', 0))
        
        # 2. Calculate Gradient Norm (Proxy for "how hard is it trying to learn?")
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.history['grad_norms'].append(total_norm)
        
        # 3. Calculate Parameter Movement (Euclidean distance from last step)
        current_params = [p.data.clone() for p in model.parameters()]
        if self.prev_params is not None:
            dist = 0.0
            for p_new, p_old in zip(current_params, self.prev_params):
                dist += torch.norm(p_new - p_old).item()
            self.history['param_dist'].append(dist)
        self.prev_params = current_params

        return self._calculate_score()

    def _calculate_score(self):
        """Computes the weighted sum of saturation signals."""
        if len(self.history['val_loss']) < self.window_size:
            return 0.0, False # Not enough data yet

        scores = {}
        
        # A. Val Loss Stagnation (Is loss flattening?)
        # Logic: If variance of recent losses is low, we are stagnating.
        loss_std = np.std(self.history['val_loss'])
        loss_mean = np.mean(self.history['val_loss'])
        # Score approaches 1.0 as std -> 0
        scores['val_loss_stagnation'] = 1.0 / (1.0 + (loss_std / (loss_mean + 1e-6)) * 100)

        # B. Gradient Norm Decay (Are gradients vanishing?)
        # Logic: Low gradients mean convergence.
        avg_grad = np.mean(self.history['grad_norms'])
        # Normalize (assuming initial grads were ~1.0-10.0, low is <0.1)
        scores['grad_norm_decay'] = 1.0 / (1.0 + avg_grad)

        # C. Parameter Movement (Are weights stable?)
        if len(self.history['param_dist']) > 0:
            avg_move = np.mean(self.history['param_dist'])
            scores['param_movement'] = 1.0 / (1.0 + avg_move * 10)
        else:
            scores['param_movement'] = 0.0

        # D. Task Saturation (Is accuracy high and stable?)
        acc = list(self.history['accuracies'])
        if len(acc) >= 2:
            # High accuracy + Low improvement = High Score
            current_acc = acc[-1]
            improvement = acc[-1] - acc[0]
            if improvement < 0.01 and current_acc > 0.5: # Hardcoded heuristic
                scores['task_perf_sat'] = 1.0
            else:
                scores['task_perf_sat'] = current_acc # Higher acc contributes to saturation score
        else:
            scores['task_perf_sat'] = 0.0

        # E. (Simplified) Placeholders for expensive metrics
        # Fisher Info & Rep Drift are computationally heavy, setting to neutral for Phase 1
        scores['fisher_info'] = 0.5 
        scores['rep_drift'] = 0.5
        scores['opt_state'] = 0.5

        # --- FINAL WEIGHTED SUM ---
        final_score = sum(self.weights[k] * scores.get(k, 0) for k in self.weights)
        
        # Check Trigger
        triggered = False
        if final_score > self.threshold:
            self.consecutive_triggers += 1
        else:
            self.consecutive_triggers = 0
            
        if self.consecutive_triggers >= self.patience:
            triggered = True
            
        return final_score, triggered

    def reset(self):
        """Call this after freezing to reset stats for the new layer."""
        self.history = {k: deque(maxlen=self.window_size) for k in self.history}
        self.prev_params = None
        self.consecutive_triggers = 0