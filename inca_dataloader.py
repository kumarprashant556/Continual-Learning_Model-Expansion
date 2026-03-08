import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

class StreamDataset(Dataset):
    """
    Causal LM Dataset combining both raw stream text AND probe QA pairs.
    This exposes the model to the QA format during training so it can
    better rank answers during evaluation.
    """
    def __init__(self, data, probe_data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.examples = []
        
        # Add raw stream text
        for item in data:
            text = item.get("text", "")
            if len(text) > 10:
                self.examples.append(text)
        
        # Add QA format: "Question: {q}\nAnswer: {correct_answer}"
        # This teaches the model the QA structure and correct answers
        for probe in probe_data:
            question = probe.get('question', '')
            answer_text = probe.get('answer_text', '')
            
            if question and answer_text:
                qa_text = f"Question: {question}\nAnswer: {answer_text}"
                self.examples.append(qa_text)
        
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # For Causal LM (GPT), Labels = Input_IDs
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": enc["input_ids"].squeeze()
        }

class INCALoader:
    """
    The Time-Machine.
    Iterates through the dataset week-by-week or month-by-month.
    Can aggregate multiple months into single training period for more probe data.
    """
    def __init__(self, data_root, tokenizer: PreTrainedTokenizer, batch_size=4, max_seq_len=512, aggregate_weeks=12):
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.aggregate_weeks = aggregate_weeks  # Aggregate N weeks at a time (default: 12 weeks = ~3 months)
        # This ensures ~100-150 probes per evaluation (20 probes/week × 5-8 weeks minimum)
        
        self.stream_dir = self.data_root / "stream"
        self.probes_dir = self.data_root / "probes"
        
        # 1. Index all available weeks
        # We look at stream files to define the timeline
        stream_files = sorted(list(self.stream_dir.glob("*.jsonl")))
        self.timeline = [f.stem for f in stream_files] # ['20200103', '20200110'...]
        
        print(f"[INCALoader] Found {len(self.timeline)} chronological steps.")
        print(f"[INCALoader] Aggregating {self.aggregate_weeks} week(s) per training period for more probe data.")

    def _load_jsonl(self, path):
        data = []
        if not path.exists(): return data
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except: continue
        return data

    def __iter__(self):
        """
        Yields (step_id, train_loader, probe_data) for evaluation periods.
        
        Each period aggregates aggregate_weeks consecutive weeks to ensure sufficient
        probe data for fine-grained accuracy measurement.
        
        Example: With 20 probes/week and aggregate_weeks=12:
        - 240 total probes per period
        - Accuracy granularity = 1/240 = 0.42% (vs 5% with single week)
        - Can see improvements in 0.42% steps instead of 5% jumps
        """
        idx = 0
        while idx < len(self.timeline):
            # Collect data for aggregate_weeks consecutive weeks
            aggregated_streams = []
            aggregated_probes = []
            step_ids = []
            
            for i in range(self.aggregate_weeks):
                if idx + i >= len(self.timeline):
                    break
                
                step_id = self.timeline[idx + i]
                step_ids.append(step_id)
                
                # Load stream
                stream_path = self.stream_dir / f"{step_id}.jsonl"
                if stream_path.exists():
                    raw_stream = self._load_jsonl(stream_path)
                    aggregated_streams.extend(raw_stream)
                
                # Load probes
                probe_path = self.probes_dir / f"{step_id}.jsonl"
                if probe_path.exists():
                    probes = self._load_jsonl(probe_path)
                    aggregated_probes.extend(probes)
            
            # Skip if no data
            if len(aggregated_streams) == 0:
                idx += self.aggregate_weeks
                continue
            
            # Create dataset and loader from aggregated data
            # Now includes both raw text AND probe QA pairs
            train_ds = StreamDataset(aggregated_streams, aggregated_probes, self.tokenizer, self.max_seq_len)
            
            train_loader = DataLoader(
                train_ds, 
                batch_size=self.batch_size, 
                shuffle=True,
                num_workers=0
            )
            
            # Use first step_id as identifier
            period_label = f"{step_ids[0]}" if len(step_ids) == 1 else f"{step_ids[0]}-{step_ids[-1]}"
            
            num_probes = len(aggregated_probes)
            accuracy_granularity = 100.0 / num_probes if num_probes > 0 else 0
            
            print(f"[INCALoader] Period: {period_label} ({len(aggregated_streams)} stream, {num_probes} probes, {accuracy_granularity:.2f}% granularity)")
            
            yield period_label, train_loader, aggregated_probes
            
            idx += self.aggregate_weeks

    def get_specific_week(self, week_id):
        """Helper to jump to a specific date for debugging"""
        if week_id not in self.timeline:
            return None, None
            
        stream_path = self.stream_dir / f"{week_id}.jsonl"
        probe_path = self.probes_dir / f"{week_id}.jsonl"
        
        probes = self._load_jsonl(probe_path)
        train_ds = StreamDataset(self._load_jsonl(stream_path), probes, self.tokenizer, self.max_seq_len)
        
        return DataLoader(train_ds, batch_size=self.batch_size), probes