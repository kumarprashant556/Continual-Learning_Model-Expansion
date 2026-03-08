# Plan: Continual Learning Benchmarking for INCA-GPT2

This plan outlines a robust benchmarking strategy for your INCA-GPT2 continual learning architecture, focusing on temporal, knowledge-evolving, and domain-adaptive NLP tasks. It selects the best datasets and papers for both general continual learning and your specific interests (history, news, medical, QA, and language modeling).

## 1. Define Task Suite

Select tasks that cover diverse NLP challenges:
- **Language Modeling**: Continual pretraining on streaming text
- **Open-Domain QA**: Knowledge-intensive question answering (current focus with RealtimeQA)
- **Text Classification**: Topic classification, sentiment analysis with domain drift
- **Named Entity Recognition (NER)**: Entity recognition with evolving entity types
- **Domain Adaptation**: Adaptation across different text domains (news → medical → social media)

---

## 2. Select and Prepare Datasets

### 2.1 Primary Datasets

| Dataset | Domain | Task | Size | Temporal | Why Use? |
|---------|--------|------|------|----------|----------|
| **TemporalWiki / Wiki-Stream** | Wikipedia | Knowledge tracking, LM | ~1.5B tokens | Monthly snapshots | Knowledge evolution, history tracking |
| **CC-News** | News | LM, Classification | ~100B+ tokens | Daily crawls | News stream, topic drift, event detection |
| **AG News** | News | Classification | ~120K articles | By year | Quick topic drift benchmark |
| **CORD-19** | Biomedical | LM, QA, Classification | ~1M papers | Weekly updates | Medical innovation tracking |
| **PubMed Abstracts** | Biomedical | LM, NER, Classification | ~30M abstracts | Monthly updates | Domain adaptation, medical continual learning |
| **Amazon Reviews** | E-commerce | Sentiment, Classification | ~130M reviews | By year | Sentiment drift, long-term domain adaptation |
| **WMT News** | News (multilingual) | Machine Translation | ~50M sent pairs | Yearly | Multilingual, domain adaptation |
| **RealtimeQA** | News + Wikipedia | Open-Domain QA | ~90 probes/week | Weekly | Current use, knowledge QA over time |

### 2.2 Multitask Benchmarks

- **GLUE/SuperGLUE**: Split by domain or artificially by time
- **XTREME**: Multilingual multitask (classification, NER, QA, etc.)
- **TACRED**: Relation extraction with temporal splits
- **OntoNotes**: NER, coreference with temporal splits

### 2.3 Additional Specialized Datasets

- **QuAC**: Dialog-based QA with context and memory
- **SQuAD 2.0**: Reading comprehension (static, but useful for domain adaptation)
- **BioASQ**: Biomedical QA
- **OpenBookQA**: Science QA

---

## 3. Implement Data Loaders

Extend `INCALoader` to support multiple datasets with unified time-based iteration:

### 3.1 Proposed Architecture

```
DataLoader Registry:
├── RealtimeQALoader (current)
├── TemporalWikiLoader
├── NewsStreamLoader (CC-News, AG News)
├── BiomedicalStreamLoader (PubMed, CORD-19)
├── AmazonReviewsLoader (sentiment, classification)
├── WMTNewsLoader (multilingual)
└── GLUETimeLoader (multitask)
```

Each loader should:
- Support time-based iteration (weekly, monthly, yearly)
- Aggregate multiple time steps for sufficient probe/task data
- Expose task-specific metrics (accuracy for classification, F1 for NER, etc.)
- Handle missing files gracefully

### 3.2 Key Implementation Details

- Default aggregation: `aggregate_weeks=12` (ensures ~100-240 samples per evaluation)
- Support both **stream data** (for LM training) and **probe data** (for evaluation)
- For classification/NER: Use labeled data as both train and eval probes

---

## 4. Benchmark Against State-of-the-Art

### 4.1 Baseline Methods

Compare INCA-GPT2 against standard continual learning approaches:

| Method | Type | Key Idea |
|--------|------|----------|
| **Fine-Tuning** | Naive | Update all parameters (baseline for forgetting) |
| **EWC** | Regularization | Penalize changes to important parameters |
| **Learning without Forgetting (LwF)** | Distillation | Maintain old task performance via KD |
| **GEM** | Memory | Constrain gradient to avoid increasing old task loss |
| **Experience Replay (ER)** | Memory | Store and replay old samples |
| **AdapterCL** | Adapter | Task-specific adapters for continual learning |
| **ER + Adapter** | Hybrid | Combine replay with adapters |

### 4.2 Evaluation Frameworks

- **Avalanche**: (https://avalanche.continualai.org/) Standardized CL benchmarks and metrics
- **CL-Bench**: Continual learning benchmark suite for NLP

---

## 5. Evaluation Protocol

### 5.1 Metrics

For **each task/dataset**, track:

1. **Accuracy Metrics**
   - **Task Accuracy**: Current task performance
   - **Average Accuracy (AA)**: Mean accuracy on all seen tasks
   - **Forgetting (BWT)**: Backward transfer / task degradation over time
   - **Forward Transfer (FWT)**: Performance improvement on new tasks due to prior learning

2. **Loss-Based Metrics**
   - Training loss
   - Validation loss (on current task)
   - Replay loss (if using experience replay)

3. **Stability Metrics**
   - Accuracy granularity (probes / evaluation period)
   - Accuracy variance (std dev over epochs)
   - Plateau detection accuracy (when does saturation occur?)

### 5.2 Evaluation Schedule

- Evaluate every **aggregated period** (e.g., every 12 weeks)
- Track all previously seen tasks for backward transfer
- Publish results in table format:
  ```
  Week | LM Loss | RealtimeQA Acc | Topic Acc | NER F1 | Avg Acc | Forgetting
  -----|---------|----------------|-----------|--------|---------|----------
  ```

### 5.3 Visualizations

- **Line plots**: Accuracy over time per task
- **Heatmaps**: Accuracy matrix (tasks × time steps) to see forgetting
- **Stability plots**: Loss variance, plateau detection triggers
- **Comparison plots**: INCA vs. baselines (EWC, ER, LwF, etc.)

---

## 6. Paper Reproduction and Comparison

Compare your results against published continual learning benchmarks:

### 6.1 Key Papers to Reference

1. **"Continual Learning in Natural Language Processing: A Survey"** (Delange et al., 2021)
   - Comprehensive overview of CL techniques in NLP
   - Metrics: AA, BWT, FWT

2. **"Avalanche: an End-to-End Library for Continual Learning"** (Lomonaco et al., 2021)
   - Reference implementation of CL benchmarks
   - Use their evaluation protocol for standardization

3. **"CL-Bench: A Continual Learning Benchmark Suite for NLP"** (d'Autume et al., 2023)
   - Standardized NLP continual learning benchmarks
   - Reference results for multiple datasets

4. **"TemporalWiki: A Benchmark for Knowledge-Intensive Temporal Reasoning"** (Jung et al., 2022)
   - Knowledge evolution tracking
   - Temporal QA evaluation protocol

5. **"StreamingQA: A Benchmark for Adaptation to New Knowledge over Time"** (Jang et al., 2021)
   - Real-time QA benchmarking (directly relevant to your work)
   - Forward transfer metrics for QA

6. **"Experience Replay for Continual Learning"** (Buzzega et al., 2020)
   - Strong empirical baseline for CL in NLP/vision

7. **"Adapter-Based Continual Learning"** (Rusu et al., 2021)
   - Parameter-efficient continual learning
   - Good comparison point for your selector mechanism

### 6.2 Expected Baselines

Based on published work:
- **Fine-tuning**: ~40-60% accuracy, high forgetting
- **EWC**: ~55-70% accuracy, moderate forgetting
- **ER (small buffer)**: ~65-75% accuracy, low forgetting
- **ER (large buffer)**: ~75-85% accuracy, minimal forgetting
- **INCA-GPT2 (expected)**: ~70-80%+ (with selector + replay mechanism)

---

## 7. Implementation Roadmap

### Phase 1: Core Setup (1-2 weeks)
- [ ] Extend `INCALoader` for multiple datasets
- [ ] Implement baseline methods (EWC, LwF, GEM)
- [ ] Set up evaluation framework

### Phase 2: Primary Benchmarking (2-3 weeks)
- [ ] RealtimeQA (current work)
- [ ] TemporalWiki + CC-News
- [ ] Multitask (GLUE splits)

### Phase 3: Extended Benchmarking (2-4 weeks)
- [ ] PubMed + CORD-19 (medical domain)
- [ ] Amazon Reviews (sentiment drift)
- [ ] WMT News (multilingual)

### Phase 4: Analysis & Writing (1-2 weeks)
- [ ] Generate comparison tables and plots
- [ ] Write results and discussion
- [ ] Publication/preprint submission

---

## 8. Further Considerations

### 8.1 Dataset Licensing and Access
- **PubMed**: Free, but large (~250GB). Use sampling or abstracts only.
- **CORD-19**: Free, open access.
- **CC-News**: Free via Common Crawl.
- **Amazon Reviews**: Free from various academic sources.
- **WMT News**: Free, open data (through WMT shared task).

### 8.2 Compute Resources
- **GPU**: 1x A100 (40GB) or 2x V100 (32GB each) for efficient training
- **Storage**: ~500GB-1TB for all datasets
- **Time**: Continual training is sequential; estimate 2-4 weeks for full pipeline

### 8.3 Task Diversity Strategy
- Start with **language modeling** + **QA** (current focus)
- Add **classification** (topic, sentiment)
- Add **NER** (entity tracking)
- Add **dialog** (if time permits)

### 8.4 Hyperparameter Tuning
- Aggregate weeks: `aggregate_weeks=12` (default, may adjust per dataset)
- QA loss weight: `beta=0.2` in `CombinedLoss`
- Replay buffer: `capacity=500` (adjust based on memory)
- Learning rate: `lr=1e-5` (may need dataset-specific tuning)

---

## 9. Success Metrics

### INCA-GPT2 is successful if:
1. ✅ Achieves >70% accuracy on RealtimeQA (vs 40% baseline)
2. ✅ Shows <5% forgetting on previous tasks (vs >20% for fine-tuning)
3. ✅ Outperforms EWC/LwF baselines on 3+ datasets
4. ✅ Demonstrates forward transfer on unseen domains
5. ✅ Accuracy granularity shows ~0.5% steps (not 5% jumps) with 12-week aggregation

---

## 10. Next Steps

1. **Immediate**: Run training with fixed `aggregate_weeks=12` (see ACCURACY_GRANULARITY_FIX.md)
2. **Short-term**: Prepare TemporalWiki and CC-News loaders
3. **Medium-term**: Implement baseline methods (EWC, ER)
4. **Long-term**: Execute full benchmarking suite

Start with RealtimeQA + TemporalWiki as primary validation, then expand.
