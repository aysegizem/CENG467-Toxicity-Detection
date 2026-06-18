# Context-Aware Toxicity Detection and Mitigation in Competitive Multiplayer Game Chats

**Course:** CENG 467 — Natural Language Understanding and Generation  
**Institution:** İzmir Institute of Technology (İYTE)  
**Student:** Ayşe Gizem Akkoç — 300201028

---

## Overview

Standard NLP toxicity models produce high false-positive rates in competitive gaming environments because they have no way of distinguishing aggressive-sounding gaming jargon ("kill", "shoot", "feed") from genuine harassment. This project builds and evaluates a two-phase moderation pipeline:

1. **Detection** — fine-tuned RoBERTa classifier that correctly contextualizes gaming vocabulary
2. **Mitigation** — BART-based text detoxification that rewrites toxic messages into neutral equivalents

The core design goal is **false-positive reduction**: avoiding the unfair penalization of players for ordinary in-game communication.

---

## Results Summary

### Phase 1 — Toxicity Detection

| Model | Toxic Precision | Toxic Recall | Toxic F1 |
|---|---|---|---|
| Naive Bayes (TF-IDF) | 0.92 | 0.51 | 0.66 |
| Zero-Shot Llama-3 | 0.00 | 0.00 | 0.00 |
| Pre-trained BERT (`unitary/toxic-bert`) | 0.99 | 0.94 | 0.96 |
| **Fine-tuned RoBERTa (proposed)** | **1.00** | **0.72** | **0.83** |

All results on a balanced 500-sample test set. The fine-tuned RoBERTa achieved **zero false positives** on the unseen balanced test set, directly addressing the core challenge of the project. Recall of 0.72 reflects a known limitation of the constrained training setup (2,000 samples, 1 epoch, CPU).

### Phase 2 — Text Detoxification (BART)

| Metric | Score |
|---|---|
| Successful Toxicity Alleviation (STA) | 68% |
| Average ROUGE-L | 0.5874 |

BART handles simple profanity well but struggles with implicit threats and domain-specific gaming context. See the report for detailed failure case analysis.

---

## Models & Tools

| Component | Model / Library |
|---|---|
| Detection baseline | `unitary/toxic-bert` |
| Proposed classifier | `roberta-base` (fine-tuned) |
| Detoxification | `s-nlp/bart-base-detox` |
| LLM supplement | Llama-3.1 via Groq API |
| STA judge | `unitary/toxic-bert` |
| Framework | PyTorch, Hugging Face Transformers, scikit-learn |

---

## Dataset

[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (~159,000 samples), aggregated into binary labels (toxic / non-toxic). A 2,000-sample subset was used for fine-tuning due to local hardware constraints.

---

## Repository Structure

```
├── notebooks/
│   ├── naive_bayes_baseline.ipynb
│   ├── zero_shot_llm_baseline.ipynb
│   ├── pretrained_bert_baseline.ipynb
│   └── roberta_finetuning.ipynb
│   ├── text_detoxification.ipynb
│   └── smart_detox_llm.ipynb
├── data/
│   └── train.csv
├── report/
│   └── CENG467_Final_Report.pdf
└── README.md   
└── requirements.txt
```

## Reproducing the Experiments

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Groq API key** (only required for LLM notebooks):
```bash
export GROQ_API_KEY="your_key_here"
```

Each notebook is self-contained. Run them in order within each phase.

## Key Findings

- Bag-of-Words models miss ~50% of toxic content and lack contextual understanding
- Zero-shot LLMs exhibit a strong safety bias and fail to flag any toxicity even with strict prompts
- Fine-tuned RoBERTa achieves zero false positives but misses ~28% of toxic messages under constrained training — scaling data and epochs is the primary improvement path
- BART detoxification achieves 68% STA but fails on implicit real-world threats and produces truncated outputs for complex inputs
- Prompt engineering alone is insufficient for safe generative moderation
```
