# Context-Aware Toxicity Detection and Mitigation in Competitive Multiplayer Game Chats

**Course:** CENG 467 Natural Language Understanding and Generation  
**Institution:** İzmir Institute of Technology (İYTE)  

## Project Overview
Moderating player chats in competitive online games is a highly challenging NLP task. Standard toxicity detection models often produce high false-positive rates by misclassifying common gaming jargon (e.g., "kill", "shoot", "destroy") as toxic or violent behavior. 

This project aims to build a context-aware NLP pipeline to accurately distinguish between competitive banter and genuine toxicity. By utilizing domain-specific datasets (such as the Jigsaw Toxic Comment Classification Challenge paired with multiplayer chat corpora), this study explores how Transformer-based architectures (e.g., RoBERTa) and text detoxification models can effectively moderate game chats without penalizing standard gamer communication.

## Key Objectives
* Implement and evaluate baseline ML models (Naive Bayes/SVM with TF-IDF) and Zero-Shot LLMs.
* Fine-tune a Transformer model (RoBERTa) specifically for the gaming domain to reduce false positives.
* Explore Natural Language Generation (NLG) techniques to detoxify harmful text while preserving the original semantic intent.
