# SMS Spam Detection: A Comparative Study of NLP Classification Models

## Case Study Report

---

## Table of Contents
1. [Introduction](#1-introduction)
2. [Dataset Description](#2-dataset-description)
3. [Methodology](#3-methodology)
4. [Results](#4-results)
5. [Comparative Analysis](#5-comparative-analysis)
6. [Discussion](#6-discussion)
7. [Conclusion](#7-conclusion)
8. [References](#8-references)

---

## 1. Introduction

### 1.1 Problem Statement

SMS spam has become a significant problem in mobile communication, causing inconvenience to users and potential security risks. The goal of this project is to develop and compare multiple machine learning models for automatically classifying SMS messages as spam or ham (legitimate).

### 1.2 Objectives

1. Implement a comprehensive text preprocessing pipeline for SMS data
2. Develop and train five different classification models:
   - Logistic Regression
   - Multinomial Naive Bayes
   - Random Forest
   - Support Vector Machine (SVM)
   - LSTM Neural Network
3. Evaluate models using standard classification metrics
4. Provide comparative analysis and recommendations

### 1.3 Significance

Automated spam detection is crucial for:
- Protecting users from phishing and scam attempts
- Reducing unwanted message clutter
- Improving mobile communication experience
- Demonstrating practical NLP applications

---

## 2. Dataset Description

### 2.1 Source

**SMS Spam Collection Dataset**
- **Origin**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

### 2.2 Dataset Statistics

| Attribute | Value |
|-----------|-------|
| Total Messages | 5,574 |
| Ham (Legitimate) | 4,827 (86.6%) |
| Spam | 747 (13.4%) |
| Average Message Length | 80 characters |
| Average Word Count | 15 words |

### 2.3 Class Distribution

The dataset exhibits class imbalance with approximately 87% ham and 13% spam messages. This imbalance was addressed during model evaluation by using stratified sampling and focusing on metrics like F1-score rather than just accuracy.

---

## 3. Methodology

### 3.1 Data Preprocessing Pipeline

**Pipeline Steps:**
1. Text Cleaning (lowercase, remove URLs, special characters)
2. Tokenization (split into words)
3. Stopword Removal (remove common words)
4. Lemmatization (convert to base form)

### 3.2 Feature Extraction

**TF-IDF Vectorization (Traditional ML)**
- Max Features: 5,000
- N-gram Range: (1, 2)

**Sequence Tokenization (LSTM)**
- Max Words: 5,000
- Max Sequence Length: 100

### 3.3 Models Implemented

1. **Logistic Regression**: Linear classifier, fast and interpretable
2. **Naive Bayes**: Probabilistic classifier, very fast training
3. **Random Forest**: Ensemble method, provides feature importance
4. **SVM**: Maximum margin classifier, effective in high dimensions
5. **LSTM**: Deep learning model, captures sequential patterns

### 3.4 Train-Test Split

- Training: 80% (4,459 samples)
- Test: 20% (1,115 samples)
- Stratified sampling to maintain class distribution

---

## 4. Results

### 4.1 Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.970 | 0.953 | 0.912 | 0.932 | 0.989 |
| Naive Bayes | 0.961 | 0.987 | 0.793 | 0.879 | 0.982 |
| Random Forest | 0.973 | 0.994 | 0.841 | 0.911 | 0.991 |
| SVM | 0.980 | 0.972 | 0.924 | 0.948 | 0.993 |
| LSTM | 0.979 | 0.951 | 0.931 | 0.941 | 0.992 |

### 4.2 Training Time Comparison

| Model | Training Time |
|-------|---------------|
| Naive Bayes | ~0.05s |
| Logistic Regression | ~0.15s |
| Random Forest | ~2.50s |
| SVM | ~5.00s |
| LSTM | ~45.00s |

---

## 5. Comparative Analysis

### 5.1 Key Findings

1. **SVM achieves the highest F1-score** (0.948) providing the best balance
2. **LSTM follows closely** (0.941) with good generalization
3. **Naive Bayes is fastest** but has lower recall
4. **All models exceed 96% accuracy**

### 5.2 Trade-offs

- **Speed vs Accuracy**: Naive Bayes is fastest, SVM is most accurate
- **Interpretability**: Logistic Regression is most interpretable
- **Complexity**: LSTM requires more resources but captures context

---

## 6. Discussion

### 6.1 Insights

1. Traditional ML performs comparably to deep learning on this dataset
2. Text preprocessing significantly impacts performance
3. SVM with linear kernel works excellently for text classification
4. Class imbalance requires careful metric selection

### 6.2 Limitations

- Dataset size may limit deep learning potential
- Models may not generalize to other spam types
- Spam patterns evolve over time

---

## 7. Conclusion

### 7.1 Summary

| Rank | Model | F1-Score | Recommendation |
|------|-------|----------|----------------|
| 1 | SVM | 0.948 | Best for production |
| 2 | LSTM | 0.941 | Best for larger datasets |
| 3 | Logistic Regression | 0.932 | Best speed/accuracy balance |
| 4 | Random Forest | 0.911 | Good for feature analysis |
| 5 | Naive Bayes | 0.879 | Best for speed |

### 7.2 Recommendations

- **Production**: Use SVM with TF-IDF features
- **Quick Prototyping**: Use Naive Bayes
- **Research**: Experiment with LSTM and transformers

---

## 8. References

1. UCI ML Repository - SMS Spam Collection Dataset
2. Scikit-learn Documentation
3. TensorFlow Documentation
4. NLTK Documentation

---

*Report prepared as part of NLP Case Study Assessment*

**Repository**: https://github.com/Sahilsonii/NLP-CA3
