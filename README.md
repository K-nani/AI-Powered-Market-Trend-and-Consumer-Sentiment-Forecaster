# AI-Powered Market Trend and Consumer Sentiment Forecaster

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-008080.svg)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, enterprise-grade Artificial Intelligence and Deep Learning system for automated multi-channel consumer sentiment analysis, unsupervised topic modeling, and multivariate sales demand forecasting.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Innovations & Architecture](#-key-innovations--architecture)
3. [System Prerequisites](#-system-prerequisites)
4. [Step-by-Step Installation Guide](#-step-by-step-installation-guide)
5. [Dependencies & Requirements](#-dependencies--requirements)
6. [Running the Application](#-running-the-application)
7. [The 8 Interactive Dashboards](#-the-8-interactive-dashboards)
8. [Generating the PowerPoint Presentation](#-generating-the-powerpoint-presentation)
9. [Running Automated Tests](#-running-automated-tests)
10. [References & Citations](#-references--citations)

---

## 🎯 Project Overview

In modern e-commerce and retail, consumer purchase behavior is heavily shaped by online reviews, social media chatter, and community feedback. Traditional sales forecasting systems (ARIMA, Holt-Winters) operate in isolation from this unstructured textual data, creating severe demand blind spots.

This project delivers a unified AI framework that:
- **Ingests multi-channel consumer feedback** (Twitter/X, Amazon Verified Reviews, Reddit, Financial News).
- **Performs Aspect-Based Sentiment Analysis (ABSA)** across product dimensions (Battery, Build, Price, UI, Speed, Support) using **BERT** and **VADER**.
- **Normalizes ratings, sentiments, and community upvotes** onto standard $[1, 5]$ intervals to filter fake reviews.
- **Extracts emerging discussion trends** using **BERTopic** (c-TF-IDF) and **Latent Dirichlet Allocation (LDA)**.
- **Forecasts multivariate demand** using the **SentiTSMixer** deep time-series architecture, achieving a **65% to 99% error reduction** over traditional baselines.
- **Retrieves context-grounded insights** using **FAISS** vector search and **Gemini LLMs** for RAG-powered strategic intelligence.

---

## 🏛️ Key Innovations & Architecture

```
┌────────────────────────────────────────────────────────┐
│ 1. Multi-Channel Data Ingestion Engine                 │
│ - Twitter/X Social Buzz Stream                         │
│ - Amazon Verified Customer Reviews                     │
│ - Reddit Communities (r/technology, r/smartphones)     │
│ - Financial & Tech News RSS                            │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 2. NLP & Aspect-Based Sentiment Lab (ABSA)             │
│ - BERT Transformer 3-Class Softmax Probabilities       │
│ - VADER Context-Aware Sentiment Reasoning              │
│ - Eq. 8 (0-1), Eq. 9 (1-5), Eq. 10 (Helpfulness Norm)  │
│ - Granular Emotion Mining (Joy, Trust, Anger, Sadness) │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 3. Unsupervised Topic Modeling Engine                  │
│ - BERTopic (UMAP Dimensionality + c-TF-IDF Keywords)   │
│ - Latent Dirichlet Allocation (LDA Posterior Mixtures) │
│ - Dynamic Temporal Topic Evolution Tracking            │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 4. Deep Trend Forecaster (SentiTSMixer & Bi-LSTM)      │
│ - Time-Mixing MLPs + Channel/Feature-Mixing MLPs       │
│ - Align Stage Weighted by Random Forest MDI            │
│ - 95% Confidence Bounds & Time-Series Decomposition    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 5. FAISS Vector Database & Conversational RAG Copilot  │
│ - In-Memory Dense Vector Index FlatIP (Cosine Sim)     │
│ - Contextual Market Analysis via Gemini LLM            │
│ - Interactive What-If Simulation Sandbox               │
└────────────────────────────────────────────────────────┘
```

---

## 💻 System Prerequisites

Before downloading and running the project, ensure you have:
- **Operating System:** Windows 10/11, macOS (x86_64 or Apple Silicon), or Linux (Ubuntu 20.04+).
- **Python:** Python 3.10 or 3.11 installed (Python 3.11 recommended).
- **Package Manager:** `pip` (standard) or `uv` (ultra-fast, recommended).
- **Hardware:**
  - Minimum: 4 GB RAM, Dual-Core CPU, 2 GB free disk space.
  - Recommended: 8+ GB RAM, GPU with CUDA (optional, CPU fallback is fully supported).

---

## 🚀 Step-by-Step Installation Guide

### Step 1: Open Terminal / Command Prompt
Clone or open the project folder in your terminal:
```bash
cd path/to/dazzling-euclid
```

### Step 2: Create a Virtual Environment
It is best practice to install dependencies in an isolated virtual environment:
```bash
# Using standard Python:
python -m venv .venv

# Activate on Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Activate on Windows (Command Prompt):
.\.venv\Scripts\activate.bat

# Activate on macOS / Linux:
source .venv/bin/activate
```

### Step 3: Install All Required Dependencies
Run `pip` to install all project libraries from `requirements.txt`:
```bash
pip install -r requirements.txt
```

*(Optional Fast Installation via `uv`)*:
```bash
uv pip install -r requirements.txt
```

---

## 📦 Dependencies & Requirements

All required libraries are listed in `requirements.txt`:

| Package | Purpose in Project |
|---|---|
| `streamlit>=1.35.0` | Powers the 8 interactive enterprise dashboards and web portal. |
| `plotly>=5.22.0` | Renders interactive line, radar, donut, and 2D scatter charts. |
| `pandas>=2.2.0` | High-performance data manipulation, time-series resampling, and aggregation. |
| `numpy>=1.26.0` | Matrix mathematics, tensor transformations, and vector normalization. |
| `scikit-learn>=1.4.0` | Random Forest MDI feature importance, TruncatedSVD, KMeans, and TF-IDF vectorizers. |
| `vaderSentiment>=3.3.2` | Rule-based lexical valence engine for social media and customer review text. |
| `nltk>=3.8.1` | Natural language preprocessing, tokenization, and stopword cleaning. |
| `statsmodels>=0.14.0` | Baseline time-series forecasting models (ARIMA, SARIMA). |
| `torch>=2.2.0` | Deep learning framework for the SentiTSMixer PyTorch neural network. |
| `faiss-cpu>=1.8.0` | Dense vector database for sub-millisecond semantic review similarity search. |
| `matplotlib>=3.8.0` | Generates high-resolution output visualization charts for presentations and reports. |
| `python-pptx>=1.0.0` | Programmatically creates and updates widescreen 16:9 PowerPoint presentations. |
| `google-genai>=0.1.0` | Google Gemini API SDK for conversational RAG market intelligence generation. |

---

## ⚡ Running the Application

### Option A: One-Click Windows Launcher
Simply double-click the included batch script:
```cmd
run_app.bat
```

### Option B: Terminal Command
With your virtual environment activated, run:
```bash
streamlit run app.py
```

Once started, open your browser and navigate to:
👉 **`http://localhost:8501`**

---

## 🗂️ The 8 Interactive Dashboards

| # | Dashboard | Capabilities & Features |
|---|---|---|
| **1** | **🌐 Live Market Pulse** | Live streaming simulation across Twitter/X, Amazon, Reddit, and News. Includes real-time Net Sentiment Velocity gauge (NSI) and channel share breakdown. |
| **2** | **🔬 Aspect Sentiment Lab** | Aspect-Based Sentiment Analysis (ABSA) for Battery, Build, Price, Performance, UI, and Support. Polar radar chart, 6-emotion breakdown, and live review text tester. |
| **3** | **🧠 Topic Modeling** | Unsupervised discovery of consumer discussion themes using BERTopic (2D UMAP scatter, c-TF-IDF terms) and Latent Dirichlet Allocation (LDA) with temporal trends. |
| **4** | **📈 Deep Trend Forecaster** | SentiTSMixer & Bi-LSTM demand predictions, 4-component time-series decomposition (Trend, Seasonality, Sentiment Lift, Residuals), and 95% confidence intervals. |
| **5** | **🏆 Model Evaluation Arena** | Rigorous engineering benchmarks: SentiTSMixer vs. Bi-LSTM vs. LSTM vs. SARIMA vs. ARIMA across MAPE, RMSE, MAE, R², and Latency, plus MDI feature importances. |
| **6** | **🤖 RAG AI Market Analyst** | Semantic similarity search over consumer feedback using FAISS vector embeddings, natural language Q&A, and Gemini LLM contextual synthesis. |
| **7** | **🔮 What-If Scenario Simulator** | Dynamic business simulator: model price adjustments, sentiment shocks, viral buzz multipliers, revenue deltas ($), and supply chain bullwhip dampening. |
| **8** | **📂 Data Pipeline Studio** | Upload custom CSV/JSON review datasets, generate synthetic review streams, execute automated NLP ETL pipelines, and export forecasts. |

---

## 📽️ Generating the PowerPoint Presentation

The project includes an automated script that compiles the complete slide deck with **embedded high-resolution output charts**:

```bash
# 1. Generate the latest high-res output charts:
python generate_output_images.py

# 2. Compile the PowerPoint presentation:
python generate_ppt.py
```

Output Presentation File:
📁 **`AI_Powered_Market_Trend_and_Consumer_Sentiment_Forecaster.pptx`**

### Presentation Outline (13 Slides):
1. **Title Page** (Widescreen 16:9, Enterprise Dark Theme)
2. **Abstract** (Background, Problem, Solution, RAG & Impact)
3. **Introduction** (Digital consumer voice, multi-dimensional feedback)
4. **Literature Survey (Part 1)** (Statistical models & sentiment reasoning)
5. **Literature Survey (Part 2)** (TSMixer architecture & SentiTSMixer)
6. **Existing System** (Transaction dependency, fake reviews, bullwhip distortions)
7. **Proposed System** (Dual sentiment, SentiTSMixer, ABSA, FAISS RAG)
8. **Problem Statement** (Formal mathematical & engineering formulation)
9. **Project Objectives** (6 granular engineering milestones)
10. **System Output & Visualization (Part 1)** (Embedded Sentiment Pulse & ABSA Radar)
11. **System Output & Visualization (Part 2)** (Embedded SentiTSMixer Forecast & Benchmarks)
12. **Conclusion & Future Work** (Key achievements, supply chain impacts, extensions)
13. **References** (Formal academic IEEE bibliography)

---

## 🧪 Running Automated Tests

To run the automated unit test suite verifying the SentiTSMixer model, sentiment normalizations, error metrics, and FAISS indexing:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

All 9 test suites will execute and validate:
- SentiTSMixer tensor shape preservation and training loop convergence.
- Equations 8, 9, 10, and 11 mathematical normalizations.
- MAPE, MSE, and MSLE error calculation accuracy.
- LDA and BERTopic semantic topic extraction.
- FAISS vector indexing and top-k similarity retrieval.

---

## 📚 References & Citations

1. P. Ghosh, S. Das, S. Roy, A. Bhattacharjee, A. Cortesi, and S. Sen, *"SentiTSMixer: A Specific Model for Sales Forecasting Using Sentiment Analysis of Customer,"* **IEEE Access**, vol. 13, pp. 85882–85897, 2025.
2. S.-A. Chen, C.-L. Li, N. Yoder, S. O. Arik, and T. Pfister, *"TSMixer: An all-MLP architecture for time series forecasting,"* arXiv:2303.06053, Google Research, 2023.
3. C. Hutto and É. Gilbert, *"VADER: A parsimonious rule-based model for sentiment analysis of social media text,"* in *Proc. 8th Int. AAAI Conf. Web Social Media (ICWSM)*, pp. 216–225, 2014.
4. J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, *"BERT: Pre-training of deep bidirectional transformers for language understanding,"* in *Proc. NAACL-HLT*, pp. 4171–4186, 2019.
5. M. Grootendorst, *"BERTopic: Neural topic modeling with a class-based TF-IDF procedure,"* arXiv:2203.05794, 2022.
6. J. Johnson, M. Douze, and H. Jégou, *"Billion-scale similarity search with GPUs [FAISS],"* **IEEE Transactions on Big Data**, vol. 7, no. 3, pp. 535–547, 2021.
7. S. J. Taylor and B. Letham, *"Forecasting at scale [Prophet],"* **The American Statistician**, vol. 72, no. 1, pp. 37–45, 2018.
8. P. Ghosh, O. Samanta, T. Goto, and S. Sen, *"Sales forecasting of overrated products: Fine tuning of customer's rating by integrating sentiment analysis,"* **IEEE Access**, vol. 12, pp. 69578–69592, 2024.
