# 📱 Smartphone AI Analyzer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

**Turning raw specs and YouTube comments into actionable intelligence.**

A comprehensive data science dashboard that combines **Artificial Neural Networks (ANN)** for hardware benchmarking and **Natural Language Processing (NLP)** for real-time sentiment analysis of smartphone reviews.
<img width="2558" height="1262" alt="Screenshot 2026-01-30 154940" src="https://github.com/user-attachments/assets/a2a16b4a-2833-4405-8d1c-fe6ec57c1398" />


---

## 🚀 Key Features

### 1. ⚡ Hardware Intelligence (ANN)
* **Predictive Modeling:** Uses a PyTorch Neural Network trained on global dataset specifications.
* **Input Factors:** Analyzes RAM, Storage, and Market Price.
* **Output:** Predicts the potential Antutu Benchmark score with a visual gauge meter comparing it to the actual database score.

### 2. 💬 Netizen Verdict (Real-Time NLP)
* **Live Scraping:** Connects to YouTube to find the most relevant review video for the selected device.
* **Sentiment Analysis:** Extracts comments and analyzes public opinion using a lexicon-based approach.
* **Visualization:** Displays a "Buy/Pass" verdict, sentiment distribution charts, and chat-style comment feeds.

### 3. 🔍 Smart Data Pipeline
* **Automated Cleaning:** Handles currency conversion (INR/IDR to USD), text normalization, and fuzzy matching.
* **Brand Filtering:** Cascading dropdowns to filter devices by Brand (Samsung, Xiaomi, etc.) and Model.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit, Plotly (Interactive Charts)
* **Deep Learning:** PyTorch (Feedforward Neural Network)
* **Data Processing:** Pandas, NumPy, Scikit-Learn (Scalers)
* **Scraping & NLP:** `Youtube-python`, `youtube-comment-downloader`, `thefuzz`
* **Utilities:** Joblib (Model Serialization), TQDM

---

## 📂 Project Structure

```bash
smartphone-ai-analyzer/
├── data/
│   ├── raw/               # Raw datasets from Kaggle
│   └── processed/         # Cleaned CSVs for training
├── src/
│   ├── app.py             # Main Streamlit Dashboard Application
│   ├── train.py           # PyTorch Training Script (ANN)
│   ├── data_pipeline.py   # ETL Pipeline (Cleaning & Merging)
│   └── youtube_scraper.py # NLP & Scraping Modules
├── model_antutu.pth       # Trained PyTorch Model
├── scaler_x.pkl           # Feature Scaler
├── scaler_y.pkl           # Target Scaler
├── requirements.txt       # Project Dependencies
└── README.md              # Documentation
```

## ⚡ Quick Start

Get the application running locally in minutes.

### 1. Installation
Clone the repository and install the required Python packages.

```bash
# Clone the repo
git clone https://github.com/r3nyah/smartphone-ai-analyzer.git
cd smartphone-ai-analyzer

# Install dependencies
pip install -r requirements.txt
```

### 2. Data & Training (Optional)
These steps are only necessary if you want to rebuild the dataset or retrain the model from scratch. The repo comes with pre-trained models.
```bash
# Re-scrape and clean data from Kaggle sources
python src/data_pipeline.py

# Retrain the PyTorch Neural Network
python src/train.py
```

### 3. Launch App
Start the Streamlit dashboard server.
```bash
streamlit run src/app.py
```

## 🧠 System Architecture
### 🤖 1. Hardware Prediction Engine (ANN)
To estimate the potential performance of a device, we utilize a Feedforward Neural Network (Multi-Layer Perceptron) built with PyTorch. The model creates a non-linear mapping between raw hardware specifications and synthetic benchmark scores.
- Input Dimensions: 3 (RAM, Storage Capacity, Market Price)
- Hidden Layers: 3 Dense Layers with ReLU activation for feature extraction.
- Output: Single continuous variable (Antutu Score).
```
graph LR
    I[Input Layer] --> H1[Hidden 1]
    H1 --> H2[Hidden 2]
    H2 --> H3[Hidden 3]
    H3 --> O[Output Layer]

    subgraph Architecture ["Feature Extraction Flow"]
    direction TB
    I["RAM, Storage, Price"]
    H1["Dense (64) + ReLU"]
    H2["Dense (128) + ReLU"]
    H3["Dense (64) + ReLU"]
    O["Predicted Score"]
    end
    
    style I fill:#2ecc71,stroke:#27ae60,color:white
    style O fill:#3498db,stroke:#2980b9,color:white
    style H1 fill:#f1c40f,stroke:#f39c12
    style H2 fill:#f1c40f,stroke:#f39c12
    style H3 fill:#f1c40f,stroke:#f39c12
```

### 💬 2. Sentiment Intelligence Engine (NLP)
This module executes a real-time ETL (Extract, Transform, Load) pipeline to gauge public opinion. By scraping and analyzing the latest comments from Indonesian YouTube review videos, the system generates a dynamic "Netizen Verdict."
- Data Source: YouTube Data API (via wrapper).
- Methodology: Lexicon-based sentiment polarity scoring.
- Logic: Aggregates compound scores to classify consensus as Positive, Neutral, or Negative.
```
flowchart TD
    Start([User Selects Device]) --> Search{Search Engine}
    Search -->|Query: 'Review Device Indonesia'| Video[Fetch Top Video Metadata]
    Video --> Scrape[Scrape Top 40 Comments]
    Scrape --> NLP[Sentiment Analysis Engine]
    
    NLP -->|Score > 0| Pos[Positive Sentiment]
    NLP -->|Score < 0| Neg[Negative Sentiment]
    
    Pos & Neg --> Viz[Final Verdict & Visualization]
    
    style Start fill:#9b59b6,stroke:#8e44ad,color:white
    style Viz fill:#34495e,stroke:#2c3e50,color:white
```
