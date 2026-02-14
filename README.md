# 🚀 Narrative Nexus  
### 🧠 Dynamic NLP-Powered Customer Review Intelligence Platform  

🔗 **Live Demo:**  
https://narrative-nexus-lyuetvyqfgkrchwlbk3evj.streamlit.app/

---

## ✨ Overview

**Narrative Nexus** is an end-to-end Natural Language Processing (NLP) web application that analyzes customer reviews and transforms raw text into actionable insights.

The system integrates machine learning models, topic modeling, and interactive visualization into a unified Streamlit dashboard.

This project demonstrates practical implementation of:

- Supervised Sentiment Classification  
- Topic Modeling using LDA  
- Extractive Text Summarization  
- Keyword & Word Cloud Visualization  
- Interactive Data Visualization  

---

## 🎯 Core Features

### 🔵 1. Sentiment Detection
- Classifies reviews into **Positive / Neutral / Negative**
- Displays prediction confidence
- Visual sentiment distribution chart

---

### 🟣 2. Extractive Summarization
- LSA-based summary generation
- Highlights key review insights
- Produces concise, meaningful summaries

---

### 🟢 3. Topic Modeling (LDA)
- Identifies hidden themes within reviews
- Displays topic probability distribution
- Shows top representative keywords per topic

---

### 🟡 4. Word Cloud Visualization
- Removes stopwords
- Highlights dominant keywords
- Visual representation of frequent terms

---

### 🔴 5. Unified Analytics Dashboard
- Sentiment pie chart
- Topic probability bar chart
- Keyword importance visualization
- Word cloud
- All insights in a single interface

---

## 🏗 Architecture Overview

```
Narrative-Nexus/
│
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── model.pkl
    ├── vectorizer.pkl
    ├── label_encoder.pkl
    ├── lda_model_v1.gensim
    ├── lda_model_v1.gensim.state
    ├── lda_model_v1.gensim.expElogbeta.npy
    └── dictionary_v1.gensim
```

---

## ⚙️ Technology Stack

| Category | Tools |
|----------|-------|
| Programming Language | Python |
| Web Framework | Streamlit |
| ML Model | TF-IDF + Logistic Regression |
| Topic Modeling | Gensim LDA |
| NLP Processing | NLTK |
| Visualization | Matplotlib, WordCloud |

---

## 🧠 Machine Learning Components

- TF-IDF Vectorization for feature extraction  
- Logistic Regression for sentiment classification  
- Label Encoding for multi-class prediction  
- LDA (Latent Dirichlet Allocation) for topic discovery  
- Pre-trained models integrated into production-ready app  

---

## 🛠 Installation

```bash
git clone https://github.com/ManishaGurugubelli/Narrative-Nexus.git
cd Narrative-Nexus
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶ Run Locally

```bash
streamlit run app.py
```

---

## 📦 Model Files

Pre-trained ML and LDA models are included inside the `models/` directory for demonstration purposes.

---

## 💡 Learning Highlights

- End-to-end NLP pipeline development  
- Multi-model integration in a single web application  
- Handling imbalanced sentiment classes  
- Topic coherence optimization  
- Deployment-ready Streamlit architecture  
- Cloud deployment using Streamlit Community Cloud  

---

## 👩‍💻 Author

**Manisha Gurugubelli**

Originally developed during internship and independently refined for portfolio presentation.

