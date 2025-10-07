
# 🤖 AI Projects Portfolio — Instant Software Solutions

This repository contains projects developed as part of my **AI & Data Science course** at **Instant Software Solutions**, where I explored hands-on implementations of Natural Language Processing, Machine Learning, and AI model deployment using real-world datasets and modern tools.

---

## 🧠 Project 1: Text Emotion Detection using Transformers

### 📘 Overview
This project focuses on **Emotion Detection** in text using **Transformers** (DistilBERT model). It identifies emotions like **joy, anger, sadness, love**, and others from user text or datasets. The model is fine-tuned using Hugging Face Transformers and deployed interactively for real-time predictions.

### ⚙️ Features
✅ Preprocesses and tokenizes text data using Hugging Face Tokenizer  
✅ Fine-tunes a transformer model for multi-class emotion classification  
✅ Supports evaluation with metrics like accuracy, F1-score, precision, and recall  
✅ Easily deployable for real-world text emotion applications  

### 🧩 Tech Stack
| Component | Technology |
|------------|-------------|
| Programming Language | Python |
| Framework | Hugging Face Transformers |
| Libraries | TensorFlow, scikit-learn, Pandas, NumPy, Matplotlib |
| Platform | Google Colab / Jupyter Notebook |

### 🚀 How to Run
1️⃣ Install dependencies  
```bash
pip install transformers tensorflow sklearn pandas numpy matplotlib
```

2️⃣ Train the model  
```python
python train.py
```

3️⃣ Test saved model  
```python
python test_model.py
```

### 🧠 Model Details
- Base model: **DistilBERT**
- Dataset: Custom-labeled emotion dataset (text + emotion labels)
- Classes: 4 detected emotions (Positive, Negative, Neutral, Mixed)

### 💾 Saving and Reusing the Model
You can save and reload the fine-tuned transformer model folder directly (no need to retrain).  
Just upload and load it using:  
```python
model = TFAutoModelForSequenceClassification.from_pretrained("saved_model_folder")
```

### 💡 Learning Outcomes
This project enhanced my understanding of:
- Transformer fine-tuning with labeled data  
- Emotion classification from text inputs  
- Model saving/loading and deployment  
- Evaluation metrics and performance tracking

---

## 🧠 Project 2: AI Text Summarizer & Question Answering App

### 📘 Overview
The AI Text Summarizer & Question Answering App is an interactive NLP tool built using **Streamlit** and **Hugging Face Transformers** that allows users to:
- Upload PDF or TXT files, or paste text manually  
- Automatically summarize long documents using smart chunking  
- Ask questions about the uploaded content using an integrated Question Answering model  

It demonstrates practical applications of NLP in document understanding and automation.

### ⚙️ Features
✅ Summarizes large or complex text (automatic chunking)  
✅ Supports `.pdf`, `.txt`, and direct text input  
✅ Provides interactive Q&A from summarized text  
✅ Clean Streamlit UI for real-time use  
✅ Uses state-of-the-art NLP models from Hugging Face  

### 🧩 Tech Stack
| Component | Technology |
|------------|-------------|
| Programming Language | Python |
| Web Framework | Streamlit |
| NLP Models | Hugging Face Transformers (BART, DistilBERT) |
| Libraries | pdfplumber, torch, transformers |
| Platform | Google Colab / Local Deployment |

### 🚀 How to Run
1️⃣ Install dependencies  
```bash
pip install streamlit transformers torch pdfplumber
```

2️⃣ Run the app  
```bash
streamlit run app.py
```

3️⃣ Open in browser at `http://localhost:8501/`

### 📂 Project Structure
```
📁 AI_Text_Summarizer_QA/
│
├── app.py                 # Main Streamlit app
├── requirements.txt       # Required packages
└── sample_text.txt        # Example test text
```

### 🧠 Models Used
- **Summarization:** `facebook/bart-large-cnn`  
- **Question Answering:** `deepset/roberta-base-squad2`

### 💡 Learning Outcomes
This project helped me strengthen my understanding of:
- Transformer-based NLP models (BART, DistilBERT, RoBERTa)  
- Handling large documents with chunking techniques  
- Deploying interactive AI applications with Streamlit  
- Combining summarization and Q&A tasks in one workflow  

---

### 🌟 Acknowledgment
Developed as part of my training projects with **Instant Software Solutions** — a leading AI and Data Science training center that focuses on hands-on learning and real-world applications.
