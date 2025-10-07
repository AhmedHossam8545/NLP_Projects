🧠 Project 2: AI Text Summarizer & Question Answering App
📘 Overview

This project is part of my AI & Data Science course at Instant Software Solutions
.
The AI Text Summarizer & Question Answering App is an interactive NLP tool built using Streamlit and Hugging Face Transformers that allows users to:

Upload PDF or TXT files, or paste any text manually.

Automatically summarize long documents using smart chunking.

Ask questions about the uploaded content using an integrated Question Answering model.

It demonstrates practical applications of Natural Language Processing (NLP) in document understanding and automation.

⚙️ Features

✅ Summarizes large or complex text (with automatic chunking for long inputs)
✅ Supports .pdf, .txt, and direct text input
✅ Provides interactive Q&A from the summarized text
✅ Clean Streamlit UI for real-time use
✅ Uses state-of-the-art NLP models from Hugging Face

🧩 Tech Stack
Component	Technology
Programming Language	Python
Web Framework	Streamlit
NLP Models	Hugging Face Transformers (BART, DistilBERT)
Libraries	pdfplumber, torch, transformers
Platform	Google Colab / Local Deployment
🚀 How to Run
1️⃣ Install Dependencies
pip install streamlit transformers torch pdfplumber

2️⃣ Run the App
streamlit run app.py

3️⃣ Open in Browser

Streamlit will display a local URL (e.g. http://localhost:8501/).
Open it to interact with the app.

📂 Project Structure
📁 AI_Text_Summarizer_QA/
│
├── app.py                 # Main Streamlit app
├── requirements.txt       # Required packages
└── sample_text.txt        # Example test text

🧠 Models Used

Summarization: facebook/bart-large-cnn

Question Answering: deepset/roberta-base-squad2

Both models are loaded dynamically via the Hugging Face transformers library.

🧾 Example Usage

1️⃣ Upload a long PDF or text file
2️⃣ Click Summarize Text → app generates an abstract summary
3️⃣ Enter a question → app extracts a precise answer from the text

💡 Learning Outcomes

This project helped me strengthen my understanding of:

Transformer-based NLP models (BART, DistilBERT, RoBERTa)

Handling large documents safely with chunking techniques

Deploying interactive AI applications with Streamlit

Combining summarization and Q&A tasks in one workflow

🌟 Acknowledgment

Developed as part of my training project with Instant Software Solutions — a leading AI and data training center that focuses on hands-on learning and real-world applications.
