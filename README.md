# 📄 PDF + GK Chatbot 🤖

This is a Retrieval-Augmented Generation (RAG) chatbot powered by a fine-tuned open-source language model and your own PDF document (e.g., resume, academic papers). Built using **LangChain**, **Streamlit**, and **Hugging Face Transformers**, this app allows you to ask natural language questions and get intelligent responses based on both your document and general knowledge.

## 🚀 Features

- Ask questions about a PDF document (e.g., your resume)
- Uses `flan-t5-large` (free Hugging Face model) as the language model
- Retrieves relevant document chunks via vector similarity (Chroma DB)
- Clean and interactive Streamlit interface
- Lightweight and free (no OpenAI or paid API required)


## 🧠 Tech Stack

- Python
- Streamlit
- LangChain
- HuggingFace Hub + Transformers
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Chroma (for vector storage)
- dotenv

## ⚙️ Setup Instructions

## 1. Clone the Repo

```bash
git clone https://github.com/kbphys96/Project-1-Rag_ChatBot.git
cd Project-1-Rag_ChatBot

```
## 2. Install Dependencies

```bash
pip install -r requirements.txt

```
## 3. Add Hugging Face API Token
Go to huggingface.co/settings/tokens

Create a token (read access is enough)

Create a .env file with the following line:

```bash
HUGGINGFACEHUB_API_TOKEN=*******************
```

## 4. Run the App

```bash
streamlit run app.py
```

### 📝 Example Prompt
arduino
Copy
Edit
"Tell me about my educational background."
"Which AI tools have I worked with?"
"What are my skills in NLP?"



### 📌 Notes
Uses UnstructuredPDFLoader to parse text from your uploaded PDF.

The document is chunked and embedded using sentence-transformers/all-MiniLM-L6-v2.

Model: google/flan-t5-large (free, zero-cost on Hugging Face Hub).



### 🔐 Security
No sensitive data is logged or transmitted externally.

Your Hugging Face token is stored locally in .env.

### 📧 Author
Kulbhushan Nautiyal
📬 kbphys96@gmail.com
🔗 LinkedIn
🐙 GitHub
