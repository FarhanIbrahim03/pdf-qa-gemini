# 📘 PDF Q&A Assistant (RAG + Gemini)

A Streamlit-based AI assistant that allows you to **upload any PDF** and **ask questions** about its content — powered by **Google Gemini** and **LangChain Retrieval-Augmented Generation (RAG)**.

---

## 🚀 Features
- 📄 Upload PDFs and extract text automatically  
- 🤖 Ask questions about your document using Gemini LLM  
- 🧠 Semantic search with FAISS and Hugging Face embeddings  
- 🪄 Summarize long PDFs instantly  
- 🔒 API key protected (stored locally via `.env`)

---

## 🧩 Tech Stack
| Component | Technology |
|------------|-------------|
| LLM | Google Gemini (`langchain-google-genai`) |
| Embeddings | Hugging Face `all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| Framework | LangChain |
| Frontend | Streamlit |
| PDF Parsing | PyPDF2 |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone this repository
```bash
git clone https://github.com/FarhanIbrahim03/pdf-qa-gemini.git
cd pdf-qa-assistant


### 2️⃣ Create a virtual environment
python -m venv .venv
.venv\Scripts\activate     # On Windows
source .venv/bin/activate  # On Mac/Linux

### 3️⃣ Install dependencies
pip install -r requirements.txt

### 4️⃣ Add your Gemini API key
Create a .env file in the root directory:
GOOGLE_API_KEY=your_gemini_api_key_here

### 5️⃣ Run the app
streamlit run app.py
```


🧠 How It Works

1. The uploaded PDF is read and converted into text using PyPDF2.
2. Text is split into chunks using LangChain Text Splitter.
3. Each chunk is embedded into vector form using Hugging Face Sentence Transformers.
4. All vectors are stored in a FAISS vector database.
5. When a query is asked, relevant chunks are retrieved and passed to Gemini to generate an answer.

🧾 Example

User: “Summarize the key points of this PDF.”
Assistant: “This document covers AI agent optimization, multi-agent systems, and context utilization improvements.”

🖼️ Screenshot
<img width="2420" height="1452" alt="image" src="https://github.com/user-attachments/assets/a4d58ce5-1709-4e72-85a7-b1c191c2b39f" />

🧑‍💻 Author

Farhan M.
🎓 Data Science | AI/ML Enthusiast | Gen AI
<p align="center">
  <a href="https://www.linkedin.com/in/farhan-m03/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
  <a href="https://github.com/FarhanIbrahim03" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-%2312100E.svg?&style=for-the-badge&logo=github&logoColor=white" />
  </a>
</p>

🪶 License

This project is open-source under the MIT License.

