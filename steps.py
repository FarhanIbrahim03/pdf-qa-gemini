"""
===========================================
🔹 RAG Workflow using PDF Documents (LangChain)
===========================================

Step-by-step process of how a PDF becomes an answer:

1️⃣ Extract Text (PyPDF2)
    - Load and extract text content from PDF files.
    - Example: Research papers, manuals, reports, etc.

2️⃣ Split Text into Chunks (LangChain Splitter)
    - Divide the extracted text into smaller, overlapping chunks.
    - Helps LLMs process long documents effectively.

3️⃣ Create Embeddings (HuggingFace / Sentence Transformers)
    - Convert each chunk into a numeric vector (embedding).
    - These embeddings represent semantic meaning.

4️⃣ Store & Index (FAISS)
    - Store all embeddings in a FAISS vector database.
    - Enables fast similarity search (retrieval).

5️⃣ Retrieve Relevant Chunks (LangChain Retriever)
    - When a user asks a question, search FAISS for the most similar chunks.

6️⃣ Generate Answer (LLM)
    - Send retrieved chunks + question to an LLM (e.g., Gemini, OpenAI, or local HF model).
    - LLM combines the context and generates a relevant answer.

===========================================
🧠 Packages Involved:
-------------------------------------------
- PyPDF2 ..................... PDF text extraction
- langchain .................. Framework for retrieval and QA
- langchain-community ........ Embedding + retriever support
- sentence-transformers ...... Text embedding model
- faiss-cpu .................. Vector storage and similarity search

===========================================
"""

