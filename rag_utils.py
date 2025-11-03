from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import streamlit as st


# 1 Load and read the pdf
def load_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()
    return pdf_text


# 2 Split into chunks
def split_text(pdf_text: str, pdf_path: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.create_documents([pdf_text], metadatas=[{'source': pdf_path}])
    return docs


# 3 Create/Load FAISS DB
def build_vector_db(docs):
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(docs, embedding_model)
    vector_db.save_local("faiss_index")
    return vector_db


@st.cache_resource
def load_vector_db(embedding_model):
    return FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)


# 4 Conversation with gemini
def ask_question(vector_db, query: str):
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

    retriever = vector_db.as_retriever(search_kwargs={"k": 2})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    response = qa_chain({'query': query})
    return response['result'], [doc.metadata for doc in response['source_documents']]
