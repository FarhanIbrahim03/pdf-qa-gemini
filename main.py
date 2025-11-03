from PyPDF2 import PdfReader

pdf_path = "ai agents.pdf"

reader = PdfReader(pdf_path)

pdf_text = ""
for page in reader.pages:
    pdf_text += page.extract_text()

# print("pdf text extracted successfully")
# print(pdf_text[:500])

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.create_documents([pdf_text], metadatas=[{'source':pdf_path}])

print(f"Number of chunks created: {len(docs)}")
print(f"Sample chunk: {docs[0].page_content[:300]}")
print(f"meta data: {docs[0].metadata}")


from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = FAISS.from_documents(docs, embedding=embedding_model)
vector_db.save_local("faiss_index")

print("Embeddings saved successfully")

from langchain_community.vectorstores import FAISS

vector_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

retriever = vector_db.as_retriever(search_kwargs={"k":2})


import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.chains import RetrievalQA

from langchain.chains import RetrievalQA
# from langchain.chains.retrieval_qa.base import RetrievalQA
# from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

query = "Summarize the key topics covered in this PDF"
response = qa_chain({"query": query})

print(f"Question: {query}")
print(f"Answer: {response}")

print("Source chunks")
for doc in response["source_documents"]:
    print("-", doc.metadata)


