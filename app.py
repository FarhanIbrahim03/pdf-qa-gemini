import streamlit as st
from rag_utils import load_pdf, split_text, build_vector_db, ask_question

st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="🤖",
    layout="centered")

st.title("📚 PDF Q&A - Powered by Gemini")
st.markdown(
    """
    Upload a PDF, and ask questions about its content.  
    This app uses **LangChain + FAISS + Gemini** to retrieve and generate intelligent answers from your document.
    """
)
st.divider()

pdf_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

if pdf_file:
    uploaded_name = pdf_file.name

    with st.spinner("Processing your PDF... ⏳"):
        # Save uploaded file locally
        with open(uploaded_name, "wb") as f:
            f.write(pdf_file.read())

        # Load and process
        pdf_text = load_pdf(uploaded_name)
        docs = split_text(pdf_text, uploaded_name)
        vector_db = build_vector_db(docs)

    st.success(f"✅ PDF '{uploaded_name}' processed successfully!")
    st.info(f"📑 Created **{len(docs)}** text chunks for retrieval.")

    st.divider()

    # 💬 --- Question Input ---
    query = st.text_input("🔍 Ask a question about your PDF:")

    col1, col2 = st.columns(2)
    with col1:
        get_answer = st.button("💡 Get Answer")
    with col2:
        summarize = st.button("📝 Summarize PDF")

    if get_answer and query.strip():
        with st.spinner("Generating answer... 🤔"):
            answer, sources = ask_question(vector_db, query)
        st.markdown("### ✅ Answer")
        st.write(answer)

        st.markdown("### 📚 Sources")
        for src in sources:
            st.write(f"- {src['source']}")

    elif summarize:
        with st.spinner("Summarizing your document... ✨"):
            summary, _ = ask_question(vector_db, "Summarize this document briefly")
        st.markdown("### 📝 Summary")
        st.write(summary)

else:
    st.info("👆 Please upload a PDF to begin.")
