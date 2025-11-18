import os
import streamlit as st
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Collect URLs from the sidebar
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()
index_path = "faiss_index"  # Directory to save the FAISS index
llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.6)

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','], chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorindex_hf = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)
    
    # Save the FAISS index to disk
    vectorindex_hf.save_local(index_path)
    st.success("Index built and saved. You can now ask a question below.")

# Always show the "Question" input box
query = st.text_input("Question:")
if query:
    if os.path.exists(index_path):
        # Recreate the embeddings object
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Load the FAISS index with the embeddings
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
    else:
        st.error("Index not found. Please click 'Process URLs' first.")