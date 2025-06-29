import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub

from dotenv import load_dotenv
import os
load_dotenv()

# Load free embedding model (no API key needed)
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}  # Use 'cuda' if you have GPU
)

# Load and index the PDF
@st.cache_resource
def create_db():
    loader = UnstructuredPDFLoader("Kulbhushan_Nautiyal_Resume.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    db = Chroma.from_documents(chunks, embedding, persist_directory="db")
    db.persist()
    return db

# Load or create DB
db = create_db()
retriever = db.as_retriever()

# Set up free LLM (requires HuggingFace token)
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",  # Free model
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),  # Get from huggingface.co/settings/tokens
    model_kwargs={"temperature": 0.5, "max_length": 512}
)

# Set up chain
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Streamlit UI
st.title("📄 Free Physics Document Chatbot  + General Knowledge 🤖")
query = st.text_input("Ask a question about your document or anything!")

if query:
    result = chain({"question": query}, return_only_outputs=True)
    st.markdown("### ✅ Answer:")
    st.write(result["answer"])