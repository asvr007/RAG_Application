import os
import uuid
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from huggingface_hub import login

# Load environment variables
load_dotenv()

APIKEY = os.getenv("GROQ_API_KEY")
if not APIKEY:
    APIKEY = st.secrets["GROQ_API_KEY"]

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    HF_TOKEN = st.secrets["HF_TOKEN"]

login(token=HF_TOKEN)


@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2"
    )


embedding = load_embedding_model()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=APIKEY
)


def process_doc_to_chromadb(file_path):

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )

    chunks = splitter.split_documents(documents)

    # Create a unique temporary directory
    temp_db_dir = tempfile.mkdtemp()

    # Create a unique collection name
    collection_name = str(uuid.uuid4())

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=temp_db_dir,
        collection_name=collection_name
    )

    return vectordb, temp_db_dir


def answer_the_question(user_question, vectordb):

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 25,
            "lambda_mult": 0.5
        }
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    response = qa_chain.invoke(
        {"query": user_question}
    )

    return response["result"]
