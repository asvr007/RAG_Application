#Functions related to RAG
import streamlit as st
import os
import shutil
from dotenv import load_dotenv

# from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
# from sentence_transformers import SentenceTransformer
from huggingface_hub import login

#load env variables from .env file
load_dotenv()
APIKEY = os.getenv("GROK_API_KEY")

if not APIKEY:
    APIKEY = st.secrets["GROQ_API_KEY"]
    
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    HF_TOKEN = st.secrets["HF_TOKEN"]
    
login(token=HF_TOKEN)
working_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(working_dir, "doc_vectorstore")

@st.cache_resource
def load_embedding_model():    
#Load the embeding model
# embedding = SentenceTransformer('sentence-transformers/all-miniLM-L12-v2')
    return HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L12-v2")
    
embedding = load_embedding_model()
#Load the Llama-3.3-70B model from Groq
llm = ChatGroq( model = "llama-3.3-70b-versatile", temperature=0, api_key=APIKEY)


def process_doc_to_chromadb(file_path):
    #load the PDF document using UnstructuredPDFloader
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    #Split the text into chunks for embedidngs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 200)
    chunks = text_splitter.split_documents(documents)

    #Remove previous vector database
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    #Create croma database
    chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=persist_directory)
    #Store the embeddings in a chroma vector database
    # vector_db = Chroma.from_documents(documents = texts, embedding = embedding, persist_directory=f"{working_dir}/doc_vectorstore")


def answer_the_question(user_question):
    #Load the persistent chroma vector database
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    #Create a retriever for document_search
    retriever = vectordb.as_retriever(search_kwargs={"k":4})

    #Create a RetrievalQA chain to answer user questions using Llama-3.3-70b-versatile
    qa_chain = RetrievalQA.from_chain_type(llm = llm, chain_type = "stuff", retriever = retriever)

    response = qa_chain.invoke({"query" : user_question})

    return  response["result"]
