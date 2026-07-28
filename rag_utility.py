#Functions related to RAG
import streamlit as st
import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
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

@st.cache_resource
def load_embedding_model():    
#Load the embeding model
    return HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L12-v2")
    
embedding = load_embedding_model()
#Load the Llama-3.3-70B model from Groq
llm = ChatGroq( model = "llama-3.3-70b-versatile", temperature=0, api_key=APIKEY)

#Process document
def process_doc_to_chromadb(file_path):
    #load the PDF document using UnstructuredPDFloader
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    #Split the text into chunks for embedidngs
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 300)
    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(documents=chunks, embedding = embedding)
    return vectordb


def answer_the_question(user_question, vectordb):

    #Create a retriever for document_search
    retriever = vectordb.as_retriever(search_type="mmr, 
        search_kwargs={"k":10, "fetch_k" : 25, "lambda_mult" : 0.5})

    #Create a RetrievalQA chain to answer user questions using Llama-3.3-70b-versatile
    qa_chain = RetrievalQA.from_chain_type(llm = llm, chain_type = "stuff", retriever = retriever)

    response = qa_chain.invoke({"query" : user_question})

    return  response["result"]
