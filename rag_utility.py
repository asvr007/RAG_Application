#Functions related to RAG

import os
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

HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)
working_dir = os.path.dirname(os.path.abspath(__file__))

#Load the embeding model
# embedding = SentenceTransformer('sentence-transformers/all-miniLM-L12-v2')
embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L12-v2")

#Load the Llama-3.3-70B model from Groq
llm = ChatGroq( model = "llama-3.3-70b-versatile", temperature=0, api_key=APIKEY)


def process_doc_to_chromadb(file_path):
    #load the PDF document using UnstructuredPDFloader
    loader = PyPDFLoader(f"{file_path}")
    documents = loader.load()

    #Split the text into chunks for embedidngs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 200)
    texts = text_splitter.split_documents(documents)

    #Store the embeddings in a chroma vector database
    vector_db = Chroma.from_documents(documents = texts, embedding = embedding, persist_directory=f"{working_dir}/doc_vectorstore")

    return 0

def answer_the_question(user_question):
    #Load the persistent chroma vector database
    vectordb = Chroma(persist_directory=f"{working_dir}/doc_vectorstore", embedding_function=embedding)

    #Create a retriever for document_search
    retriever = vectordb.as_retriever()

    #Create a RetrievalQA chain to answer user questions using Llama-3.3-70b-versatile
    qa_chain = RetrievalQA.from_chain_type(llm = llm, chain_type = "stuff", retriever = retriever)

    response = qa_chain.invoke({"query" : user_question})
    answer = response["result"]

    return answer
