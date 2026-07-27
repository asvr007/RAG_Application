import os
import streamlit as st
from rag_utility import process_doc_to_chromadb, answer_the_question

#set the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))
# working_dir = os.getcwd()

st.title("Llama -3.3-70B Versatile - Document RAG")

#File Uploader
uploaded_file = st.file_uploader("Upload a PDF file ", type=["pdf"])

if uploaded_file is not None:
    #define save_path
    save_path = os.path.join(working_dir, uploaded_file.name)

    #Save the file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    process_document = process_doc_to_chromadb(save_path)
    st.info("Document Processed Successfully")

#Text widget to get the user question
user_question = st.text_area("Ask your question  about the document")

if st.button("Answer"):
    answer = answer_the_question(user_question)
    st.markdown("### Llama-3.3-70B Versatile response")
    st.markdown(answer)
