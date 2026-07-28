import os
import streamlit as st
from rag_utility import process_doc_to_chromadb, answer_the_question
import tempfile

#set the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))
# working_dir = os.getcwd()

st.title("Llama -3.3-70B Versatile - Document RAG")

#Session_State
if "vectordb" not in st.session_state:
    st.session_state.vetordb = None

#File Uploader
uploaded_file = st.file_uploader("Upload a PDF file ", type=["pdf"])

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_pdf_path = tmp_file.name

    # process_document = process_doc_to_chromadb(save_path)
    with st.spinner("Processing the document "):
        try:
            vectordb = process_doc_to_chromadb(temp_pdf_path)
            st.session_state.processed_file = uploaded_file.name
            st.success("Document processed succesfully")
        except Exception as e:
            st.error(f"Error while processing document: \n\n{e}")
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

#Text widget to get the user question
user_question = st.text_area("Ask your question  about the document")

if st.button("Answer"):
    if uploaded_file is None:
        st.warning("Please upload a PDF Document first")
    elif not user_question.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Generating answer..."):
            try:
                answer = answer_the_question(user_question)
                st.markdown("## Llama-3.3-70B Response")
                st.markdown("answer")
            except Exception as e:
                st.error(f"Error while generating answer:\n\n{e}")
