import os
import shutil
import tempfile
import streamlit as st

from rag_utility import (
    process_doc_to_chromadb,
    answer_the_question
)

working_dir = os.path.dirname(os.path.abspath(__file__))

st.title("Llama-3.3-70B Versatile - Document RAG")

# ---------------- Session State ---------------- #

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

if "db_dir" not in st.session_state:
    st.session_state.db_dir = None


# ---------------- Upload PDF ---------------- #

uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.getbuffer())
        temp_pdf_path = tmp_file.name

    with st.spinner("Processing document..."):

        try:

            # Delete previous vector database
            if st.session_state.vectordb is not None:

                try:
                    st.session_state.vectordb.delete_collection()
                except Exception:
                    pass

                st.session_state.vectordb = None

            # Delete previous temporary folder
            if (
                st.session_state.db_dir is not None
                and os.path.exists(st.session_state.db_dir)
            ):
                shutil.rmtree(
                    st.session_state.db_dir,
                    ignore_errors=True
                )

                st.session_state.db_dir = None

            # Process new document
            vectordb, db_dir = process_doc_to_chromadb(
                temp_pdf_path
            )

            st.session_state.vectordb = vectordb
            st.session_state.db_dir = db_dir
            st.session_state.processed_file = uploaded_file.name

            st.success("Document processed successfully!")

        except Exception as e:

            st.error(f"Processing Error\n\n{e}")

        finally:

            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)


# ---------------- Ask Question ---------------- #

user_question = st.text_area(
    "Ask your question about the document"
)

if st.button("Answer"):

    if st.session_state.vectordb is None:

        st.warning("Please upload a PDF first.")

    elif not user_question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Generating answer..."):

            try:

                answer = answer_the_question(
                    user_question,
                    st.session_state.vectordb
                )

                st.markdown("## Llama-3.3-70B Response")
                st.write(answer)

            except Exception as e:

                st.error(f"Error\n\n{e}")
