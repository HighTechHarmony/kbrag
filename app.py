# app.py
import os
import tempfile
import time
import streamlit as st
from streamlit_chat import message
from rag import ChatPDF

st.set_page_config(page_title="RAG with Local DeepSeek R1")


def display_messages():
    """Display the chat history."""
    st.subheader("Chat History")


    # There is a method called st.expander that should encapsulate the part of the message that is encapsulated
    # in <think></think> tags. 

    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        if "<think>" in msg:
            # Extract the hidden and visible parts
            hidden_text = msg.split("<think>")[1].split("</think>")[0]
            visible_text = msg.split("</think>")[1]

            # Display hidden text with st.expander (if you still want it):
            with st.expander(f"See hidden thoughts Part {i}"):
                message(hidden_text, is_user=is_user, key=f"hidden_{i}")

            # Display the visible text without st.expander, using a unique key
            message(visible_text, is_user=is_user, key=f"visible_{i}")
        else:
            message(msg, is_user=is_user, key=f"default_{i}")


    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    """Process the user input and generate an assistant response."""
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner("Thinking..."):
            try:
                agent_text = st.session_state["assistant"].ask(
                    user_text,
                    k=st.session_state["retrieval_k"],
                    score_threshold=st.session_state["retrieval_threshold"],
                    verbosity = st.session_state["verbosity"]
                )
            except ValueError as e:
                agent_text = str(e)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_file():
    """Handle file upload and ingestion."""
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}..."):
            t0 = time.time()
            st.session_state["assistant"].ingest(file_path)
            t1 = time.time()

        st.session_state["messages"].append(
            (f"Ingested {file.name} in {t1 - t0:.2f} seconds", False)
        )
        os.remove(file_path)


def page():
    """Main app page layout."""
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()

    st.header("RAG with Local DeepSeek R1")

    st.subheader("Upload a Document")
    st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    # Retrieval settings
    st.subheader("Settings")
    st.session_state["retrieval_k"] = st.slider(
        "Number of Retrieved Results (k)", min_value=1, max_value=10, value=5
    )
    st.session_state["retrieval_threshold"] = st.slider(
        "Similarity Score Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05
    )

    st.session_state["verbosity"] = st.slider(
        "Verbosity (target # of sentences, 0 = no limit)", min_value=0, max_value=10, value=0, step=1
    )

    # Display messages and text input
    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

    # Clear all and Clear chat buttons side by side
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear Chat Only"):
            st.session_state["messages"] = []

    with col2:
        if st.button("Clear Storage"):
            st.session_state["messages"] = []
            st.session_state["assistant"].clear()
            

    # Display the contents of the vector store
    if st.button("View samples in Vector Store"):
        st.write("Excerpts of data in Vector Store:")

        # Call the vs_list method and capture the return value
        document_sources = st.session_state["assistant"].vs_samples()

        # Print the list of document sources
        st.write(document_sources)
        

    

if __name__ == "__main__":
    page()
