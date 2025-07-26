import os
import streamlit as st
import requests
from dotenv import load_dotenv
import io

load_dotenv()

st.set_page_config(
    page_title="Bengali PDF RAG",
    page_icon="📚",
    layout="wide"
)

st.title("Bengali PDF RAG System")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                # Call the FastAPI endpoint to process the PDF
                files = {"file": ("document.pdf", uploaded_file.getvalue(), "application/pdf")}
                response = requests.post("http://localhost:8000/process", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Successfully processed {result['stored_count']} chunks!")
                else:
                    st.error("Error processing PDF")

# Main chat interface
st.header("Chat with your Documents")

# Chat input
question = st.text_input("Ask a question about your documents:")
top_k = st.slider("Number of relevant chunks to consider:", min_value=1, max_value=10, value=5)

if question:
    if st.button("Ask"):
        # Prepare the chat request
        payload = {
            "question": question,
            "top_k": top_k
        }
        
        with st.spinner("Getting answer..."):
            # Call the FastAPI chat endpoint
            response = requests.post("http://localhost:8000/chat", json=payload)
            
            if response.status_code == 200:
                answer = response.json()["answer"]
                
                # Add to chat history
                st.session_state.chat_history.append({"question": question, "answer": answer})
            else:
                st.error("Error getting response")

# Display chat history
st.header("Chat History")
for chat in reversed(st.session_state.chat_history):
    st.write("**Q:** " + chat["question"])
    st.write("**A:** " + chat["answer"])
    st.markdown("---")