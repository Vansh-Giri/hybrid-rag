import streamlit as st
import requests

# Point to your running FastAPI server
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Hybrid RAG System", layout="centered")

st.title("Technical Knowledge Base Q&A")
st.markdown("Ask questions about your documents using Hybrid Retrieval and local LLM generation. The system remembers your recent conversation for follow-up questions.")

# Sidebar for System controls and Retrieval Settings
with st.sidebar:
    st.header("Retrieval Settings")
    
    # Dynamic Alpha Slider
    alpha_val = st.slider(
        "Hybrid Weight (Alpha)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.1, 
        help="0.0 = Keyword Only (Sparse/BM25), 1.0 = Semantic Only (Dense/FAISS)"
    )
    
    # Top-K Slider
    top_k_val = st.number_input("Top-K Sources", min_value=1, max_value=10, value=3)
    
    st.divider()
    
    st.header("System Management")
    st.write("Initialize the dense and sparse indexes before querying.")
    
    if st.button("Index Documents"):
        with st.spinner("Loading models and indexing chunks..."):
            try:
                res = requests.post(f"{API_BASE_URL}/index")
                if res.status_code == 200:
                    st.success(res.json().get("message", "Indexing complete!"))
                else:
                    st.error(f"Error: {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend. Is FastAPI running?")
                
    st.divider()
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If the message is from the assistant and contains sources, display them in an expander
        if msg["role"] == "assistant" and "sources" in msg:
            latency = msg.get("latency", "N/A")
            with st.expander(f"View Sources (Completed in {latency}s)"):
                for i, source in enumerate(msg["sources"]):
                    st.write(f"**Source {i+1}:** {source['source']} (Page {source['page']}) - Score: {source['score']}")
                    st.write(source.get("text_snippet", ""))

# Chat Input Trigger
if prompt := st.chat_input("What would you like to know? (e.g., What is the default port for PostgreSQL?)"):
    
    # 1. Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Process Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                # Extract history for the API (exclude the prompt we just appended)
                chat_history = [
                    {"role": m["role"], "content": m["content"]} 
                    for m in st.session_state.messages[:-1]
                ]
                
                payload = {
                    "query": prompt, 
                    "top_k": top_k_val,
                    "alpha": alpha_val,
                    "history": chat_history
                }
                
                response = requests.post(f"{API_BASE_URL}/query", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    latency = data.get("latency_seconds", "N/A")
                    
                    # Display Answer
                    st.markdown(answer)
                    
                    # Display Sources inside an expander
                    with st.expander(f"View Sources (Completed in {latency}s)"):
                        for i, source in enumerate(sources):
                            st.write(f"**Source {i+1}:** {source['source']} (Page {source['page']}) - Score: {source['score']}")
                            text_snippet = source.get("text_snippet", "Used as context for generation.")
                            st.write(text_snippet)
                            
                    # Append assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "latency": latency
                    })
                    
                elif response.status_code == 400:
                    st.warning("System is not indexed. Please click 'Index Documents' in the sidebar first.")
                else:
                    st.error(f"API Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend API. Ensure 'python -m uvicorn api.main:app' is running.")