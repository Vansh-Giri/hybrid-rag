import streamlit as st
import requests

# Point to your running FastAPI server
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Hybrid RAG System", page_icon="🔍", layout="centered")

st.title("Technical Knowledge Base Q&A")
st.markdown("Ask questions about your documents using Hybrid Retrieval and local LLM generation.")

# Sidebar for System controls and Retrieval Settings
with st.sidebar:
    st.header("⚙️ Retrieval Settings")
    
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
    
    st.header("🗄️ System Management")
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

# Main Query Interface
with st.form("query_form"):
    query = st.text_input("What would you like to know?", placeholder="e.g., What is the default port for PostgreSQL?")
    submit_button = st.form_submit_button("Generate Answer")

if submit_button:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            try:
                # Updated payload with dynamic parameters
                payload = {
                    "query": query, 
                    "top_k": top_k_val,
                    "alpha": alpha_val
                }
                response = requests.post(f"{API_BASE_URL}/query", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Display Answer and Latency
                    st.subheader("Answer")
                    st.info(data["answer"])
                    
                    latency = data.get("latency_seconds", "N/A")
                    st.caption(f"⏱️ Request completed in {latency}s (Alpha: {alpha_val})")
                    
                    # Display Sources
                    st.subheader("Sources Retrieved")
                    for i, source in enumerate(data["sources"]):
                        with st.expander(f"Source {i+1}: {source['source']} (Page {source['page']}) - Score: {source['score']}"):
                            # Depending on what your API returns, display the text snippet if available
                            text_snippet = source.get("text_snippet", "Used as context for generation.")
                            st.write(text_snippet)
                
                elif response.status_code == 400:
                    st.warning("System is not indexed. Please click 'Index Documents' in the sidebar first.")
                else:
                    st.error(f"API Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend API. Ensure 'python api/main.py' is running.")