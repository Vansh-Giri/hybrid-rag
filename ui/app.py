import streamlit as st
import requests

API_BASE_URL = "http://127.0.0.1:8000"

# 1. Clean Page Configuration (Wide layout for modern feel)
st.set_page_config(page_title="Hybrid RAG System", layout="wide")

# 2. Hide Streamlit's default clutter
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            /* Optional: Adjust main padding to make it look more like a chat app */
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Retrieval Settings")
    alpha_val = st.slider(
        "Hybrid Weight (Alpha)", 
        min_value=0.0, max_value=1.0, value=0.5, step=0.1, 
        help="0.0 = Sparse/BM25, 1.0 = Dense/FAISS"
    )
    top_k_val = st.number_input("Top-K Sources", min_value=1, max_value=10, value=3)
    
    st.divider()
    
    st.header("Inference Engine")
    provider_choice = st.selectbox(
        "Primary Model",
        options=["gemini", "groq", "ollama"],
        format_func=lambda x: {
            "gemini": "Gemini 2.5 Flash (Google)",
            "groq": "Llama 3 8B (Groq Fast API)",
            "ollama": "Phi-4 Mini (Local VRAM)"
        }.get(x, x)
    )
    
    st.divider()
    st.header("System Management")
    if st.button("Index Documents", use_container_width=True):
        with st.spinner("Indexing..."):
            try:
                res = requests.post(f"{API_BASE_URL}/index")
                if res.status_code == 200:
                    st.success(res.json().get("message", "Indexing complete!"))
                else:
                    st.error(f"Error: {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to backend.")
                
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("Document Intelligence System")
st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat history (with their specific dropdowns)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Clean expander for sources on assistant messages
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            latency = msg.get("latency", "N/A")
            fallback_note = " | Fallback Engaged" if msg.get("used_fallback") else ""
            
            with st.expander(f"View Retrieved Sources ({latency}s{fallback_note})"):
                for i, source in enumerate(msg["sources"]):
                    st.markdown(f"**Source {i+1}:** `{source['source']}` (Page {source['page']}) | *Score: {source['score']:.4f}*")
                    if source.get("text_snippet"):
                        st.caption(f'"{source["text_snippet"]}"')
                    st.divider()

# Handle new user input
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Show assistant response
    with st.chat_message("assistant"):
        with st.spinner(f"Generating answer using {provider_choice.upper()}..."):
            try:
                # Prepare history for context
                chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                
                payload = {
                    "query": prompt, 
                    "top_k": top_k_val,
                    "alpha": alpha_val,
                    "history": chat_history,
                    "provider": provider_choice
                }
                
                response = requests.post(f"{API_BASE_URL}/query", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    latency = data.get("latency_seconds", "N/A")
                    used_fallback = data.get("used_fallback", False)
                    
                    if used_fallback:
                        st.warning(f"Primary model ({provider_choice}) failed. Fallback model was used successfully.")
                    
                    # Display the text answer
                    st.markdown(answer)
                    
                    # Display the clean expander for the new sources
                    if sources:
                        with st.expander(f"View Retrieved Sources ({latency}s)"):
                            for i, source in enumerate(sources):
                                st.markdown(f"**Source {i+1}:** `{source['source']}` (Page {source['page']}) | *Score: {source['score']:.4f}*")
                                if source.get("text_snippet"):
                                    st.caption(f'"{source["text_snippet"]}"')
                                st.divider()
                                
                    # Save to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "latency": latency,
                        "used_fallback": used_fallback
                    })
                    
                elif response.status_code == 400:
                    st.warning("System is not indexed. Please click 'Index Documents' in the sidebar.")
                else:
                    st.error(f"API Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend API. Is FastAPI running?")