import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Evaluation Arena", layout="wide")
st.title("RAG Evaluation Arena")
st.markdown("Compare model responses and evaluate retrieval metrics (Precision/Recall/Latency) in real-time.")

API_URL = "http://127.0.0.1:8000/query"

# Define your ground truth dataset here
GROUND_TRUTH = {
    "What is the equation for Scaled Dot-Product Attention?": "attention_paper.pdf",
    "Why did the authors choose to use Multi-Head Attention instead of a single attention function?": "attention_paper.pdf",
    "What is the default port number that the PostgreSQL server listens on?": "postgres_docs.pdf",
    "What is the primary purpose of the Write-Ahead Log (WAL) in PostgreSQL?": "postgres_docs.pdf"
}

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Test Parameters")
    selected_query = st.selectbox("Select Benchmark Query", list(GROUND_TRUTH.keys()))
    expected_source = GROUND_TRUTH[selected_query]
    
    st.write(f"**Target Document:** `{expected_source}`")
    st.divider()
    
    top_k = st.number_input("Top-K Chunks", min_value=1, max_value=10, value=3)
    alpha = st.slider("Hybrid Alpha (0=Sparse, 1=Dense)", 0.0, 1.0, 0.5)
    
    models_to_test = st.multiselect(
        "Select Models to Compare",
        ["gemini", "groq", "ollama"],
        default=["gemini", "ollama"]
    )
    
    run_test = st.button("Run Evaluation", type="primary", use_container_width=True)

# --- Main Interface ---
if run_test:
    if not models_to_test:
        st.warning("Please select at least one model to test.")
    else:
        st.subheader(f"Query: {selected_query}")
        
        # Create dynamic columns based on how many models are selected
        cols = st.columns(len(models_to_test))
        
        metrics_data = []
        
        for idx, model in enumerate(models_to_test):
            with cols[idx]:
                st.markdown(f"### {model.upper()}")
                with st.spinner(f"Querying {model}..."):
                    try:
                        payload = {
                            "query": selected_query,
                            "top_k": top_k,
                            "alpha": alpha,
                            "provider": model,
                            "history": []
                        }
                        
                        res = requests.post(API_URL, json=payload)
                        
                        if res.status_code == 200:
                            data = res.json()
                            
                            # Calculate Metrics on the fly
                            retrieved_sources = [s["source"] for s in data["sources"]]
                            
                            is_hit = any(expected_source in source for source in retrieved_sources)
                            recall = 1.0 if is_hit else 0.0
                            
                            relevant_count = sum(1 for source in retrieved_sources if expected_source in source)
                            precision = relevant_count / top_k
                            
                            metrics_data.append({
                                "Model": model.upper(),
                                "Precision": f"{precision:.2f}",
                                "Recall": f"{recall:.2f}",
                                "Latency (s)": f"{data['latency_seconds']:.2f}s",
                                "Fallback": "Yes" if data.get("used_fallback") else "No"
                            })
                            
                            # Display Answer
                            st.info(data["answer"])
                            
                            # Display Sources
                            with st.expander("View Retrieved Sources"):
                                for s in data["sources"]:
                                    match_icon = "✅" if expected_source in s["source"] else "❌"
                                    st.write(f"{match_icon} **{s['source']}** (Score: {s['score']})")
                        else:
                            st.error(f"Backend Error: {res.text}")
                            
                    except requests.exceptions.ConnectionError:
                        st.error("Connection Error: Is the FastAPI server running?")
        
        # --- Metrics Table ---
        if metrics_data:
            st.divider()
            st.subheader("Retrieval & Generation Metrics")
            df = pd.DataFrame(metrics_data)
            st.dataframe(df, use_container_width=True)