import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# --- PATH FIX: Go up 3 levels to reach the root 'hybrid-rag' directory ---
current_file = os.path.abspath(__file__)                 # eval_dashboard.py
pages_dir = os.path.dirname(current_file)                # ui/pages/
ui_dir = os.path.dirname(pages_dir)                      # ui/
root_dir = os.path.dirname(ui_dir)                       # hybrid-rag/ (Root)

sys.path.append(root_dir)

# Now it can successfully find config.py in the root directory!
from config import settings

# 1. Clean Page Configuration
st.set_page_config(page_title="RAG Evaluation Arena", layout="wide")

# 2. Hide Streamlit's default clutter to match the main app
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("RAG Evaluation Arena")
st.markdown("Upload documents, compare model generations, visualize latency, and trace Reciprocal Rank Fusion (RRF) metrics.")
st.markdown("---")

API_URL = f"http://{settings.API_HOST}:{settings.API_PORT}/query"
INDEX_URL = f"http://{settings.API_HOST}:{settings.API_PORT}/index"

# Define your ground truth dataset here
GROUND_TRUTH = {
    "What is the equation for Scaled Dot-Product Attention?": "attention_paper.pdf",
    "Why did the authors choose to use Multi-Head Attention instead of a single attention function?": "attention_paper.pdf",
    "What is the default port number that the PostgreSQL server listens on?": "postgres_docs.pdf",
    "What is the primary purpose of the Write-Ahead Log (WAL) in PostgreSQL?": "postgres_docs.pdf",
    "define adam optimizer": "attention_paper.pdf"
}

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader("Upload New PDF/TXT", type=["pdf", "txt"], accept_multiple_files=True)
    
    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join(settings.DATA_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        st.warning("New documents detected! The system must be re-indexed to retrieve them.")
        
    if st.button("Re-Index Database", use_container_width=True):
        with st.spinner("Processing & Indexing Documents... This may take a moment."):
            try:
                res = requests.post(INDEX_URL, timeout=300)
                if res.status_code == 200:
                    st.success("Successfully Re-Indexed!")
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")

    st.divider()

    st.header("Evaluation Parameters")
    
    query_mode = st.radio("Query Mode", ["Benchmark Questions", "Custom Query"])
    
    if query_mode == "Benchmark Questions":
        selected_query = st.selectbox("Select Benchmark Query", list(GROUND_TRUTH.keys()))
        expected_source = GROUND_TRUTH[selected_query]
    else:
        selected_query = st.text_area("Enter Custom Query", placeholder="Type your question here...")
        
        # Dynamically fetch available files for ground truth selection
        available_files = []
        if os.path.exists(settings.DATA_DIR):
            available_files = [f for f in os.listdir(settings.DATA_DIR) if f.endswith(('.pdf', '.txt'))]
            
        if not available_files:
            available_files = ["No documents found"]
            
        expected_source = st.selectbox("Expected Ground Truth Document", available_files, help="Which document SHOULD contain the answer? Used for computing Recall/Precision.")

    st.info(f"**Target Ground Truth Document:**\n`{expected_source}`")
    st.divider()
    
    st.subheader("Hybrid RAG Tuning")
    top_k = st.number_input("Top-K Chunks (MMR Filtered)", min_value=1, max_value=10, value=3)
    alpha = st.slider("Hybrid Alpha", 0.0, 1.0, 0.5, help="0.0 = Pure BM25 (Sparse) | 1.0 = Pure FAISS (Dense)")
    
    models_to_test = st.multiselect(
        "Select Models to Compare",
        ["gemini", "groq", "ollama"],
        default=["gemini", "groq", "ollama"]
    )
    
    run_test = st.button("Run Evaluation", type="primary", use_container_width=True)

# --- Main Interface ---
if run_test:
    if not models_to_test:
        st.warning("Please select at least one model to test.")
    elif not selected_query.strip():
        st.warning("Please enter a query.")
    else:
        st.subheader(f"Query: *\"{selected_query}\"*")
        st.write("") # Spacer
        
        cols = st.columns(len(models_to_test))
        metrics_data = []
        
        # --- API Calls and Data Gathering ---
        for idx, model in enumerate(models_to_test):
            with cols[idx]:
                st.markdown(f"### {model.upper()}")
                with st.spinner(f"Awaiting {model.upper()}..."):
                    try:
                        payload = {
                            "query": selected_query,
                            "top_k": top_k,
                            "alpha": alpha,
                            "provider": model,
                            "history": [],
                            "bypass_cache": True # FORCE FRESH GENERATION
                        }
                        
                        res = requests.post(API_URL, json=payload)
                        
                        if res.status_code == 200:
                            data = res.json()
                            
                            # Metrics Calculation
                            retrieved_sources = [s["source"] for s in data["sources"]]
                            is_hit = any(expected_source in source for source in retrieved_sources)
                            recall = 1.0 if is_hit else 0.0
                            relevant_count = sum(1 for source in retrieved_sources if expected_source in source)
                            precision = relevant_count / top_k if top_k > 0 else 0
                            
                            latency = data['latency_seconds']
                            
                            metrics_data.append({
                                "Model": model.upper(),
                                "Precision@K": precision,
                                "Recall@K": recall,
                                "Latency (s)": latency,
                                "Fallback Used": data.get("used_fallback", False)
                            })
                            
                            # Display Answer in a clean box
                            st.info(data["answer"])
                            
                            # Display Clean RRF Trace
                            with st.expander("View RRF Retrieval Trace", expanded=False):
                                st.markdown("**Maximal Marginal Relevance (MMR) Output:**")
                                for s in data["sources"]:
                                    match_icon = "Y" if expected_source in s["source"] else "N"
                                    st.markdown(f"{match_icon} **{s['source']}** (Page {s['page']}) | *Score: {s['score']:.4f}*")
                                    st.divider()
                                    
                        else:
                            st.error(f"Backend Error: {res.text}")
                            
                    except requests.exceptions.ConnectionError:
                        st.error("Connection Error: Is the FastAPI server running?")
        
        # --- Analytics Dashboard ---
        if metrics_data:
            st.markdown("---")
            st.header("Performance Analytics")
            df = pd.DataFrame(metrics_data)
            
            # KPI Cards
            kpi1, kpi2, kpi3 = st.columns(3)
            avg_latency = df["Latency (s)"].mean()
            avg_precision = df["Precision@K"].mean()
            
            kpi1.metric("Average Latency", f"{avg_latency:.2f}s")
            kpi2.metric("System Precision@K", f"{avg_precision:.2f}")
            kpi3.metric("Alpha Weight (Dense/Sparse)", f"{alpha} / {1.0-alpha}")
            
            # Charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Latency Comparison Chart
                fig_latency = px.bar(
                    df, 
                    x="Model", 
                    y="Latency (s)", 
                    color="Model",
                    title="Inference Latency Comparison",
                    text_auto='.2s'
                )
                fig_latency.update_layout(showlegend=False)
                st.plotly_chart(fig_latency, use_container_width=True)
                
            with chart_col2:
                # Precision/Recall Radar or Grouped Bar
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Bar(
                    x=df['Model'],
                    y=df['Precision@K'],
                    name='Precision@K',
                    marker_color='indianred'
                ))
                fig_pr.add_trace(go.Bar(
                    x=df['Model'],
                    y=df['Recall@K'],
                    name='Recall@K',
                    marker_color='lightsalmon'
                ))
                fig_pr.update_layout(
                    title='Retrieval Accuracy per Model',
                    barmode='group'
                )
                st.plotly_chart(fig_pr, use_container_width=True)