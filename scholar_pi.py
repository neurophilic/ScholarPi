import os
import sqlite3
import json
import hashlib
import time
from datetime import datetime, timedelta
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
import fitz  # PyMuPDF
from groq import Groq, RateLimitError

# --- 1. CONFIGURATION & ENVIRONMENT ---
st.set_page_config(page_title="Dynamic Epistemic Cartography", page_icon="🔭", layout="wide")

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_TEXT_TOKENS = 6000
EPOCH_DAYS = 30
SEED_NUMBER = 42

BASE_DIR = os.path.abspath('./Scientometric_VQR_System')
os.makedirs(BASE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, 'dynamic_science_map.db')

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("⚠️ API Key not found! Please configure your environment variables or Streamlit Secrets.")
    st.stop()
client = Groq(api_key=GROQ_API_KEY)

# --- 2. DATABASE INITIALIZATION ---
@st.cache_resource
def init_system():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    
    # Table for individual paper assessments (VQR aligned)
    cursor.execute('''CREATE TABLE IF NOT EXISTS papers 
                      (file_hash TEXT PRIMARY KEY, filename TEXT, 
                       c1 REAL, c2 REAL, c3 REAL, c4 REAL, 
                       c5 REAL, c6 REAL, c7 REAL, c8 REAL,
                       keywords TEXT, final_score REAL, timestamp DATETIME)''')
                       
    # Table for historical 30-day epoch EWM weights
    cursor.execute('''CREATE TABLE IF NOT EXISTS epoch_weights 
                      (epoch_id INTEGER PRIMARY KEY AUTOINCREMENT, 
                       w1 REAL, w2 REAL, w3 REAL, w4 REAL, 
                       w5 REAL, w6 REAL, w7 REAL, w8 REAL, timestamp DATETIME)''')
    
    # Initialize default uniform weights if database is empty (12.5% each)
    cursor.execute("SELECT COUNT(*) FROM epoch_weights")
    if cursor.fetchone()[0] == 0:
        cursor.execute('''INSERT INTO epoch_weights (w1, w2, w3, w4, w5, w6, w7, w8, timestamp) 
                          VALUES (0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, ?)''', 
                       (datetime.now().isoformat(),))
    conn.commit()
    return conn

conn = init_system()

# --- 3. ENTROPY WEIGHT METHOD (EWM) ALGORITHM ---
def calculate_ewm_weights(matrix):
    """
    Calculates objective criteria weights using Shannon Entropy.
    Formula: W_j = d_j / SUM(d_j) where d_j = 1 - E_j
    """
    m, n = matrix.shape
    if m <= 1:
        return np.ones(n) / n 
    
    # Min-Max Normalization to bounded [0, 1] space
    norm_matrix = np.zeros_like(matrix)
    for j in range(n):
        col = matrix[:, j]
        c_min, c_max = col.min(), col.max()
        if c_max - c_min > 1e-9:
            norm_matrix[:, j] = (col - c_min) / (c_max - c_min)
        else:
            norm_matrix[:, j] = 0.5 

    # Calculate proportions p_ij
    col_sums = norm_matrix.sum(axis=0)
    col_sums[col_sums == 0] = 1e-9 
    p_matrix = norm_matrix / col_sums
    
    # Calculate Information Entropy (E_j)
    p_matrix_eps = np.where(p_matrix == 0, 1e-12, p_matrix)
    entropy = - (1.0 / np.log(m)) * np.sum(p_matrix * np.log(p_matrix_eps), axis=0)
    
    # Calculate Divergence (d_j) and Final Weights (W_j)
    d = 1.0 - entropy
    d_sum = d.sum()
    if d_sum == 0:
        return np.ones(n) / n
    
    return d / d_sum

def trigger_epoch_recalculation():
    """Evaluates temporal delta and recalculates MCDA weights every 30 days."""
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp FROM epoch_weights ORDER BY epoch_id DESC LIMIT 1")
    last_epoch_str = cursor.fetchone()[0]
    last_epoch_date = datetime.fromisoformat(last_epoch_str)
    
    if datetime.now() - last_epoch_date >= timedelta(days=EPOCH_DAYS):
        st.toast(f"⏳ {EPOCH_DAYS}-Day Epoch reached. Executing EWM recalibration...", icon="🔄")
        
        target_date = (datetime.now() - timedelta(days=EPOCH_DAYS)).isoformat()
        cursor.execute("SELECT c1, c2, c3, c4, c5, c6, c7, c8 FROM papers WHERE timestamp >= ?", (target_date,))
        rows = cursor.fetchall()
        
        if len(rows) > 5: # Statistical validity threshold
            matrix = np.array(rows)
            new_weights = calculate_ewm_weights(matrix)
            
            cursor.execute('''INSERT INTO epoch_weights (w1, w2, w3, w4, w5, w6, w7, w8, timestamp) 
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                           (*new_weights, datetime.now().isoformat()))
            conn.commit()
            st.toast("✅ Criteria weights successfully adapted to current scientific variance.", icon="📈")

# --- 4. SEMANTIC LLM EXTRACTION ---
def evaluate_paper(text, model):
    """Leverages LLM to perform Semantic VQR-aligned proxy extraction."""
    prompt = f"""Read the following academic text excerpt. You are an expert peer reviewer assessing for originality, methodological rigor, and societal impact.
Evaluate the paper across the following 8 criteria, assigning a strict numerical score from 0.0 to 10.0 for each.
Return ONLY a valid JSON object matching this exact structure:
{{
    "C1_Originality": <float>,
    "C2_Methodological_Rigor": <float>,
    "C3_Interdisciplinary": <float>,
    "C4_Societal_Impact": <float>,
    "C5_Open_Science": <float>,
    "C6_Literature_Integration": <float>,
    "C7_Empirical_Density": <float>,
    "C8_Future_Actionability": <float>,
    "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5"]
}}
Text: {text[:MAX_TEXT_TOKENS]}"""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model, temperature=0.1, seed=SEED_NUMBER, response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def process_upload(file_bytes, filename):
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    cursor = conn.cursor()
    
    cursor.execute("SELECT final_score, c1, c2, c3, c4, c5, c6, c7, c8 FROM papers WHERE file_hash=?", (file_hash,))
    cached = cursor.fetchone()
    if cached:
        return cached[0], list(cached[1:9]), True, False
        
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = " ".join([page.get_text() for page in doc])
    
    used_fallback = False
    try:
        raw_scores = evaluate_paper(text, PRIMARY_MODEL)
    except RateLimitError:
        used_fallback = True
        st.toast(f"⚠️ Rate limit exceeded. Failing over to {FALLBACK_MODEL}.", icon="🔄")
        time.sleep(2)
        raw_scores = evaluate_paper(text, FALLBACK_MODEL)
        
    cursor.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM epoch_weights ORDER BY epoch_id DESC LIMIT 1")
    weights = cursor.fetchone()
    
    scores = [
        raw_scores.get("C1_Originality", 5.0), 
        raw_scores.get("C2_Methodological_Rigor", 5.0), 
        raw_scores.get("C3_Interdisciplinary", 5.0), 
        raw_scores.get("C4_Societal_Impact", 5.0),
        raw_scores.get("C5_Open_Science", 5.0), 
        raw_scores.get("C6_Literature_Integration", 5.0),
        raw_scores.get("C7_Empirical_Density", 5.0), 
        raw_scores.get("C8_Future_Actionability", 5.0)
    ]
    
    # Calculate final dynamic score via dot product (Score Vector · Weight Vector)
    final_score = float(np.dot(scores, weights))
    
    cursor.execute('''INSERT INTO papers 
                      (file_hash, filename, c1, c2, c3, c4, c5, c6, c7, c8, keywords, final_score, timestamp) 
                      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                   (file_hash, filename, *scores, json.dumps(raw_scores.get("keywords", [])), final_score, datetime.now().isoformat()))
    conn.commit()
    
    trigger_epoch_recalculation()
    return final_score, scores, False, used_fallback

# --- 5. TOPOLOGICAL SCIENCE MAPPING ---
def generate_trend_network():
    """Builds a semantic co-occurrence network map of current epoch trends."""
    cursor = conn.cursor()
    target_date = (datetime.now() - timedelta(days=EPOCH_DAYS)).isoformat()
    cursor.execute("SELECT keywords FROM papers WHERE timestamp >= ?", (target_date,))
    
    G = nx.Graph()
    for row in cursor.fetchall():
        try:
            keywords = [k.lower().strip() for k in json.loads(row[0])]
            for i in range(len(keywords)):
                for j in range(i+1, len(keywords)):
                    if G.has_edge(keywords[i], keywords[j]):
                        G[keywords[i]][keywords[j]]['weight'] += 1
                    else:
                        G.add_edge(keywords[i], keywords[j], weight=1)
        except:
            continue
            
    if len(G.nodes) == 0:
        return None

    pos = nx.spring_layout(G, k=0.5, seed=42)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    
    node_x, node_y, text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(f"{node} (Freq: {G.degree(node)})")
        
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=text,
                            textposition="top center", hoverinfo='text',
                            marker=dict(showscale=True, colorscale='Viridis', size=12,
                                        color=[G.degree(n) for n in G.nodes()], 
                                        line_width=2))
                                        
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(title='Emerging Scientific Trends (30-Day Keyword Co-Occurrence)',
                                     showlegend=False, hovermode='closest',
                                     margin=dict(b=0,l=0,r=0,t=40),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig

# --- 6. USER INTERFACE ---
st.title("🔭 Dynamic Epistemic Cartography")
st.markdown("An AI-driven, temporally dynamic research evaluation engine powered by MCDA Entropy Weighting.")

tab1, tab2, tab3 = st.tabs(["📄 Document Evaluation", "📊 Science Trend Map", "⚙️ System Dynamics"])

with tab1:
    uploaded_file = st.file_uploader("Upload an Academic Paper (PDF)", type=["pdf"])
    if uploaded_file and st.button("Evaluate Manuscript", type="primary"):
        with st.spinner("Analyzing semantic structure and applying dynamic MCDA algorithms..."):
            score, raw_scores, cached, used_fallback = process_upload(uploaded_file.read(), uploaded_file.name)
            
            if cached:
                st.info("ℹ️ Retrieved calculated metrics from decentralized cache.")
            else:
                model_used = FALLBACK_MODEL if used_fallback else PRIMARY_MODEL
                st.success(f"✅ Algorithms executed successfully via `{model_used}` semantic extraction.")

            st.metric("Dynamic Quality Score (VQR-Aligned)", f"{score:.2f} / 10.0")
            
            st.markdown("### Raw Extracted Criteria")
            labels = ["C1: Originality", "C2: Methodological Rigor", "C3: Interdisciplinary Synthesis", 
                      "C4: Societal Impact", "C5: Open Science", "C6: Literature Integration", 
                      "C7: Empirical Density", "C8: Future Actionability"]
            cols = st.columns(4)
            for i, col in enumerate(cols * 2):
                if i < 8:
                    col.metric(labels[i], f"{raw_scores[i]:.1f}/10")

with tab2:
    st.subheader("Global Epistemic Cartography")
    st.write("This topological network visualizes keyword co-occurrences from all manuscripts processed in the current 30-day epoch. Distant nodes clustering together signify highly disruptive interdisciplinary convergence.")
    fig = generate_trend_network()
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Insufficient data in the current epoch to generate a trend map. Upload more documents.")

with tab3:
    st.subheader("Dynamic Entropy Weights (Current Epoch)")
    st.write("Criteria weights are dynamically recalculated every 30 days based on data dispersion (Entropy). Criteria where the global scientific community achieves uniform consensus drop in weight, while highly variable criteria (disruptive frontiers) increase in weight to differentiate excellence.")
    
    cursor = conn.cursor()
    cursor.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8, timestamp FROM epoch_weights ORDER BY epoch_id DESC LIMIT 1")
    weights = cursor.fetchone()
    
    if weights:
        st.caption(f"Last Epoch Recalculation: {weights[8]}")
        cols = st.columns(4)
        labels = ["C1: Originality", "C2: Method Rigor", "C3: Interdisciplinary", "C4: Societal Impact", 
                  "C5: Open Science", "C6: Lit Integration", "C7: Empirical Density", "C8: Actionability"]
        for i, col in enumerate(cols * 2):
            if i < 8:
                col.metric(labels[i], f"{(weights[i]*100):.1f}%")
