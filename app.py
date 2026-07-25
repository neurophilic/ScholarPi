import os
import re
import json
import time
import requests
import colorsys
import tempfile
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import EPOCH_BLOCK_SIZE
from blockchain import init_system, verify_chain_integrity, calculate_merkle_root
from math_engine import get_pi_float
from ai_engine import process_single_pdf, PiBlockchainDataset, PiBrainLSTM
from services import (
    search_arxiv, download_arxiv_pdf, create_radar_comparison,
    generate_latex_report, generate_bibtex, export_to_csv, export_to_excel,
    get_portfolio_stats
)

st.set_page_config(page_title="π-Index Assessment Engine", layout="wide", page_icon="π")

# --- Database Connection Cache ---
@st.cache_resource
def get_db_connection():
    return init_system()

# --- UI Utilities ---
def verify_orcid_live(orcid_id):
    try:
        url = f"https://pub.orcid.org/v3.0/{orcid_id}/person"
        headers = {"Accept": "application/json"}
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            name_data = response.json().get('name', {})
            if name_data:
                given = name_data.get('given-names', {}).get('value', '') if name_data.get('given-names') else ''
                family = name_data.get('family-name', {}).get('value', '') if name_data.get('family-name') else ''
                return True, f"{given} {family}".strip() or "Verified Researcher (Name Private)"
            return True, "Verified Researcher (Name Private)"
        return False, "ORCID ID not found on public registry."
    except Exception as e:
        return False, f"API Error: {str(e)}"

def generate_interactive_bubble_chart(user_id, target_author=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if target_author and target_author != "All Authors":
        cursor.execute("SELECT fields, subfields, final_score FROM papers_assessment WHERE user_id=? AND author_name LIKE ?", (user_id, f"%{target_author}%"))
    else:
        cursor.execute("SELECT fields, subfields, final_score FROM papers_assessment WHERE user_id=?", (user_id,))
        
    data = cursor.fetchall()
    html_string, table_html = "", ""
    if not data: return html_string, table_html
    
    all_topics = []
    for fields_json, subfields_json, final_score in data:
        try:
            fields = [f.title().strip() for f in json.loads(fields_json)]
            subfields = [s.title().strip() for s in json.loads(subfields_json)]
            score = float(final_score) if final_score else 50.0
            
            for f in fields: all_topics.append({'topic': f, 'weight': score})
            for s in subfields: all_topics.append({'topic': s, 'weight': score})
        except: continue
            
    if not all_topics: return html_string, table_html
    
    df_topics = pd.DataFrame(all_topics)
    topic_counts = df_topics.groupby(['topic'])['weight'].sum().reset_index(name='weight')
    if topic_counts.empty: return html_string, table_html
        
    unique_topics = topic_counts['topic'].unique()
    
    def get_color(i, n):
        h, s, v = i/n if n > 0 else 0, 0.7, 0.9
        rgb = colorsys.hsv_to_rgb(h, s, v)
        return '#%02x%02x%02x' % tuple(int(x * 255) for x in rgb)
    
    color_map = {topic: get_color(i, len(unique_topics)) for i, topic in enumerate(unique_topics)}
    
    net = Network(height='600px', width='100%', bgcolor='#ffffff', font_color='#2c3e50', notebook=False)
    physics_options = """{ "physics": { "barnesHut": { "gravitationalConstant": -1000, "centralGravity": 1, "springLength": 100, "avoidOverlap": 1.0 }, "stabilization": { "enabled": true, "iterations": 200 } } }"""
    net.set_options(physics_options)
    
    for _, row in topic_counts.iterrows():
        node_size = 30 + (row['weight'] * 2.5) 
        net.add_node(n_id=row['topic'], label=' ', title=f"Topic: {row['topic']} | Weight: {row['weight']:.1f}", size=node_size, physics=True, color=color_map[row['topic']])
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
        net.save_graph(tmp_file.name)
        with open(tmp_file.name, 'r', encoding='utf-8') as f: html_string = f.read()
    os.remove(tmp_file.name)

    unique_network_id = f"pi_network_{int(time.time() * 1000)}"
    html_string = html_string.replace('mynetwork', unique_network_id)

    table_html = "<style>.table-big { width: 100%; font-size: 14px; border-collapse: collapse; margin-top: 10px; font-family: sans-serif; } .table-big th { background-color: #2c3e50; color: white; padding: 8px; text-align: left; } .table-big td { padding: 8px; border-bottom: 1px solid #ecf0f1; } .color-box { width: 30px; height: 30px; border-radius: 4px; display: inline-block; } </style>"
    table_html += "<div class='legend-container'><table class='table-big'><thead><tr><th style='width: 25%; text-align: center;'>Color</th><th>Topic</th><th style='text-align: right;'>Weight</th></tr></thead><tbody>"
    for _, row in topic_counts.sort_values(by="weight", ascending=False).iterrows():
        table_html += f"<tr><td style='text-align: center;'><div class='color-box' style='background-color:{color_map[row['topic']]};'></div></td><td>{row['topic']}</td><td style='text-align: right;'>{row['weight']:.1f}</td></tr>"
    table_html += "</tbody></table></div>"
    
    return html_string, table_html

# --- Session State Initialization ---
if 'assessment_update_token' not in st.session_state: 
    st.session_state['assessment_update_token'] = time.time()
if 'orcid_id' not in st.session_state:
    st.session_state.orcid_id = "0000-0000-0000-0000"
    st.session_state.orcid_name = ""
    st.session_state.is_authenticated = False
if 'arxiv_results' not in st.session_state:
    st.session_state.arxiv_results = []
if 'selected_arxiv_paper' not in st.session_state:
    st.session_state.selected_arxiv_paper = None

# --- UI LAYOUT ---
st.sidebar.title("🔐 System Access")

if not st.session_state.is_authenticated:
    st.sidebar.markdown("### Authenticate via ORCID")
    manual_orcid = st.sidebar.text_input("Enter ORCID iD", placeholder="XXXX-XXXX-XXXX-XXXX")
    if st.sidebar.button("🔗 Validate & Connect"):
        clean_orcid = manual_orcid.strip()
        if re.match(r'^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$', clean_orcid):
            with st.sidebar.status("Connecting to ORCID Registry..."):
                is_valid, user_name = verify_orcid_live(clean_orcid)
            if is_valid:
                st.session_state.orcid_id, st.session_state.orcid_name, st.session_state.is_authenticated = clean_orcid, user_name, True
                st.rerun()
            else: 
                st.sidebar.error(user_name)
        else: 
            st.sidebar.error("Invalid ORCID format. Expected: 0000-0000-0000-0000")
else:
    st.sidebar.success("✅ Securely Connected")
    st.sidebar.markdown(f"**Researcher:** {st.session_state.orcid_name}\n\n**ORCID iD:** `{st.session_state.orcid_id}`")
    if st.sidebar.button("🔓 Disconnect Session"):
        st.session_state.is_authenticated, st.session_state.orcid_name = False, ""
        st.session_state.orcid_id = "0000-0000-0000-0000"
        st.rerun()

current_user = st.session_state.orcid_id

st.title("π-Index Assessment Engine")
st.markdown("*Upload papers, define your scope, and let the π-Index filter noise to reveal true research quality.*")

# --- Portfolio Stats (if authenticated) ---
if st.session_state.is_authenticated:
    conn = get_db_connection()
    stats = get_portfolio_stats(conn, current_user)
    if stats['total_papers'] > 0:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Papers Assessed", stats['total_papers'])
        c2.metric("Avg π-Index", f"{stats['avg_score']:.1f}")
        c3.metric("Top Score", f"{stats['max_score']:.1f}")
        c4.metric("Logic Integrity", f"{stats['avg_logic']:.1f}%")
        c5.metric("Fields Covered", stats['unique_fields'])
        st.divider()

with st.expander("📖 View π-Index Grading Criteria & Theoretical Formulations"):
    st.markdown("### Evaluation Metrics & Adversarial Logic Engine")
    st.markdown(r"""
    **Adversarial Logic Gap ($\Delta_{Logic}$):** Before a final score is validated, the system maps the paper's reasoning structure. It penalizes the paper exponentially if the author's conclusions exceed empirical support.
    $$ L_i = (\mathcal{P}_{valid} \cdot \mathcal{E}_{strength}) \cdot \exp\left(-\left(2 \cdot \max(0, \mathcal{C}_{reach} - \mathcal{E}_{strength}) + 1.5 \cdot \lambda_{jumps}\right)\right) \times 100 $$
    """)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**C1: Originality**\nEvaluates uniqueness through epistemic gradient fields.")
        st.markdown(r"$$O = \varpi_1 \cdot \frac{\mathcal{H}_{novel} \otimes \mathcal{K}_{epistemic}}{\zeta \cdot \mathcal{I}_{existing} + \epsilon} \times 60$$")
        st.markdown("**C2: Methodological Rigor**\nAssesses robustness via error-covariance tensors.")
        st.markdown(r"$$R = \varpi_2 \cdot \left( 1 - \frac{\Sigma_{error}}{\mu_{signal} + \epsilon} \right) \cdot \rho_k \cdot \Gamma(1.5) \times 140$$")
        st.markdown("**C3: Interdisciplinary**\nMeasures bridge capacity using generalized Rényi entropy.")
        st.markdown(r"$$I = \varpi_3 \cdot \left( -\ln\sum p_j^2 + bridge\_capacity \right) \times 55$$")
        st.markdown("**C4: Societal Impact**\nProjects applications utilizing fractional stochastic integration.")
        st.markdown(r"$$S = \varpi_4 \cdot \frac{1}{\Gamma(q)} \cdot Utility \cdot e^{-decay} \times 150$$")
    with col2:
        st.markdown("**C5: Open Science Potential**\nGauges transparency via multi-objective integration.")
        st.markdown(r"$$O_s = \varpi_5 \cdot \frac{0.7 \cdot D_{open} + 0.3 \cdot J_{code}}{max(D_{total}, 1)} \times P_{FAIR} \times 180$$")
        st.markdown("**C6: Literature Integration**\nEvaluates embedding via non-Euclidean PageRank.")
        st.markdown(r"$$L = \varpi_6 \cdot e^{-1.5 \cdot d_g} \cdot R_\xi \cdot PR_\xi \times 180$$")
        st.markdown("**C7: Empirical Density**\nEvaluates data depth utilizing Fisher information metrics.")
        st.markdown(r"$$E_d = \varpi_7 \cdot \tanh\left(\frac{I_{Fisher} \cdot KL_{div}}{V_{baseline} \cdot \omega_{data} + \epsilon}\right) \times \sum\lambda\kappa \times 80$$")
        st.markdown("**C8: Future Actionability**\nDetermines continuation potential using Lyapunov exponents.")
        st.markdown(r"$$F_a = \varpi_8 \cdot \frac{1}{1 + e^{-(\eta - 5\Lambda_{Lyapunov})}} \times 100$$")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📤 Batch Assessment", 
    "🔍 Research Discovery", 
    "📊 Scope Cartography", 
    "⚖️ Paper Comparison",
    "⛓️ Active Epoch & Blockchain", 
    "🧠 π-Brain Neural Network"
])

# ==================== TAB 1: BATCH ASSESSMENT ====================
with tab1:
    research_scope = st.text_input(
        "Define your specific Research Topic / Scope (Optional)", 
        placeholder="e.g., Application of deep learning in vascular imaging...",
        help="Adding a scope enables Scope Drift calculation and recommendation spectrum."
    )
    
    col_up1, col_up2 = st.columns([3, 1])
    with col_up1:
        uploaded_files = st.file_uploader(
            "Upload Academic Papers (PDFs)", 
            type=["pdf"], 
            accept_multiple_files=True,
            help="Upload one or more PDFs to assess. Duplicate papers are automatically cached."
        )
    with col_up2:
        st.write("")
        st.write("")
        if st.button("🚀 Run Batch Assessment", type="primary", use_container_width=True):
            if not uploaded_files: 
                st.warning("Please upload at least one academic paper (PDF) to proceed.")
            else:
                results_list = []
                progress_bar, status_text = st.progress(0), st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Analyzing {i+1} of {len(uploaded_files)}: {file.name}...")
                    
                    try:
                        title, author_name, score, logic_integrity, drift, rec, fields, subfields, scores_dict, eval_hash = process_single_pdf(
                            file.read(), file.name, research_scope, current_user
                        )
                        
                        record = {
                            "No.": i + 1, 
                            "File Name": file.name, 
                            "Title": title,
                            "Primary Author": author_name, 
                            "Fields": ", ".join(fields),
                            "Subfields": ", ".join(subfields),
                            "Logic Integrity (%)": round(logic_integrity, 1), 
                            "π-Index (0-100)": round(score, 1),
                            "Eval Hash": eval_hash
                        }
                        
                        if research_scope.strip():
                            record.update({
                                "Scope": research_scope, 
                                "Recommendation": rec, 
                                "Scope Drift %": round(drift, 1) if drift != "N/A" else "N/A"
                            })
                        
                        for j in range(8):
                            key = f"C{j+1}"
                            record[key] = round(scores_dict.get(list(scores_dict.keys())[j], 0.0), 1)
                        
                        results_list.append(record)
                    except Exception as e:
                        st.error(f"Failed to process {file.name}: {str(e)}")
                        record = {
                            "No.": i + 1, 
                            "File Name": file.name, 
                            "Title": "ERROR",
                            "Primary Author": "N/A", 
                            "π-Index (0-100)": 0.0,
                            "Error": str(e)
                        }
                        results_list.append(record)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                status_text.success(f"✅ Batch processing complete! {len(results_list)} papers evaluated.")
                st.session_state['latest_assessment_results'] = pd.DataFrame(results_list)
                st.session_state['assessment_update_token'] = time.time()
                st.session_state['last_trained_blocks'] = -1
                
                # Auto-export options
                if results_list:
                    csv_data = export_to_csv(pd.DataFrame(results_list))
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_data,
                        file_name=f"pi_index_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
    if 'latest_assessment_results' in st.session_state:
        st.subheader("Latest Assessment Results")
        st.dataframe(st.session_state['latest_assessment_results'], use_container_width=True, hide_index=True)
        
        # Individual exports
        st.markdown("#### Individual Paper Exports")
        selected_idx = st.selectbox(
            "Select a paper to export:", 
            range(len(st.session_state['latest_assessment_results'])),
            format_func=lambda i: f"{i+1}. {st.session_state['latest_assessment_results'].iloc[i]['Title'][:50]}"
        )
        
        if selected_idx is not None:
            row = st.session_state['latest_assessment_results'].iloc[selected_idx]
            c1, c2, c3 = st.columns(3)
            
            scores_dict = {f"C{i+1}": row.get(f"C{i+1}", 0.0) for i in range(8)}
            
            with c1:
                latex_report = generate_latex_report(
                    row['Title'], row['Primary Author'], 
                    row['π-Index (0-100)'], row['Logic Integrity (%)'],
                    scores_dict, row['Eval Hash']
                )
                st.download_button(
                    "📄 LaTeX Report", latex_report,
                    file_name=f"pi_report_{row['Eval Hash'][:8]}.tex",
                    mime="text/plain"
                )
            with c2:
                bibtex = generate_bibtex(row['Title'], row['Primary Author'], row['Eval Hash'])
                st.download_button(
                    "📚 BibTeX Citation", bibtex,
                    file_name=f"pi_citation_{row['Eval Hash'][:8]}.bib",
                    mime="text/plain"
                )
            with c3:
                if 'Scope Drift %' in row:
                    st.metric("Scope Drift", f"{row['Scope Drift %']}%")

    st.markdown("### 📜 Assessment History")
    if st.session_state.is_authenticated:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT title, author_name, scope, final_score, timestamp, eval_hash, logic_score 
            FROM papers_assessment 
            WHERE user_id=? 
            ORDER BY timestamp DESC 
            LIMIT 50
        """, (current_user,))
        history_data = cursor.fetchall()
        if history_data: 
            df_hist = pd.DataFrame(history_data, columns=[
                "Paper Title", "Primary Author", "Scope", "π-Index Score", 
                "Date", "Evaluation Hash", "Logic Integrity"
            ])
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
            
            # Excel export for history
            excel_data = export_to_excel(df_hist)
            st.download_button(
                "📊 Download Full History (Excel)",
                excel_data,
                file_name=f"pi_history_{current_user}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else: 
            st.info("No assessment history found.")
    else: 
        st.warning("Please connect your ORCID iD in the sidebar to view history.")

# ==================== TAB 2: ARXIV RESEARCH DISCOVERY ====================
with tab2:
    st.subheader("🔍 ArXiv Research Discovery")
    st.markdown("Search ArXiv and directly assess papers through the π-Index pipeline.")
    
    arxiv_col1, arxiv_col2 = st.columns([3, 1])
    with arxiv_col1:
        arxiv_query = st.text_input("Search Query", placeholder="e.g., 'transformer architecture medical imaging'")
    with arxiv_col2:
        max_results = st.number_input("Max Results", min_value=1, max_value=20, value=5)
    
    if st.button("🔎 Search ArXiv", type="primary"):
        with st.spinner("Searching ArXiv..."):
            results = search_arxiv(arxiv_query, max_results)
            st.session_state.arxiv_results = results
            if not results:
                st.warning("No results found. Try a different query.")
            else:
                st.success(f"Found {len(results)} papers")
    
    if st.session_state.arxiv_results:
        st.markdown("### Results")
        for i, paper in enumerate(st.session_state.arxiv_results):
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f"**{paper['title']}**")
                    st.caption(f"Authors: {paper['authors']}")
                    with st.expander("Abstract"):
                        st.write(paper['summary'])
                with c2:
                    if st.button("📥 Assess", key=f"assess_arxiv_{i}"):
                        with st.spinner(f"Downloading and assessing: {paper['title'][:40]}..."):
                            pdf_bytes = download_arxiv_pdf(paper['pdf_url'])
                            if pdf_bytes:
                                title, author_name, score, logic_integrity, drift, rec, fields, subfields, scores_dict, eval_hash = process_single_pdf(
                                    pdf_bytes, f"arxiv_{paper['title'][:30]}.pdf", research_scope, current_user
                                )
                                st.success(f"Assessment Complete! π-Index: {score:.1f}")
                                st.json({
                                    "Title": title, "Author": author_name, "π-Index": round(score, 1),
                                    "Logic Integrity": round(logic_integrity, 1), "Recommendation": rec,
                                    "Fields": fields, "Subfields": subfields
                                })
                            else:
                                st.error("Failed to download PDF from ArXiv.")

# ==================== TAB 3: SCOPE CARTOGRAPHY ====================
with tab2:
    pass  # Placeholder to avoid syntax issues; actual tab3 content below

with tab3:
    st.subheader("🗺️ Epistemic Bubbles (Author & Portfolio Cartography)")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT author_name FROM papers_assessment WHERE user_id=?", (current_user,))
    user_authors = sorted(list(set([row[0].strip() for row in cursor.fetchall() if row[0] and row[0].strip()])))
    
    selected_author = None
    if user_authors:
        filter_choice = st.selectbox(
            "Filter Cartography by Primary Author:", 
            ["All Authors"] + user_authors, 
            key=f"author_filter_dropdown_{st.session_state['assessment_update_token']}"
        )
        if filter_choice != "All Authors": 
            selected_author = filter_choice

    interactive_html, table_html = generate_interactive_bubble_chart(current_user, target_author=selected_author)
    if interactive_html:
        col1, col2 = st.columns([3, 1])
        with col1: 
            components.html(interactive_html, height=620, scrolling=True)
        with col2: 
            st.markdown("### Legend")
            st.markdown(table_html, unsafe_allow_html=True)
    else: 
        st.info("Awaiting sufficient data. Assess some papers to generate your research landscape.")

# ==================== TAB 4: PAPER COMPARISON ====================
with tab4:
    st.subheader("⚖️ Paper Comparison Engine")
    st.markdown("Select two papers from your assessment history to compare across all 8 criteria.")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT eval_hash, title, author_name, final_score, c1, c2, c3, c4, c5, c6, c7, c8 
        FROM papers_assessment 
        WHERE user_id=? 
        ORDER BY timestamp DESC 
        LIMIT 100
    """, (current_user,))
    papers = cursor.fetchall()
    
    if len(papers) < 2:
        st.info("You need at least 2 assessed papers to use the comparison feature.")
    else:
        paper_options = {f"{p[1][:50]}... ({p[2]})" if len(p[1]) > 50 else f"{p[1]} ({p[2]})": p for p in papers}
        
        c1, c2 = st.columns(2)
        with c1:
            selected_paper_1 = st.selectbox("Paper A", list(paper_options.keys()), key="comp_a")
        with c2:
            selected_paper_2 = st.selectbox("Paper B", list(paper_options.keys()), index=min(1, len(paper_options)-1), key="comp_b")
        
        if selected_paper_1 and selected_paper_2 and selected_paper_1 != selected_paper_2:
            p1 = paper_options[selected_paper_1]
            p2 = paper_options[selected_paper_2]
            
            scores1 = {
                "C1_Originality": p1[4], "C2_Methodological_Rigor": p1[5],
                "C3_Interdisciplinary": p1[6], "C4_Societal_Impact": p1[7],
                "C5_Open_Science_Potential": p1[8], "C6_Literature_Integration": p1[9],
                "C7_Empirical_Density": p1[10], "C8_Future_Actionability": p1[11]
            }
            scores2 = {
                "C1_Originality": p2[4], "C2_Methodological_Rigor": p2[5],
                "C3_Interdisciplinary": p2[6], "C4_Societal_Impact": p2[7],
                "C5_Open_Science_Potential": p2[8], "C6_Literature_Integration": p2[9],
                "C7_Empirical_Density": p2[10], "C8_Future_Actionability": p2[11]
            }
            
            fig = create_radar_comparison(p1[1], scores1, p2[1], scores2)
            st.plotly_chart(fig, use_container_width=True)
            
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            with comp_col1:
                st.metric("Paper A π-Index", f"{p1[3]:.1f}")
            with comp_col2:
                st.metric("Paper B π-Index", f"{p2[3]:.1f}")
            with comp_col3:
                diff = p1[3] - p2[3]
                st.metric("Difference", f"{diff:+.1f}", delta="A higher" if diff > 0 else "B higher")
        else:
            st.warning("Please select two different papers.")

# ==================== TAB 5: BLOCKCHAIN EXPLORER ====================
with tab5:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT block_height, w1, w2, w3, w4, w5, w6, w7, w8, model_used, eval_hash, block_hash, timestamp, previous_hash 
        FROM blockchain_por_weights 
        ORDER BY block_height DESC 
        LIMIT 1
    """)
    epoch_data = cursor.fetchone()
    
    if epoch_data:
        block_height = epoch_data[0]
        weights = epoch_data[1:9]
        model_used = epoch_data[9]
        eval_hash = epoch_data[10]
        block_hash = epoch_data[11]
        timestamp = epoch_data[12]
        previous_hash = epoch_data[13]
        
        cursor.execute("SELECT COUNT(DISTINCT eval_hash) FROM blockchain_por_weights WHERE eval_hash != 'genesis'")
        total_papers_processed = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM blockchain_por_weights WHERE eval_hash != 'genesis'")
        total_blocks = cursor.fetchone()[0]

        st.markdown(f"""
        **📦 Ledger Status:** `{total_blocks}` blocks | **📝 Papers:** `{total_papers_processed}` | 
        **🔗 Block Size:** `{EPOCH_BLOCK_SIZE}` | **🤖 Model:** `{model_used}` | 
        **⬆️ Height:** `{block_height}` | **π Acc:** `{get_pi_float(block_height)}`
        """)
        
        # Chain integrity check
        is_valid, invalid_block = verify_chain_integrity(conn)
        if is_valid:
            st.success("✅ Chain integrity verified. All hashes and links are valid.")
        else:
            st.error(f"❌ Chain integrity compromised at block {invalid_block}!")
        
        # Merkle root of current weights
        merkle_root = calculate_merkle_root(list(weights))
        st.caption(f"Current Epoch Merkle Root: `{merkle_root}`")
        
        cols = st.columns(4)
        labels = [("C1", r"$\varpi_1$"), ("C2", r"$\varpi_2$"), ("C3", r"$\varpi_3$"), ("C4", r"$\varpi_4$"), 
                  ("C5", r"$\varpi_5$"), ("C6", r"$\varpi_6$"), ("C7", r"$\varpi_7$"), ("C8", r"$\varpi_8$")]
        for i, col in enumerate(cols * 2):
            if i < 8:
                col.markdown(f"**{labels[i][0]} ({labels[i][1]})**")
                col.markdown(f"<h3 style='margin-top:0px; margin-bottom:5px;'>{weights[i]:.6f}</h3>", unsafe_allow_html=True)
                
        st.markdown("### 🔍 PoR Blockchain Explorer")
        explore_col1, explore_col2 = st.columns([3, 1])
        with explore_col1: 
            search_query = st.text_input("Enter Document Evaluation Hash or Block Hash to verify ledger record...")
        with explore_col2: 
            st.write("")
            st.write("")
            search_btn = st.button("🔎 Verify Record", use_container_width=True)
            
        if search_btn and search_query:
            cursor.execute("""
                SELECT block_height, timestamp, model_used, validator_node, block_hash, eval_hash, 
                       w1, w2, w3, w4, w5, w6, w7, w8, previous_hash 
                FROM blockchain_por_weights 
                WHERE block_hash=? OR eval_hash=?
            """, (search_query, search_query))
            record = cursor.fetchone()
            if record:
                st.success("✅ Valid Block Found on Ledger!")
                st.json({
                    "Block Height": record[0], 
                    "Timestamp": record[1], 
                    "Model Used": record[2], 
                    "Validator Node": record[3],
                    "Block Hash": record[4], 
                    "Evaluation Hash": record[5], 
                    "Previous Hash": record[14],
                    "Weights": dict(zip([f"w{i+1}" for i in range(8)], record[6:14]))
                })
            else: 
                st.error("❌ No block matching that signature was found on the ledger.")
                
        # Block history table
        st.markdown("### 📜 Recent Block History")
        cursor.execute("""
            SELECT block_height, timestamp, eval_hash, model_used, block_hash 
            FROM blockchain_por_weights 
            ORDER BY block_height DESC 
            LIMIT 10
        """)
        history = cursor.fetchall()
        if history:
            st.dataframe(
                pd.DataFrame(history, columns=["Height", "Timestamp", "Eval Hash", "Model", "Block Hash"]),
                use_container_width=True, hide_index=True
            )

# ==================== TAB 6: PI-BRAIN NEURAL NETWORK ====================
with tab6:
    st.subheader("🧠 π-Brain: Meta-Learning on the PoR Blockchain")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights ORDER BY block_height ASC")
    historical_rows = cursor.fetchall()
    
    lookback_window = 5
    if len(historical_rows) < lookback_window + 2:
        st.warning(f"Not enough blockchain data to train the meta-model. Need at least {lookback_window + 2} blocks (currently {len(historical_rows)}).")
        st.info("Assess more papers to generate additional epoch blocks.")
    else:
        current_block_count = len(historical_rows)
        train_col, info_col = st.columns([2, 1])
        
        with train_col:
            if st.button("🚀 Train / Refresh π-Brain Model", type="primary") or \
               ('last_trained_blocks' not in st.session_state or st.session_state.last_trained_blocks != current_block_count):
                
                with st.spinner("Training LSTM on blockchain weight evolution..."):
                    weight_data = np.array(historical_rows, dtype=np.float32)
                    dataset = PiBlockchainDataset(weight_data, lookback_window)
                    dataloader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False)
                    
                    # CRITICAL FIX: Use single model instance
                    model = PiBrainLSTM()
                    loss_function = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    epochs = 200
                    
                    model.train()
                    for epoch in range(epochs):
                        total_loss = 0
                        for seq, target in dataloader:
                            optimizer.zero_grad()
                            output = model(seq)
                            loss = loss_function(output, target)
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()
                        
                        avg_loss = total_loss / len(dataloader)
                        if epoch % 20 == 0 or epoch == epochs - 1:
                            status_text.text(f"Epoch {epoch}/{epochs} | MSE Loss: {avg_loss:.6f}")
                            progress_bar.progress((epoch + 1) / epochs)
                    
                    model.eval()
                    with torch.no_grad():
                        last_sequence = torch.tensor(weight_data[-lookback_window:], dtype=torch.float32).unsqueeze(0)
                        prediction = model(last_sequence).squeeze().numpy()
                        
                        st.session_state.predicted_next_weights = prediction
                        st.session_state.current_weights = weight_data[-1]
                        st.session_state.last_trained_blocks = current_block_count
                        
                st.success("π-Brain training complete!")
        
        with info_col:
            st.markdown("""
            **Model Architecture:**
            - LSTM Hidden: 32 units
            - Linear: 32 → 16 → 8
            - Activation: ReLU + Softmax
            - Output: 8-dimensional weight vector
            """)

        if 'predicted_next_weights' in st.session_state:
            df_compare = pd.DataFrame({
                "Current Active Weights": st.session_state.current_weights, 
                "Predicted Next Epoch": st.session_state.predicted_next_weights
            }, index=[
                "C1: Originality", "C2: Methodological Rigor", "C3: Interdisciplinary", 
                "C4: Societal Impact", "C5: Open Science", "C6: Literature Integration", 
                "C7: Empirical Density", "C8: Future Actionability"
            ])
            st.bar_chart(df_compare, height=400)
            
            pred_sum = sum(st.session_state.predicted_next_weights)
            current_sum = sum(st.session_state.current_weights)
            c1, c2 = st.columns(2)
            c1.metric("Current Weight Sum", f"{current_sum:.4f}", "Target: 8.0")
            c2.metric("Predicted Weight Sum", f"{pred_sum:.4f}", f"Δ {pred_sum-current_sum:+.4f}")
            
            if abs(pred_sum - 8.0) > 0.5:
                st.warning("⚠️ Predicted weights deviate significantly from the normalization target of 8.0. Consider retraining.")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
    "Framework Author: Ali Vafadar Yengejeh | Università degli Studi di Milano-Bicocca<br>"
    "π-Index Assessment Engine v2.0 | Enhanced Edition"
    "</div>", 
    unsafe_allow_html=True
)
