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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import EPOCH_BLOCK_SIZE
from blockchain import init_system, generate_blockchain_pi
from ai_engine import process_single_pdf, PiBlockchainDataset, PiBrainLSTM
from services import fetch_doi_metadata, download_pdf_from_url, generate_rebuttal_strategy

st.set_page_config(page_title="π-Index Assessment Engine", layout="wide")

@st.cache_resource
def get_db_connection():
    return init_system()

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

def generate_interactive_bubble_chart(target_author=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if target_author and target_author != "All Authors":
        cursor.execute("SELECT fields, subfields, final_score FROM papers_assessment WHERE author_name LIKE ?", (f"%{target_author}%",))
    else:
        cursor.execute("SELECT fields, subfields, final_score FROM papers_assessment")
        
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
        net.add_node(n_id=row['topic'], label=' ', title=f"Topic: {row['topic']} | Weight: {row['weight']}", size=node_size, physics=True, color=color_map[row['topic']])
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
        net.save_graph(tmp_file.name)
        with open(tmp_file.name, 'r', encoding='utf-8') as f: html_string = f.read()
    os.remove(tmp_file.name)

    unique_network_id = f"pi_network_{int(time.time() * 1000)}"
    html_string = html_string.replace('mynetwork', unique_network_id)

    table_html = "<style>.table-big { width: 100%; font-size: 14px; border-collapse: collapse; margin-top: 10px; font-family: sans-serif; } .table-big th { background-color: #2c3e50; color: white; padding: 8px; text-align: left; } .table-big td { padding: 8px; border-bottom: 1px solid #ecf0f1; } .color-box { width: 30px; height: 30px; border-radius: 4px; display: inline-block; } </style>"
    table_html += "<div class='legend-container'><table class='table-big'><thead><tr><th style='width: 25%; text-align: center;'>Color</th><th>Topic</th></tr></thead><tbody>"
    for _, row in topic_counts.sort_values(by="weight", ascending=False).iterrows():
        table_html += f"<tr><td style='text-align: center;'><div class='color-box' style='background-color:{color_map[row['topic']]};'></div></td><td>{row['topic']}</td></tr>"
    table_html += "</tbody></table></div>"
    
    return html_string, table_html

st.sidebar.title("System Access")

if 'assessment_update_token' not in st.session_state: st.session_state['assessment_update_token'] = time.time()
if 'orcid_id' not in st.session_state:
    st.session_state.orcid_id = "0000-0000-0000-0000"
    st.session_state.orcid_name = ""
    st.session_state.eth_wallet = "None"
    st.session_state.is_authenticated = False

if not st.session_state.is_authenticated:
    st.sidebar.markdown("### Authenticate")
    manual_orcid = st.sidebar.text_input("Enter ORCID iD", placeholder="XXXX-XXXX-XXXX-XXXX")
    wallet_input = st.sidebar.text_input("Ethereum Wallet Address (For $PIC Rewards)", placeholder="0x...")
    
    if st.sidebar.button("🔗 Validate & Connect"):
        clean_orcid = manual_orcid.strip()
        if re.match(r'^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$', clean_orcid):
            with st.sidebar.status("Connecting to ORCID Registry..."):
                is_valid, user_name = verify_orcid_live(clean_orcid)
            if is_valid:
                st.session_state.orcid_id, st.session_state.orcid_name, st.session_state.is_authenticated = clean_orcid, user_name, True
                st.session_state.eth_wallet = wallet_input.strip() if wallet_input.strip() else "None"
                st.rerun()
            else: st.sidebar.error(user_name)
        else: st.sidebar.error("Invalid ORCID format.")
else:
    st.sidebar.success("Securely Connected")
    st.sidebar.markdown(f"**Researcher:** {st.session_state.orcid_name}\n**ORCID iD:** `{st.session_state.orcid_id}`")
    st.sidebar.markdown(f"**ETH Wallet:** `{st.session_state.eth_wallet[:6]}...{st.session_state.eth_wallet[-4:]}`")
    if st.sidebar.button("Disconnect Session"):
        st.session_state.is_authenticated, st.session_state.orcid_name = False, ""
        st.session_state.eth_wallet = "None"
        st.rerun()

current_user = st.session_state.orcid_id
current_wallet = st.session_state.eth_wallet

st.title("π-Index Assessment Engine")
st.markdown("**Upload papers, define your scope of research, let π-index filter noise and have better results**")

with st.expander("View π-Index Grading Criteria (Math to Plain English Translation)"):
    st.markdown("### Evaluation Metrics & Adversarial Logic Engine")
    st.markdown(r"""
    **Adversarial Logic Gap ($\Delta_{Logic}$):** 
    *Plain English:* We map the paper's reasoning structure before giving a final score. If the authors make claims that aren't supported by their own evidence, the system exponentially penalizes the paper.
    $$ L_i = (\mathcal{P}_{valid} \cdot \mathcal{E}_{strength}) \cdot \exp\left(-\left(2 \cdot \max(0, \mathcal{C}_{reach} - \mathcal{E}_{strength}) + 1.5 \cdot \lambda_{jumps}\right)\right) \times \frac{1}{1 + e^{-\Delta Premise}} $$
    """)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**C1: Originality**\n*Plain English:* Does this paper disrupt existing knowledge (high score), or is it mostly derivative of older work (low score)?")
        st.markdown(r"$$O = \varpi_1 \cdot \lim_{\Delta t \to 0} \oint_{\partial \Omega} \frac{\nabla \times (\mathcal{H}_{novel} \otimes \mathcal{K}_{epistemic})}{\iint_{\mathcal{M}} \sum_{i=1}^{N} |Z_i| \, dV} \cdot e^{-0.1 \zeta} $$")
        st.markdown("**C2: Methodological Rigor**\n*Plain English:* Are the methods statistically sound, and is the risk of a fundamental flaw minimized?")
        st.markdown(r"$$R = \varpi_2 \cdot \left( 1 - \frac{\mathrm{tr}(\boldsymbol{\Sigma}_{error} \boldsymbol{\Lambda}^{-1})}{\det(\boldsymbol{\mu}_{signal} \otimes \mathbf{W})} \right) \cdot \mathbb{E}[\rho_k] $$")
        st.markdown("**C3: Interdisciplinary**\n*Plain English:* How well does the research bridge multiple disciplines together rather than staying in an isolated silo?")
        st.markdown(r"$$I = \varpi_3 \cdot \left( \frac{1}{1-\alpha} \ln \left( \sum_{j=1}^{K} p_j^\alpha \right) + \sum_{i,j} \frac{A_{ij} \phi_i \phi_j}{\sqrt{d_i d_j}} \right) \cdot bridge\_capacity $$")
        st.markdown("**C4: Societal Impact**\n*Plain English:* What is the predicted long-term, real-world utility of the research findings?")
        st.markdown(r"$$S = \varpi_4 \cdot \frac{1}{\Gamma(q)} \int_{t_0}^{t_\infty} (t_\infty - \tau)^{q-1} e^{-\gamma(\tau) \tau} \cdot \Theta\left[ \sum_{v \in \mathcal{V}} \omega_v U_v(\tau, \mathbf{x}) \right] d\tau $$")
    with col2:
        st.markdown("**C5: Open Science Potential**\n*Plain English:* Rewards transparency, specifically the sharing of open-source datasets and verifiable code.")
        st.markdown(r"$$O_s = \varpi_5 \cdot \frac{\sum_{\ell \in \mathcal{L}} \alpha_\ell \mathcal{D}_{open}^{(\ell)} + \beta \iint_{\mathcal{C}} \nabla \cdot \mathbf{J}_{code} \, dV}{\max \left[ \mathcal{N}_{\text{datasets}}, 1 \right]} $$")
        st.markdown("**C6: Literature Integration**\n*Plain English:* Assesses how firmly grounded the paper is in foundational literature without being completely reliant on it.")
        st.markdown(r"$$L = \varpi_6 \cdot \frac{1}{\mathcal{N}} \sum_{i=1}^{\mathcal{N}} \int_{\mathcal{M}} e^{-\lambda d_g(x_i, x_{core})} R(x_i) \sqrt{g} \, dx_i \cdot \frac{\text{PR}(x_i)}{\sum PR} $$")
        st.markdown("**C7: Empirical Density**\n*Plain English:* Measures the sheer depth and volume of the underlying data analyzed.")
        st.markdown(r"$$E_d = \varpi_7 \cdot \tanh \left( \frac{\det \mathcal{I}_{Fisher}(\hat{\theta}) \cdot \mathbb{E}_{P}\left[\log\frac{P}{Q}\right]}{\mathcal{V}_{baseline} \cdot \oint_\Gamma K(\mathbf{x}) \, d\ell} \right) $$")
        st.markdown("**C8: Future Actionability**\n*Plain English:* Predicts whether the paper will trigger a cascade of actionable future research.")
        st.markdown(r"$$F_a = \varpi_8 \cdot \frac{1}{\mathcal{Z}} \int_{\mathcal{X}} \frac{1}{1 + \exp\left(-\sum_{k=1}^K w_k(\eta_k(\mathbf{x}) - \eta_{0,k}) + \Lambda_{Lyapunov}\right)} d\mu(\mathbf{x}) $$")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Assessment & DOI Import", "📊 Global Map of Science", "🚀 Super Features", "⛓️ Active Epoch constants", "🧠 π-Brain Neural Network"])

with tab1:
    st.subheader("Document Assessment & Import")
    research_scope = st.text_input("Define your specific Research Topic / Scope (Optional)", placeholder="e.g., Application of deep learning in vascular imaging...")
    
    col_up, col_doi = st.columns(2)
    with col_up:
        st.markdown("#### Upload Local PDF")
        uploaded_files = st.file_uploader("Upload Academic Papers", type=["pdf"], accept_multiple_files=True)
    with col_doi:
        st.markdown("#### Import via Unpaywall (DOI)")
        doi_input = st.text_input("Enter Document Object Identifier (DOI)", placeholder="10.1038/s41586-020-2649-2")
    
    if st.button("🚀 Run Assessment Pipeline & Mint Rewards", type="primary", use_container_width=True):
        if not uploaded_files and not doi_input.strip():
            st.warning("Please upload a PDF or provide a valid DOI to proceed.")
        else:
            results_list = []
            progress_bar, status_text = st.progress(0), st.empty()
            
            if doi_input.strip():
                status_text.text(f"Resolving DOI: {doi_input}...")
                metadata = fetch_doi_metadata(doi_input)
                if metadata and metadata['pdf_url']:
                    pdf_bytes = download_pdf_from_url(metadata['pdf_url'])
                    if pdf_bytes:
                        status_text.text(f"Assessing Open Access document from DOI...")
                        title, author_name, score, logic_integrity, drift, rec, fields, subfields, scores_dict, eval_hash, coins, tx_hash, zk_proof = process_single_pdf(
                            pdf_bytes, f"DOI_{doi_input.replace('/', '_')}.pdf", research_scope, current_user, current_wallet
                        )
                        record = {
                            "Source": "DOI", "Title": title, "Primary Author": author_name, 
                            "π-Index": round(score, 1), "$PIC Minted": coins, "zk-SNARK": f"{zk_proof[:10]}..."
                        }
                        results_list.append(record)
                    else: st.error("Failed to securely download PDF from the Open Access source.")
                else: st.error("Failed to resolve DOI or no Open Access PDF is publicly available.")
            
            if uploaded_files:
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Analyzing uploaded file {i+1} of {len(uploaded_files)}: {file.name}...")
                    title, author_name, score, logic_integrity, drift, rec, fields, subfields, scores_dict, eval_hash, coins, tx_hash, zk_proof = process_single_pdf(
                        file.read(), file.name, research_scope, current_user, current_wallet
                    )
                    
                    record = {
                        "Source": "File", "Title": title, "Primary Author": author_name, 
                        "π-Index": round(score, 1), "$PIC Minted": coins, "zk-SNARK": f"{zk_proof[:10]}..."
                    }
                    results_list.append(record)
                    progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.success("Pipeline processing complete!")
            if results_list:
                st.session_state['latest_assessment_results'] = pd.DataFrame(results_list)
                st.session_state['assessment_update_token'] = time.time()
                st.session_state['last_trained_blocks'] = -1
            
    if 'latest_assessment_results' in st.session_state:
        st.dataframe(st.session_state['latest_assessment_results'], use_container_width=True, hide_index=True)

    st.markdown("### Your Assessment & Reward History")
    if st.session_state.is_authenticated:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT title, author_name, scope, final_score, coins_minted, zk_proof, tx_hash FROM papers_assessment WHERE user_id=? ORDER BY timestamp DESC LIMIT 20", (current_user,))
        history_data = cursor.fetchall()
        if history_data: st.dataframe(pd.DataFrame(history_data, columns=["Paper Title", "Primary Author", "Scope", "π-Index Score", "$PIC Minted", "zk-SNARK Proof", "Eth Tx Hash"]), use_container_width=True, hide_index=True)
        else: st.info("No assessment history found.")
    else: st.warning("Please connect your ORCID iD in the sidebar.")

with tab2:
    st.subheader("Global Map of Science (Ledger-Driven Cartography)")
    st.markdown("This map is permanently updated by every user assessing documents on the blockchain ledger, forming an unalterable topological view of current scientific trends.")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT author_name FROM papers_assessment")
    all_global_authors = sorted(list(set([row[0].strip() for row in cursor.fetchall() if row[0] and row[0].strip()])))
    
    selected_author = None
    if all_global_authors:
        filter_choice = st.selectbox("Filter Global Cartography by Author:", ["All Authors"] + all_global_authors, key=f"author_filter_dropdown_{st.session_state['assessment_update_token']}")
        if filter_choice != "All Authors": selected_author = filter_choice

    interactive_html, table_html = generate_interactive_bubble_chart(target_author=selected_author)
    if interactive_html:
        col1, col2 = st.columns([3, 1])
        with col1: components.html(interactive_html, height=620, scrolling=True)
        with col2: st.markdown("### Legend"); st.markdown(table_html, unsafe_allow_html=True)
    else: st.info("Awaiting sufficient data for this selection.")

with tab3:
    st.subheader("🚀 System Super Features")
    st.markdown("Use these advanced utilities to gain strategic insights into your evaluations.")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT eval_hash, title, author_name, c1, c2, c3, c4, c5, c6, c7, c8 FROM papers_assessment WHERE user_id=? ORDER BY timestamp DESC", (current_user,))
    user_papers = cursor.fetchall()
    
    if not user_papers:
        st.info("You must assess at least one paper to unlock the Super Features.")
    else:
        paper_options = {f"{p[1][:50]}... ({p[2]})" if len(p[1]) > 50 else f"{p[1]} ({p[2]})": p for p in user_papers}
        selected_super_paper = st.selectbox("Select a paper to analyze:", list(paper_options.keys()))
        
        if st.button("🛡️ Generate AI Peer Review Defense Strategy"):
            paper_data = paper_options[selected_super_paper]
            scores = {
                "C1_Originality": paper_data[3], "C2_Methodological_Rigor": paper_data[4],
                "C3_Interdisciplinary": paper_data[5], "C4_Societal_Impact": paper_data[6],
                "C5_Open_Science_Potential": paper_data[7], "C6_Literature_Integration": paper_data[8],
                "C7_Empirical_Density": paper_data[9], "C8_Future_Actionability": paper_data[10]
            }
            rebuttal = generate_rebuttal_strategy(scores)
            st.success("Defense Strategy Generated Successfully.")
            st.markdown(rebuttal)

with tab4:
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT block_height, w1, w2, w3, w4, w5, w6, w7, w8, model_used, eval_hash, block_hash, por_proof, formulas_hash FROM blockchain_por_weights ORDER BY block_height DESC LIMIT 1")
        epoch_data = cursor.fetchone()
    except Exception:
        epoch_data = None
    
    if epoch_data:
        block_height, weights, model_used, eval_hash, block_hash, por_proof, formulas_hash = epoch_data[0], epoch_data[1:9], epoch_data[9], epoch_data[10], epoch_data[11], epoch_data[12], epoch_data[13]
        cursor.execute("SELECT COUNT(DISTINCT eval_hash) FROM blockchain_por_weights WHERE eval_hash != 'genesis'")
        total_papers_processed = cursor.fetchone()[0]

        # Fetch the algorithmic Pi value from the blockchain
        current_pi_accuracy = generate_blockchain_pi(block_height)

        st.markdown(f"**Processed:** `{total_papers_processed}` | **Block Size:** `{EPOCH_BLOCK_SIZE}` | **Model:** `{model_used}` | **Block:** `{block_height}` | **Pi Algorithmic Precision:** `{current_pi_accuracy}`")
        
        cols = st.columns(4)
        labels = [("C1", r"$\varpi_1$"), ("C2", r"$\varpi_2$"), ("C3", r"$\varpi_3$"), ("C4", r"$\varpi_4$"), ("C5", r"$\varpi_5$"), ("C6", r"$\varpi_6$"), ("C7", r"$\varpi_7$"), ("C8", r"$\varpi_8$")]
        for i, col in enumerate(cols * 2):
            if i < 8:
                col.markdown(f"**{labels[i][0]} ({labels[i][1]})**")
                col.markdown(f"<h3 style='margin-top:0px; margin-bottom:5px;'>{weights[i]:.6f}</h3>", unsafe_allow_html=True)
                
        st.markdown("### PoR Blockchain Explorer")
        st.info(f"**Latest Proof-of-Research:** `{por_proof}` successfully verified and sealed to block `{block_hash}`.")
        st.caption(f"**Unalterable Criteria State Hash:** `{formulas_hash}` (Guarantees grading mathematical constants cannot be tampered with).")
        
        explore_col1, explore_col2 = st.columns([3, 1])
        with explore_col1: search_query = st.text_input("Enter Document Evaluation Hash or Block Hash to verify ledger record...")
        with explore_col2: st.write(""); st.write(""); search_btn = st.button("Verify Record")
            
        if search_btn and search_query:
            try:
                cursor.execute("SELECT * FROM blockchain_por_weights WHERE block_hash=? OR eval_hash=?", (search_query, search_query))
                record = cursor.fetchone()
                if record:
                    st.success("Valid Block Found on Ledger!")
                    st.json({"Block Height": record[0], "Timestamp": record[9], "Model Used": record[14], "Validator Node": record[11], "Block Hash": record[12], "Evaluation Hash": record[13], "PoR Signature": record[15], "Formulas Hash": record[16], "Weights": dict(zip([f"w{i+1}" for i in range(8)], record[1:9]))})
                else: st.error("No block matching that signature was found on the ledger.")
            except:
                st.error("Error reading database schema. Try refreshing the app.")

with tab5:
    st.subheader("π-Brain: Meta-Learning on the PoR Blockchain")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights ORDER BY block_height ASC")
    historical_rows = cursor.fetchall()
    
    lookback_window = 5
    if len(historical_rows) < lookback_window + 2:
        st.warning(f"Not enough blockchain data to train the meta-model. You need at least {lookback_window + 2} blocks.")
    else:
        current_block_count = len(historical_rows)
        if 'last_trained_blocks' not in st.session_state or st.session_state.last_trained_blocks != current_block_count:
            weight_data = np.array(historical_rows, dtype=np.float32)
            dataset = PiBlockchainDataset(weight_data, lookback_window)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
            
            model, loss_function, optimizer = PiBrainLSTM(), nn.MSELoss(), optim.Adam(PiBrainLSTM().parameters(), lr=0.001)
            progress_bar, status_text = st.progress(0), st.empty()
            epochs = 200
            
            model.train()
            for epoch in range(epochs):
                total_loss = 0
                for seq, target in dataloader:
                    optimizer.zero_grad()
                    loss = loss_function(model(seq), target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if epoch % 10 == 0 or epoch == epochs - 1:
                    status_text.text(f"Training Epoch {epoch}/{epochs} | MSE Loss: {total_loss / len(dataloader):.6f}")
                    progress_bar.progress((epoch + 1) / epochs)
            
            model.eval()
            with torch.no_grad():
                st.session_state.predicted_next_weights = model(torch.tensor(weight_data[-lookback_window:], dtype=torch.float32).unsqueeze(0)).squeeze().numpy()
                st.session_state.current_weights = weight_data[-1]
                st.session_state.last_trained_blocks = current_block_count
        else:
            st.info("Meta-model is cached and up-to-date with the latest blockchain ledger.")

        df_compare = pd.DataFrame({"Current Active Weights": st.session_state.current_weights, "Predicted Next Epoch": st.session_state.predicted_next_weights}, index=["C1: Originality", "C2: Methodological Rigor", "C3: Interdisciplinary", "C4: Societal Impact", "C5: Open Science", "C6: Literature Integration", "C7: Empirical Density", "C8: Future Actionability"])
        st.bar_chart(df_compare, height=400)
        st.markdown(f"**Mathematical Constraint Check:** Predicted Sum = `{sum(st.session_state.predicted_next_weights):.6f}` / `8.0`")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 0.8em;'>Framework Author: Ali Vafadar Yengejeh | Università degli Studi di Milano-Bicocca</div>", unsafe_allow_html=True)
