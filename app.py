import os
import re
import json
import time
import hashlib
import tempfile
import shutil
import colorsys
import logging
import traceback
import urllib.parse
import requests
from datetime import datetime
from collections import deque

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pyvis.network import Network
import altair as alt

import streamlit as st
import streamlit.components.v1 as components
from web3 import Web3
from eth_account.messages import encode_defunct

from config import (
    BASE_DIR, EPOCH_BLOCK_SIZE, PIQ_CONTRACT_ADDRESS, REGISTRY_CONTRACT_ADDRESS, 
    HOT_TOPICS, ORCID_CLIENT_ID, ORCID_CLIENT_SECRET, ORCID_REDIRECT_URI
)
from database import get_db_connection
from ledger import restore_state_from_web3, generate_blockchain_pi, get_sepolia_explorer_url
from integrations import (
    clean_author_name, is_likely_institution, fetch_doi_metadata, 
    fetch_semantic_scholar_pdf, download_pdf_from_url, search_openalex_topics,
    fetch_core_text_by_doi, create_virtual_pdf_from_text
)
from brain import ( # 
    process_single_pdf, generate_rebuttal_strategy, PidyneLSTM, # 
    PidyneBlockchainDataset, generate_scilem_fallback_report, reset_scilem, # 
    evaluate_scilem_analysis_report # 
) # 

w3 = Web3()
OWNER_ID = "0x1Af8D9A120b02D0983590587364F8705e6942356"

st.set_page_config(page_title="Pi-Index Assessment Engine", layout="wide")

# --- INITIALIZATION & SESSION STATE ---
if "app_logs" not in st.session_state: st.session_state.app_logs = deque(maxlen=50)
def add_log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.app_logs.appendleft(f"[{timestamp}] {msg}")
    logging.info(f"[{timestamp}] {msg}")

def safe_get_sepolia_url(tx):
    if not tx or not isinstance(tx, str) or not tx.startswith("0x") or len(tx) != 66: return None
    try: return get_sepolia_explorer_url(tx, "tx")
    except Exception: return None

def safe_float(val, default=0.0):
    if val is None: return default
    if isinstance(val, (int, float)): return float(val)
    try: return float(val)
    except ValueError:
        try:
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
            return float(nums[0]) if nums else default
        except Exception: return default

def get_author_piq_dict():
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT author_name, piq_minted, eth_book FROM papers_assessment")
        data = cursor.fetchall()
    finally:
        conn.close()
    author_piq, author_book = {}, {}
    for authors_str, piq, eth_book in data:
        clean_authors = clean_author_name(authors_str)
        if not clean_authors or clean_authors.lower() in ["unidentified", "unknown", "research scholar"] or is_likely_institution(clean_authors): continue
        alist = [a.strip() for a in clean_authors.split(",") if a.strip()]
        if not alist: continue
        share = safe_float(piq, 0.0) / len(alist)
        for a in alist:
            author_piq[a] = author_piq.get(a, 0.0) + share
            author_book[a] = eth_book if eth_book and w3.is_address(eth_book) else "Unbound / Escrow"
    return author_piq, author_book

def preprocess_pdf_layout(pdf_bytes, fname): return pdf_bytes
def rbot(topic_key): return f"<span class='scilem-trigger' data-query='{topic_key}' title='Ask Scilem' style='cursor: pointer !important; opacity:0.8;'>[?]</span>" # 

if "web3_wallet" not in st.session_state: st.session_state.web3_wallet = None
if "orcid_profile" not in st.session_state: st.session_state.orcid_profile = None
if "researcher_name" not in st.session_state: st.session_state.researcher_name = "Anonymous Researcher"
if "free_evals_used" not in st.session_state: st.session_state["free_evals_used"] = 0
if "assessment_update_token" not in st.session_state: st.session_state["assessment_update_token"] = time.time()
if "reset_token" not in st.session_state: st.session_state["reset_token"] = 0
if "evaluated_papers_buffer" not in st.session_state: st.session_state["evaluated_papers_buffer"] = []
if "download_errors" not in st.session_state: st.session_state["download_errors"] = []
if "is_running" not in st.session_state: st.session_state["is_running"] = False
if "cancel_requested" not in st.session_state: st.session_state["cancel_requested"] = False
if "session_temp_dir" not in st.session_state:
    st.session_state["session_temp_dir"] = tempfile.mkdtemp()
    add_log(f"Temporary volume allocated: {st.session_state['session_temp_dir']}")
if "scilem_messages" not in st.session_state: # 
    st.session_state.scilem_messages = [{"role": "assistant", "content": "**Welcome! I am Scilem.** Ask any research question or check criteria ratings."}] # 

if "state_restored" not in st.session_state:
    restore_state_from_web3()
    st.session_state["state_restored"] = True
    add_log("Synchronized state with Sepolia Ethereum Ledger.")

has_web3 = bool(st.session_state.web3_wallet and w3.is_address(st.session_state.web3_wallet))
has_orcid = bool(st.session_state.orcid_profile)
current_user = st.session_state.orcid_profile if has_orcid else (st.session_state.web3_wallet if has_web3 else "Anonymous")
valid_book_address = st.session_state.web3_wallet if has_web3 else "0x0000000000000000000000000000000000000000"

# --- GLOBAL DIALOGS ---
@st.dialog("The Pi-Index Framework: Next-Gen Architecture & CoARA Compliance Workflow", width="large")
def framework_workflow_dialog():
    st.markdown("Pi-Index filters noise and yields quantitative results strictly aligned with **Responsible Research Assessment (RRA)** and **CoARA** (Coalition for Advancing Research Assessment) guidelines.")

@st.dialog("Criterion Details & Adversarial Logic Engine", width="medium")
def criterion_details_dialog(c_id, title, q_key, weight_val, sym, desc, formula):
    st.markdown(f"### {c_id}: {title}")
    st.markdown(f"**Current Epoch Weight (\\varpi_{{{sym}}}):** `{weight_val:.6f}`")
    st.markdown(f"{desc} {rbot(q_key)}", unsafe_allow_html=True)
    st.markdown(formula)
    st.markdown("---")
    st.markdown(r"**Adversarial Logic Gap ($\Delta_{Logic}$):** Evaluates reasoning structure and penalizes claims unsupported by evidence.")
    st.markdown(r"$$ L_i = \left( (\mathcal{P}_{valid} \cdot \mathcal{E}_{strength}) \cdot \exp\left(-\left(2 \cdot \max(0, \mathcal{C}_{reach} - \mathcal{E}_{strength}) + 1.5 \cdot \lambda_{jumps}\right)\right) \right) \times \frac{1}{1 + e^{-\Delta Premise}} + \lambda \cdot \text{vapri} $$") # 

@st.dialog("Detailed Research Integrity Dossier", width="large")
def more_details_dialog(item):
    st.subheader(f"{item.get('title', 'Unknown')} by {clean_author_name(item.get('author_name', 'Unknown'))}")
    st.write(f"**Evaluation Hash:** `{item.get('eval_hash', '0x0')}`\n**piQ Minted:** `{safe_float(item.get('piq'), 0.0)}`") # 
    if item.get("evidence_report_text"): st.markdown("### Evidence Report\n" + item["evidence_report_text"])

@st.dialog("AI Peer Review Defense Strategy", width="medium")
def defense_strategy_dialog(scores_dict):
    with st.spinner("Synthesizing adversarial defense strategy..."):
        st.markdown(generate_rebuttal_strategy(scores_dict))

# --- SIDEBAR UI ---
st.sidebar.title("System Access & Sync")
if not has_web3: st.sidebar.button("Connect MetaMask", use_container_width=True)
else: st.sidebar.success(f"Web3 Linked: `{st.session_state.web3_wallet[:6]}...{st.session_state.web3_wallet[-4:]}`")

if not has_orcid: st.sidebar.button("Link ORCID Account", use_container_width=True)
else: st.sidebar.success(f"ORCID Linked: `{st.session_state.orcid_profile}`")

st.sidebar.markdown(f"**Researcher:** {st.session_state.researcher_name}")
if st.sidebar.button("Unlink / Reset Session", use_container_width=True):
    st.session_state.web3_wallet, st.session_state.orcid_profile, st.session_state.researcher_name = None, None, "Anonymous Researcher"
    st.rerun()

st.sidebar.markdown("---")
with st.sidebar.expander("Live System Monitor", expanded=True):
    st.code("\n".join(st.session_state.app_logs) if st.session_state.app_logs else "No active logs...", language="bash")

with st.sidebar.expander("🧠 Scilem Assistant", expanded=False): # 
    for message in st.session_state.scilem_messages: # 
        with st.chat_message(message["role"], avatar="🧠" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])
    if prompt := st.chat_input("Ask Scilem..."): # 
        st.session_state.scilem_messages.append({"role": "user", "content": prompt}) # 
        st.session_state.scilem_messages.append({"role": "assistant", "content": evaluate_scilem_analysis_report(prompt)}) # 
        st.rerun()
    if has_web3 and st.session_state.web3_wallet.lower() == OWNER_ID.lower() and st.button("Reset Scilem (Owner)", use_container_width=True): # 
        st.success(reset_scilem()) # 

# --- PAGE FUNCTIONS ---
def page_assessment():
    conn_cnt = get_db_connection()
    total_analyzed_count = conn_cnt.execute("SELECT COUNT(*) FROM papers_assessment").fetchone()[0]
    conn_cnt.close()

    c1, c2 = st.columns([4, 2], vertical_alignment="center")
    c1.markdown("<h1 style='margin-bottom:0;'>📄 Manuscript Intake & Processing</h1>", unsafe_allow_html=True)
    c2.markdown(f"<div style='float: right; background-color: #0f172a; color: white; padding: 6px 16px; border-radius: 20px; font-weight: 600;'>Total Analyzed: <span style='color: #60a5fa;'>{total_analyzed_count}</span></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### Assess a Manuscript")
        free_evals_used = st.session_state.get("free_evals_used", 0)
        stake_amount = True
        
        if free_evals_used > 0:
            if not has_web3:
                st.warning("🔒 **Free Trial Completed:** Please connect your **Web3 Ethereum Wallet** in the sidebar to continue.")
                stake_amount = False
            else:
                stake_amount = st.checkbox("Stake 0.1 piQ to Process", value=True) # 

        t_local, t_doi = st.tabs(["📄 Local Upload", "🔗 DOI Lookup"])
        selected_uploaded_files = []
        
        with t_local:
            uploaded_files = st.file_uploader("Upload Local PDF(s)", type=["pdf"], accept_multiple_files=True)
            if uploaded_files:
                for f in uploaded_files:
                    if st.checkbox(f"Local File: {f.name}", value=True): selected_uploaded_files.append(f)

        with t_doi:
            doi_input = st.text_input("Enter a DOI", placeholder="10.1000/xyz123")
            include_doi = st.checkbox("Include this DOI in pipeline", disabled=not doi_input.strip())

        if st.session_state["is_running"]:
            c1, c2 = st.columns([4, 1], gap="medium")
            c1.button("Working...", type="primary", use_container_width=True, disabled=True)
            if c2.button("Stop", type="secondary", use_container_width=True):
                st.session_state["is_running"], st.session_state["cancel_requested"] = False, True
                st.rerun()

            with st.status("Initializing Assessment Pipeline...", expanded=True) as status_box:
                try:
                    if st.session_state.get("snap_include_doi") and st.session_state.get("snap_doi") and not st.session_state["cancel_requested"]:
                        doi_snap = st.session_state["snap_doi"]
                        status_box.update(label=f"Resolving DOI: {doi_snap}...")
                        metadata = fetch_doi_metadata(doi_snap)
                        pdf_bytes = download_pdf_from_url(metadata["pdf_url"]) if metadata and metadata.get("pdf_url") else None
                        if not pdf_bytes: pdf_bytes = download_pdf_from_url(fetch_semantic_scholar_pdf(doi_snap))
                        if not pdf_bytes: pdf_bytes = create_virtual_pdf_from_text(fetch_core_text_by_doi(doi_snap))

                        if pdf_bytes:
                            status_box.update(label="Assessing document...")
                            res = process_single_pdf(preprocess_pdf_layout(pdf_bytes, f"DOI_{doi_snap}.pdf"), f"DOI_{doi_snap}.pdf", "", current_user, valid_book_address, provided_doi=doi_snap)
                            if res:
                                item = {"title": res[0], "author_name": clean_author_name(res[1]), "score": res[2], "logic_integrity": res[3], "scores_dict": res[8], "eval_hash": res[9], "piq": res[10], "tx_hash": res[11], "zk_proof": res[12], "h_idx": res[14], "i10_idx": res[15], "repro_score": res[16], "filename": f"DOI_{doi_snap}.pdf", "warnings": res[18], "consensus_raw": res[19], "evidence_report_text": res[20], "scilem_rating": res[21]} # 
                                st.session_state["evaluated_papers_buffer"].insert(0, item)
                                st.session_state["free_evals_used"] += 1
                        else:
                            st.session_state["download_errors"].append({"title": doi_snap, "doi": doi_snap, "url": f"https://doi.org/{doi_snap}"})

                    snap_files = st.session_state.get("snap_files", [])
                    for i, (fname, fpath) in enumerate(snap_files):
                        if st.session_state["cancel_requested"]: break
                        status_box.update(label=f"Analyzing {fname}...")
                        with open(fpath, "rb") as in_f: raw_bytes = in_f.read()
                        res = process_single_pdf(preprocess_pdf_layout(raw_bytes, fname), fname, "", current_user, valid_book_address)
                        if res:
                            item = {"title": res[0], "author_name": clean_author_name(res[1]), "score": res[2], "logic_integrity": res[3], "scores_dict": res[8], "eval_hash": res[9], "piq": res[10], "tx_hash": res[11], "zk_proof": res[12], "h_idx": res[14], "i10_idx": res[15], "repro_score": res[16], "filename": fname, "warnings": res[18], "consensus_raw": res[19], "evidence_report_text": res[20], "scilem_rating": res[21]} # 
                            st.session_state["evaluated_papers_buffer"].insert(0, item)
                            st.session_state["free_evals_used"] += 1

                    status_box.update(label="Complete.", state="complete" if not st.session_state["cancel_requested"] else "error")
                    time.sleep(1)
                finally:
                    st.session_state["is_running"], st.session_state["cancel_requested"] = False, False
                    st.session_state["reset_token"] += 1
                    st.rerun()

        else:
            if st.button("Run Assessment Pipeline", type="primary", use_container_width=True):
                if free_evals_used >= 1 and (not has_web3 or not stake_amount):
                    st.error("Free trial limit reached. Connect Web3 and stake 0.1 piQ.") # 
                elif not selected_uploaded_files and not (include_doi and doi_input.strip()):
                    st.warning("Please tick at least one source to assess.")
                else:
                    saved_files = []
                    for f in selected_uploaded_files:
                        f_path = os.path.join(st.session_state["session_temp_dir"], f.name)
                        with open(f_path, "wb") as out_f: out_f.write(f.getvalue())
                        saved_files.append((f.name, f_path))
                    st.session_state["snap_files"], st.session_state["snap_doi"], st.session_state["snap_include_doi"] = saved_files, doi_input, include_doi
                    st.session_state["is_running"] = True
                    st.rerun()

    if st.session_state["evaluated_papers_buffer"] or st.session_state.get("download_errors"):
        st.markdown("### Assessment Results")
        for err_idx, err in enumerate(st.session_state.get("download_errors", [])):
            st.warning(f"**Failed DOI:** `{err['doi']}` (Publisher restricts direct access)")

        for item_idx, item in enumerate(st.session_state["evaluated_papers_buffer"]):
            with st.container(border=True):
                c_info, c_actions = st.columns([6, 4])
                c_info.markdown(f"**{item['title']}** — *{item['author_name']}*\n\n**Score: {item['score']:.2f} | piQ: {item['piq']}**") # 
                ac1, ac2, ac3 = c_actions.columns([3, 3, 1])
                if ac1.button("More Details", key=f"det_{item['eval_hash']}_{item_idx}", use_container_width=True): more_details_dialog(item)
                if ac2.button("Suggest Defense", key=f"strat_{item['eval_hash']}_{item_idx}", use_container_width=True): defense_strategy_dialog(item['scores_dict'])
                if ac3.button("❌", key=f"del_{item['eval_hash']}_{item_idx}"):
                    st.session_state["evaluated_papers_buffer"].pop(item_idx)
                    st.rerun()
                    
    st.markdown("---")
    if st.button("The Pi-Index Framework Workflow", use_container_width=True): framework_workflow_dialog()

def page_analytics():
    st.markdown("### Pidyne Brain & Epoch Forecasting")
    lookback = st.selectbox("Lookback Window", ["1 Epoch", "3 Epochs", "5 Epochs"], index=1)
    actual_lookback = int(lookback.split()[0])
    
    conn = get_db_connection()
    historical_rows = conn.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights ORDER BY block_height ASC").fetchall()
    
    if len(historical_rows) < 2:
        st.warning("Not enough blockchain data to train meta-model. Need at least 2 blocks.")
    else:
        weight_data = np.array([[safe_float(v, 1.0) for v in r] for r in historical_rows], dtype=np.float32)
        
        df_hist = pd.DataFrame(historical_rows[-(actual_lookback + 1):], columns=["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"])
        df_hist.index.name = "Block"
        df_melted = df_hist.reset_index().melt('Block', var_name='Criterion', value_name='Weight')
        st.altair_chart(alt.Chart(df_melted).mark_line(point=True).encode(x='Block:O', y='Weight:Q', color='Criterion:N'), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### Pi Quotient (piQ) Leaderboard") # 
        data = pd.read_sql_query("SELECT author_name, piq_minted FROM papers_assessment", conn)
        author_piq = {}
        for _, row in data.iterrows():
            ca = clean_author_name(row["author_name"])
            if ca and ca.lower() not in ["unidentified", "unknown"] and not is_likely_institution(ca):
                for a in [x.strip() for x in ca.split(",")]: author_piq[a] = author_piq.get(a, 0.0) + float(row["piq_minted"] or 0) # 
        if author_piq:
            st.dataframe(pd.DataFrame(sorted(author_piq.items(), key=lambda x: x[1], reverse=True)[:20], columns=["Author", "piQ"]), hide_index=True) # 

    with col2:
        st.markdown("### piX Top Papers")
        df_pix = pd.read_sql_query("SELECT title, author_name, final_score FROM papers_assessment ORDER BY final_score DESC LIMIT 20", conn)
        st.dataframe(df_pix, hide_index=True)
    conn.close()

def page_explorer():
    st.markdown("### Proof-of-Research Blockchain Explorer")
    conn = get_db_connection()
    search_q = st.text_input("Search Ledger by Eval Hash, Block Hash, Title, or Author:")

    if search_q.strip():
        q_term = f"%{search_q.strip()}%"
        rows = conn.execute("""
            SELECT p.title, p.author_name, p.filename, p.final_score, p.logic_score, 
                   p.c1, p.c2, p.c3, p.c4, p.c5, p.c6, p.c7, p.c8, 
                   p.piq_minted, p.tx_hash, p.zk_proof, p.mdar_adherence_score, 
                   p.rrid_valid_count, p.reproducibility_score, p.eval_hash,
                   p.consensus_data, p.evidence_report, p.scilem_score
            FROM papers_assessment p
            LEFT JOIN blockchain_por_weights b ON p.eval_hash = b.eval_hash
            WHERE b.block_hash LIKE ? OR p.eval_hash LIKE ? OR p.title LIKE ? OR p.author_name LIKE ?
            LIMIT 10
        """, (q_term, q_term, q_term, q_term)).fetchall()
        
        if rows:
            for r in rows:
                with st.expander(f"{r[0]} - {r[1]} (Score: {r[3]:.2f})"):
                    st.write(f"**Hash:** `{r[19]}`")
                    if st.button("Full Dossier", key=r[19]):
                        more_details_dialog({
                            "title": r[0], "author_name": r[1], "score": r[3], "logic_integrity": r[4], 
                            "scores_dict": {"C1": r[5], "C2": r[6], "C3": r[7], "C4": r[8], "C5": r[9], "C6": r[10], "C7": r[11], "C8": r[12]},
                            "eval_hash": r[19], "piq": r[13], "tx_hash": r[14], "zk_proof": r[15], # 
                            "h_idx": r[16], "i10_idx": r[17], "repro_score": r[18], "filename": r[2], 
                            "consensus_raw": json.loads(r[20]) if r[20] else {}, "evidence_report_text": r[21], "scilem_rating": r[22] # 
                        })
        else:
            st.error("No matching ledger records found.")
    else:
        st.markdown("### Latest Assessed Papers")
        df = pd.read_sql_query("SELECT title as Title, author_name as Author, final_score as Score, eval_hash as Hash FROM papers_assessment ORDER BY timestamp DESC LIMIT 20", conn)
        st.dataframe(df, use_container_width=True, hide_index=True)
    conn.close()

# --- ROUTING/NAVIGATION ---
pg = st.navigation([
    st.Page(page_assessment, title="Assess Manuscript", icon="📄"),
    st.Page(page_analytics, title="Analytics & Map", icon="🌐"),
    st.Page(page_explorer, title="Blockchain Explorer", icon="⛓️"),
])
pg.run()
