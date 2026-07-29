import os
import json
import time
import hashlib
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

from config import ORCID_CLIENT_ID, ORCID_CLIENT_SECRET, ORCID_REDIRECT_URI
from database import get_db_connection
from integrations import clean_author_name
from brain import process_single_pdf, generate_rebuttal_strategy, evaluate_scilem_analysis_report
from ledger import safe_get_sepolia_url
from web3 import Web3
from eth_account.messages import encode_defunct

w3 = Web3()

st.set_page_config(page_title="Pi-Index Brain", layout="wide")

custom_ui_code = """
<style>
h1, h2, h3, h4 { color: #0f172a !important; font-family: -apple-system, sans-serif !important; font-weight: 600 !important; }
button[kind="primary"] { background-color: #000080 !important; border-color: #000080 !important; color: #ffffff !important; }
</style>
"""
components.html(custom_ui_code, height=0, width=0)

# Initialize Session States
if "web3_wallet" not in st.session_state: st.session_state.web3_wallet = None
if "orcid_profile" not in st.session_state: st.session_state.orcid_profile = None
if "researcher_name" not in st.session_state: st.session_state.researcher_name = "Anonymous Researcher"
if "scilem_messages" not in st.session_state:
    st.session_state.scilem_messages = [{"role": "assistant", "content": "**Welcome! I am Scilem.** Ask any research question."}]

st.sidebar.title("Identity & Sync")
if st.session_state.web3_wallet:
    st.sidebar.success(f"Web3 Linked: {st.session_state.web3_wallet[:6]}...{st.session_state.web3_wallet[-4:]}")
else:
    st.sidebar.warning("Web3 Wallet Not Connected. Staking disabled.")
    # You can add the SIWE button logic here from app.py

with st.sidebar.expander("🧠 Scilem Assistant", expanded=True):
    for message in st.session_state.scilem_messages:
        st.chat_message(message["role"]).markdown(message["content"])
    
    floating_prompt = st.text_input("Ask Scilem...", key="scilem_input")
    if st.button("Send") and floating_prompt:
        st.session_state.scilem_messages.append({"role": "user", "content": floating_prompt})
        scilem_neural_reply = evaluate_scilem_analysis_report(floating_prompt)
        st.session_state.scilem_messages.append({"role": "assistant", "content": scilem_neural_reply})
        st.rerun()

st.title("🧠 Pidyne Brain & Adversarial LLM Jury")

conn = get_db_connection()
cur = conn.cursor()
cur.execute("SELECT id, file_name, source_type, source_val, timestamp FROM ingestion_queue WHERE status='pending'")
pending_papers = cur.fetchall()

if not pending_papers:
    st.info("No pending manuscripts in queue. Submit papers via the Intake Engine.")
else:
    st.subheader(f"Pending Queue ({len(pending_papers)} Manuscripts)")
    for pid, fname, src_type, src_val, ts in pending_papers:
        with st.container(border=True):
            cols = st.columns([4, 1])
            cols[0].markdown(f"**{fname}** (Source: {src_type} | Queued: {ts})")
            if cols[1].button("Run Assessment", key=f"run_{pid}", type="primary"):
                cur.execute("SELECT raw_bytes FROM ingestion_queue WHERE id=?", (pid,))
                raw_bytes = cur.fetchone()[0]
                
                with st.status(f"Evaluating {fname} via Multi-LLM Consensus & Scilem...", expanded=True) as status_box:
                    current_user = st.session_state.orcid_profile if st.session_state.orcid_profile else (st.session_state.web3_wallet if st.session_state.web3_wallet else "Anonymous")
                    valid_book = st.session_state.web3_wallet if st.session_state.web3_wallet else "0x0000000000000000000000000000000000000000"
                    
                    res = process_single_pdf(
                        raw_bytes, fname, "", current_user, valid_book, email="None", provided_doi=src_val if src_type == 'doi' else "None"
                    )
                    
                    if res:
                        cur.execute("UPDATE ingestion_queue SET status='completed' WHERE id=?", (pid,))
                        conn.commit()
                        status_box.update(label="Evaluation Complete!", state="complete")
                        st.success(f"Score: {res[2]:.2f} | piQ Minted: {res[10]}")
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        status_box.update(label="Evaluation Failed.", state="error")
conn.close()

# Include criterion dialog logic to ensure \text{vapri} and vapri render correctly
def get_criteria_info(weights):
    tw1, tw2, tw3, tw4, tw5, tw6, tw7, tw8 = weights
    return [
        ("C1", "Originality", "c1: originality", tw1, "1", "Semantic distance from literature corpus penalized by generative AI laundering heuristics.", r"$$ C_1 = \varpi_1 \cdot \mathcal{D}_{semantic}(P_{target}, P_{corpus}) \times (1 - \lambda_{laundering}) + \text{vapri} $$"),
    ]