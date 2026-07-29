import os
import re
import json
import time
import hashlib
import tempfile
import urllib.parse
import requests
from datetime import datetime
from collections import deque

import streamlit as st
import streamlit.components.v1 as components
from web3 import Web3
from eth_account.messages import encode_defunct

from config import (
    ORCID_CLIENT_ID, ORCID_CLIENT_SECRET, ORCID_REDIRECT_URI
)
from database import get_db_connection
from ledger import restore_state_from_web3, get_sepolia_explorer_url
from integrations import clean_author_name
from brain import generate_rebuttal_strategy, reset_scilem, evaluate_scilem_analysis_report

w3 = Web3()
OWNER_ID = "0x1Af8D9A120b02D0983590587364F8705e6942356"

def add_log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {msg}"
    st.session_state.app_logs.appendleft(log_entry)

def safe_get_sepolia_url(tx):
    if not tx or not isinstance(tx, str) or not tx.startswith("0x") or len(tx) != 66:
        return None
    try:
        return get_sepolia_explorer_url(tx, "tx")
    except Exception:
        return None

def safe_float(val, default=0.0):
    if val is None: return default
    if isinstance(val, (int, float)): return float(val)
    try:
        return float(val)
    except ValueError:
        try:
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
            return float(nums[0]) if nums else default
        except Exception:
            return default

def rbot(topic_key):
    return f"<span class='scilem-trigger' data-query='{topic_key}' title='Ask Scilem' style='cursor: pointer !important; opacity:0.8;'>[?]</span>"

@st.dialog("AI Peer Review Defense Strategy", width="medium")
def defense_strategy_dialog(scores_dict):
    with st.spinner("Synthesizing adversarial defense strategy..."):
        rebuttal = generate_rebuttal_strategy(scores_dict)
    st.markdown(rebuttal)

@st.dialog("Detailed Research Integrity Dossier", width="large")
def more_details_dialog(item):
    title = item.get("title", "Unknown Title")
    author_name = clean_author_name(item.get("author_name", "Unknown"))
    score = safe_float(item.get("score"), 0.0)
    logic_integrity = safe_float(item.get("logic_integrity"), 75.0)
    scores_dict = item.get("scores_dict", {})
    used_weights = item.get("used_weights", [1.0]*8)
    eval_hash = item.get("eval_hash", "0x0")
    piq = safe_float(item.get("piq"), 0.0)
    tx_hash = item.get("tx_hash", "None")
    zk_proof = item.get("zk_proof", "None")
    mdar_score = safe_float(item.get("h_idx"), 0.0)
    rrid_count = int(safe_float(item.get("i10_idx"), 0))
    repro_score = safe_float(item.get("repro_score"), 0.0)
    filename = item.get("filename", "N/A")
    warnings = item.get("warnings", [])
    evidence_report_text = item.get("evidence_report_text", "")
    author_book = "0x" + hashlib.sha256(author_name.encode()).hexdigest()[:40]

    st.subheader(f"{title} by {author_name}")
    if warnings:
        st.warning(f"⚠️ **Manuscript Flagged with {len(warnings)} Warning Check(s):**")
        for w in warnings: st.markdown(f"- {w}")

    st.markdown("### Overview & Ledger")
    st.write(f"**File Name:** `{filename}`")
    st.write(f"**Evaluation Hash (Paper Address):** `{eval_hash}`")
    st.write(f"**Unique Book Address:** `{author_book}`")
    st.write(f"**piQ Minted:** `{piq}`")
    st.markdown(f"**zk-SNARK Proof:** `{zk_proof}`", unsafe_allow_html=True)
    
    tx_url = safe_get_sepolia_url(tx_hash)
    tx_disp_val = tx_hash if tx_hash and str(tx_hash).strip() not in ["None", ""] else "Not Connected / No Book / Missing PK"
    if tx_url: st.markdown(f"**Tx Hash:** [`{tx_disp_val}`]({tx_url})")
    else: st.write(f"**Tx Hash:** `{tx_disp_val}`")

    st.markdown(f"**Executable Reproducibility Score:** `{repro_score * 100:.1f}%`", unsafe_allow_html=True)
    st.markdown(f"**SciScore MDAR Adherence:** `{mdar_score * 100:.1f}%` | **Valid RRIDs:** `{rrid_count}`", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Synthesized Evidence Report")
    if evidence_report_text:
        st.markdown(evidence_report_text)
        st.download_button(
            label="Download Final Evidence Report (.md)",
            data=evidence_report_text,
            file_name=f"Evidence_Report_{eval_hash[:10]}.md",
            mime="text/markdown",
            use_container_width=True,
            key=f"dl_report_modal_{eval_hash}_{time.time()}"
        )
    else:
        st.info("No synthesized evidence report generated for this manuscript.")

def setup_global_state_and_sidebar():
    if "app_logs" not in st.session_state: st.session_state.app_logs = deque(maxlen=50)
    if "web3_wallet" not in st.session_state: st.session_state.web3_wallet = None
    if "orcid_profile" not in st.session_state: st.session_state.orcid_profile = None
    if "researcher_name" not in st.session_state: st.session_state.researcher_name = "Anonymous Researcher"
    if "free_evals_used" not in st.session_state: st.session_state["free_evals_used"] = 0
    if "state_restored" not in st.session_state:
        restore_state_from_web3()
        st.session_state["state_restored"] = True
        add_log("Synchronized state with Sepolia Ethereum Ledger.")
    if "assessment_update_token" not in st.session_state: st.session_state["assessment_update_token"] = time.time()
    if "reset_token" not in st.session_state: st.session_state["reset_token"] = 0
    if "evaluated_papers_buffer" not in st.session_state: st.session_state["evaluated_papers_buffer"] = []
    if "download_errors" not in st.session_state: st.session_state["download_errors"] = []
    if "is_running" not in st.session_state: st.session_state["is_running"] = False
    if "cancel_requested" not in st.session_state: st.session_state["cancel_requested"] = False
    if "session_temp_dir" not in st.session_state: 
        st.session_state["session_temp_dir"] = tempfile.mkdtemp()
        add_log(f"Temporary volume allocated: {st.session_state['session_temp_dir']}")
    if "scilem_messages" not in st.session_state:
        st.session_state.scilem_messages = [{"role": "assistant", "content": "**Welcome! I am Scilem.** Ask any research question or check criteria ratings."}]

    if "restore_orcid" in st.query_params:
        st.session_state.orcid_profile = st.query_params.get("restore_orcid")
        if st.query_params.get("restore_orcid_name"):
            st.session_state.researcher_name = st.query_params.get("restore_orcid_name")

    if "siwe_address" in st.query_params:
        raw_address = st.query_params.get("siwe_address")
        if raw_address and w3.is_address(raw_address):
            st.session_state.web3_wallet = w3.to_checksum_address(raw_address)
            st.toast(f"MetaMask Linked: {st.session_state.web3_wallet[:6]}...{st.session_state.web3_wallet[-4:]}")
        st.query_params.clear()
        st.rerun()

    if "code" in st.query_params:
        auth_code = st.query_params.get("code")
        returned_state = st.query_params.get("state")
        if returned_state and returned_state != "none" and w3.is_address(returned_state):
            st.session_state.web3_wallet = w3.to_checksum_address(returned_state)
        try:
            res = requests.post("https://orcid.org/oauth/token", data={
                "client_id": ORCID_CLIENT_ID, "client_secret": ORCID_CLIENT_SECRET,
                "grant_type": "authorization_code", "code": auth_code, "redirect_uri": ORCID_REDIRECT_URI
            }, headers={"Accept": "application/json"})
            if res.status_code == 200:
                data = res.json()
                if data.get("orcid"):
                    st.session_state.orcid_profile = data.get("orcid")
                    st.session_state.researcher_name = data.get("name") or f"ORCID Scholar ({data.get('orcid')[-4:]})"
                    st.toast(f"ORCID Linked: {st.session_state.researcher_name}")
        except Exception:
            pass
        st.query_params.clear()
        st.rerun()

    custom_ui_code = """
    <style>
    h1, h2, h3, h4, h5, h6 { color: #0f172a !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; font-weight: 600 !important; }
    [data-testid="stSidebar"] { background-color: #f8fafc !important; border-right: 1px solid #e2e8f0 !important; }
    button[kind="primary"] { background-color: #000080 !important; color: #ffffff !important; }
    button[kind="secondary"] { background-color: #dc2626 !important; color: #ffffff !important; }
    </style>
    """
    components.html(custom_ui_code, height=0, width=0)

    client_ip = "127.0.0.1"
    try:
        headers = st.context.headers
        client_ip = headers.get("X-Forwarded-For") or headers.get("X-Real-Ip") or "127.0.0.1"
        if "," in client_ip: client_ip = client_ip.split(",")[0].strip()
    except Exception: pass

    conn_ip = get_db_connection()
    try:
        cur_ip = conn_ip.cursor()
        if not cur_ip.execute("SELECT ip_address FROM auto_ip_tracking WHERE ip_address=?", (client_ip,)).fetchone():
            cur_ip.execute("INSERT INTO auto_ip_tracking (ip_address, first_seen) VALUES (?, ?)", (client_ip, datetime.now().isoformat()))
            conn_ip.commit()
    finally:
        conn_ip.close()

    st.sidebar.title("System Access & Sync")
    has_web3 = bool(st.session_state.web3_wallet and w3.is_address(st.session_state.web3_wallet))
    has_orcid = bool(st.session_state.orcid_profile)
    
    current_orcid_js = st.session_state.orcid_profile if st.session_state.orcid_profile else ""
    current_orcid_name_js = st.session_state.researcher_name if st.session_state.researcher_name != "Anonymous Researcher" else ""
    state_payload = st.session_state.web3_wallet if has_web3 else "none"
    orcid_auth_url = f"https://orcid.org/oauth/authorize?client_id={ORCID_CLIENT_ID}&response_type=code&scope=/authenticate&redirect_uri={ORCID_REDIRECT_URI}&state={state_payload}"

    mm_button_html = f"""
        <button id="connect-mm-btn" type="button" style="width: 100%; background: linear-gradient(135deg, #f6851b, #e2761b); color: white; border: none; padding: 10px 14px; border-radius: 8px; font-weight: 700; font-size: 13px; cursor: pointer;">Connect MetaMask</button>
        <script>
        document.getElementById('connect-mm-btn').addEventListener('click', async () => {{
            let provider = window.ethereum;
            if(provider) {{
                try {{
                    const accounts = await provider.request({{ method: 'eth_requestAccounts' }});
                    const targetUrl = new URL(window.top.location.href.split('?')[0]);
                    targetUrl.searchParams.set("siwe_address", accounts[0]);
                    if ("{current_orcid_js}") targetUrl.searchParams.set("restore_orcid", "{current_orcid_js}");
                    if ("{current_orcid_name_js}") targetUrl.searchParams.set("restore_orcid_name", "{current_orcid_name_js}");
                    window.open(targetUrl.href, '_blank');
                }} catch (err) {{}}
            }}
        }});
        </script>
    """
    
    with st.sidebar:
        if not has_web3:
            components.html(mm_button_html, height=50)
        else:
            st.success(f"Web3 Linked: `{st.session_state.web3_wallet[:6]}...{st.session_state.web3_wallet[-4:]}`")

        if not has_orcid:
            st.markdown(f"""<a href="{orcid_auth_url}" target="_blank" style="width: 100%; background: #A6CE39; color: #ffffff; padding: 10px 14px; border-radius: 8px; font-weight: 700; font-size: 13px; text-align: center; text-decoration: none; display: block;">Link ORCID Account</a>""", unsafe_allow_html=True)
            st.write("")
        else:
            st.success(f"ORCID Linked: `{st.session_state.orcid_profile}`")

        if has_web3 or has_orcid:
            conn_hist = get_db_connection()
            total_user_piq = 0.0
            try:
                cur_h = conn_hist.cursor()
                clauses, params = [], []
                if has_web3: clauses.extend(["eth_book = ?"]); params.append(st.session_state.web3_wallet)
                if has_orcid: clauses.extend(["user_id = ?"]); params.append(st.session_state.orcid_profile)
                if clauses:
                    piq_rows = cur_h.execute(f"SELECT DISTINCT eval_hash, piq_minted FROM papers_assessment WHERE {' OR '.join(clauses)}", tuple(params)).fetchall()
                    total_user_piq = sum(safe_float(r[1], 0.0) for r in piq_rows if r[1])
            finally:
                conn_hist.close()
            st.markdown(f"**Researcher:** {st.session_state.researcher_name}\n\n**TOTAL piQ AWARDED:** `{total_user_piq:.2f} piQ`")

            if st.button("Unlink / Reset Session", use_container_width=True):
                st.session_state.web3_wallet = None
                st.session_state.orcid_profile = None
                st.session_state.researcher_name = "Anonymous Researcher"
                st.rerun()

        st.markdown("---")
        with st.expander("Live System Monitor", expanded=True):
            log_text = "\n".join(st.session_state.app_logs)
            st.code(log_text if log_text else "No active logs...", language="bash")

        with st.expander("🧠 Scilem Assistant", expanded=False):
            fc = st.container(height=220)
            with fc:
                for msg in st.session_state.scilem_messages:
                    st.chat_message(msg["role"], avatar="🧠" if msg["role"]=="assistant" else "👤").markdown(msg["content"])
            with st.form("scilem_form", clear_on_submit=False):
                prompt = st.text_input("Ask Scilem...", label_visibility="collapsed")
                if st.form_submit_button("Send") and prompt.strip():
                    st.session_state.scilem_messages.append({"role": "user", "content": prompt})
                    st.session_state.scilem_messages.append({"role": "assistant", "content": evaluate_scilem_analysis_report(prompt)})
                    st.rerun()
            if has_web3 and st.session_state.web3_wallet and OWNER_ID.lower() == st.session_state.web3_wallet.lower():
                if st.button("Reset Scilem (Owner)", use_container_width=True):
                    reset_scilem()
                    st.session_state.scilem_messages = [{"role": "assistant", "content": "**Scilem reset.**"}]
                    st.rerun()
