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
from brain import (
    process_single_pdf, generate_rebuttal_strategy, PidyneLSTM, 
    PidyneBlockchainDataset, generate_scilem_fallback_report, reset_scilem,
    evaluate_scilem_analysis_report
)

w3 = Web3()
OWNER_ID = "0x1Af8D9A120b02D0983590587364F8705e6942356"

st.set_page_config(page_title="Pi-Index Assessment Engine", layout="wide")

# ==========================================
# 1. INITIALIZATION & SESSION STATE
# ==========================================
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

def rbot(topic_key):
    return f"<span class='scilem-trigger' data-query='{topic_key}' title='Ask Scilem' style='cursor: pointer !important; opacity:0.8;'>[?]</span>"

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
if "scilem_messages" not in st.session_state:
    st.session_state.scilem_messages = [{"role": "assistant", "content": "**Welcome! I am Scilem.** Ask any research question or check criteria ratings."}]

if "state_restored" not in st.session_state:
    restore_state_from_web3()
    st.session_state["state_restored"] = True
    add_log("Synchronized state with Sepolia Ethereum Ledger.")

# ==========================================
# 2. AUTHENTICATION & QUERY PARAMS
# ==========================================
if "restore_orcid" in st.query_params:
    st.session_state.orcid_profile = st.query_params.get("restore_orcid")
    r_name = st.query_params.get("restore_orcid_name")
    if r_name: st.session_state.researcher_name = r_name

if "siwe_address" in st.query_params:
    raw_address = st.query_params.get("siwe_address")
    raw_signature = st.query_params.get("siwe_signature")
    raw_message = st.query_params.get("siwe_message")
    if raw_address and w3.is_address(raw_address):
        clean_wallet = w3.to_checksum_address(raw_address)
        authenticated = False
        if raw_signature and raw_message:
            try:
                decoded_msg = urllib.parse.unquote(raw_message)
                signable_msg = encode_defunct(text=decoded_msg)
                recovered_address = w3.eth.account.recover_message(signable_msg, signature=raw_signature)
                if recovered_address.lower() == clean_wallet.lower():
                    authenticated = True
                    add_log(f"MetaMask Identity Cryptographically Authenticated via SIWE: {clean_wallet}")
            except Exception as e: add_log(f"SIWE signature verification fallback: {str(e)}")
        st.session_state.web3_wallet = clean_wallet
        st.toast(f"MetaMask Linked: {clean_wallet[:6]}...{clean_wallet[-4:]}")
    st.query_params.clear()
    st.rerun()

if "code" in st.query_params:
    auth_code = st.query_params.get("code")
    returned_state = st.query_params.get("state")
    if returned_state and returned_state != "none" and w3.is_address(returned_state):
        st.session_state.web3_wallet = w3.to_checksum_address(returned_state)
    try:
        token_url = "https://orcid.org/oauth/token"
        headers = {"Accept": "application/json"}
        payload = {"client_id": ORCID_CLIENT_ID, "client_secret": ORCID_CLIENT_SECRET, "grant_type": "authorization_code", "code": auth_code, "redirect_uri": ORCID_REDIRECT_URI}
        response = requests.post(token_url, data=payload, headers=headers)
        if response.status_code == 200:
            orcid_data = response.json()
            real_orcid = orcid_data.get("orcid")
            real_name = orcid_data.get("name")
            if real_orcid:
                st.session_state.orcid_profile = real_orcid
                st.session_state.researcher_name = real_name if real_name else f"ORCID Scholar ({real_orcid[-4:]})"
                st.toast(f"ORCID Linked: {st.session_state.researcher_name}")
                add_log(f"ORCID Profile Successfully Authenticated: {real_orcid}")
            else: st.error("Authentication failed: ORCID identifier not returned.")
        else:
            err_desc = response.json().get('error_description', 'Invalid Code')
            st.error(f"ORCID Verification Error: {err_desc}")
            add_log(f"ORCID Auth Error: {err_desc}")
    except Exception as e: st.error(f"Failed to connect to ORCID API: {str(e)}")
    st.query_params.clear()
    st.rerun()

has_web3 = bool(st.session_state.web3_wallet and w3.is_address(st.session_state.web3_wallet))
has_orcid = bool(st.session_state.orcid_profile)
current_user = st.session_state.orcid_profile if has_orcid else (st.session_state.web3_wallet if has_web3 else "Anonymous")
valid_book_address = st.session_state.web3_wallet if has_web3 else "0x0000000000000000000000000000000000000000"

# ==========================================
# 3. GLOBAL UI INJECTIONS & SIDEBAR
# ==========================================
custom_ui_code = """
<style>
h1, h2, h3, h4, h5, h6 { color: #0f172a !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important; font-weight: 600 !important; letter-spacing: -0.02em !important; }
hr { border-color: #e2e8f0 !important; margin: 1.5rem 0 !important; }
[data-testid="stHeaderActionElements"] { display: none !important; }
[data-testid="stSidebar"] { background-color: #f8fafc !important; border-right: 1px solid #e2e8f0 !important; overflow-y: auto !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px !important; border: 1px solid #e2e8f0 !important; background-color: #ffffff !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important; transition: box-shadow 0.2s ease-in-out, transform 0.2s ease-in-out !important; padding: 0.5rem !important; }
button[kind="primary"], [data-testid="baseButton-primary"] { background-color: #000080 !important; border-color: #000080 !important; color: #ffffff !important; }
button[kind="primary"]:hover, [data-testid="baseButton-primary"]:hover { background-color: #00005b !important; border-color: #00005b !important; color: #ffffff !important; }
.stButton>button { border-radius: 8px !important; font-weight: 600 !important; letter-spacing: 0.01em !important; }
[data-testid="stExpander"] { border-radius: 10px !important; border: 1px solid #e2e8f0 !important; background-color: #ffffff !important; }
iframe { border: none !important; border-radius: 8px !important; outline: none !important; }
.pyvis-map-wrapper iframe { width: 100% !important; height: 600px !important; display: block !important; }
</style>
"""
components.html(custom_ui_code, height=0, width=0)

current_orcid_js = st.session_state.orcid_profile if st.session_state.orcid_profile else ""
current_orcid_name_js = st.session_state.researcher_name if st.session_state.researcher_name != "Anonymous Researcher" else ""
state_payload = st.session_state.web3_wallet if has_web3 else "none"
orcid_auth_url = f"https://orcid.org/oauth/authorize?client_id={ORCID_CLIENT_ID}&response_type=code&scope=/authenticate&redirect_uri={ORCID_REDIRECT_URI}&state={state_payload}"

# Plain, unstyled button HTML structure for MetaMask connection
mm_button_html = f"""
    <div style="width: 100%; box-sizing: border-box;">
        <button id="connect-mm-btn" type="button" style="width: 100%; cursor: pointer;">
            Connect MetaMask
        </button>
        <div id="mm-status" style="margin-top: 4px; font-size: 11px; color: #dc2626; text-align: center;"></div>
    </div>
    <script>
    function getEthereumProvider() {{
        let provider = window.ethereum;
        if (!provider && window.parent) {{ try {{ provider = window.parent.ethereum; }} catch(e) {{}} }}
        if (!provider && window.top) {{ try {{ provider = window.top.ethereum; }} catch(e) {{}} }}
        if (provider && provider.providers) {{ provider = provider.providers.find(p => p.isMetaMask) || provider; }}
        return provider;
    }}
    const mmBtn = document.getElementById('connect-mm-btn');
    if (mmBtn) {{
        mmBtn.addEventListener('click', async () => {{
            const statusDiv = document.getElementById('mm-status');
            statusDiv.innerText = "Connecting...";
            const provider = getEthereumProvider();
            if (!provider) {{ statusDiv.innerText = "MetaMask not detected!"; return; }}
            try {{
                const accounts = await provider.request({{ method: 'eth_requestAccounts' }});
                if (!accounts || accounts.length === 0) return;
                const account = accounts[0];
                statusDiv.innerText = "Signing...";
                const domain = "ScholarPi";
                const nonce = Math.floor(Math.random() * 100000000);
                const message = `${{domain}} wants you to sign in with your Ethereum account:\\n${{account}}\\n\\nSign in with Ethereum to authenticate session.\\n\\nNonce: ${{nonce}}\\nIssued At: ${{new Date().toISOString()}}`;
                let signature = null;
                try {{
                    const hexMessage = '0x' + Array.from(new TextEncoder().encode(message)).map(b => b.toString(16).padStart(2, '0')).join('');
                    signature = await provider.request({{ method: 'personal_sign', params: [hexMessage, account] }});
                }} catch (e) {{}}
                const targetUrl = new URL(window.top.location.href.split('?')[0]);
                targetUrl.searchParams.set("siwe_address", account);
                if (signature) {{ targetUrl.searchParams.set("siwe_signature", signature); targetUrl.searchParams.set("siwe_message", encodeURIComponent(message)); }}
                const currentOrcid = "{current_orcid_js}"; const currentOrcidName = "{current_orcid_name_js}";
                if (currentOrcid) targetUrl.searchParams.set("restore_orcid", currentOrcid);
                if (currentOrcidName) targetUrl.searchParams.set("restore_orcid_name", currentOrcidName);
                window.open(targetUrl.href, '_blank');
                statusDiv.innerHTML = "Verified! Sync completed in the newly opened tab.";
            }} catch (err) {{ statusDiv.innerText = err.message || "Rejected."; }}
        }});
    }}
    </script>
"""

st.sidebar.title("System Access & Sync")
with st.sidebar:
    if not has_web3: components.html(mm_button_html, height=55)
    else: st.success(f"Web3 Linked: `{st.session_state.web3_wallet[:6]}...{st.session_state.web3_wallet[-4:]}`")

    if not has_orcid: st.link_button("Link ORCID Account", orcid_auth_url, use_container_width=True)
    else: st.success(f"ORCID Linked: `{st.session_state.orcid_profile}`")

if not (has_web3 and has_orcid):
    st.sidebar.info("**Dual-Auth Synchronization Guide:**\n• **Link Both First:** Connect both your MetaMask wallet and your ORCID account below before running assessments.\n• **Seamless Rewards:** When both are active, your evaluation history and rewards merge automatically.")

if has_web3 or has_orcid:
    conn_hist = get_db_connection()
    total_user_piq = 0.0
    try:
        cur_h = conn_hist.cursor()
        clauses, params = [], []
        if has_web3: clauses.append("eth_book = ?"); params.append(st.session_state.web3_wallet)
        if has_orcid: clauses.append("user_id = ?"); params.append(st.session_state.orcid_profile)
        if clauses:
            cur_h.execute(f"SELECT DISTINCT eval_hash, piq_minted FROM papers_assessment WHERE {' OR '.join(clauses)}", tuple(params))
            total_user_piq = sum(safe_float(r[1], 0.0) for r in cur_h.fetchall() if r[1])
    finally:
        conn_hist.close()

    status_line = "**Synced Status:** Active Sync\n\n" if (has_web3 and has_orcid) else ""
    st.sidebar.markdown(f"**Researcher:** {st.session_state.researcher_name}\n\n{status_line}**TOTAL piQ AWARDED:** `{total_user_piq:.2f} piQ`")

    if st.sidebar.button("Unlink / Reset Session", use_container_width=True):
        st.session_state.web3_wallet, st.session_state.orcid_profile, st.session_state.researcher_name = None, None, "Anonymous Researcher"
        st.rerun()

st.sidebar.markdown("---")
with st.sidebar.expander("Live System Monitor", expanded=True):
    st.code("\n".join(st.session_state.app_logs) if st.session_state.app_logs else "No active logs...", language="bash")

with st.sidebar.expander("🧠 Scilem Assistant", expanded=False):
    floating_chat_container = st.container(height=220)
    with floating_chat_container:
        for idx, message in enumerate(st.session_state.scilem_messages):
            st.chat_message(message["role"], avatar="🧠" if message["role"] == "assistant" else "👤").markdown(message["content"])

    with st.form(key="scilem_sidebar_form", clear_on_submit=False):
        f_cols = st.columns([3, 1])
        floating_prompt = f_cols[0].text_input("Ask Scilem...", value="", label_visibility="collapsed")
        if f_cols[1].form_submit_button("Send") and floating_prompt.strip():
            st.session_state.scilem_messages.append({"role": "user", "content": floating_prompt})
            st.session_state.scilem_messages.append({"role": "assistant", "content": evaluate_scilem_analysis_report(floating_prompt)})
            st.rerun()

    if has_web3 and st.session_state.web3_wallet.lower() == OWNER_ID.lower() and st.button("Reset Scilem (Owner)", use_container_width=True):
        msg = reset_scilem()
        st.session_state.scilem_messages = [{"role": "assistant", "content": "**Scilem has been reset.** Neural weights and context cleared to baseline by Web3 owner."}]
        add_log(msg)
        st.toast(msg, icon="🧠")
        time.sleep(0.5)
        st.rerun()

# ==========================================
# 4. SHARED FUNCTIONS & DIALOGS
# ==========================================
def refine_science_field(s):
    s_lower = s.lower()
    if any(k in s_lower for k in ["blockchain", "smart contract", "crypto", "ledger"]): return "Computer Science > Blockchain & Distributed Systems"
    elif any(k in s_lower for k in ["machine learning", "deep learning", "neural", "ai", "artificial intelligence"]): return "Computer Science > Artificial Intelligence & Machine Learning"
    elif any(k in s_lower for k in ["algorithm", "software", "computation", "cyber", "data", "information"]): return "Computer Science > Algorithms & Software Engineering"
    elif any(k in s_lower for k in ["quantum", "optics", "photonics"]): return "Physics > Quantum Mechanics & Optics"
    elif any(k in s_lower for k in ["energy", "mechanics", "thermodynamics", "physics"]): return "Physics > Applied Mechanics & Energy Systems"
    elif any(k in s_lower for k in ["polymer", "catalysis", "molecule", "chemical", "chemistry"]): return "Chemistry > Chemical Synthesis & Molecular Catalysis"
    elif any(k in s_lower for k in ["genetics", "genomics", "gene", "biology"]): return "Life Sciences > Genetics & Genomics"
    elif any(k in s_lower for k in ["cellular", "protein", "molecular biology"]): return "Life Sciences > Molecular & Cellular Biology"
    elif any(k in s_lower for k in ["ecology", "ecosystem", "biodiversity"]): return "Life Sciences > Ecology & Evolutionary Biology"
    elif any(k in s_lower for k in ["clinical", "hospital", "patient", "disease", "pharmac", "medical", "medicine"]): return "Medical Sciences > Clinical Medicine & Pharmacology"
    elif any(k in s_lower for k in ["biomedical", "neuroscience", "cardiac"]): return "Medical Sciences > Biomedical Research"
    elif any(k in s_lower for k in ["climate", "carbon", "atmosphere", "meteorology", "earth"]): return "Earth Sciences > Climate Science & Meteorology"
    elif any(k in s_lower for k in ["geology", "ocean", "seismic"]): return "Earth Sciences > Geology & Earth Systems"
    elif any(k in s_lower for k in ["economics", "finance", "market", "social"]): return "Social Sciences > Economics & Quantitative Finance"
    elif any(k in s_lower for k in ["sociology", "psychology", "policy", "management"]): return "Social Sciences > Behavioral & Policy Studies"
    elif any(k in s_lower for k in ["math", "statistics", "algebra", "probability", "calculus"]): return "Mathematics & Statistics > Applied Mathematics & Statistics"
    elif any(k in s_lower for k in ["engineering", "robotics", "materials", "civil", "electrical"]): return "Engineering & Technology > Applied Engineering & Materials Science"
    else: return f"Engineering & Technology > Applied Technical Research ({s.title()})"

@st.cache_data(ttl=3600)
def render_bubble_chart_clean(target_author, repulsion=-3000, spring_len=180, size_scale=1.5, central_grav=0.15, _db_token=0):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT fields, subfields, final_score, author_name FROM papers_assessment")
        data = cursor.fetchall()
    finally: conn.close()

    html_string, table_html = "", ""
    if not data: return html_string, table_html

    topic_aggregates = {}
    exclude_terms = {"general", "general science", "unspecified domain", "unspecified sub-domain", "core research topic"}
    for fields_json, subfields_json, final_score, author_str in data:
        cleaned_author = clean_author_name(author_str)
        if target_author and target_author != "All Authors" and target_author not in cleaned_author: continue
        try:
            raw_subfields = [s.title().strip() for s in json.loads(subfields_json)]
            score = safe_float(final_score, 50.0)
            for rs in raw_subfields:
                if rs and rs.lower() not in exclude_terms:
                    s = refine_science_field(rs)
                    if s not in topic_aggregates: topic_aggregates[s] = {"weight_sum": 0.0, "frequency": 0}
                    topic_aggregates[s]["weight_sum"] += score
                    topic_aggregates[s]["frequency"] += 1
        except: continue

    if not topic_aggregates: topic_aggregates["Computer Science > Algorithms & Software Engineering"] = {"weight_sum": 50.0, "frequency": 1}
    if len(topic_aggregates) > 15:
        sorted_topics = sorted(topic_aggregates.items(), key=lambda x: (x[1]["frequency"], x[1]["weight_sum"]), reverse=True)
        topic_aggregates = dict(sorted_topics[:15])

    unique_topics = list(topic_aggregates.keys())
    major_fields_dict = {}
    for topic in unique_topics:
        major = [p.strip() for p in topic.split('>')][0]
        if major not in major_fields_dict: major_fields_dict[major] = []
        major_fields_dict[major].append(topic)

    major_keys = sorted(list(major_fields_dict.keys()))
    color_map = {}
    for i, major in enumerate(major_keys):
        h = i / len(major_keys) if len(major_keys) > 0 else 0
        subfields = sorted(major_fields_dict[major])
        n_subs = len(subfields)
        for j, topic in enumerate(subfields):
            if n_subs <= 1: s, v = 0.7, 0.9
            else:
                ratio = j / (n_subs - 1)
                s = 0.4 + (0.5 * ratio)
                v = 0.95 - (0.35 * ratio)
            rgb = colorsys.hsv_to_rgb(h, s, v)
            color_map[topic] = "#%02x%02x%02x" % tuple(int(x * 255) for x in rgb)

    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#2c3e50", notebook=False)
    physics_options = f"""{{ "physics": {{ "barnesHut": {{ "gravitationalConstant": {repulsion}, "centralGravity": {central_grav}, "springLength": {spring_len}, "springConstant": 0.005, "damping": 1.0, "avoidOverlap": 2.0 }}, "stabilization": {{ "enabled": true, "iterations": 2500, "fit": true }} }} }}"""
    net.set_options(physics_options)

    for topic, metrics in topic_aggregates.items():
        avg_weight = metrics["weight_sum"] / metrics["frequency"]
        node_size = max(35, (25 + (avg_weight * 3.0)) * size_scale)
        base_col = color_map[topic]
        net.add_node(n_id=topic, label=" ", title=f"Field: {topic} | Frequency: {metrics['frequency']} | Avg Weight/Score: {avg_weight:.1f}", size=node_size, shape="dot", physics=True, font={"color": "rgba(0,0,0,0)", "size": 0}, color={"background": base_col, "border": "#1a1a1a", "highlight": {"background": base_col, "border": "#000000"}, "hover": {"background": base_col, "border": "#000000"}}, shadow={"enabled": True, "color": "rgba(0,0,0,0.5)", "size": 6, "x": 3, "y": 3})
        
    for i, t1 in enumerate(unique_topics):
        for j, t2 in enumerate(unique_topics):
            if i < j and t1.split(">")[0].strip() == t2.split(">")[0].strip(): net.add_edge(t1, t2, color="rgba(150,150,150,0.2)")

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".html")
    os.close(tmp_fd) 
    try:
        net.save_graph(tmp_name)
        with open(tmp_name, "r", encoding="utf-8") as f: html_string = f.read()
    finally:
        if os.path.exists(tmp_name): os.remove(tmp_name)

    gradient_injection = f"""<style type="text/css"> body, html {{ margin: 0; padding: 0; border: none; overflow: hidden; width: 100%; height: 600px; }} canvas {{ background: radial-gradient(circle at 50% 50%, #ffffff 0%, #f0f2f5 100%); border: none !important; outline: none !important; width: 100% !important; height: 600px !important; }} #mynetwork, .vis-network, .card-body {{ border: none !important; box-shadow: none !important; margin: 0 !important; width: 100% !important; height: 600px !important; }} </style> <!-- reload_timestamp: {time.time()} --> </head>"""
    html_string = html_string.replace("</head>", gradient_injection).replace("mynetwork", f"pi_network_{int(time.time() * 1000)}")

    table_html = "<style>.table-compact { width: 100%; font-size: 12px; border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif; } .table-compact th { background-color: #f8fafc; color: #475569; padding: 6px 8px; text-align: left; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; border-bottom: 2px solid #e2e8f0; position: sticky; top: 0; z-index: 1; } .table-compact td { padding: 6px 8px; border-bottom: 1px solid #f1f5f9; color: #1e293b; } .color-box { width: 14px; height: 14px; border-radius: 3px; display: inline-block; box-shadow: 0 1px 2px rgba(0,0,0,0.1); } </style><div style='max-height: 220px; overflow-y: auto; border: 1px solid #e2e8f0; border-radius: 8px;'><table class='table-compact'><thead><tr><th style='width: 15%; text-align: center;'>Color</th><th>Science Field</th><th style='text-align: center;'>Freq</th><th style='text-align: center;'>Avg Weight</th></tr></thead><tbody>"
    for topic, metrics in sorted(topic_aggregates.items(), key=lambda x: x[1]["frequency"], reverse=True):
        avg_w = metrics["weight_sum"] / metrics["frequency"]
        table_html += f"<tr><td style='text-align: center;'><div class='color-box' style='background-color:{color_map[topic]};'></div></td><td><b>{topic}</b></td><td style='text-align: center;'>{metrics['frequency']}</td><td style='text-align: center;'>{avg_w:.1f}</td></tr>"
    table_html += "</tbody></table></div>"

    return html_string, table_html

def get_criteria_info(weights):
    tw1, tw2, tw3, tw4, tw5, tw6, tw7, tw8 = weights
    return [
        ("C1", "Originality", "c1: originality", tw1, "1", "Semantic distance from literature corpus penalized by generative AI laundering heuristics.", r"$$ C_1 = \varpi_1 \cdot \mathcal{D}_{semantic}(P_{target}, P_{corpus}) \times (1 - \lambda_{laundering}) + \text{vapri} $$"),
        ("C2", "Methodological Rigor", "c2: methodological rigor", tw2, "2", "Deterministic adherence to MDAR reporting standards and valid RRIDs via SciScore.", r"$$ C_2 = \varpi_2 \cdot \mathcal{I}_{blinding} + \varpi_2 \cdot \mathcal{I}_{randomization} + \varpi_2 \cdot \mathcal{I}_{power\_calc} + \varpi_2 \cdot \left(\frac{N_{RRID\_valid}}{N_{RRID\_expected} + \epsilon}\right) $$"),
        ("C3", "Interdisciplinary Synergy", "c3: interdisciplinary synergy", tw3, "3", "Measures cross-disciplinary integration and entropy across scientific domains.", r"$$ C_3 = \varpi_3 \cdot -\sum_{i=1}^{k} p_i \ln(p_i) $$"),
        ("C4", "Societal Impact", "c4: societal impact", tw4, "4", "Evaluates broader societal and open infrastructure contributions.", r"$$ C_4 = \varpi_4 \cdot \Theta\left[ \sum_{v \in \mathcal{V}} \omega_v U_v(\tau, \mathbf{x}) \right] $$"),
        ("C5", "Open Science", "c5: open science", tw5, "5", "Evaluates open data, open code, and containerized reproducibility.", r"$$ C_5 = \varpi_5 \cdot (\beta_1 \cdot \mathcal{V}_{data} + \beta_2 \cdot \mathcal{V}_{code} + \beta_3 \cdot \mathcal{Z}_{container}) $$"),
        ("C6", "Literature Integration", "c6: literature integration", tw6, "6", "Evaluates citation polarity and integration with existing foundational literature.", r"$$ C_6 = \varpi_6 \cdot \frac{1}{\mathcal{N}} \sum_{i=1}^{\mathcal{N}} \text{Polarity}(x_i) \cdot \text{PR}(x_i) $$"),
        ("C7", "Empirical Density", "c7: empirical density", tw7, "7", "Assesses empirical sample strength and baseline variance.", r"$$ C_7 = \varpi_7 \cdot \tanh \left( \frac{n_{\text{valid}} \cdot \text{Cohort Strength}}{\text{Baseline Variance}} \right) $$"),
        ("C8", "Future Actionability", "c8: future actionability", tw8, "8", "Evaluates future research actionability and adherence to FAIR principles.", r"$$ C_8 = \varpi_8 \cdot \frac{1}{\mathcal{Z}} \int_{\mathcal{X}} \text{FAIR\_Score}(\mathbf{x}) \, d\mu(\mathbf{x}) $$"),
    ]

@st.dialog("Criterion Details & Adversarial Logic Engine", width="medium")
def criterion_details_dialog(c_id, title, q_key, weight_val, sym, desc, formula):
    st.markdown(f"### {c_id}: {title}")
    st.markdown(rf"**Current Epoch Weight ($\varpi_{sym}$):** `{weight_val:.6f}`")
    st.markdown(f"{desc} {rbot(q_key)}", unsafe_allow_html=True)
    st.markdown(formula)
    st.markdown("---")
    st.markdown("**Adversarial Logic Gap ($\Delta_{Logic}$):** Evaluates reasoning structure and penalizes claims unsupported by evidence.", unsafe_allow_html=True)
    st.markdown(r"$$ L_i = \left( (\mathcal{P}_{valid} \cdot \mathcal{E}_{strength}) \cdot \exp\left(-\left(2 \cdot \max(0, \mathcal{C}_{reach} - \mathcal{E}_{strength}) + 1.5 \cdot \lambda_{jumps}\right)\right) \right) \times \frac{1}{1 + e^{-\Delta Premise}} + \lambda \cdot \text{vapri} $$")

@st.dialog("Detailed Research Integrity Dossier", width="large")
def more_details_dialog(item):
    title, author_name = item.get("title", "Unknown"), clean_author_name(item.get("author_name", "Unknown"))
    st.subheader(f"{title} by {author_name}")
    st.write(f"**Evaluation Hash:** `{item.get('eval_hash', '0x0')}`")
    st.write(f"**piQ Minted:** `{safe_float(item.get('piq'), 0.0)}`")
    
    if item.get("warnings"):
        st.warning(f"⚠️ **Manuscript Flagged with {len(item['warnings'])} Warning Check(s):**")
        for w in item["warnings"]: st.markdown(f"- {w}")

    st.markdown("---")
    if item.get("consensus_raw") and isinstance(item["consensus_raw"], dict):
        st.markdown("### Multi-LLM Extractions")
        llm_cols = st.columns(2, gap="medium")
        for idx, llm_key in enumerate(["llama", "mistral", "qwen", "gemini", "scilem"]):
            with llm_cols[idx % 2]:
                data = item["consensus_raw"].get(llm_key, {})
                with st.container(border=True):
                    st.markdown(f"**Model: {llm_key.upper()}**")
                    if llm_key == "scilem":
                        st.markdown(f"**Engine Status:** Active (Local PyTorch Neural Network)\n**Structural Analysis:** {data.get('opinion', 'Scilem structural analysis active.')}")
                    elif data.get('api_failed', False):
                        st.markdown(f"**Status:** Rate / Credit Limit Hit\n**Opinion:** {data.get('opinion', 'No opinion extracted.')}")
                    else:
                        st.markdown(f"**Extracted Title:** `{data.get('title', 'N/A')}`\n**Extracted Authors:** `{data.get('authors', 'N/A')}`\n**Opinion:** {data.get('opinion', 'No opinion extracted.')}")
    
    if item.get("evidence_report_text"):
        st.markdown("---")
        st.markdown("### Synthesized Evidence Report")
        st.markdown(item["evidence_report_text"])

@st.dialog("AI Peer Review Defense Strategy", width="medium")
def defense_strategy_dialog(scores_dict):
    with st.spinner("Synthesizing adversarial defense strategy..."):
        st.markdown(generate_rebuttal_strategy(scores_dict))

def render_breakdown_item(item, index):
    title, author_name = item["title"], clean_author_name(item["author_name"])
    with st.container(border=True):
        col_info, col_actions = st.columns([6, 4], gap="medium")
        with col_info:
            warn_badge = f" ⚠️ *({len(item.get('warnings', []))} warning checks active)*" if item.get('warnings') else ""
            st.markdown(f"**{title}** — *{author_name}*{warn_badge}")
            st.markdown(f"**Score: {safe_float(item['score'], 0.0):.2f} | piQ: {safe_float(item['piq'], 0.0)}**")
        with col_actions:
            c_det, c_strat, c_del = st.columns([3, 3, 1], gap="small")
            if c_det.button("More Details", key=f"more_det_{index}_{item['eval_hash']}", use_container_width=True): more_details_dialog(item)
            if c_strat.button("Suggest Defense", key=f"gen_strat_{index}_{item['eval_hash']}", use_container_width=True): defense_strategy_dialog(item['scores_dict'])
            if c_del.button("❌", key=f"close_eval_{index}_{item['eval_hash']}", help="Close this result"):
                st.session_state["evaluated_papers_buffer"].pop(index)
                st.rerun()

# ==========================================
# 5. PAGE DEFINITIONS
# ==========================================

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
                stake_amount = st.checkbox("Stake 0.1 piQ to Process", value=True)

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
            r1, r2 = st.columns([4, 1], gap="medium")
            r1.button("Working...", type="primary", use_container_width=True, disabled=True)
            if r2.button("Stop", type="secondary", use_container_width=True):
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
                                item = {"title": res[0], "author_name": clean_author_name(res[1]), "score": res[2], "logic_integrity": res[3], "scores_dict": res[8], "eval_hash": res[9], "piq": res[10], "tx_hash": res[11], "zk_proof": res[12], "h_idx": res[14], "i10_idx": res[15], "repro_score": res[16], "filename": f"DOI_{doi_snap}.pdf", "warnings": res[18], "consensus_raw": res[19], "evidence_report_text": res[20], "scilem_rating": res[21]}
                                st.session_state["evaluated_papers_buffer"].insert(0, item)
                                st.session_state["free_evals_used"] += 1
                        else:
                            st.session_state["download_errors"].append({"title": doi_snap, "doi": doi_snap, "url": f"https://doi.org/{doi_snap}"})

                    for fname, fpath in st.session_state.get("snap_files", []):
                        if st.session_state["cancel_requested"]: break
                        status_box.update(label=f"Analyzing {fname}...")
                        with open(fpath, "rb") as in_f: raw_bytes = in_f.read()
                        res = process_single_pdf(preprocess_pdf_layout(raw_bytes, fname), fname, "", current_user, valid_book_address)
                        if res:
                            item = {"title": res[0], "author_name": clean_author_name(res[1]), "score": res[2], "logic_integrity": res[3], "scores_dict": res[8], "eval_hash": res[9], "piq": res[10], "tx_hash": res[11], "zk_proof": res[12], "h_idx": res[14], "i10_idx": res[15], "repro_score": res[16], "filename": fname, "warnings": res[18], "consensus_raw": res[19], "evidence_report_text": res[20], "scilem_rating": res[21]}
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
                if free_evals_used >= 1 and (not has_web3 or not stake_amount): st.error("Free trial limit reached. Connect Web3 and stake 0.1 piQ.")
                elif not selected_uploaded_files and not (include_doi and doi_input.strip()): st.warning("Please tick at least one source to assess.")
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
            render_breakdown_item(item, item_idx)

def page_analytics():
    st.markdown("<h1 style='margin-bottom:0;'>🌐 Map & Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    top_c1, top_c2 = st.columns(2, gap="large")
    with top_c1:
        st.markdown("### Pidyne Forecast")
        lookback = st.selectbox("Lookback Window", ["1 Epoch", "3 Epochs", "5 Epochs"], index=1)
        actual_lookback = int(lookback.split()[0])
        
        conn = get_db_connection()
        historical_rows = conn.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights ORDER BY block_height ASC").fetchall()
        
        if len(historical_rows) < 2:
            st.warning("Not enough blockchain data to train meta-model. Need at least 2 blocks.")
        else:
            weight_data = np.array([[safe_float(v, 1.0) for v in r] for r in historical_rows], dtype=np.float32)
            dataset = PidyneBlockchainDataset(weight_data, actual_lookback)
            dataloader = DataLoader(dataset, batch_size=min(4, max(1, len(dataset))), shuffle=False)
            model = PidyneLSTM()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            model.train()
            for _ in range(300):
                for seq, target in dataloader:
                    optimizer.zero_grad()
                    loss = nn.MSELoss()(model(seq), target)
                    loss.backward()
                    optimizer.step()
            model.eval()
            with torch.no_grad():
                raw_pred = model(torch.tensor(weight_data[-actual_lookback:], dtype=torch.float32).unsqueeze(0)).squeeze().numpy()
                predicted = weight_data[-1] + (raw_pred - weight_data[-1]) * 20.0
                next_weights = np.clip(predicted, 0.01, 7.9) * (8.0 / np.sum(np.clip(predicted, 0.01, 7.9)))

            df_hist = pd.DataFrame(historical_rows[-(actual_lookback + 1):], columns=["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"])
            df_hist.index.name = "Block"
            df_melted = df_hist.reset_index().melt('Block', var_name='Criterion', value_name='Weight')
            st.altair_chart(alt.Chart(df_melted).mark_line(point=True).encode(x='Block:O', y='Weight:Q', color='Criterion:N'), use_container_width=True)

            with st.container(border=True):
                st.markdown(f"**Ledger Forecast (Raw Sum = {sum(next_weights):.6f}/8.0):**")
                crit_info = get_criteria_info(next_weights)
                cols1 = st.columns(4, gap="small")
                for idx, c_data in enumerate(crit_info[:4]):
                    if cols1[idx].button(f"{c_data[0]}: {c_data[3]:.5f}", key=f"btn_crit_{c_data[0]}", use_container_width=True): criterion_details_dialog(*c_data)
                cols2 = st.columns(4, gap="small")
                for idx, c_data in enumerate(crit_info[4:]):
                    if cols2[idx].button(f"{c_data[0]}: {c_data[3]:.5f}", key=f"btn_crit_{c_data[0]}", use_container_width=True): criterion_details_dialog(*c_data)

    with top_c2:
        st.markdown("### Global Map of Science")
        interactive_html_top, table_html_top = render_bubble_chart_clean("All Authors", -3000, 180, 1.5, 0.15, st.session_state['assessment_update_token'])
        with st.container():
            if interactive_html_top:
                st.markdown("<div class='pyvis-map-wrapper'>", unsafe_allow_html=True)
                components.html(interactive_html_top, height=600, scrolling=False)
                st.markdown("</div>", unsafe_allow_html=True)
            else: st.info("Awaiting sufficient data for map visualization.")
            
        with st.expander("Map Legend & Data"): st.markdown(table_html_top, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### Pi Quotient (piQ) Leaderboard")
        data = pd.read_sql_query("SELECT author_name, piq_minted FROM papers_assessment", conn)
        author_piq = {}
        for _, row in data.iterrows():
            ca = clean_author_name(row["author_name"])
            if ca and ca.lower() not in ["unidentified", "unknown"] and not is_likely_institution(ca):
                for a in [x.strip() for x in ca.split(",")]: author_piq[a] = author_piq.get(a, 0.0) + float(row["piq_minted"] or 0)
        if author_piq: st.dataframe(pd.DataFrame(sorted(author_piq.items(), key=lambda x: x[1], reverse=True)[:20], columns=["Author", "piQ"]), hide_index=True)

    with col2:
        st.markdown("### piX Top Papers")
        df_pix = pd.read_sql_query("SELECT title, author_name, final_score FROM papers_assessment ORDER BY final_score DESC LIMIT 20", conn)
        st.dataframe(df_pix, hide_index=True)
    conn.close()

def page_explorer():
    st.markdown("<h1 style='margin-bottom:0;'>⛓️ Ledger Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
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
                            "eval_hash": r[19], "piq": r[13], "tx_hash": r[14], "zk_proof": r[15],
                            "h_idx": r[16], "i10_idx": r[17], "repro_score": r[18], "filename": r[2], 
                            "consensus_raw": json.loads(r[20]) if r[20] else {}, "evidence_report_text": r[21], "scilem_rating": r[22]
                        })
        else: st.error("No matching ledger records found.")
    else:
        st.markdown("### Latest Assessed Papers")
        df = pd.read_sql_query("SELECT title as Title, author_name as Author, final_score as Score, eval_hash as Hash FROM papers_assessment ORDER BY timestamp DESC LIMIT 20", conn)
        st.dataframe(df, use_container_width=True, hide_index=True)
    conn.close()

def page_diagram():
    st.markdown("<h1 style='margin-bottom:0;'>📊 Framework Architecture</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "Pi-Index filters noise and yields quantitative results strictly aligned with **Responsible Research Assessment (RRA)** and **CoARA** (Coalition for Advancing Research Assessment) guidelines.\n\n"
        "### Architecture Flowchart & Whitepaper DOI\n\n"
        "Read the foundational framework whitepaper and preprints via [Ali Vafadar Yengejeh's ResearchGate Profile](https://www.researchgate.net/profile/Ali-Vafadar-Yengejeh).\n\n"
        "The enhanced system architecture flow below details the decentralized intake, ZK double-blind reviewer assignment, SciScore deterministic parsing, Item Response Theory (IRT) calibration, and smart contract slashing mechanisms."
    )
    st.graphviz_chart("""
    digraph PiIndexSystemOverview {
        rankdir=TB; compound=true; fontname="Helvetica,Arial,sans-serif"; node [fontname="Helvetica,Arial,sans-serif", style=filled, margin=0.2]; edge [fontname="Helvetica,Arial,sans-serif", fontsize=10];
        node [shape=box, fillcolor="#f8f9fa", color="#2c3e50", penwidth=1.5];

        subgraph cluster_intake { label = "1. Unified Multi-Source Intake & ZK-Identity Registry (ZIP-600)"; style = rounded; color = "#34495e"; fillcolor = "#ecf0f1"; Auth [label="Researcher Authentication\n• ORCID iD / W3C DID Verification\n• ZK-Email Institutional Proof", fillcolor="#aed6f1"]; Intake [label="Multi-Source Ingestion Engine\n• Local Binary PDFs Extraction\n• Unpaywall DOI Resolver\n• OpenAlex Topic API Search", fillcolor="#aed6f1"]; TempDisk [label="Temp Disk State Management\n• Streamlit Render Protection\n• Buffered Binary Writes", fillcolor="#aed6f1", style="dashed,filled"]; ZKBlind [label="ZK Double-Blind Assignment\n• Merkle Tree Non-Membership Proofs\n• Anonymous Author Shielding", fillcolor="#aed6f1"]; Auth -> Intake -> TempDisk -> ZKBlind; }
        subgraph cluster_eval { label = "2. Core Evaluation & Adversarial Analysis Pipeline (CoARA/RRA)"; style = rounded; color = "#27ae60"; fillcolor = "#e8f8f5"; PyMuPDF [label="PyMuPDF Layout Sort\n• Spatial Reading Extraction\n• Mathematical Integrity Safeguard", fillcolor="#a3e4d7", style="dashed,filled"]; SciParser [label="Deterministic SciScore API\n• MDAR Reporting Adherence\n• Valid RRIDs Count Extraction", fillcolor="#a3e4d7"]; Retry [label="Multi-LLM Consensus Engine\n• Llama, Mistral, Qwen, Gemini & Scilem Analysis\n• Synthesized Evidence Report", fillcolor="#a3e4d7", style="dashed,filled"]; IRTCalib [label="Item Response Theory Calibration\n• Counterfactual Stress Testing\n• Variance & Difficulty Mapping", fillcolor="#a3e4d7"]; Criteria [label="8 Transparent Criteria Rubrics\n• C1 Originality to C8 FAIR Actionability\n• Formulaic Score Computation", fillcolor="#a3e4d7"]; Logic [label="Adversarial Logic Integrity Matrix\n• Premise Validity & Evidence Strength\n• AI Hallucination & Laundering Penalty", fillcolor="#a3e4d7"]; PyMuPDF -> SciParser -> Retry -> IRTCalib -> Criteria -> Logic; }
        subgraph cluster_blockchain { label = "3. Blockchain Consensus, Cryptographic Proofs & Slashing Tokenomics"; style = rounded; color = "#8e44ad"; fillcolor = "#f4ecf7"; PoR [label="Proof-of-Research (PoR) Validation\n• Dynamic Epoch Weight Shifting\n• Formulas Hash Stamping & SHA-256 Block", fillcolor="#d7bde2"]; Slashing [label="Anti-Laundering Slashing Guard\n• Smart Contract piQ Burn for Fraud\n• Stake Penalty Enforcement", fillcolor="#f5b7b1"]; Mint [label="Soulbound Token Minting\n• Author-Specific Book Address (eth_book)\n• Shared Paper Address (eval_hash) & Tx Hash", fillcolor="#d7bde2"]; PoR -> Slashing -> Mint; }
        subgraph cluster_outputs { label = "4. User Interface, Cartography & Institutional Policy Support"; style = rounded; color = "#d35400"; fillcolor = "#fef5e7"; Dossier [label="CoARA & DORA-Aligned Dossier\n• Markdown Research Integrity Report\n• AI Defense Rebuttal Strategy", fillcolor="#f8c471"]; Cartography [label="Global Map of Science\n• Ledger PyVis Network Cartography\n• Author & Topic Bubble Filtering", fillcolor="#f8c471"]; PidyneBrain [label="Pidyne LSTM Meta-Learning\n• PyTorch Temporal Weight Prediction\n• Calibration Drift & Epoch Forecasting", fillcolor="#f8c471"]; }

        ZKBlind -> PyMuPDF [lhead=cluster_eval, label="Processed Manuscript Text"]; Logic -> PoR [lhead=cluster_blockchain, label="Audited Score & Hashes"]; Mint -> Dossier [lhead=cluster_outputs, label="Ledger Seal & Tokens"]; Mint -> Cartography; Mint -> PidyneBrain;
    }
    """)
    st.markdown("---")
    st.markdown("""
    ### CoARA Compliance & Core Pillars
    *   **Diverse Research Outputs (C5 & C8):** Moving beyond traditional journal impact factors, Pi-Index structurally evaluates open datasets, code repositories, and containerized executable environments.
    *   **Qualitative & Quantitative Balance (C1-C8):** Algorithms act as auditors, not replacements for peer review. They standardize empirical rigor (e.g., RRID usage, MDAR adherence) while an adversarial logic matrix maps qualitative reasoning structure.
    *   **Transparency & Researcher Sovereignty:** Complete evaluation weights, logic states, and criteria scores are irreversibly hashed and stored on the Ethereum (Sepolia) blockchain. Researchers retain sovereign ownership of their academic profile via DID/ORCID integration.
    """)
    st.markdown("<div style='text-align: center; color: gray; font-size: 0.9em; padding-bottom: 5px;'>Framework Author: Ali Vafadar Yengejeh | Universita degli Studi di Milano-Bicocca</div>", unsafe_allow_html=True)

# ==========================================
# 6. ROUTING/NAVIGATION
# ==========================================
pg = st.navigation([
    st.Page(page_assessment, title="Assess Manuscript", icon="📄"),
    st.Page(page_analytics, title="Analytics & Map", icon="🌐"),
    st.Page(page_explorer, title="Blockchain Explorer", icon="⛓️"),
    st.Page(page_diagram, title="Architecture Diagram", icon="📊"),
])
pg.run()
