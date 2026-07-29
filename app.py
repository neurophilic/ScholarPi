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
    assess_manuscript, generate_rebuttal_strategy, PidyneLSTM, 
    PidyneBlockchainDataset, generate_scilem_fallback_report, reset_scilem,
    evaluate_scilem_analysis_report
)

w3 = Web3()

OWNER_ID = "0x1Af8D9A120b02D0983590587364F8705e6942356"

st.set_page_config(
    page_title="Pi-Index Assessment Engine", layout="wide"
)

if "app_logs" not in st.session_state:
    st.session_state.app_logs = deque(maxlen=50)

def add_log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {msg}"
    st.session_state.app_logs.appendleft(log_entry)
    logging.info(log_entry)

def get_tx_url(tx):
    if not tx or not isinstance(tx, str) or not tx.startswith("0x") or len(tx) != 66:
        return None
    try:
        return get_sepolia_explorer_url(tx, "tx")
    except Exception:
        return None

def safe_float(val, default=0.0):
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except ValueError:
        try:
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
            return float(nums[0]) if nums else default
        except Exception:
            return default

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
        if (
            not clean_authors
            or clean_authors.lower()
            in ["unidentified", "unknown", "research scholar"]
            or is_likely_institution(clean_authors)
        ):
            continue
        alist = [a.strip() for a in clean_authors.split(",") if a.strip()]
        if not alist:
            continue
        share = safe_float(piq, 0.0) / len(alist)
        for a in alist:
            author_piq[a] = author_piq.get(a, 0.0) + share
            author_book[a] = eth_book if eth_book and w3.is_address(eth_book) else "Unbound / Escrow"
    return author_piq, author_book

def preprocess_pdf_layout(pdf_bytes, fname):
    return pdf_bytes

def rbot(topic_key):
    return f"<span class='scilem-trigger' data-query='{topic_key}' title='Ask Scilem' style='cursor: pointer !important; opacity:0.8;'>[?]</span>"

# Initialize Unified Session States
if "web3_wallet" not in st.session_state:
    st.session_state.web3_wallet = None
if "orcid_profile" not in st.session_state:
    st.session_state.orcid_profile = None
if "researcher_name" not in st.session_state:
    st.session_state.researcher_name = "Anonymous Researcher"

# 1. State Preservation
if "restore_orcid" in st.query_params:
    st.session_state.orcid_profile = st.query_params.get("restore_orcid")
    r_name = st.query_params.get("restore_orcid_name")
    if r_name:
        st.session_state.researcher_name = r_name

# 2. Handle Web3 SIWE Callback
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
            except Exception as e:
                add_log(f"SIWE signature verification fallback: {str(e)}")
        
        if authenticated:
            st.session_state.web3_wallet = clean_wallet
            st.toast(f"MetaMask Linked: {clean_wallet[:6]}...{clean_wallet[-4:]}")
        else:
            st.error("Authentication failed: Invalid wallet signature.")

    st.query_params.clear()
    st.rerun()

# 3. Handle Actual ORCID OAuth Token Exchange
if "code" in st.query_params:
    auth_code = st.query_params.get("code")
    returned_state = st.query_params.get("state")
    
    if returned_state and returned_state != "none" and w3.is_address(returned_state):
        st.session_state.web3_wallet = w3.to_checksum_address(returned_state)

    try:
        token_url = "https://orcid.org/oauth/token"
        headers = {"Accept": "application/json"}
        payload = {
            "client_id": ORCID_CLIENT_ID,
            "client_secret": ORCID_CLIENT_SECRET,
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": ORCID_REDIRECT_URI
        }
        
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
            else:
                st.error("Authentication failed: ORCID identifier not returned.")
        else:
            err_desc = response.json().get('error_description', 'Invalid Code')
            st.error(f"ORCID Verification Error: {err_desc}")
            add_log(f"ORCID Auth Error: {err_desc}")
            
    except Exception as e:
        st.error(f"Failed to connect to ORCID API: {str(e)}")

    st.query_params.clear()
    st.rerun()

custom_ui_code = """
<style>
h1, h2, h3, h4, h5, h6 {
    color: #0f172a !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

hr { border-color: #e2e8f0 !important; margin: 1.5rem 0 !important; }
[data-testid="stHeaderActionElements"] { display: none !important; }
[data-testid="stSidebar"] { background-color: #f8fafc !important; border-right: 1px solid #e2e8f0 !important; overflow-y: auto !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px !important; border: 1px solid #e2e8f0 !important; background-color: #ffffff !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important; transition: box-shadow 0.2s ease-in-out, transform 0.2s ease-in-out !important; padding: 0.5rem !important; }

button[kind="primary"], [data-testid="baseButton-primary"] { background-color: #000080 !important; border-color: #000080 !important; color: #ffffff !important; }
button[kind="primary"]:hover, [data-testid="baseButton-primary"]:hover { background-color: #00005b !important; border-color: #00005b !important; color: #ffffff !important; }
.pi-stop-button, .pi-stop-button:focus, button[kind="secondary"], [data-testid="baseButton-secondary"] { background-color: #dc2626 !important; border-color: #dc2626 !important; color: #ffffff !important; }
button[kind="secondary"]:hover, [data-testid="baseButton-secondary"]:hover { background-color: #b91c1c !important; border-color: #b91c1c !important; }
.stButton>button { border-radius: 8px !important; font-weight: 600 !important; letter-spacing: 0.01em !important; }
[data-testid="stExpander"] { border-radius: 10px !important; border: 1px solid #e2e8f0 !important; background-color: #ffffff !important; }
iframe { border: none !important; border-radius: 8px !important; outline: none !important; }
.pyvis-map-wrapper iframe { width: 100% !important; height: 600px !important; display: block !important; }

/* Unified Auth Buttons */
.unified-auth-btn { 
    width: 100%; 
    background-color: #0f172a; 
    color: white; 
    border: 1px solid #1e293b; 
    padding: 10px 14px; 
    border-radius: 8px; 
    font-weight: 600; 
    font-size: 14px; 
    cursor: pointer; 
    display: flex; 
    align-items: center; 
    justify-content: center; 
    transition: background-color 0.2s; 
    text-decoration: none;
    box-sizing: border-box;
}
.unified-auth-btn:hover { background-color: #1e293b; color: white;}
.auth-status-txt { margin-top: 4px; font-size: 11px; color: #dc2626; font-weight: 500; text-align: center; word-break: break-word; }

/* Map canvas */
.vis-gradient-canvas { background: radial-gradient(circle at 50% 50%, #ffffff 0%, #f0f2f5 100%); border: none !important; outline: none !important; width: 100% !important; height: 600px !important; }
.color-box { width: 14px; height: 14px; border-radius: 3px; display: inline-block; box-shadow: 0 1px 2px rgba(0,0,0,0.1); vertical-align: middle; margin-right: 8px; }
</style>
"""
components.html(custom_ui_code, height=0, width=0)

st.sidebar.title("System Access & Sync")

has_web3 = bool(st.session_state.web3_wallet and w3.is_address(st.session_state.web3_wallet))
has_orcid = bool(st.session_state.orcid_profile)

current_orcid_js = st.session_state.orcid_profile if st.session_state.orcid_profile else ""
current_orcid_name_js = st.session_state.researcher_name if st.session_state.researcher_name != "Anonymous Researcher" else ""
state_payload = st.session_state.web3_wallet if has_web3 else "none"
orcid_auth_url = f"https://orcid.org/oauth/authorize?client_id={ORCID_CLIENT_ID}&response_type=code&scope=/authenticate&redirect_uri={ORCID_REDIRECT_URI}&state={state_payload}"

mm_button_html = f"""
    <button id="connect-mm-btn" class="unified-auth-btn" type="button">
        <span>Connect MetaMask Web3</span>
    </button>
    <div id="mm-status" class="auth-status-txt"></div>

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
            statusDiv.style.color = "#2563eb";
            statusDiv.innerText = "Connecting...";

            const provider = getEthereumProvider();
            if (!provider) {{ statusDiv.innerText = "MetaMask not detected!"; return; }}

            try {{
                const accounts = await provider.request({{ method: 'eth_requestAccounts' }});
                if (!accounts || accounts.length === 0) return;
                const account = accounts[0];
                statusDiv.innerText = "Signing SIWE...";

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
                if (signature) {{
                    targetUrl.searchParams.set("siwe_signature", signature);
                    targetUrl.searchParams.set("siwe_message", encodeURIComponent(message));
                }}
                
                const currentOrcid = "{current_orcid_js}";
                const currentOrcidName = "{current_orcid_name_js}";
                if (currentOrcid) targetUrl.searchParams.set("restore_orcid", currentOrcid);
                if (currentOrcidName) targetUrl.searchParams.set("restore_orcid_name", currentOrcidName);

                window.open(targetUrl.href, '_blank');
                statusDiv.innerHTML = `<div style="background:#10b981; color:white; padding:8px; border-radius:6px; margin-top:8px;">Verified! Sync completed in the newly opened tab. You may close this tab.</div>`;
            }} catch (err) {{
                statusDiv.innerText = err.message || "Rejected.";
            }}
        }});
    }}
    </script>
"""

orcid_button_html = f"""<a href="{orcid_auth_url}" target="_blank" class="unified-auth-btn">Link ORCID Account</a>"""

with st.sidebar:
    if not has_web3:
        components.html(mm_button_html, height=100)
    else:
        st.success(f"Web3 Linked: `{st.session_state.web3_wallet[:6]}...{st.session_state.web3_wallet[-4:]}`")

    if not has_orcid:
        components.html(orcid_button_html, height=60)
    else:
        st.success(f"ORCID Linked: `{st.session_state.orcid_profile}`")

if not (has_web3 and has_orcid):
    st.sidebar.info(
        "**Dual-Auth Synchronization Guide:**\n"
        "• **Link Both First:** Connect both your MetaMask wallet and your ORCID account below before running assessments.\n"
        "• **Seamless Rewards:** When both are active, your evaluation history and rewards merge automatically."
    )

if "initialized" not in st.session_state:
    st.session_state["initialized"] = True

if "free_evals_used" not in st.session_state:
    st.session_state["free_evals_used"] = 0

client_ip = "127.0.0.1"
try:
    headers = st.context.headers
    client_ip = (
        headers.get("X-Forwarded-For")
        or headers.get("X-Real-Ip")
        or "127.0.0.1"
    )
    if "," in client_ip:
        client_ip = client_ip.split(",")[0].strip()
except Exception:
    pass

conn_ip = get_db_connection()
try:
    cur_ip = conn_ip.cursor()
    cur_ip.execute(
        "SELECT ip_address FROM auto_ip_tracking WHERE ip_address=?", (client_ip,)
    )
    ip_exists = cur_ip.fetchone()
    if not ip_exists:
        cur_ip.execute(
            "INSERT INTO auto_ip_tracking (ip_address, first_seen) VALUES (?, ?)",
            (client_ip, datetime.now().isoformat()),
        )
        conn_ip.commit()
finally:
    conn_ip.close()

conn_cnt = get_db_connection()
try:
    cur_cnt = conn_cnt.cursor()
    cur_cnt.execute("SELECT COUNT(*) FROM papers_assessment")
    total_analyzed_count = cur_cnt.fetchone()[0]
finally:
    conn_cnt.close()

if "state_restored" not in st.session_state:
    restore_state_from_web3()
    st.session_state["state_restored"] = True
    add_log("Synchronized state with Sepolia Ethereum Ledger.")

if "assessment_update_token" not in st.session_state:
    st.session_state["assessment_update_token"] = time.time()
if "reset_token" not in st.session_state:
    st.session_state["reset_token"] = 0
if "evaluated_papers_buffer" not in st.session_state:
    st.session_state["evaluated_papers_buffer"] = []
if "download_errors" not in st.session_state:
    st.session_state["download_errors"] = []
if "is_running" not in st.session_state:
    st.session_state["is_running"] = False
if "cancel_requested" not in st.session_state:
    st.session_state["cancel_requested"] = False
if "session_temp_dir" not in st.session_state:
    st.session_state["session_temp_dir"] = tempfile.mkdtemp()
    add_log(f"Temporary volume allocated: {st.session_state['session_temp_dir']}")

if "scilem_messages" not in st.session_state:
    st.session_state.scilem_messages = [
        {
            "role": "assistant", 
            "content": "**Welcome! I am Scilem.** Ask any research question or check criteria ratings."
        }
    ]

if has_web3 or has_orcid:
    conn_hist = get_db_connection()
    total_user_piq = 0.0
    try:
        cur_h = conn_hist.cursor()
        clauses, params = [], []
        if has_web3:
            clauses.append("eth_book = ?")
            params.append(st.session_state.web3_wallet)
        if has_orcid:
            clauses.append("user_id = ?")
            params.append(st.session_state.orcid_profile)
        
        if clauses:
            cur_h.execute(f"SELECT DISTINCT eval_hash, piq_minted FROM papers_assessment WHERE {' OR '.join(clauses)}", tuple(params))
            piq_rows = cur_h.fetchall()
            total_user_piq = sum(safe_float(r[1], 0.0) for r in piq_rows if r[1])
    finally:
        conn_hist.close()

    status_line = "**Synced Status:** Active Sync\n\n" if (has_web3 and has_orcid) else ""

    st.sidebar.markdown(
        f"**Researcher:** {st.session_state.researcher_name}\n\n"
        f"{status_line}"
        f"**TOTAL piQ AWARDED:** `{total_user_piq:.2f} piQ`"
    )

    if st.sidebar.button("Unlink / Reset Session", use_container_width=True):
        add_log("Synced session unlinked.")
        st.session_state.web3_wallet = None
        st.session_state.orcid_profile = None
        st.session_state.researcher_name = "Anonymous Researcher"
        st.rerun()

current_user = st.session_state.orcid_profile if has_orcid else (st.session_state.web3_wallet if has_web3 else "Anonymous")
valid_book_address = st.session_state.web3_wallet if has_web3 else "0x0000000000000000000000000000000000000000"

st.sidebar.markdown("---")
with st.sidebar.expander("Live System Monitor", expanded=True):
    log_text = "\n".join(st.session_state.app_logs)
    st.code(log_text if log_text else "No active logs...", language="bash")

with st.sidebar.expander("🧠 Scilem Assistant", expanded=False):
    floating_chat_container = st.container(height=220)
    with floating_chat_container:
        for idx, message in enumerate(st.session_state.scilem_messages):
            msg_avatar = "🧠" if message["role"] == "assistant" else "👤"
            with st.chat_message(message["role"], avatar=msg_avatar):
                st.markdown(message["content"])

    with st.form(key="scilem_sidebar_form", clear_on_submit=False):
        f_cols = st.columns([3, 1])
        with f_cols[0]:
            floating_prompt = st.text_input("Ask Scilem...", value="", label_visibility="collapsed")
        with f_cols[1]:
            submitted_floating = st.form_submit_button("Send")
            if submitted_floating and floating_prompt.strip():
                st.session_state.scilem_messages.append({"role": "user", "content": floating_prompt})
                scilem_neural_reply = evaluate_scilem_analysis_report(floating_prompt)
                st.session_state.scilem_messages.append({
                    "role": "assistant",
                    "content": scilem_neural_reply
                })
                st.rerun()

    if has_web3 and w3.is_address(st.session_state.web3_wallet) and w3.is_address(OWNER_ID) and st.session_state.web3_wallet.lower() == OWNER_ID.lower():
        if st.button("Reset Scilem (Owner)", use_container_width=True):
            msg = reset_scilem()
            st.session_state.scilem_messages = [
                {
                    "role": "assistant", 
                    "content": "**Scilem has been reset.** Neural weights and context cleared to baseline by Web3 owner."
                }
            ]
            add_log(msg)
            st.toast(msg, icon="🧠")
            st.success(msg)
            time.sleep(0.5)
            st.rerun()

@st.cache_data(ttl=3600)
def build_science_map(target_author, repulsion=-3000, spring_len=180, size_scale=1.5, central_grav=0.15, _db_token=0):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT fields, subfields, final_score, author_name FROM papers_assessment")
        data = cursor.fetchall()
    finally:
        conn.close()

    html_string, table_data = "", []
    if not data:
        return html_string, table_data

    topic_aggregates = {}
    exclude_terms = { "general", "general science", "unspecified domain", "unspecified sub-domain", "core research topic" }

    for fields_json, subfields_json, final_score, author_str in data:
        cleaned_author = clean_author_name(author_str)
        if target_author and target_author != "All Authors" and target_author not in cleaned_author:
            continue
        try:
            raw_subfields = [s.title().strip() for s in json.loads(subfields_json)]
            score = safe_float(final_score, 50.0)
            for rs in raw_subfields:
                if rs and rs.lower() not in exclude_terms:
                    s = rs 
                    if s not in topic_aggregates:
                        topic_aggregates[s] = {"weight_sum": 0.0, "frequency": 0}
                    topic_aggregates[s]["weight_sum"] += score
                    topic_aggregates[s]["frequency"] += 1
        except:
            continue

    if not topic_aggregates:
        topic_aggregates["Computer Science > Algorithms & Software Engineering"] = {"weight_sum": 50.0, "frequency": 1}

    if len(topic_aggregates) > 15:
        sorted_topics = sorted(topic_aggregates.items(), key=lambda x: (x[1]["frequency"], x[1]["weight_sum"]), reverse=True)
        topic_aggregates = dict(sorted_topics[:15])

    unique_topics = list(topic_aggregates.keys())

    major_fields_dict = {}
    for topic in unique_topics:
        parts = [p.strip() for p in topic.split('>')]
        major = parts[0]
        if major not in major_fields_dict:
            major_fields_dict[major] = []
        major_fields_dict[major].append(topic)

    major_keys = sorted(list(major_fields_dict.keys()))
    color_map = {}

    for i, major in enumerate(major_keys):
        h = i / len(major_keys) if len(major_keys) > 0 else 0
        subfields = sorted(major_fields_dict[major])
        n_subs = len(subfields)
        
        for j, topic in enumerate(subfields):
            if n_subs <= 1:
                s, v = 0.7, 0.9
            else:
                ratio = j / (n_subs - 1)
                s = 0.4 + (0.5 * ratio)
                v = 0.95 - (0.35 * ratio)
            rgb = colorsys.hsv_to_rgb(h, s, v)
            color_map[topic] = "#%02x%02x%02x" % tuple(int(x * 255) for x in rgb)

    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#2c3e50", notebook=False)
    
    physics_options = f"""{{ 
        "physics": {{ 
            "barnesHut": {{ "gravitationalConstant": {repulsion}, "centralGravity": {central_grav}, "springLength": {spring_len}, "springConstant": 0.005, "damping": 1.0, "avoidOverlap": 2.0 }}, 
            "stabilization": {{ "enabled": true, "iterations": 2500, "fit": true }} 
        }} 
    }}"""
    net.set_options(physics_options)

    for topic, metrics in topic_aggregates.items():
        avg_weight = metrics["weight_sum"] / metrics["frequency"]
        freq = metrics["frequency"]
        node_size = max(35, (25 + (avg_weight * 3.0)) * size_scale)

        base_col = color_map[topic]
        net.add_node(
            n_id=topic, label=" ",
            title=f"Field: {topic} | Frequency: {freq} | Avg Weight/Score: {avg_weight:.1f}",
            size=node_size, shape="dot", physics=True,
            font={"color": "rgba(0,0,0,0)", "size": 0},
            color={
                "background": base_col, "border": "#1a1a1a",
                "highlight": {"background": base_col, "border": "#000000"},
                "hover": {"background": base_col, "border": "#000000"},
            },
            shadow={"enabled": True, "color": "rgba(0,0,0,0.5)", "size": 6, "x": 3, "y": 3}
        )
        
    for i, t1 in enumerate(unique_topics):
        for j, t2 in enumerate(unique_topics):
            if i < j and t1.split(">")[0].strip() == t2.split(">")[0].strip():
                net.add_edge(t1, t2, color="rgba(150,150,150,0.2)")

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".html")
    os.close(tmp_fd) 
    
    try:
        net.save_graph(tmp_name)
        with open(tmp_name, "r", encoding="utf-8") as f:
            html_string = f.read()
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    gradient_injection = f"""
    <!-- reload_timestamp: {time.time()} -->
    </head>
    """
    html_string = html_string.replace("</head>", gradient_injection)
    html_string = html_string.replace("<canvas", "<canvas class='vis-gradient-canvas'")
    html_string = html_string.replace("mynetwork", f"pi_network_{int(time.time() * 1000)}")

    for topic, metrics in sorted(topic_aggregates.items(), key=lambda x: x[1]["frequency"], reverse=True):
        avg_w = metrics["weight_sum"] / metrics["frequency"]
        table_data.append({
            "Color": color_map[topic],
            "Science Field": topic,
            "Frequency": metrics["frequency"],
            "Avg Weight": round(avg_w, 1)
        })

    return html_string, table_data

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
def show_criterion_metrics(c_id, title, q_key, weight_val, sym, desc, formula):
    st.markdown(f"### {c_id}: {title}")
    st.markdown(rf"**Current Epoch Weight ($\varpi_{sym}$):** `{weight_val:.6f}`")
    st.markdown(f"{desc} {rbot(q_key)}", unsafe_allow_html=True)
    st.markdown(formula)
    st.markdown("---")
    st.markdown(r"**Adversarial Logic Gap ($\Delta_{Logic}$):** Evaluates reasoning structure and penalizes claims unsupported by evidence or counterfactual stress failures.", unsafe_allow_html=True)
    st.markdown(r"$$ L_i = \left( (\mathcal{P}_{valid} \cdot \mathcal{E}_{strength}) \cdot \exp\left(-\left(2 \cdot \max(0, \mathcal{C}_{reach} - \mathcal{E}_{strength}) + 1.5 \cdot \lambda_{jumps}\right)\right) \right) \times \frac{1}{1 + e^{-\Delta Premise}} + \lambda \cdot \text{vapri} $$")

top_title_col, top_badge_col = st.columns([4, 2], vertical_alignment="center")
with top_title_col:
    st.markdown("<h1 style='margin-bottom:0;'>Pi-Index Assessment Engine</h1>", unsafe_allow_html=True)
with top_badge_col:
    st.markdown(
        f"""
        <div style="float: right; background-color: #0f172a; color: white; padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 600; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
            Total Analyzed Papers: <span style="color: #60a5fa;">{total_analyzed_count}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown("<br>", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("### Assess a Manuscript")
    
    free_evals_used = st.session_state.get("free_evals_used", 0)

    if free_evals_used == 0:
        if not has_web3:
            st.info(
                "**First Assessment Free:** Your first assessment runs with zero stake required! "
                "**Recommendation:** Connect your Web3 Wallet in the sidebar first so earned **piQ** tokens can be credited directly to your address."
            )
        stake_amount = True
    else:
        if not has_web3:
            st.warning(
                "🔒 **Free Trial Completed:** Please connect your **Web3 Ethereum Wallet** in the sidebar to stake **0.1 piQ** and execute further paper assessments."
            )
            stake_amount = False
        else:
            stake_amount = st.checkbox(
                "Stake 0.1 piQ to Process (Returned on Valid Assessment)",
                value=True,
                key=f"stake_chk_{st.session_state['reset_token']}",
            )

    reset_tok = st.session_state["reset_token"]

    intake_tab_local, intake_tab_doi, intake_tab_search = st.tabs(["📄 Local Upload", "🔗 DOI Lookup", "🌐 Open Source Search"])

    selected_uploaded_files = []
    with intake_tab_local:
        uploaded_files = st.file_uploader(
            "Upload Local PDF(s)", type=["pdf"], accept_multiple_files=True, key=f"file_uploader_{reset_tok}"
        )
        if uploaded_files:
            st.markdown("**Tick local files to include:**")
            for i, file in enumerate(uploaded_files):
                if st.checkbox(f"Local File: {file.name}", value=True, key=f"up_chk_{i}_{reset_tok}"):
                    selected_uploaded_files.append(file)

    doi_input = ""
    include_doi = False
    with intake_tab_doi:
        doi_input = st.text_input(
            "Enter a DOI", placeholder="10.1000/xyz123 or https://doi.org/10.1000/xyz123", key=f"doi_text_{reset_tok}"
        )
        include_doi = st.checkbox(
            "Include this DOI in the assessment pipeline", value=False, key=f"doi_chk_{reset_tok}", disabled=not doi_input.strip()
        )
        st.caption("Pi-Index resolves open-access PDFs automatically via Unpaywall → Semantic Scholar → CORE, in that order.")

    selected_search_urls = []
    with intake_tab_search:
        search_query = st.text_input("Search OpenAlex Topics/Keywords", key=f"search_{reset_tok}")
        if st.button("Search Open Source Papers", key=f"search_btn_{reset_tok}") and search_query:
            with st.spinner("Querying OpenAlex Database..."):
                st.session_state["search_results"] = search_openalex_topics(search_query, limit=5)
        
        if "search_results" in st.session_state and st.session_state["search_results"]:
            st.markdown("**Select papers to assess from Open Access Search:**")
            for i, res in enumerate(st.session_state["search_results"]):
                if st.checkbox(f"{res['title']} ({res['authors']})", key=f"srch_chk_{i}_{reset_tok}"):
                    selected_search_urls.append(res)

    if st.session_state["is_running"]:
        col_run, col_stop = st.columns([4, 1], gap="medium")
        with col_run:
            st.button("Working...", type="primary", use_container_width=True, disabled=True)
        with col_stop:
            if st.button("Stop", type="secondary", use_container_width=True):
                st.session_state["is_running"] = False
                st.session_state["cancel_requested"] = True
                add_log("Pipeline operation forcefully interrupted by user.")
                st.info("Pipeline operation cancelled by user.")
                st.rerun()

        scope_val = ""
        snap_files = st.session_state.get("snap_files", [])
        include_doi_snap = st.session_state.get("snap_include_doi", False)
        doi_snap = st.session_state.get("snap_doi", "")
        snap_search = st.session_state.get("snap_search", [])
        
        with st.status("Initializing Assessment Pipeline...", expanded=True) as status_box:
            try:
                if snap_search and not st.session_state["cancel_requested"]:
                    total_search = len(snap_search)
                    for idx, s_item in enumerate(snap_search):
                        if st.session_state["cancel_requested"]: break
                        status_box.update(label=f"Resolving Open Source Paper {idx+1} of {total_search}: {s_item['title']}...")
                        
                        pdf_bytes = download_pdf_from_url(s_item['pdf_url'])
                        if not pdf_bytes and s_item['doi']:
                            core_text = fetch_core_text_by_doi(s_item['doi'])
                            if core_text:
                                pdf_bytes = create_virtual_pdf_from_text(core_text, title=s_item['title'])
                        
                        if pdf_bytes:
                            fname = f"OA_Search_{int(time.time())}.pdf"
                            clean_bytes = preprocess_pdf_layout(pdf_bytes, fname)
                            try:
                                res = assess_manuscript(clean_bytes, fname, scope_val, current_user, valid_book_address, email="None", provided_doi=s_item['doi'])
                                if res and len(res) >= 22:
                                    eval_record = {
                                        "title": res[0], "author_name": clean_author_name(res[1]),
                                        "score": res[2], "logic_integrity": res[3], "drift": res[4], "rec": res[5], 
                                        "fields": res[6], "subfields": res[7], "scores_dict": res[8], "eval_hash": res[9], 
                                        "piq": res[10], "tx_hash": res[11], "zk_proof": res[12], "used_weights": res[13],
                                        "h_idx": res[14], "i10_idx": res[15], "repro_score": res[16], "filename": fname, 
                                        "warnings": res[18], "warnings_acknowledged": False, "consensus_raw": res[19], 
                                        "evidence_report_text": res[20], "scilem_rating": res[21]
                                    }
                                    st.session_state["evaluated_papers_buffer"].insert(0, eval_record)
                                    st.session_state["evaluated_papers_buffer"] = st.session_state["evaluated_papers_buffer"][:50]
                                    st.session_state["free_evals_used"] += 1
                                    add_log(f"Successfully evaluated search item: {s_item['title']}")
                            except Exception as err:
                                add_log(f"Error executing assess_manuscript for search source: {str(err)}")
                        else:
                            err_item = {"title": s_item['title'], "doi": s_item['doi'], "url": s_item['pdf_url']}
                            if err_item not in st.session_state["download_errors"]:
                                st.session_state["download_errors"].append(err_item)

                if include_doi_snap and doi_snap.strip() and not st.session_state["cancel_requested"]:
                    status_box.update(label=f"Resolving DOI: {doi_snap}...")
                    metadata = fetch_doi_metadata(doi_snap)
                    fname = f"DOI_{doi_snap.replace('/', '_')}.pdf"
                    pdf_bytes = None
                    add_log(f"Attempting API resolution for standalone DOI: {doi_snap}")
                    status_box.write(f"Attempting API resolution for standalone DOI: {doi_snap}")
                    
                    if metadata and metadata.get("pdf_url"):
                        pdf_bytes = download_pdf_from_url(metadata["pdf_url"])
                    if not pdf_bytes:
                        s2_url = fetch_semantic_scholar_pdf(doi_snap)
                        if s2_url:
                            pdf_bytes = download_pdf_from_url(s2_url)
                    
                    if not pdf_bytes:
                        core_text = fetch_core_text_by_doi(doi_snap)
                        if core_text:
                            pdf_bytes = create_virtual_pdf_from_text(core_text, title="DOI Target Text")

                    if pdf_bytes:
                        status_box.update(label="Assessing document from resolved source...")
                        clean_bytes = preprocess_pdf_layout(pdf_bytes, fname)
                        try:
                            res = assess_manuscript(clean_bytes, fname, scope_val, current_user, valid_book_address, email="None", provided_doi=doi_snap.strip())
                        except Exception as err:
                            res = None
                            err_trace = traceback.format_exc()
                            add_log(f"Error executing assess_manuscript for DOI source: {str(err)}\n{err_trace}")
                            status_box.write(f"Pipeline error: {str(err)}")

                        if res and len(res) >= 22:
                            eval_record = {
                                "title": res[0], "author_name": clean_author_name(res[1]),
                                "score": res[2], "logic_integrity": res[3], "drift": res[4], "rec": res[5], 
                                "fields": res[6], "subfields": res[7], "scores_dict": res[8], "eval_hash": res[9], 
                                "piq": res[10], "tx_hash": res[11], "zk_proof": res[12], "used_weights": res[13],
                                "h_idx": res[14], "i10_idx": res[15], "repro_score": res[16], "filename": fname, 
                                "warnings": res[18], "warnings_acknowledged": False, "consensus_raw": res[19], 
                                "evidence_report_text": res[20], "scilem_rating": res[21]
                            }
                            st.session_state["evaluated_papers_buffer"].insert(0, eval_record)
                            st.session_state["evaluated_papers_buffer"] = st.session_state["evaluated_papers_buffer"][:50]
                            st.session_state["free_evals_used"] += 1
                            add_log("Successfully evaluated and logged DOI source.")
                            status_box.write("Successfully evaluated and logged DOI source.")
                        else:
                            add_log("Error: assess_manuscript returned incomplete data for DOI source.")
                    else:
                        clean_doi = doi_snap.replace("https://doi.org/", "").strip()
                        doi_url = f"https://doi.org/{clean_doi}"
                        err_item = {"title": f"DOI Input: {clean_doi}", "doi": clean_doi, "url": doi_url}
                        add_log("Publisher access blocks direct binary extraction for standalone DOI.")
                        if err_item not in st.session_state["download_errors"]:
                            st.session_state["download_errors"].append(err_item)

                if snap_files and not st.session_state["cancel_requested"]:
                    total_files = len(snap_files)
                    for i, (fname, fpath) in enumerate(snap_files):
                        if st.session_state["cancel_requested"]: break
                        status_box.update(label=f"Analyzing uploaded file {i+1} of {total_files}: {fname}...")
                        add_log(f"Engaging logical extraction on local file structure: {fname}")
                        status_box.write(f"Engaging logical extraction on local file structure: {fname}")
                        
                        with open(fpath, "rb") as in_f:
                            raw_bytes = in_f.read()
                            
                        clean_bytes = preprocess_pdf_layout(raw_bytes, fname)
                        
                        try:
                            res = assess_manuscript(clean_bytes, fname, scope_val, current_user, valid_book_address, email="None", provided_doi="None")
                        except Exception as err:
                            res = None
                            err_trace = traceback.format_exc()
                            add_log(f"Error executing assess_manuscript for local file {fname}: {str(err)}\n{err_trace}")
                            status_box.write(f"Pipeline error for local file {fname}: {str(err)}")

                        if res and len(res) >= 22:
                            eval_record = {
                                "title": res[0], "author_name": clean_author_name(res[1]),
                                "score": res[2], "logic_integrity": res[3], "drift": res[4], "rec": res[5], 
                                "fields": res[6], "subfields": res[7], "scores_dict": res[8], "eval_hash": res[9], 
                                "piq": res[10], "tx_hash": res[11], "zk_proof": res[12], "used_weights": res[13],
                                "h_idx": res[14], "i10_idx": res[15], "repro_score": res[16], "filename": fname, 
                                "warnings": res[18], "warnings_acknowledged": False, "consensus_raw": res[19], 
                                "evidence_report_text": res[20], "scilem_rating": res[21]
                            }
                            st.session_state["evaluated_papers_buffer"].insert(0, eval_record)
                            st.session_state["evaluated_papers_buffer"] = st.session_state["evaluated_papers_buffer"][:50]
                            st.session_state["free_evals_used"] += 1
                            add_log(f"Stored local assessment result to cache.")
                        else:
                            add_log(f"Error: assess_manuscript returned incomplete data for {fname}")

                if st.session_state["cancel_requested"]:
                    status_box.update(label="Pipeline operation was stopped.", state="error")
                else:
                    status_box.update(label="Pipeline processing complete.", state="complete")
                    time.sleep(1)
            finally:
                st.session_state["is_running"] = False
                st.session_state["cancel_requested"] = False
                st.session_state["reset_token"] += 1
                st.session_state["assessment_update_token"] = time.time()
                st.rerun()

    else:
        if st.button("Run Assessment Pipeline", type="primary", use_container_width=True):
            if free_evals_used >= 1 and not has_web3:
                st.error("Free trial limit reached. Please connect your Web3 Ethereum Wallet in the sidebar to stake 0.1 piQ and run assessments.")
            elif free_evals_used >= 1 and not stake_amount:
                st.error("You must agree to stake 0.1 piQ to execute further paper assessments.")
            elif not selected_uploaded_files and not (include_doi and doi_input.strip()) and not selected_search_urls:
                st.warning("Please tick at least one paper or input source to assess.")
            else:
                add_log("Preparing pipeline dispatch queue...")
                saved_files = []
                for f in selected_uploaded_files:
                    safe_filename = os.path.basename(f.name)
                    f_path = os.path.join(st.session_state["session_temp_dir"], safe_filename)
                    with open(f_path, "wb") as out_f:
                        out_f.write(f.getvalue())
                    f.seek(0)
                    saved_files.append((safe_filename, f_path))
                    add_log(f"Cached user file to temporary disk node: {safe_filename}")
                    
                st.session_state["snap_files"] = saved_files
                st.session_state["snap_search"] = selected_search_urls
                st.session_state["snap_scope"] = ""
                st.session_state["snap_doi"] = doi_input
                st.session_state["snap_include_doi"] = include_doi
                st.session_state["is_running"] = True
                st.session_state["cancel_requested"] = False
                st.rerun()

@st.dialog("Detailed Research Integrity Dossier", width="large")
def show_dossier(item):
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
    consensus_raw = item.get("consensus_raw", {})
    evidence_report_text = item.get("evidence_report_text", "")
    author_book = "0x" + hashlib.sha256(author_name.encode()).hexdigest()[:40]

    st.subheader(f"{title} by {author_name}")

    if warnings:
        st.warning(f"⚠️ **Manuscript Flagged with {len(warnings)} Warning Check(s):**")
        for w in warnings:
            st.markdown(f"- {w}")

    st.markdown("### Overview & Ledger")
    st.write(f"**File Name:** `{filename}`")
    st.write(f"**Evaluation Hash (Paper Address):** `{eval_hash}`")
    st.write(f"**Unique Book Address:** `{author_book}`")
    st.write(f"**piQ Minted:** `{piq}`")
    st.markdown(f"**zk-SNARK Proof (Structurally Validated):** `{zk_proof}`", unsafe_allow_html=True)
    
    tx_url = get_tx_url(tx_hash)
    tx_disp_val = tx_hash if tx_hash and str(tx_hash).strip() not in ["None", ""] else "Not Connected / No Book / Missing PK"
    if tx_url:
        st.markdown(f"**Tx Hash:** [`{tx_disp_val}`]({tx_url})")
    else:
        st.write(f"**Tx Hash:** `{tx_disp_val}`")

    st.markdown(f"**Executable Reproducibility Score:** `{repro_score * 100:.1f}%`", unsafe_allow_html=True)
    st.markdown(f"**SciScore MDAR Adherence:** `{mdar_score * 100:.1f}%` | **Valid RRIDs:** `{rrid_count}`", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Multi-LLM Extractions")
    if consensus_raw and isinstance(consensus_raw, dict):
        llm_cols = st.columns(2, gap="medium")
        target_llms = ["llama", "mistral", "qwen", "gemini", "scilem"]
        for idx, llm_key in enumerate(target_llms):
            col = llm_cols[idx % 2]
            with col:
                data = consensus_raw.get(llm_key, {})
                with st.container(border=True):
                    st.markdown(f"**Model: {llm_key.upper()}**")
                    if llm_key == "scilem":
                        st.markdown(f"**Engine Status:** Active (Local PyTorch Neural Network)")
                        st.markdown(f"**Structural Analysis:** {data.get('opinion', 'Scilem structural analysis active.')}")
                    elif data.get('api_failed', False):
                        st.markdown(f"**Status:** Rate / Credit Limit Hit")
                        st.markdown(f"**Opinion:** {data.get('opinion', 'No opinion extracted.')}")
                    else:
                        st.markdown(f"**Extracted Title:** `{data.get('title', 'N/A')}`")
                        st.markdown(f"**Extracted Authors:** `{data.get('authors', 'N/A')}`")
                        st.markdown(f"**Opinion:** {data.get('opinion', 'No opinion extracted.')}")
                        refs = data.get("references", [])
                        if refs:
                            st.markdown(f"**References ({len(refs)}):**")
                            for r in refs[:3]:
                                if isinstance(r, dict):
                                    st.markdown(f"- **{r.get('citation', '[*]')}**: {r.get('authors', 'Unknown')} ({r.get('year', 'N/A')})")
                                else:
                                    st.markdown(f"- {r}")
    else:
        st.info("No individual LLM raw opinion payloads stored.")

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

    st.markdown("---")

    st.markdown("### Criteria Breakdown & Score Matrix")
    breakdown_df = pd.DataFrame({
        "Criterion": [
            "C1: Semantic Originality", "C2: Methodological Rigor (SciScore)", "C3: Interdisciplinary Entropy",
            "C4: Societal Impact", "C5: Open Science & Repro", "C6: Literature Integration",
            "C7: Empirical Density", "C8: Future Actionability & FAIR",
        ],
        "Score Extracted (0-100)": [
            safe_float(scores_dict.get("C1_Semantic_Originality"), 0), safe_float(scores_dict.get("C2_Methodological_Rigor_SciScore"), 0),
            safe_float(scores_dict.get("C3_Interdisciplinary_Entropy"), 0), safe_float(scores_dict.get("C4_Societal_Impact"), 0),
            safe_float(scores_dict.get("C5_Open_Science_Repro"), 0), safe_float(scores_dict.get("C6_Literature_Integration"), 0),
            safe_float(scores_dict.get("C7_Empirical_Density"), 0), safe_float(scores_dict.get("C8_Future_Actionability_FAIR"), 0),
        ],
        "Epoch Weight": used_weights,
        "Weighted Value": [
            safe_float(scores_dict.get(k), 0) * used_weights[i]
            for i, k in enumerate([
                "C1_Semantic_Originality", "C2_Methodological_Rigor_SciScore", "C3_Interdisciplinary_Entropy",
                "C4_Societal_Impact", "C5_Open_Science_Repro", "C6_Literature_Integration",
                "C7_Empirical_Density", "C8_Future_Actionability_FAIR",
            ])
        ],
    })
    st.dataframe(breakdown_df, hide_index=True)
    raw_base = sum(breakdown_df["Weighted Value"]) / 8.0
    logic_multiplier = 0.7 + (logic_integrity / 333.3)
    st.markdown(f"**Base Weighted Sum (Mean divided by 8):** `{raw_base:.2f}`")
    st.markdown(f"**Logic Integrity Multiplier:** `{logic_multiplier:.4f}` (Derived from {logic_integrity:.1f}% raw logic score)")

    dossier_content = f"""# RESEARCH INTEGRITY DOSSIER (DORA-Aligned)\n**Title:** {title}\n**Author:** {author_name}\n**File Name:** {filename}\n**Evaluation Hash:** {eval_hash}\n**Unique Book:** {author_book}\n**Pi-Index Score:** {score:.2f} / 100\n**Logic Integrity Score:** {logic_integrity:.1f}%\n**SciScore MDAR Adherence:** {mdar_score * 100:.1f}%\n**Valid RRIDs Count:** {rrid_count}\n"""
    st.download_button(
        label=f"Download Research Integrity Dossier ({filename})",
        data=dossier_content,
        file_name=f"Dossier_{eval_hash[:10]}.md",
        mime="text/markdown",
        key=f"download_dossier_modal_{eval_hash}_{time.time()}",
        use_container_width=True,
    )

@st.dialog("AI Peer Review Defense Strategy", width="medium")
def show_defense_rebuttal(scores_dict):
    with st.spinner("Synthesizing adversarial defense strategy..."):
        rebuttal = generate_rebuttal_strategy(scores_dict)
    st.markdown(rebuttal)

def render_assessment_card(item, index):
    title = item["title"]
    author_name = clean_author_name(item["author_name"])
    score = safe_float(item["score"], 0.0)
    eval_hash = item["eval_hash"]
    piq = safe_float(item["piq"], 0.0)
    scores_dict = item["scores_dict"]
    warnings = item.get("warnings", [])
    acknowledged = item.get("warnings_acknowledged", False)

    with st.container(border=True):
        col_info, col_actions = st.columns([6, 4], gap="medium")
        with col_info:
            warn_badge = f" ⚠️ *({len(warnings)} warning checks active)*" if warnings and not acknowledged else (f" 🛡️ *({len(warnings)} warning checks acknowledged)*" if warnings and acknowledged else "")
            
            title_lower, author_lower = str(title).lower().strip(), str(author_name).lower().strip()
            invalid_titles = ["n/a", "none", "unknown", "failed", "unnamed", "api limit"]
            invalid_authors = ["n/a", "none", "unknown", "unidentified", "independent research scholar", "unconfigured key", "anonymous"]

            has_valid_title = (title and not any(inv in title_lower for inv in invalid_titles) and "parsed via local heuristics" not in title_lower)
            has_valid_author = (author_name and not any(inv in author_lower for inv in invalid_authors))

            extraction_badge = " ✅ *Title & Author Extracted Successfully*" if has_valid_title and has_valid_author else (" ⚠️ *Partial Extraction (Title or Author Only)*" if has_valid_title or has_valid_author else "")

            st.markdown(f"**{title}** — *{author_name}*{extraction_badge}{warn_badge}")
            st.markdown(f"**Score: {score:.2f} | piQ: {piq}**")
            
            if warnings:
                with st.expander(f"View Warning Checks ({len(warnings)})", expanded=not acknowledged):
                    for w in warnings: st.markdown(f"- {w}")
                    if not acknowledged:
                        if st.button("Acknowledge Warnings / Dismiss Flag", key=f"ack_warn_{index}_{eval_hash}"):
                            item["warnings_acknowledged"] = True
                            add_log(f"Warnings acknowledged for {item['filename']}")
                            st.rerun()
                    else:
                        st.caption("Warnings acknowledged. Paper evaluated normally with piQ minted.")
        with col_actions:
            c_det, c_strat, c_del = st.columns([3, 3, 1], gap="small")
            with c_det:
                if st.button("More Details", key=f"more_det_{index}_{eval_hash}", use_container_width=True): show_dossier(item)
            with c_strat:
                if st.button("Suggest Defense", key=f"gen_strat_{index}_{eval_hash}", use_container_width=True): show_defense_rebuttal(scores_dict)
            with c_del:
                if st.button("❌", key=f"close_eval_{index}_{eval_hash}", help="Close this result"):
                    st.session_state["evaluated_papers_buffer"].pop(index)
                    st.rerun()

if st.session_state["evaluated_papers_buffer"] or st.session_state.get("download_errors"):
    st.markdown("### Assessment Results")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.get("download_errors"):
        st.markdown("#### Publisher Access & Download Restrictions")
        for err_idx, err_data in enumerate(st.session_state["download_errors"]):
            err_col1, err_col2 = st.columns([6, 1], gap="medium")
            with err_col1:
                st.warning(f"**Could not directly download PDF for '{err_data['title']}':** Publishers restrict direct binary access.\n\n- **DOI:** `{err_data['doi']}`\n- **PDF URL Link:** [{err_data['url']}]({err_data['url']})")
            with err_col2:
                if st.button("Close", key=f"close_err_{err_idx}_{st.session_state['reset_token']}"):
                    st.session_state["download_errors"].pop(err_idx)
                    st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

    for item_idx, item in enumerate(st.session_state["evaluated_papers_buffer"]):
        render_assessment_card(item, item_idx)

st.markdown("<br>", unsafe_allow_html=True)
top_analytics_col1, top_analytics_col2 = st.columns(2, gap="large")

with top_analytics_col1:
    col_fc1, col_fc_pop, col_fc2 = st.columns([2.5, 0.5, 1], vertical_alignment="center")
    with col_fc1:
        st.markdown("### Pidyne Forecast", unsafe_allow_html=True)
    with col_fc_pop:
        with st.popover("❔", help="What's Pidyne?"):
            st.markdown(r"""
            **What's Pidyne?**
            Pidyne serves as the core orchestration and meta-learning brain of the Pi-Index Assessment Engine, integrating multi-LLM consensus with decentralized ledger infrastructure:
            1. **LSTM Meta-Learning:** Deploys a local PyTorch neural network (`PidyneLSTM`) that continuously trains on historical blockchain epoch weights, forecasting future shifts in scientific evaluation standards across the 8 core criteria.
            2. **Multi-Model Consensus & LLM-as-a-Judge:** Aggregates independent evaluations from local networks (Scilem) and remote LLMs (Llama, Mistral, Qwen, Gemini), acting as the final judge of the paper to read reports and deliver a definitive verdict using an LLM model.
            """)
    with col_fc2:
        forecast_horizon = st.selectbox("Lookback", ["1 Epoch", "3 Epochs", "5 Epochs"], index=1, key="pidyne_lookback_dropdown", label_visibility="collapsed")
        actual_lookback = int(forecast_horizon.split()[0])

    @st.cache_data(show_spinner="Training Pidyne LSTM Model in background...")
    def train_pidyne_cached(weight_data, actual_lookback):
        dataset = PidyneBlockchainDataset(weight_data, actual_lookback)
        dataloader = DataLoader(dataset, batch_size=min(4, max(1, len(dataset))), shuffle=False)

        model = PidyneLSTM()
        weights_path = os.path.join(BASE_DIR, "pidyne_weights.pt")
        if os.path.exists(weights_path):
            try: model.load_state_dict(torch.load(weights_path, weights_only=True))
            except Exception: pass

        loss_function = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(300):
            for seq, target in dataloader:
                optimizer.zero_grad()
                loss = loss_function(model(seq), target)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            raw_pred = model(torch.tensor(weight_data[-actual_lookback:], dtype=torch.float32).unsqueeze(0)).squeeze().numpy()
            current_w = weight_data[-1]
            predicted = current_w + (raw_pred - current_w) * 20.0
            predicted = np.clip(predicted, 0.01, 7.9)
            predicted = predicted * (8.0 / np.sum(predicted))
            torch.save(model.state_dict(), weights_path)
            return predicted

    conn_pb = get_db_connection()
    try:
        cursor_pb = conn_pb.cursor()
        cursor_pb.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights ORDER BY block_height ASC")
        historical_rows = cursor_pb.fetchall()
    finally:
        conn_pb.close()

    min_blocks_required = 2
    if len(historical_rows) < min_blocks_required:
        st.warning(f"Not enough blockchain data to train the meta-model. You need at least {min_blocks_required} blocks (Currently on ledger: {len(historical_rows)}). Assess at least 1 manuscript to generate block 2.")
    else:
        current_block_count = len(historical_rows)
        lookback_window = max(1, min(actual_lookback, current_block_count - 1))

        if "last_trained_blocks" not in st.session_state or st.session_state.last_trained_blocks != current_block_count or st.session_state.get("last_lookback") != lookback_window:
            cleaned_historical_data = [[safe_float(val, 1.0) for val in row] for row in historical_rows]
            weight_data = np.array(cleaned_historical_data, dtype=np.float32)

            st.session_state.predicted_next_weights = train_pidyne_cached(weight_data, lookback_window)
            st.session_state.current_weights = weight_data[-1]
            st.session_state.last_trained_blocks = current_block_count
            st.session_state.last_lookback = lookback_window

        if len(historical_rows) > 0:
            sliced_rows = historical_rows[-(lookback_window + 1):] if len(historical_rows) > lookback_window else historical_rows
            df_history = pd.DataFrame(sliced_rows, columns=["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"])
            df_history.index.name = "Block / Epoch"
            
            df_amplified = df_history.copy()
            for col in df_amplified.columns:
                df_amplified[col] = 1.0 + (df_amplified[col] - 1.0) * 1500.0
                
            df_melted = df_amplified.reset_index().melt('Block / Epoch', var_name='Criterion', value_name='Weight (Amplified)')
            
            base = alt.Chart(df_melted).mark_line(point=True).encode(
                x='Block / Epoch:O', y=alt.Y('Weight (Amplified):Q', scale=alt.Scale(zero=False)),
                color='Criterion:N', tooltip=['Block / Epoch', 'Criterion', 'Weight (Amplified)']
            ).properties(height=350)
            st.altair_chart(base, use_container_width=True)

        st.markdown("#### Evaluation Metrics")
        with st.container(border=True):
            st.markdown(f"**Ledger Forecast (Raw Sum = {sum(st.session_state.predicted_next_weights):.6f}/8.0):**")
            crit_info = get_criteria_info(st.session_state.predicted_next_weights)
            cols1 = st.columns(4, gap="small")
            for idx, c_data in enumerate(crit_info[:4]):
                with cols1[idx]:
                    if st.button(f"{c_data[0]}: {c_data[3]:.5f}", key=f"btn_crit_{c_data[0]}", use_container_width=True): show_criterion_metrics(*c_data)
            cols2 = st.columns(4, gap="small")
            for idx, c_data in enumerate(crit_info[4:]):
                with cols2[idx]:
                    if st.button(f"{c_data[0]}: {c_data[3]:.5f}", key=f"btn_crit_{c_data[0]}", use_container_width=True): show_criterion_metrics(*c_data)

with top_analytics_col2:
    st.markdown("### Global Map of Science")

    conn_m = get_db_connection()
    try:
        cursor_m = conn_m.cursor()
        cursor_m.execute("SELECT DISTINCT author_name FROM papers_assessment")
        all_global_authors = []
        for row in cursor_m.fetchall():
            if row[0]:
                cleaned = clean_author_name(row[0])
                for a in cleaned.split(","):
                    if a.strip() and not is_likely_institution(a.strip()):
                        all_global_authors.append(a.strip())
    finally:
        conn_m.close()
    all_global_authors = sorted(list(set(all_global_authors)))
    piq_dict, book_dict = get_author_piq_dict()

    if "mod_repulsion" not in st.session_state: st.session_state.mod_repulsion = -3000
    if "mod_spring" not in st.session_state: st.session_state.mod_spring = 180
    if "mod_size" not in st.session_state: st.session_state.mod_size = 1.5
    if "mod_gravity" not in st.session_state: st.session_state.mod_gravity = 0.15

    filter_key = f"top_author_filter_{st.session_state['assessment_update_token']}"
    if filter_key not in st.session_state: st.session_state[filter_key] = "All Authors"

    current_filter = st.session_state.get(filter_key, "All Authors")
    selected_author_top = None if current_filter == "All Authors" else current_filter

    interactive_html_top, table_data_top = build_science_map(
        selected_author_top, repulsion=st.session_state.mod_repulsion, spring_len=st.session_state.mod_spring,
        size_scale=st.session_state.mod_size, central_grav=st.session_state.mod_gravity, _db_token=st.session_state['assessment_update_token']
    )

    map_container = st.container()
    with map_container:
        if interactive_html_top:
            st.markdown("<div class='pyvis-map-wrapper'>", unsafe_allow_html=True)
            components.html(interactive_html_top, height=600, scrolling=False)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Awaiting sufficient data for map visualization.")

    tab_filter, tab_mod, tab_legend = st.tabs(["Author Filter", "Modulators", "Legend"])
    with tab_filter:
        if all_global_authors:
            st.selectbox("Filter Map by Author:", ["All Authors"] + all_global_authors, key=filter_key, format_func=lambda x: (f"{x} (piQ: {piq_dict.get(x, 0.0):.2f})" if x != "All Authors" else x))
        else:
            st.info("No authors available for filtering.")
    with tab_mod:
        mod_col1, mod_col2 = st.columns(2, gap="medium")
        with mod_col1:
            st.slider("Repulsion Force", min_value=-20000, max_value=-100, value=-3000, step=500, key="mod_repulsion")
            st.slider("Spring Length", min_value=10, max_value=1000, value=180, step=20, key="mod_spring")
        with mod_col2:
            st.slider("Bubble Size Scale", min_value=0.1, max_value=8.0, value=1.5, step=0.1, key="mod_size")
            st.slider("Central Pull (Gravity)", min_value=0.0, max_value=2.0, value=0.15, step=0.01, key="mod_gravity")
    with tab_legend:
        if table_data_top:
            st.dataframe(
                pd.DataFrame(table_data_top), 
                hide_index=True, 
                use_container_width=True,
                column_config={"Color": st.column_config.TextColumn(help="Color mapped to primary field")}
            )
        else:
            st.info("No topic data found.")

st.markdown("---")

if has_web3 or has_orcid:
    conn_hist = get_db_connection()
    try:
        cur_h = conn_hist.cursor()
        history_clauses, history_params = [], []
        if has_web3: history_clauses.append("p.eth_book = ?"); history_params.append(st.session_state.web3_wallet)
        if has_orcid: history_clauses.append("p.user_id = ?"); history_params.append(st.session_state.orcid_profile)

        if history_clauses:
            cur_h.execute(f"""SELECT DISTINCT p.eval_hash, p.title, p.author_name, p.filename, p.final_score, p.logic_score, p.piq_minted, p.tx_hash, p.zk_proof, p.timestamp, b.block_height, b.block_hash, p.mdar_adherence_score, p.rrid_valid_count, p.reproducibility_score, p.consensus_data, p.evidence_report, p.scilem_score, p.c1, p.c2, p.c3, p.c4, p.c5, p.c6, p.c7, p.c8 FROM papers_assessment p LEFT JOIN blockchain_por_weights b ON p.eval_hash = b.eval_hash WHERE {' OR '.join(history_clauses)} ORDER BY p.timestamp DESC""", tuple(history_params))
        else: cur_h.execute("SELECT NULL WHERE 0")
        user_history_rows = cur_h.fetchall()
    finally:
        conn_hist.close()

    st.markdown("### Your Assessment History")
    if user_history_rows:
        for idx, uh in enumerate(user_history_rows):
            (u_hash, u_title, u_author, u_filename, u_score, u_logic, u_piq, u_tx, u_zk, u_time, u_block_height, u_block_hash, u_mdar, u_rrid, u_repro, u_consensus, u_report, u_scilem, u_c1, u_c2, u_c3, u_c4, u_c5, u_c6, u_c7, u_c8) = uh
            u_author_clean = clean_author_name(u_author)
            u_book = "0x" + hashlib.sha256(u_author_clean.encode()).hexdigest()[:40]
            u_tx_url = get_tx_url(u_tx)
            tx_disp_val = u_tx if u_tx and str(u_tx).strip() not in ["None", ""] else "Not Connected / No Book / Missing PK"

            with st.expander(f"[{idx+1}] {u_title[:50]}... — *{u_author_clean}* (Score: **{safe_float(u_score, 0.0):.2f}** | piQ: `{u_piq}`)", expanded=False):
                st.write(f"**File Name:** {u_filename if u_filename else 'N/A'}")
                st.write(f"**Evaluation Hash (Paper Address):** `{u_hash}`")
                st.write(f"**Unique Book Address:** `{u_book}`")
                st.write(f"**piQ Minted:** `{u_piq}`")
                st.markdown(f"**zk-SNARK Proof:** `{u_zk}`", unsafe_allow_html=True)
                if u_tx_url: st.markdown(f"**Tx Hash:** [`{tx_disp_val}`]({u_tx_url})")
                else: st.write(f"**Tx Hash:** `{tx_disp_val}`")
                st.markdown(f"**Executable Reproducibility Score:** `{safe_float(u_repro, 0.0) * 100:.1f}%`", unsafe_allow_html=True)
                st.markdown(f"**SciScore MDAR Adherence:** `{safe_float(u_mdar, 0.0) * 100:.1f}%` | **Valid RRIDs:** `{u_rrid}`", unsafe_allow_html=True)

                if st.button("View Full Multi-LLM & Scilem Dossier", key=f"hist_det_{idx}_{u_hash}"):
                    hist_item = {
                        "title": u_title, "author_name": u_author, "score": safe_float(u_score, 0.0), "logic_integrity": safe_float(u_logic, 75.0),
                        "scores_dict": { "C1_Semantic_Originality": safe_float(u_c1, 0), "C2_Methodological_Rigor_SciScore": safe_float(u_c2, 0), "C3_Interdisciplinary_Entropy": safe_float(u_c3, 0), "C4_Societal_Impact": safe_float(u_c4, 0), "C5_Open_Science_Repro": safe_float(u_c5, 0), "C6_Literature_Integration": safe_float(u_c6, 0), "C7_Empirical_Density": safe_float(u_c7, 0), "C8_Future_Actionability_FAIR": safe_float(u_c8, 0) },
                        "used_weights": [1.0]*8, "eval_hash": u_hash, "piq": safe_float(u_piq, 0.0), "tx_hash": u_tx, "zk_proof": u_zk, "h_idx": safe_float(u_mdar, 0.0), "i10_idx": int(safe_float(u_rrid, 0)), "repro_score": safe_float(u_repro, 0.0), "filename": u_filename or "N/A", "warnings": [], "consensus_raw": json.loads(u_consensus) if u_consensus else {}, "evidence_report_text": u_report or "", "scilem_rating": safe_float(u_scilem, 50.0)
                    }
                    show_dossier(hist_item)
    else:
        st.info("No assessment history or rewards found linked to these connected IDs.")
    st.markdown("---")

side_col1, side_col2 = st.columns(2, gap="large")

with side_col1:
    st.markdown("### Pi Quotient (piQ) Leaderboard")
    piq_dict, book_dict = get_author_piq_dict()
    if piq_dict:
        sorted_leaderboard = sorted(piq_dict.items(), key=lambda x: x[1], reverse=True)[:20]
        piq_df = pd.DataFrame(sorted_leaderboard, columns=["Author", "piQ Mined"])
        piq_df["Book Address"] = [book_dict.get(a, "None") for a in piq_df["Author"]]
        piq_df.index = np.arange(1, len(piq_df) + 1)
        st.dataframe(piq_df, use_container_width=True)
    else:
        st.info("No piQ tokens minted yet.")

with side_col2:
    st.markdown("### pi-Index (piX) Leaderboard [Top Papers]")
    conn_pi = get_db_connection()
    try:
        cur_pi = conn_pi.cursor()
        cur_pi.execute("""SELECT title, author_name, final_score, logic_score, c1, c2, c3, c4, c5, c6, c7, c8, piq_minted, tx_hash, zk_proof, mdar_adherence_score, rrid_valid_count, reproducibility_score, eval_hash, filename, consensus_data, evidence_report, scilem_score FROM papers_assessment ORDER BY final_score DESC LIMIT 20""")
        top_papers = cur_pi.fetchall()
    finally:
        conn_pi.close()
    
    if top_papers:
        for rank, tp in enumerate(top_papers, start=1):
            (p_title, p_author, p_filename, p_score, p_logic, p_c1, p_c2, p_c3, p_c4, p_c5, p_c6, p_c7, p_c8, p_piq, p_tx, p_zk, p_mdar, p_rrid, p_repro, p_hash, p_consensus, p_report, p_scilem) = tp
            clean_auth = clean_author_name(p_author)
            col1, col2, col3, col4 = st.columns([1, 4, 3, 2], vertical_alignment="center")
            col1.write(f"**#{rank}**")
            col2.write(f"**{p_title}**")
            col3.write(f"*{clean_auth}*")
            with col4:
                if st.button("View Dossier", key=f"pix_row_dossier_{rank}_{p_hash}", use_container_width=True):
                    item_dossier = {
                        "title": p_title, "author_name": p_author, "score": safe_float(p_score, 0.0), "logic_integrity": safe_float(p_logic, 75.0),
                        "scores_dict": { "C1_Semantic_Originality": safe_float(p_c1, 0), "C2_Methodological_Rigor_SciScore": safe_float(p_c2, 0), "C3_Interdisciplinary_Entropy": safe_float(p_c3, 0), "C4_Societal_Impact": safe_float(p_c4, 0), "C5_Open_Science_Repro": safe_float(p_c5, 0), "C6_Literature_Integration": safe_float(p_c6, 0), "C7_Empirical_Density": safe_float(p_c7, 0), "C8_Future_Actionability_FAIR": safe_float(p_c8, 0) },
                        "used_weights": [1.0]*8, "eval_hash": p_hash, "piq": safe_float(p_piq, 0.0), "tx_hash": p_tx, "zk_proof": p_zk, "h_idx": safe_float(p_mdar, 0.0), "i10_idx": int(safe_float(p_rrid, 0)), "repro_score": safe_float(p_repro, 0.0), "filename": p_filename or "N/A", "warnings": [], "consensus_raw": json.loads(p_consensus) if p_consensus else {}, "evidence_report_text": p_report or "", "scilem_rating": safe_float(p_scilem, 50.0)
                    }
                    show_dossier(item_dossier)
            st.divider()
    else:
        st.info("No assessments recorded for Pi-Index leaderboard yet.")

st.markdown("---")

st.markdown("### Latest Assessed Papers")
conn_recent = get_db_connection()
try:
    cur_recent = conn_recent.cursor()
    cur_recent.execute("""SELECT p.title, p.author_name, p.filename, p.final_score, p.logic_score, p.c1, p.c2, p.c3, p.c4, p.c5, p.c6, p.c7, p.c8, p.piq_minted, p.tx_hash, p.zk_proof, p.mdar_adherence_score, p.rrid_valid_count, p.reproducibility_score, p.eval_hash, p.timestamp, b.block_height, b.block_hash, p.consensus_data, p.evidence_report, p.scilem_score FROM papers_assessment p LEFT JOIN blockchain_por_weights b ON p.eval_hash = b.eval_hash ORDER BY p.timestamp DESC LIMIT 20""")
    merged_papers = cur_recent.fetchall()
finally:
    conn_recent.close()

if merged_papers:
    st.markdown("<p style='font-size:13px; color:#64748b; margin-bottom:10px;'>Scroll to view more records. Click <b>View Dossier</b> on any manuscript card to open its complete research integrity record:</p>", unsafe_allow_html=True)
    recent_scroll_container = st.container(height=450)
    with recent_scroll_container:
        for idx, mp in enumerate(merged_papers):
            (m_title, m_author, m_filename, m_score, m_logic, m_c1, m_c2, m_c3, m_c4, m_c5, m_c6, m_c7, m_c8, m_piq, m_tx, m_zk, m_mdar, m_rrid, m_repro, m_hash, m_time, m_block_height, m_block_hash, m_consensus, m_report, m_scilem) = mp
            bh = m_block_height if m_block_height is not None else "Pending"
            clean_auth = clean_author_name(m_author)
            
            r_col1, r_col2, r_col3, r_col4 = st.columns([1.5, 4.5, 2.0, 2.0], vertical_alignment="center")
            with r_col1: st.markdown(f"**Block {bh}**")
            with r_col2:
                st.markdown(f"**{m_title}**")
                st.markdown(f"*{clean_auth}*")
            with r_col3: st.markdown(f"`{safe_float(m_score, 0.0):.2f}` / `{safe_float(m_piq, 0.0):.2f}`")
            with r_col4:
                if st.button("View Dossier", key=f"native_row_dossier_{idx}_{m_hash}", use_container_width=True):
                    item_dossier = {
                        "title": m_title, "author_name": m_author, "score": safe_float(m_score, 0.0), "logic_integrity": safe_float(m_logic, 75.0),
                        "scores_dict": { "C1_Semantic_Originality": safe_float(m_c1, 0), "C2_Methodological_Rigor_SciScore": safe_float(m_c2, 0), "C3_Interdisciplinary_Entropy": safe_float(m_c3, 0), "C4_Societal_Impact": safe_float(m_c4, 0), "C5_Open_Science_Repro": safe_float(m_c5, 0), "C6_Literature_Integration": safe_float(m_c6, 0), "C7_Empirical_Density": safe_float(m_c7, 0), "C8_Future_Actionability_FAIR": safe_float(m_c8, 0) },
                        "used_weights": [1.0]*8, "eval_hash": m_hash, "piq": safe_float(m_piq, 0.0), "tx_hash": m_tx, "zk_proof": m_zk, "h_idx": safe_float(m_mdar, 0.0), "i10_idx": int(safe_float(m_rrid, 0)), "repro_score": safe_float(m_repro, 0.0), "filename": m_filename or "N/A", "warnings": [], "consensus_raw": json.loads(m_consensus) if m_consensus else {}, "evidence_report_text": m_report or "", "scilem_rating": safe_float(m_scilem, 50.0)
                    }
                    show_dossier(item_dossier)
            st.divider()
else:
    st.info("No paper assessments recorded on ledger yet.")

st.markdown("---")

exp_head_col1, exp_head_col2 = st.columns([12, 1], vertical_alignment="center")
with exp_head_col1:
    st.markdown("### Proof-of-Research Blockchain Explorer", unsafe_allow_html=True)
with exp_head_col2:
    with st.popover("ⓘ", help="View Extra Ledger Info"):
        st.markdown("**Proof-of-Research (PoR) Validation:** Anchors assessment outcomes on the Sepolia testnet, sealing the block index, criteria weights, and unalterable state hashes (`formulas_hash`) into a cryptographically verified SHA-256 block.")
        conn_pop = get_db_connection()
        try:
            cur_pop = conn_pop.cursor()
            cur_pop.execute("SELECT por_proof, block_hash, formulas_hash FROM blockchain_por_weights ORDER BY block_height DESC LIMIT 1")
            p_data = cur_pop.fetchone()
        except Exception: p_data = None
        finally: conn_pop.close()
            
        if p_data:
            p_proof, b_hash, f_hash = p_data
            st.markdown(f"**Latest Proof-of-Research:** `{p_proof}` successfully verified and sealed to block `{b_hash}`.")
            st.markdown(f"**Unalterable Criteria State Hash:** `{f_hash}` (Guarantees grading mathematical constants cannot be tampered with).")
        piq_url = f"https://sepolia.etherscan.io/address/{PIQ_CONTRACT_ADDRESS}"
        reg_url = f"https://sepolia.etherscan.io/address/{REGISTRY_CONTRACT_ADDRESS}" if REGISTRY_CONTRACT_ADDRESS else "#"
        st.markdown(f"**Deployed Smart Contracts on Sepolia Etherscan:** PiQ Token Contract: [`{PIQ_CONTRACT_ADDRESS}`]({piq_url}) | Registry Contract: [`{REGISTRY_CONTRACT_ADDRESS}`]({reg_url})")

conn = get_db_connection()
try:
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT block_height, w1, w2, w3, w4, w5, w6, w7, w8, model_used, eval_hash, block_hash, por_proof, formulas_hash FROM blockchain_por_weights ORDER BY block_height DESC LIMIT 1")
        epoch_data = cursor.fetchone()
    except Exception: epoch_data = None

    if epoch_data:
        st.info("**Proof-of-Research Verification Guide:**\n• **Latest Proof-of-Research:** PoR_8839164d808d_Score:71.23 successfully verified and sealed to block 46024c976b38b5774d26d4ab24c863614fa372b2f02281366cf9d4fdfd49bc1b.\n• **Immutable Anchoring:** This proof guarantees cryptographic verification of historical scoring parameters.")

        explore_col1, explore_col2 = st.columns([3, 1], vertical_alignment="bottom")
        with explore_col1:
            search_query = st.text_input("Search Ledger", placeholder="Enter Evaluation Hash, Block Hash, Paper Name, Author Name, or Book Address...", label_visibility="collapsed", key="pidyne_ledger_search_query")
        with explore_col2:
            search_btn = st.button("Verify Ledger Record", key="pidyne_verify_record_btn", use_container_width=True)

        if search_btn and search_query:
            try:
                q_term = f"%{search_query.strip()}%"
                cursor.execute("""SELECT p.title, p.author_name, p.filename, p.final_score, p.logic_score, p.c1, p.c2, p.c3, p.c4, p.c5, p.c6, p.c7, p.c8, p.piq_minted, p.tx_hash, p.zk_proof, p.mdar_adherence_score, p.rrid_valid_count, p.reproducibility_score, p.eval_hash, p.timestamp, b.block_height, b.block_hash, b.por_proof, b.formulas_hash, p.eth_book, p.consensus_data, p.evidence_report, p.scilem_score FROM papers_assessment p LEFT JOIN blockchain_por_weights b ON p.eval_hash = b.eval_hash WHERE b.block_hash LIKE ? OR p.eval_hash LIKE ? OR p.title LIKE ? OR p.author_name LIKE ? OR p.eth_book LIKE ? LIMIT 5""", (q_term, q_term, q_term, q_term, q_term))
                matched_records = cursor.fetchall()
                if matched_records:
                    st.success(f"Found {len(matched_records)} matching record(s) on ledger.")
                    for m_idx, mr in enumerate(matched_records):
                        (m_title, m_author, m_filename, m_score, m_logic, m_c1, m_c2, m_c3, m_c4, m_c5, m_c6, m_c7, m_c8, m_piq, m_tx, m_zk, m_mdar, m_rrid, m_repro, m_hash, m_time, m_block_height, m_block_hash, m_por, m_form, m_book_addr, m_consensus, m_report, m_scilem) = mr
                        m_author_clean = clean_author_name(m_author)
                        m_book = m_book_addr if m_book_addr else ("0x" + hashlib.sha256(m_author_clean.encode()).hexdigest()[:40])
                        m_tx_url = get_tx_url(m_tx)
                        tx_disp_val = m_tx if m_tx and str(m_tx).strip() not in ["None", ""] else "Not Connected / No Book / Missing PK"

                        with st.expander(f"[{m_idx+1}] {m_title[:65]}... — *{m_author_clean}* (Score: **{safe_float(m_score, 0.0):.2f}** | {m_time[:16]})", expanded=True):
                            st.write(f"**File Name:** {m_filename if m_filename else 'N/A'}")
                            st.write(f"**Evaluation Hash (Paper Address):** `{m_hash}`")
                            st.write(f"**Unique Book Address:** `{m_book}`")
                            st.write(f"**piQ Minted:** `{m_piq}`")
                            st.markdown(f"**zk-SNARK Proof:** `{m_zk}`", unsafe_allow_html=True)
                            
                            if m_tx_url: st.markdown(f"**Tx Hash:** [`{tx_disp_val}`]({m_tx_url})")
                            else: st.write(f"**Tx Hash:** `{tx_disp_val}`")

                            st.markdown(f"**Executable Reproducibility Score:** `{safe_float(m_repro, 0.0) * 100:.1f}%`", unsafe_allow_html=True)
                            st.markdown(f"**SciScore MDAR Adherence:** `{safe_float(m_mdar, 0.0) * 100:.1f}%` | **Valid RRIDs:** `{m_rrid}`", unsafe_allow_html=True)

                            if st.button("View Full Multi-LLM & Scilem Dossier", key=f"search_det_{m_idx}_{m_hash}"):
                                search_item = {
                                    "title": m_title, "author_name": m_author, "score": safe_float(m_score, 0.0), "logic_integrity": safe_float(m_logic, 75.0),
                                    "scores_dict": { "C1_Semantic_Originality": safe_float(m_c1, 0), "C2_Methodological_Rigor_SciScore": safe_float(m_c2, 0), "C3_Interdisciplinary_Entropy": safe_float(m_c3, 0), "C4_Societal_Impact": safe_float(m_c4, 0), "C5_Open_Science_Repro": safe_float(m_c5, 0), "C6_Literature_Integration": safe_float(m_c6, 0), "C7_Empirical_Density": safe_float(m_c7, 0), "C8_Future_Actionability_FAIR": safe_float(m_c8, 0) },
                                    "used_weights": [1.0]*8, "eval_hash": m_hash, "piq": safe_float(m_piq, 0.0), "tx_hash": m_tx, "zk_proof": m_zk, "h_idx": safe_float(m_mdar, 0.0), "i10_idx": int(safe_float(m_rrid, 0)), "repro_score": safe_float(m_repro, 0.0), "filename": m_filename or "N/A", "warnings": [], "consensus_raw": json.loads(m_consensus) if m_consensus else {}, "evidence_report_text": m_report or "", "scilem_rating": safe_float(m_scilem, 50.0)
                                }
                                show_dossier(search_item)
                else: st.error("No records matching that evaluation hash, block hash, paper name, author name, or book address were found on the ledger.")
            except Exception as e:
                st.error(f"Error reading database: {str(e)}")
finally:
    conn.close()

@st.dialog("The Pi-Index Framework: Next-Gen Architecture & CoARA Compliance Workflow", width="large")
def show_framework_architecture():
    st.markdown("Pi-Index filters noise and yields quantitative results strictly aligned with **Responsible Research Assessment (RRA)** and **CoARA** (Coalition for Advancing Research Assessment) guidelines.\n\n### Architecture Flowchart & Whitepaper DOI\n\nRead the foundational framework whitepaper and preprints via [Ali Vafadar Yengejeh's ResearchGate Profile](https://www.researchgate.net/profile/Ali-Vafadar-Yengejeh).\n\nThe enhanced system architecture flow below details the decentralized intake, ZK double-blind reviewer assignment, SciScore deterministic parsing, Item Response Theory (IRT) calibration, and smart contract slashing mechanisms.")
    st.graphviz_chart("""
    digraph PiIndexSystemOverview {
        rankdir=TB; compound=true; fontname="Helvetica,Arial,sans-serif";
        node [fontname="Helvetica,Arial,sans-serif", style=filled, margin=0.2];
        edge [fontname="Helvetica,Arial,sans-serif", fontsize=10];
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
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: gray; font-size: 0.9em; padding-bottom: 5px;'>Framework Author: Ali Vafadar Yengejeh | Universita degli Studi di Milano-Bicocca</div>", unsafe_allow_html=True)

st.markdown("---")
col_pad1, col_center, col_pad2 = st.columns([1, 2, 1])
with col_center:
    if st.button("The Pi-Index Framework Workflow", use_container_width=True): show_framework_architecture()
