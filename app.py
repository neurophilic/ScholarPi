import os
import re
import json
import time
import hashlib
import tempfile
import shutil
import colorsys
import logging
import urllib.parse
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

from config import BASE_DIR, EPOCH_BLOCK_SIZE, PIQ_CONTRACT_ADDRESS, REGISTRY_CONTRACT_ADDRESS, HOT_TOPICS
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

def safe_get_sepolia_url(tx):
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
                    st.session_state.orcid_name = "Verified Decentralized Identity (SIWE)"
                    add_log(f"MetaMask Identity Cryptographically Authenticated via SIWE: {clean_wallet}")
            except Exception as e:
                add_log(f"SIWE signature verification fallback: {str(e)}")
        
        if not authenticated:
            st.session_state.orcid_name = "Connected MetaMask Wallet"
            add_log(f"MetaMask Connected: {clean_wallet}")
            
        st.session_state.orcid_id = clean_wallet
        st.session_state.is_authenticated = True
        st.session_state.auth_method = "Web3"
        st.toast(f"MetaMask Connected: {clean_wallet[:6]}...{clean_wallet[-4:]}", icon="🦊")

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

hr {
    border-color: #e2e8f0 !important;
    margin: 1.5rem 0 !important;
}

[data-testid="stHeaderActionElements"] {
    display: none !important;
}

[data-testid="stSidebar"] {
    background-color: #f8fafc !important;
    border-right: 1px solid #e2e8f0 !important;
    overflow-y: auto !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    background-color: #ffffff !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
    transition: box-shadow 0.2s ease-in-out, transform 0.2s ease-in-out !important;
    padding: 0.5rem !important;
}

button[kind="primary"], [data-testid="baseButton-primary"] {
    background-color: #0f172a !important;
    border-color: #0f172a !important;
    color: #ffffff !important;
}
button[kind="primary"]:hover, [data-testid="baseButton-primary"]:hover {
    background-color: #1e293b !important;
    border-color: #1e293b !important;
    color: #ffffff !important;
}

.pi-stop-button, .pi-stop-button:focus {
    background-color: #dc2626 !important;
    border-color: #dc2626 !important;
    color: #ffffff !important;
}

.stButton>button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
}

[data-testid="stExpander"] {
    border-radius: 10px !important;
    border: 1px solid #e2e8f0 !important;
    background-color: #ffffff !important;
}

iframe {
    border: none !important;
    border-radius: 8px !important;
    outline: none !important;
}

.pyvis-map-wrapper iframe {
    width: 100% !important;
    height: 600px !important;
    display: block !important;
}
</style>
"""
components.html(custom_ui_code, height=0, width=0)

st.sidebar.title("System Access")

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

if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False
    st.session_state.auth_method = "Anonymous"
    st.session_state.orcid_id = "0x0000000000000000000000000000000000000000"
    st.session_state.academic_id = "None"
    st.session_state.orcid_name = "Anonymous Researcher"

def validate_orcid_did(identifier: str) -> bool:
    clean_id = identifier.strip()
    is_orcid = re.match(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$", clean_id)
    is_did = re.match(r"^did:[a-z0-9]+:[a-zA-Z0-9.\-_:]+$", clean_id)
    return bool(is_orcid or is_did)

if not st.session_state.is_authenticated:
    st.sidebar.markdown("### Web3 Authentication")
    
    metamask_ui_html = """
    <div id="mm-root" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 2px;">
        <button id="connect-mm-btn" type="button" style="
            width: 100%;
            background: linear-gradient(135deg, #f6851b, #e2761b);
            color: white;
            border: none;
            padding: 12px 16px;
            border-radius: 10px;
            font-weight: 700;
            font-size: 14px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            box-shadow: 0 4px 12px rgba(246, 133, 27, 0.25);
            transition: all 0.2s ease;
        ">
            <svg width="22" height="22" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M29.28 2.00003L18.44 10.02L20.4 4.88003L29.28 2.00003Z" fill="#E17726" stroke="#E17726" stroke-width="0.25"/>
                <path d="M2.72 2.00003L13.46 10.12L11.6 4.88003L2.72 2.00003Z" fill="#E17726" stroke="#E17726" stroke-width="0.25"/>
                <path d="M25.02 23.36L21.94 28.24L28.32 25.1L29.86 19.98L25.02 23.36Z" fill="#E17726" stroke="#E17726" stroke-width="0.25"/>
                <path d="M2.14 19.98L3.68 25.1L10.06 28.24L6.98 23.36L2.14 19.98Z" fill="#E17726" stroke="#E17726" stroke-width="0.25"/>
                <path d="M10.06 28.24L14.7 21.08L10.74 15.34L3.68 25.1L10.06 28.24Z" fill="#E27625" stroke="#E27625" stroke-width="0.25"/>
                <path d="M21.94 28.24L28.32 25.1L21.26 15.34L17.3 21.08L21.94 28.24Z" fill="#E27625" stroke="#E27625" stroke-width="0.25"/>
            </svg>
            <span>Connect MetaMask</span>
        </button>
        <div id="mm-status" style="margin-top: 6px; font-size: 12px; color: #dc2626; font-weight: 500; text-align: center; word-break: break-word;"></div>
    </div>

    <script>
    function getEthereumProvider() {
        let provider = window.ethereum;
        if (!provider && window.parent) {
            try { provider = window.parent.ethereum; } catch(e) {}
        }
        if (!provider && window.top) {
            try { provider = window.top.ethereum; } catch(e) {}
        }
        if (provider && provider.providers) {
            provider = provider.providers.find(p => p.isMetaMask) || provider;
        }
        return provider;
    }

    document.getElementById('connect-mm-btn').addEventListener('click', async () => {
        const statusDiv = document.getElementById('mm-status');
        statusDiv.style.color = "#2563eb";
        statusDiv.innerText = "Connecting to MetaMask...";

        const provider = getEthereumProvider();

        if (!provider) {
            statusDiv.style.color = "#dc2626";
            statusDiv.innerText = "MetaMask not detected! Please install or unlock MetaMask.";
            return;
        }

        try {
            const accounts = await provider.request({ method: 'eth_requestAccounts' });
            if (!accounts || accounts.length === 0) {
                statusDiv.style.color = "#dc2626";
                statusDiv.innerText = "No accounts selected in wallet.";
                return;
            }

            const account = accounts[0];
            statusDiv.innerText = "Verifying SIWE signature...";

            const domain = window.location.hostname || "localhost";
            const nonce = Math.floor(Math.random() * 100000000);
            const message = `${domain} wants you to sign in with your Ethereum account:\\n${account}\\n\\nSign in with Ethereum to authenticate session.\\n\\nNonce: ${nonce}\\nIssued At: ${new Date().toISOString()}`;

            let signature = null;
            try {
                const hexMessage = '0x' + Array.from(new TextEncoder().encode(message)).map(b => b.toString(16).padStart(2, '0')).join('');
                signature = await provider.request({
                    method: 'personal_sign',
                    params: [hexMessage, account]
                });
            } catch (signErr) {
                console.warn("SIWE signing skipped or rejected by user:", signErr);
            }

            statusDiv.innerText = "Authenticating with engine...";

            const targetWindow = window.parent || window;
            const targetUrl = new URL(targetWindow.location.href);
            targetUrl.searchParams.set("siwe_address", account);
            if (signature) {
                targetUrl.searchParams.set("siwe_signature", signature);
                targetUrl.searchParams.set("siwe_message", encodeURIComponent(message));
            }

            try {
                targetWindow.location.href = targetUrl.href;
            } catch (navErr) {
                window.location.href = targetUrl.href;
            }

        } catch (err) {
            console.error("MetaMask Connection Error:", err);
            statusDiv.style.color = "#dc2626";
            statusDiv.innerText = err.message || "Connection rejected by user.";
        }
    });
    </script>
    """
    with st.sidebar:
        components.html(metamask_ui_html, height=110)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Academic ID Alternative")
    manual_id = st.sidebar.text_input("ORCID iD or W3C DID", placeholder="0000-0000-0000-0000", label_visibility="collapsed")
    if st.sidebar.button("Connect Academic ID", use_container_width=True):
        if validate_orcid_did(manual_id):
            st.session_state.academic_id = manual_id.strip()
            st.session_state.orcid_name = "Verified Academic Researcher"
            st.session_state.is_authenticated = True
            st.session_state.auth_method = "Academic ID"
            add_log(f"Identity Authenticated via Academic ID: {manual_id.strip()}")
            st.rerun()
        else:
            st.sidebar.error("Invalid ORCID or DID format.")

    st.sidebar.info("Notice: Connect your Web3 Wallet to stake piQ tokens and receive earned rewards directly.")
else:
    st.sidebar.success("Securely Connected")
    
    conn_hist = get_db_connection()
    total_user_piq = 0.0
    try:
        cur_h = conn_hist.cursor()
        if st.session_state.auth_method == "Web3":
            cur_h.execute("SELECT piq_minted FROM papers_assessment WHERE eth_book = ?", (st.session_state.orcid_id,))
        else:
            cur_h.execute("SELECT piq_minted FROM papers_assessment WHERE user_id = ?", (st.session_state.academic_id,))
        piq_rows = cur_h.fetchall()
        total_user_piq = sum(safe_float(r[0], 0.0) for r in piq_rows if r[0])
    finally:
        conn_hist.close()
        
    auth_disp = st.session_state.orcid_id if st.session_state.auth_method == "Web3" else st.session_state.academic_id
    
    st.sidebar.markdown(
        f"**Researcher:** {st.session_state.orcid_name}\n\n"
        f"**Connected ID:** `{auth_disp[:12]}...`\n\n"
        f"**TOTAL piQ AWARDED:** `{total_user_piq:.2f} piQ`"
    )

    if st.sidebar.button("Disconnect Session", use_container_width=True):
        add_log("Session Disconnected.")
        st.session_state.is_authenticated = False
        st.session_state.auth_method = "Anonymous"
        st.session_state.orcid_name = ""
        st.session_state.orcid_id = "0x0000000000000000000000000000000000000000"
        st.session_state.academic_id = "None"
        st.rerun()

current_user = st.session_state.orcid_id if st.session_state.auth_method == "Web3" else st.session_state.academic_id
current_email = "None"
valid_book_address = current_user if (st.session_state.auth_method == "Web3" and w3.is_address(current_user)) else "0x0000000000000000000000000000000000000000"

st.sidebar.markdown("---")
with st.sidebar.expander("Live System Monitor", expanded=True):
    log_text = "\n".join(st.session_state.app_logs)
    st.code(log_text if log_text else "No active logs...", language="bash")

# Stable Sidebar Expander for Scilem Assistant (No risky DOM-stealing scripts)
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

    if (
        st.session_state.is_authenticated 
        and st.session_state.auth_method == "Web3" 
        and w3.is_address(current_user) 
        and w3.is_address(OWNER_ID) 
        and current_user.lower() == OWNER_ID.lower()
    ):
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

def refine_science_field(s):
    s_lower = s.lower()
    if any(k in s_lower for k in ["blockchain", "smart contract", "crypto", "ledger"]):
        return "Computer Science > Blockchain & Distributed Systems"
    elif any(k in s_lower for k in ["machine learning", "deep learning", "neural", "ai", "artificial intelligence"]):
        return "Computer Science > Artificial Intelligence & Machine Learning"
    elif any(k in s_lower for k in ["algorithm", "software", "computation", "cyber", "data", "information"]):
        return "Computer Science > Algorithms & Software Engineering"
    elif any(k in s_lower for k in ["quantum", "optics", "photonics"]):
        return "Physics > Quantum Mechanics & Optics"
    elif any(k in s_lower for k in ["energy", "mechanics", "thermodynamics", "physics"]):
        return "Physics > Applied Mechanics & Energy Systems"
    elif any(k in s_lower for k in ["polymer", "catalysis", "molecule", "chemical", "chemistry"]):
        return "Chemistry > Chemical Synthesis & Molecular Catalysis"
    elif any(k in s_lower for k in ["genetics", "genomics", "gene", "biology"]):
        return "Life Sciences > Genetics & Genomics"
    elif any(k in s_lower for k in ["cellular", "protein", "molecular biology"]):
        return "Life Sciences > Molecular & Cellular Biology"
    elif any(k in s_lower for k in ["ecology", "ecosystem", "biodiversity"]):
        return "Life Sciences > Ecology & Evolutionary Biology"
    elif any(k in s_lower for k in ["clinical", "hospital", "patient", "disease", "pharmac", "medical", "medicine"]):
        return "Medical Sciences > Clinical Medicine & Pharmacology"
    elif any(k in s_lower for k in ["biomedical", "neuroscience", "cardiac"]):
        return "Medical Sciences > Biomedical Research"
    elif any(k in s_lower for k in ["climate", "carbon", "atmosphere", "meteorology", "earth"]):
        return "Earth Sciences > Climate Science & Meteorology"
    elif any(k in s_lower for k in ["geology", "ocean", "seismic"]):
        return "Earth Sciences > Geology & Earth Systems"
    elif any(k in s_lower for k in ["economics", "finance", "market", "social"]):
        return "Social Sciences > Economics & Quantitative Finance"
    elif any(k in s_lower for k in ["sociology", "psychology", "policy", "management"]):
        return "Social Sciences > Behavioral & Policy Studies"
    elif any(k in s_lower for k in ["math", "statistics", "algebra", "probability", "calculus"]):
        return "Mathematics & Statistics > Applied Mathematics & Statistics"
    elif any(k in s_lower for k in ["engineering", "robotics", "materials", "civil", "electrical"]):
        return "Engineering & Technology > Applied Engineering & Materials Science"
    else:
        return f"Engineering & Technology > Applied Technical Research ({s.title()})"

def render_bubble_chart_clean(target_author, repulsion=-3000, spring_len=180, size_scale=1.5, central_grav=0.15):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT fields, subfields, final_score, author_name FROM papers_assessment")
        data = cursor.fetchall()
    finally:
        conn.close()

    html_string, table_html = "", ""
    if not data:
        return html_string, table_html

    topic_aggregates = {}
    exclude_terms = {
        "general", "general science", "unspecified domain",
        "unspecified sub-domain", "core research topic",
    }

    for fields_json, subfields_json, final_score, author_str in data:
        cleaned_author = clean_author_name(author_str)
        if (
            target_author
            and target_author != "All Authors"
            and target_author not in cleaned_author
        ):
            continue
        try:
            raw_subfields = [s.title().strip() for s in json.loads(subfields_json)]
            score = safe_float(final_score, 50.0)
            for rs in raw_subfields:
                if rs and rs.lower() not in exclude_terms:
                    s = refine_science_field(rs)
                    if s not in topic_aggregates:
                        topic_aggregates[s] = {"weight_sum": 0.0, "frequency": 0}
                    topic_aggregates[s]["weight_sum"] += score
                    topic_aggregates[s]["frequency"] += 1
        except:
            continue

    if not topic_aggregates:
        topic_aggregates["Computer Science > Algorithms & Software Engineering"] = {
            "weight_sum": 50.0,
            "frequency": 1,
        }

    if len(topic_aggregates) > 15:
        sorted_topics = sorted(
            topic_aggregates.items(),
            key=lambda x: (x[1]["frequency"], x[1]["weight_sum"]),
            reverse=True
        )
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

    net = Network(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#2c3e50",
        notebook=False,
    )
    
    physics_options = f"""{{ 
        "physics": {{ 
            "barnesHut": {{ 
                "gravitationalConstant": {repulsion}, 
                "centralGravity": {central_grav}, 
                "springLength": {spring_len}, 
                "springConstant": 0.005,
                "damping": 1.0,
                "avoidOverlap": 2.0 
            }}, 
            "stabilization": {{ 
                "enabled": true, 
                "iterations": 2500,
                "fit": true
            }} 
        }} 
    }}"""
    net.set_options(physics_options)

    for topic, metrics in topic_aggregates.items():
        avg_weight = metrics["weight_sum"] / metrics["frequency"]
        freq = metrics["frequency"]
        node_size = max(35, (25 + (avg_weight * 3.0)) * size_scale)

        base_col = color_map[topic]
        net.add_node(
            n_id=topic,
            label=" ",
            title=(
                f"Field: {topic} | Frequency: {freq} | Avg Weight/Score:"
                f" {avg_weight:.1f}"
            ),
            size=node_size,
            shape="dot",
            physics=True,
            font={"color": "rgba(0,0,0,0)", "size": 0},
            color={
                "background": base_col,
                "border": "#1a1a1a",
                "highlight": {"background": base_col, "border": "#000000"},
                "hover": {"background": base_col, "border": "#000000"},
            },
            shadow={
                "enabled": True,
                "color": "rgba(0,0,0,0.5)",
                "size": 6,
                "x": 3,
                "y": 3,
            },
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
    <style type="text/css">
        body, html {{ margin: 0; padding: 0; border: none; overflow: hidden; width: 100%; height: 600px; }}
        canvas {{
            background: radial-gradient(circle at 50% 50%, #ffffff 0%, #f0f2f5 100%);
            border: none !important;
            outline: none !important;
            width: 100% !important;
            height: 600px !important;
        }}
        #mynetwork, .vis-network, .card-body {{
            border: none !important;
            box-shadow: none !important;
            margin: 0 !important;
            width: 100% !important;
            height: 600px !important;
        }}
    </style>
    <!-- reload_timestamp: {time.time()} -->
    </head>
    """
    html_string = html_string.replace("</head>", gradient_injection)
    html_string = html_string.replace(
        "mynetwork", f"pi_network_{int(time.time() * 1000)}"
    )

    table_html = "<style>.table-big { width: 100%; font-size: 13px; border-collapse: collapse; margin-top: 10px; font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif; } .table-big th { background-color: #f8fafc; color: #475569; padding: 8px; text-align: left; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; border-bottom: 2px solid #e2e8f0; } .table-big td { padding: 8px; border-bottom: 1px solid #f1f5f9; color: #1e293b; } .color-box { width: 20px; height: 20px; border-radius: 4px; display: inline-block; box-shadow: 0 1px 2px rgba(0,0,0,0.1); } </style>"
    table_html += "<div class='legend-container'><table class='table-big'><thead><tr><th style='width: 15%; text-align: center;'>Color</th><th>Science Field</th><th style='text-align: center;'>Freq</th><th style='text-align: center;'>Avg Weight</th></tr></thead><tbody>"
    for topic, metrics in sorted(
        topic_aggregates.items(), key=lambda x: x[1]["frequency"], reverse=True
    ):
        avg_w = metrics["weight_sum"] / metrics["frequency"]
        table_html += (
            f"<tr><td style='text-align: center;'><div class='color-box'"
            f" style='background-color:{color_map[topic]};'></div></td><td><b>{topic}</b></td><td"
            f" style='text-align: center;'>{metrics['frequency']}</td><td"
            f" style='text-align: center;'>{avg_w:.1f}</td></tr>"
        )
    table_html += "</tbody></table></div>"

    return html_string, table_html


def get_criteria_info(weights):
    tw1, tw2, tw3, tw4, tw5, tw6, tw7, tw8 = weights
    return [
        ("C1", "Originality", "c1: originality", tw1, "1", "Semantic distance from literature corpus penalized by generative AI laundering heuristics.", r"$$ C_1 = \varpi_1 \cdot \mathcal{D}_{semantic}(P_{target}, P_{corpus}) \times (1 - \lambda_{laundering}) + v_{apri} $$"),
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
    st.markdown(
        r"**Adversarial Logic Gap ($\Delta_{Logic}$):** Evaluates reasoning structure and penalizes claims unsupported by evidence or counterfactual stress failures.",
        unsafe_allow_html=True
    )
    st.markdown(
        r"$$ L_i = \left( (\mathcal{P}_{valid} \cdot \mathcal{E}_{strength}) \cdot"
        r" \exp\left(-\left(2 \cdot \max(0, \mathcal{C}_{reach} -"
        r" \mathcal{E}_{strength}) + 1.5 \cdot \lambda_{jumps}\right)\right) \right)"
        r" \times \frac{1}{1 + e^{-\Delta Premise}} + \lambda \cdot v_{apri} $$"
    )

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
    reset_tok = st.session_state["reset_token"]

    research_scope = st.text_input(
        "Research Scope / Field Focus (optional)",
        placeholder="e.g. Quantum Error Correction, Oncology, Perovskite Solar Cells...",
        key=f"scope_input_{reset_tok}",
    )

    intake_tab_local, intake_tab_doi, intake_tab_alex = st.tabs(
        ["📄 Local Upload", "🔗 DOI Lookup", "🔍 OpenAlex Topic Search"]
    )

    selected_uploaded_files = []
    with intake_tab_local:
        uploaded_files = st.file_uploader(
            "Upload Local PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"file_uploader_{reset_tok}",
        )
        if uploaded_files:
            st.markdown("**Tick local files to include:**")
            for i, file in enumerate(uploaded_files):
                if st.checkbox(
                    f"Local File: {file.name}",
                    value=True,
                    key=f"up_chk_{i}_{reset_tok}",
                ):
                    selected_uploaded_files.append(file)

    doi_input = ""
    include_doi = False
    with intake_tab_doi:
        doi_input = st.text_input(
            "Enter a DOI",
            placeholder="10.1000/xyz123 or https://doi.org/10.1000/xyz123",
            key=f"doi_text_{reset_tok}",
        )
        include_doi = st.checkbox(
            "Include this DOI in the assessment pipeline",
            value=False,
            key=f"doi_chk_{reset_tok}",
            disabled=not doi_input.strip(),
        )
        st.caption(
            "Pi-Index resolves open-access PDFs automatically via Unpaywall → "
            "Semantic Scholar → CORE, in that order."
        )

    selected_alex_papers = []
    with intake_tab_alex:
        alex_choice = st.selectbox(
            "Hot Topic",
            ["Custom Search..."] + HOT_TOPICS,
            key=f"alex_topic_sel_{reset_tok}",
        )
        custom_query = ""
        if alex_choice == "Custom Search...":
            custom_query = st.text_input("Custom OpenAlex Search Query", key=f"alex_custom_{reset_tok}")
        search_term = custom_query.strip() if alex_choice == "Custom Search..." else alex_choice

        if st.button("Search OpenAlex", key=f"alex_search_btn_{reset_tok}"):
            if search_term:
                with st.spinner(f"Searching OpenAlex for '{search_term}'..."):
                    st.session_state["alex_search_results"] = search_openalex_topics(search_term, limit=15)
                    st.session_state["alex_search_query_used"] = search_term
            else:
                st.warning("Enter a custom search term or pick a hot topic first.")

        alex_results = st.session_state.get("alex_search_results", [])
        if alex_results:
            st.markdown(
                f"**{len(alex_results)} result(s) for '{st.session_state.get('alex_search_query_used', '')}':** "
                "tick papers to include in the pipeline."
            )
            for ai, ap in enumerate(alex_results):
                label = f"{ap.get('title', 'Untitled Paper')}  —  *{ap.get('authors', 'Unidentified')}*"
                if st.checkbox(label, value=False, key=f"alex_chk_{ai}_{reset_tok}"):
                    selected_alex_papers.append(ap)
        else:
            st.caption("No results yet — search a hot topic or custom query above.")

    free_evals_used = st.session_state.get("free_evals_used", 0)
    is_web3_authenticated = (
        st.session_state.is_authenticated 
        and st.session_state.auth_method == "Web3" 
        and w3.is_address(st.session_state.orcid_id)
    )

    if free_evals_used == 0:
        st.info(
            "🎁 **First Assessment Free:** Your first assessment runs with zero stake required! "
            "**Recommendation:** Connect your Web3 Wallet in the sidebar first so earned **piQ** tokens can be credited directly to your address."
        )
        stake_amount = True
    else:
        if not is_web3_authenticated:
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

        progress_bar, status_text = st.progress(0), st.empty()
        scope_val = st.session_state.get("snap_scope", "")
        snap_files = st.session_state.get("snap_files", [])
        snap_alex = st.session_state.get("snap_alex", [])
        include_doi_snap = st.session_state.get("snap_include_doi", False)
        doi_snap = st.session_state.get("snap_doi", "")

        try:
            if snap_alex and not st.session_state["cancel_requested"]:
                for p in snap_alex:
                    if st.session_state["cancel_requested"]:
                        break
                    status_text.text(f"Fetching OpenAlex paper: {p['title']}...")
                    pdf_bytes = None
                    fname = f"OpenAlex_{p['title'][:20]}.pdf"
                    p_doi = p.get("doi", "None")
                    add_log(f"Commencing open-access resolution for OpenAlex document: {fname}")

                    if p.get("pdf_url"):
                        pdf_bytes = download_pdf_from_url(p["pdf_url"])
                    if not pdf_bytes and (p.get("title") or p.get("doi")):
                        s2_url = fetch_semantic_scholar_pdf(p.get("doi") or p.get("title"))
                        if s2_url:
                            pdf_bytes = download_pdf_from_url(s2_url)
                    if not pdf_bytes and p.get("doi"):
                        metadata = fetch_doi_metadata(p["doi"])
                        if metadata and metadata.get("pdf_url"):
                            pdf_bytes = download_pdf_from_url(metadata["pdf_url"])

                    if not pdf_bytes and p_doi:
                        status_text.text("Direct download restricted. Querying CORE API fallback...")
                        core_text = fetch_core_text_by_doi(p_doi)
                        if core_text:
                            pdf_bytes = create_virtual_pdf_from_text(core_text, title=p.get('title', 'Open Access'))

                    if pdf_bytes:
                        clean_bytes = preprocess_pdf_layout(pdf_bytes, fname)
                        (
                            title, author_name, score, logic_integrity, drift, rec,
                            fields, subfields, scores_dict, eval_hash, piq, tx_hash,
                            zk_proof, used_weights, mdar_score, rrid_count, repro_score, is_cached, warnings_list,
                            consensus_raw, evidence_report_text, scilem_rating
                        ) = process_single_pdf(
                            clean_bytes, fname, scope_val, current_user, valid_book_address, current_email, p_doi,
                        )

                        eval_record = {
                            "title": title, "author_name": clean_author_name(author_name),
                            "score": score, "logic_integrity": logic_integrity, "drift": drift,
                            "rec": rec, "fields": fields, "subfields": subfields,
                            "scores_dict": scores_dict, "eval_hash": eval_hash, "piq": piq,
                            "tx_hash": tx_hash, "zk_proof": zk_proof, "used_weights": used_weights,
                            "h_idx": mdar_score, "i10_idx": rrid_count, "repro_score": repro_score,
                            "filename": fname, "warnings": warnings_list, "warnings_acknowledged": False,
                            "consensus_raw": consensus_raw, "evidence_report_text": evidence_report_text,
                            "scilem_rating": scilem_rating
                        }
                        st.session_state["evaluated_papers_buffer"].insert(0, eval_record)
                        st.session_state["evaluated_papers_buffer"] = st.session_state["evaluated_papers_buffer"][:50]
                        st.session_state["free_evals_used"] += 1
                        add_log(f"Successfully processed and recorded evaluation for {fname}")
                    else:
                        clean_doi = p_doi.replace("https://doi.org/", "").strip() if p_doi else "None"
                        doi_url = f"https://doi.org/{clean_doi}" if clean_doi and clean_doi != "None" else (p.get("pdf_url") or "N/A")
                        err_item = {"title": p.get("title", "Unknown Title"), "doi": clean_doi if clean_doi and clean_doi != "None" else "N/A", "url": doi_url}
                        add_log("Publisher access restriction encountered for OpenAlex target.")
                        if err_item not in st.session_state["download_errors"]:
                            st.session_state["download_errors"].append(err_item)

            if (
                include_doi_snap
                and doi_snap.strip()
                and not st.session_state["cancel_requested"]
            ):
                status_text.text(f"Resolving DOI: {doi_snap}...")
                metadata = fetch_doi_metadata(doi_snap)
                fname = f"DOI_{doi_snap.replace('/', '_')}.pdf"
                pdf_bytes = None
                add_log(f"Attempting API resolution for standalone DOI: {doi_snap}")
                
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
                    status_text.text("Assessing document from resolved source...")
                    clean_bytes = preprocess_pdf_layout(pdf_bytes, fname)
                    (
                        title, author_name, score, logic_integrity, drift, rec,
                        fields, subfields, scores_dict, eval_hash, piq, tx_hash,
                        zk_proof, used_weights, mdar_score, rrid_count, repro_score, is_cached, warnings_list,
                        consensus_raw, evidence_report_text, scilem_rating
                    ) = process_single_pdf(
                        clean_bytes, fname, scope_val, current_user, valid_book_address, current_email, doi_snap.strip(),
                    )

                    eval_record = {
                        "title": title, "author_name": clean_author_name(author_name),
                        "score": score, "logic_integrity": logic_integrity, "drift": drift,
                        "rec": rec, "fields": fields, "subfields": subfields,
                        "scores_dict": scores_dict, "eval_hash": eval_hash, "piq": piq,
                        "tx_hash": tx_hash, "zk_proof": zk_proof, "used_weights": used_weights,
                        "h_idx": mdar_score, "i10_idx": rrid_count, "repro_score": repro_score,
                        "filename": fname, "warnings": warnings_list, "warnings_acknowledged": False,
                        "consensus_raw": consensus_raw, "evidence_report_text": evidence_report_text,
                        "scilem_rating": scilem_rating
                    }
                    st.session_state["evaluated_papers_buffer"].insert(0, eval_record)
                    st.session_state["evaluated_papers_buffer"] = st.session_state["evaluated_papers_buffer"][:50]
                    st.session_state["free_evals_used"] += 1
                    add_log("Successfully evaluated and logged DOI source.")
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
                    if st.session_state["cancel_requested"]:
                        break
                    status_text.text(f"Analyzing uploaded file {i+1} of {total_files}: {fname}...")
                    add_log(f"Engaging logical extraction on local file structure: {fname}")
                    
                    with open(fpath, "rb") as in_f:
                        raw_bytes = in_f.read()
                        
                    clean_bytes = preprocess_pdf_layout(raw_bytes, fname)
                    
                    (
                        title, author_name, score, logic_integrity, drift, rec,
                        fields, subfields, scores_dict, eval_hash, piq, tx_hash,
                        zk_proof, used_weights, mdar_score, rrid_count, repro_score, is_cached, warnings_list,
                        consensus_raw, evidence_report_text, scilem_rating
                    ) = process_single_pdf(
                        clean_bytes, fname, scope_val, current_user, valid_book_address, current_email, "None",
                    )

                    eval_record = {
                        "title": title, "author_name": clean_author_name(author_name),
                        "score": score, "logic_integrity": logic_integrity, "drift": drift,
                        "rec": rec, "fields": fields, "subfields": subfields,
                        "scores_dict": scores_dict, "eval_hash": eval_hash, "piq": piq,
                        "tx_hash": tx_hash, "zk_proof": zk_proof, "used_weights": used_weights,
                        "h_idx": mdar_score, "i10_idx": rrid_count, "repro_score": repro_score,
                        "filename": fname, "warnings": warnings_list, "warnings_acknowledged": False,
                        "consensus_raw": consensus_raw, "evidence_report_text": evidence_report_text,
                        "scilem_rating": scilem_rating
                    }
                    st.session_state["evaluated_papers_buffer"].insert(0, eval_record)
                    st.session_state["evaluated_papers_buffer"] = st.session_state["evaluated_papers_buffer"][:50]
                    st.session_state["free_evals_used"] += 1
                    progress_bar.progress((i + 1) / total_files)
                    add_log(f"Stored local assessment result to cache.")

            if st.session_state["cancel_requested"]:
                st.warning("Pipeline operation was stopped.")
            else:
                status_text.success("Pipeline processing complete.")
                time.sleep(1)
        finally:
            st.session_state["is_running"] = False
            st.session_state["cancel_requested"] = False
            st.session_state["reset_token"] += 1
            st.session_state["assessment_update_token"] = time.time()

    else:
        if st.button("Run Assessment Pipeline", type="primary", use_container_width=True):
            if free_evals_used >= 1 and not is_web3_authenticated:
                st.error("Free trial limit reached. Please connect your Web3 Ethereum Wallet in the sidebar to stake 0.1 piQ and run assessments.")
            elif free_evals_used >= 1 and not stake_amount:
                st.error("You must agree to stake 0.1 piQ to execute further paper assessments.")
            elif (
                not selected_uploaded_files
                and not (include_doi and doi_input.strip())
                and not selected_alex_papers
            ):
                st.warning("Please tick at least one paper or input source to assess.")
            else:
                add_log("Preparing pipeline dispatch queue...")
                saved_files = []
                for f in selected_uploaded_files:
                    f_path = os.path.join(st.session_state["session_temp_dir"], f.name)
                    with open(f_path, "wb") as out_f:
                        out_f.write(f.getvalue())
                    f.seek(0)
                    saved_files.append((f.name, f_path))
                    add_log(f"Cached user file to temporary disk node: {f.name}")
                    
                st.session_state["snap_files"] = saved_files
                st.session_state["snap_scope"] = research_scope
                st.session_state["snap_doi"] = doi_input
                st.session_state["snap_include_doi"] = include_doi
                st.session_state["snap_alex"] = selected_alex_papers
                st.session_state["is_running"] = True
                st.session_state["cancel_requested"] = False
                st.rerun()

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
    st.markdown(f"**zk-SNARK Proof:** `{zk_proof}`", unsafe_allow_html=True)
    
    tx_url = safe_get_sepolia_url(tx_hash)
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
            "C1: Semantic Originality",
            "C2: Methodological Rigor (SciScore)",
            "C3: Interdisciplinary Entropy",
            "C4: Societal Impact",
            "C5: Open Science & Repro",
            "C6: Literature Integration",
            "C7: Empirical Density",
            "C8: Future Actionability & FAIR",
        ],
        "Score Extracted (0-100)": [
            safe_float(scores_dict.get("C1_Semantic_Originality"), 0),
            safe_float(scores_dict.get("C2_Methodological_Rigor_SciScore"), 0),
            safe_float(scores_dict.get("C3_Interdisciplinary_Entropy"), 0),
            safe_float(scores_dict.get("C4_Societal_Impact"), 0),
            safe_float(scores_dict.get("C5_Open_Science_Repro"), 0),
            safe_float(scores_dict.get("C6_Literature_Integration"), 0),
            safe_float(scores_dict.get("C7_Empirical_Density"), 0),
            safe_float(scores_dict.get("C8_Future_Actionability_FAIR"), 0),
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
    st.markdown(
        f"**Logic Integrity Multiplier:** `{logic_multiplier:.4f}` (Derived from"
        f" {logic_integrity:.1f}% raw logic score)"
    )

    dossier_content = f"""# RESEARCH INTEGRITY DOSSIER (DORA-Aligned)
**Title:** {title}
**Author:** {author_name}
**File Name:** {filename}
**Evaluation Hash (Paper Address):** {eval_hash}
**Unique Book Address:** {author_book}
**Final Pi-Index Score:** {score:.2f} / 100
**Logic Integrity Score:** {logic_integrity:.1f}%
**SciScore MDAR Adherence:** {mdar_score * 100:.1f}%
**Valid RRIDs Count:** {rrid_count}
"""
    st.download_button(
        label=f"Download Research Integrity Dossier ({filename})",
        data=dossier_content,
        file_name=f"Dossier_{eval_hash[:10]}.md",
        mime="text/markdown",
        key=f"download_dossier_modal_{eval_hash}_{time.time()}",
        use_container_width=True,
    )

@st.dialog("AI Peer Review Defense Strategy", width="medium")
def defense_strategy_dialog(scores_dict):
    with st.spinner("Synthesizing adversarial defense strategy..."):
        rebuttal = generate_rebuttal_strategy(scores_dict)
    st.markdown(rebuttal)

def render_breakdown_item(item, index):
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
            if warnings and not acknowledged:
                warn_badge = f" ⚠️ *({len(warnings)} warning checks active)*"
            elif warnings and acknowledged:
                warn_badge = f" 🛡️ *({len(warnings)} warning checks acknowledged)*"
            else:
                warn_badge = ""

            title_lower = str(title).lower().strip()
            author_lower = str(author_name).lower().strip()

            invalid_titles = ["n/a", "none", "unknown", "failed", "unnamed", "api limit"]
            invalid_authors = ["n/a", "none", "unknown", "unidentified", "independent research scholar", "unconfigured key", "anonymous"]

            has_valid_title = (
                title 
                and not any(inv in title_lower for inv in invalid_titles)
                and "parsed via local heuristics" not in title_lower
            )
            has_valid_author = (
                author_name 
                and not any(inv in author_lower for inv in invalid_authors)
            )

            if has_valid_title and has_valid_author:
                extraction_badge = " ✅ *Title & Author Extracted Successfully*"
            elif has_valid_title or has_valid_author:
                extraction_badge = " ⚠️ *Partial Extraction (Title or Author Only)*"
            else:
                extraction_badge = ""

            st.markdown(f"**{title}** — *{author_name}*{extraction_badge}{warn_badge}")
            st.markdown(f"**Score: {score:.2f} | piQ: {piq}**")
            
            if warnings:
                with st.expander(f"View Warning Checks ({len(warnings)})", expanded=not acknowledged):
                    for w in warnings:
                        st.markdown(f"- {w}")
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
                if st.button("More Details", key=f"more_det_{index}_{eval_hash}", use_container_width=True):
                    more_details_dialog(item)
            with c_strat:
                if st.button("Suggest Defense", key=f"gen_strat_{index}_{eval_hash}", use_container_width=True):
                    defense_strategy_dialog(scores_dict)
            with c_del:
                if st.button("❌", key=f"close_eval_{index}_{eval_hash}", help="Close this result"):
                    st.session_state["evaluated_papers_buffer"].pop(index)
                    st.rerun()

if (
    st.session_state["evaluated_papers_buffer"]
    or st.session_state.get("download_errors")
):
    st.markdown("### Assessment Results")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.get("download_errors"):
        st.markdown("#### Publisher Access & Download Restrictions")
        for err_idx, err_data in enumerate(
            st.session_state["download_errors"]
        ):
            err_col1, err_col2 = st.columns([6, 1], gap="medium")
            with err_col1:
                st.warning(
                    f"**Could not directly download PDF for '{err_data['title']}':**"
                    f" Publishers restrict direct binary access.\n\n- **DOI:**"
                    f" `{err_data['doi']}`\n- **PDF URL Link:**"
                    f" [{err_data['url']}]({err_data['url']})"
                )
            with err_col2:
                if st.button(
                    "Close",
                    key=f"close_err_{err_idx}_{st.session_state['reset_token']}",
                ):
                    st.session_state["download_errors"].pop(err_idx)
                    st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

    for item_idx, item in enumerate(st.session_state["evaluated_papers_buffer"]):
        render_breakdown_item(item, item_idx)

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
        dataloader = DataLoader(
            dataset, batch_size=min(4, max(1, len(dataset))), shuffle=False
        )

        model = PidyneLSTM()
        weights_path = os.path.join(BASE_DIR, "pidyne_weights.pt")
        if os.path.exists(weights_path):
            try:
                model.load_state_dict(torch.load(weights_path, weights_only=True))
            except Exception:
                pass

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
            raw_pred = (
                model(
                    torch.tensor(
                        weight_data[-actual_lookback:], dtype=torch.float32
                    ).unsqueeze(0)
                )
                .squeeze()
                .numpy()
            )
            current_w = weight_data[-1]
            predicted = current_w + (raw_pred - current_w) * 20.0
            predicted = np.clip(predicted, 0.01, 7.9)
            predicted = predicted * (8.0 / np.sum(predicted))
            torch.save(model.state_dict(), weights_path)
            return predicted

    conn_pb = get_db_connection()
    try:
        cursor_pb = conn_pb.cursor()
        cursor_pb.execute(
            "SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights ORDER"
            " BY block_height ASC"
        )
        historical_rows = cursor_pb.fetchall()
    finally:
        conn_pb.close()

    min_blocks_required = 2
    if len(historical_rows) < min_blocks_required:
        st.warning(
            f"Not enough blockchain data to train the meta-model. You need at least"
            f" {min_blocks_required} blocks (Currently on ledger:"
            f" {len(historical_rows)}). Assess at least 1 manuscript to generate"
            " block 2."
        )
    else:
        current_block_count = len(historical_rows)
        lookback_window = max(1, min(actual_lookback, current_block_count - 1))

        if (
            "last_trained_blocks" not in st.session_state
            or st.session_state.last_trained_blocks != current_block_count
            or st.session_state.get("last_lookback") != lookback_window
        ):
            weight_data = np.array(historical_rows, dtype=np.float32)

            st.session_state.predicted_next_weights = train_pidyne_cached(weight_data, lookback_window)
            st.session_state.current_weights = weight_data[-1]
            st.session_state.last_trained_blocks = current_block_count
            st.session_state.last_lookback = lookback_window

        if len(historical_rows) > 0:
            sliced_rows = historical_rows[-(lookback_window + 1):] if len(historical_rows) > lookback_window else historical_rows
            df_history = pd.DataFrame(
                sliced_rows,
                columns=[
                    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"
                ]
            )
            df_history.index.name = "Block / Epoch"
            
            df_amplified = df_history.copy()
            for col in df_amplified.columns:
                df_amplified[col] = 1.0 + (df_amplified[col] - 1.0) * 1500.0
                
            df_melted = df_amplified.reset_index().melt('Block / Epoch', var_name='Criterion', value_name='Weight (Amplified)')
            
            base = alt.Chart(df_melted).mark_line(point=True).encode(
                x='Block / Epoch:O',
                y=alt.Y('Weight (Amplified):Q', scale=alt.Scale(zero=False)),
                color='Criterion:N',
                tooltip=['Block / Epoch', 'Criterion', 'Weight (Amplified)']
            ).properties(height=350)
            st.altair_chart(base, use_container_width=True)

        st.markdown("#### Evaluation Metrics")
        with st.container(border=True):
            st.markdown(f"**Ledger Forecast (Raw Sum = {sum(st.session_state.predicted_next_weights):.6f}/8.0):**")
            
            crit_info = get_criteria_info(st.session_state.predicted_next_weights)
            
            cols1 = st.columns(4, gap="small")
            for idx, c_data in enumerate(crit_info[:4]):
                with cols1[idx]:
                    if st.button(f"{c_data[0]}: {c_data[3]:.5f}", key=f"btn_crit_{c_data[0]}", use_container_width=True):
                        criterion_details_dialog(*c_data)
                        
            cols2 = st.columns(4, gap="small")
            for idx, c_data in enumerate(crit_info[4:]):
                with cols2[idx]:
                    if st.button(f"{c_data[0]}: {c_data[3]:.5f}", key=f"btn_crit_{c_data[0]}", use_container_width=True):
                        criterion_details_dialog(*c_data)

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

    if "mod_repulsion" not in st.session_state:
        st.session_state.mod_repulsion = -3000
    if "mod_spring" not in st.session_state:
        st.session_state.mod_spring = 180
    if "mod_size" not in st.session_state:
        st.session_state.mod_size = 1.5
    if "mod_gravity" not in st.session_state:
        st.session_state.mod_gravity = 0.15

    filter_key = f"top_author_filter_{st.session_state['assessment_update_token']}"
    if filter_key not in st.session_state:
        st.session_state[filter_key] = "All Authors"

    current_filter = st.session_state.get(filter_key, "All Authors")
    selected_author_top = None if current_filter == "All Authors" else current_filter

    interactive_html_top, table_html_top = render_bubble_chart_clean(
        selected_author_top,
        repulsion=st.session_state.mod_repulsion,
        spring_len=st.session_state.mod_spring,
        size_scale=st.session_state.mod_size,
        central_grav=st.session_state.mod_gravity
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
            st.selectbox(
                "Filter Map by Author:",
                ["All Authors"] + all_global_authors,
                key=filter_key,
                format_func=lambda x: (
                    f"{x} (piQ: {piq_dict.get(x, 0.0):.2f})" if x != "All Authors" else x
                ),
            )
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
        st.markdown(table_html_top, unsafe_allow_html=True)

st.markdown("---")

if st.session_state.is_authenticated:
    conn_hist = get_db_connection()
    try:
        cur_h = conn_hist.cursor()
        history_clauses = []
        history_params = []
        if st.session_state.auth_method == "Web3" and w3.is_address(st.session_state.orcid_id):
            history_clauses.append("p.eth_book = ?")
            history_params.append(st.session_state.orcid_id)
        if st.session_state.auth_method == "Academic ID" and st.session_state.academic_id not in ("None", ""):
            history_clauses.append("p.user_id = ?")
            history_params.append(st.session_state.academic_id)

        if history_clauses:
            cur_h.execute(
                f"""SELECT p.title, p.author_name, p.filename, p.final_score, p.logic_score, 
                          p.piq_minted, p.tx_hash, p.zk_proof, p.eval_hash, p.timestamp,
                          b.block_height, b.block_hash, p.mdar_adherence_score, 
                          p.rrid_valid_count, p.reproducibility_score,
                          p.consensus_data, p.evidence_report, p.scilem_score,
                          p.c1, p.c2, p.c3, p.c4, p.c5, p.c6, p.c7, p.c8
                   FROM papers_assessment p
                   LEFT JOIN blockchain_por_weights b ON p.eval_hash = b.eval_hash
                   WHERE {' OR '.join(history_clauses)}
                   ORDER BY p.timestamp DESC""",
                tuple(history_params)
            )
        else:
            cur_h.execute("SELECT NULL WHERE 0")
        user_history_rows = cur_h.fetchall()
    finally:
        conn_hist.close()

    st.markdown("### Your Assessment History")

    if user_history_rows:
        for idx, uh in enumerate(user_history_rows):
            (
                u_title, u_author, u_filename, u_score, u_logic,
                u_piq, u_tx, u_zk, u_hash, u_time,
                u_block_height, u_block_hash, u_mdar, u_rrid, u_repro,
                u_consensus, u_report, u_scilem,
                u_c1, u_c2, u_c3, u_c4, u_c5, u_c6, u_c7, u_c8
            ) = uh

            u_author_clean = clean_author_name(u_author)
            u_book = "0x" + hashlib.sha256(u_author_clean.encode()).hexdigest()[:40]
            u_tx_url = safe_get_sepolia_url(u_tx)
            
            tx_disp_val = u_tx if u_tx and str(u_tx).strip() not in ["None", ""] else "Not Connected / No Book / Missing PK"

            with st.expander(
                f"[{idx+1}] {u_title[:50]}... — *{u_author_clean}* (Score: **{safe_float(u_score, 0.0):.2f}** | piQ: `{u_piq}`)",
                expanded=False,
            ):
                st.write(f"**File Name:** {u_filename if u_filename else 'N/A'}")
                st.write(f"**Evaluation Hash (Paper Address):** `{u_hash}`")
                st.write(f"**Unique Book Address:** `{u_book}`")
                st.write(f"**piQ Minted:** `{u_piq}`")
                st.markdown(f"**zk-SNARK Proof:** `{u_zk}`", unsafe_allow_html=True)
                
                if u_tx_url:
                    st.markdown(f"**Tx Hash:** [`{tx_disp_val}`]({u_tx_url})")
                else:
                    st.write(f"**Tx Hash:** `{tx_disp_val}`")

                st.markdown(f"**Executable Reproducibility Score:** `{safe_float(u_repro, 0.0) * 100:.1f}%`", unsafe_allow_html=True)
                st.markdown(f"**SciScore MDAR Adherence:** `{safe_float(u_mdar, 0.0) * 100:.1f}%` | **Valid RRIDs:** `{u_rrid}`", unsafe_allow_html=True)

                if st.button("View Full Multi-LLM & Scilem Dossier", key=f"hist_det_{idx}_{u_hash}"):
                    hist_item = {
                        "title": u_title,
                        "author_name": u_author,
                        "score": safe_float(u_score, 0.0),
                        "logic_integrity": safe_float(u_logic, 75.0),
                        "scores_dict": {
                            "C1_Semantic_Originality": safe_float(u_c1, 0), "C2_Methodological_Rigor_SciScore": safe_float(u_c2, 0),
                            "C3_Interdisciplinary_Entropy": safe_float(u_c3, 0), "C4_Societal_Impact": safe_float(u_c4, 0),
                            "C5_Open_Science_Repro": safe_float(u_c5, 0), "C6_Literature_Integration": safe_float(u_c6, 0),
                            "C7_Empirical_Density": safe_float(u_c7, 0), "C8_Future_Actionability_FAIR": safe_float(u_c8, 0)
                        },
                        "used_weights": [1.0]*8,
                        "eval_hash": u_hash,
                        "piq": safe_float(u_piq, 0.0),
                        "tx_hash": u_tx,
                        "zk_proof": u_zk,
                        "h_idx": safe_float(u_mdar, 0.0),
                        "i10_idx": int(safe_float(u_rrid, 0)),
                        "repro_score": safe_float(u_repro, 0.0),
                        "filename": u_filename or "N/A",
                        "warnings": [],
                        "consensus_raw": json.loads(u_consensus) if u_consensus else {},
                        "evidence_report_text": u_report or "",
                        "scilem_rating": safe_float(u_scilem, 50.0)
                    }
                    more_details_dialog(hist_item)
    else:
        st.info("No assessment history or rewards found linked to this authenticated ID.")
    st.markdown("---")

side_col1, side_col2 = st.columns(2, gap="large")

with side_col1:
    st.markdown("### Pi Quotient (piQ) Leaderboard [Top Authors]")
    piq_dict, book_dict = get_author_piq_dict()
    if piq_dict:
        sorted_leaderboard = sorted(piq_dict.items(), key=lambda x: x[1], reverse=True)[:20]
        
        h_c1, h_c2, h_c3, h_c4 = st.columns([0.8, 3.2, 4.5, 1.5])
        h_c1.markdown("<div style='color:#64748b; font-size:12px; font-weight:700; text-transform:uppercase;'>#</div>", unsafe_allow_html=True)
        h_c2.markdown("<div style='color:#64748b; font-size:12px; font-weight:700; text-transform:uppercase;'>Author</div>", unsafe_allow_html=True)
        h_c3.markdown("<div style='color:#64748b; font-size:12px; font-weight:700; text-transform:uppercase;'>Book Address</div>", unsafe_allow_html=True)
        h_c4.markdown("<div style='color:#64748b; font-size:12px; font-weight:700; text-transform:uppercase;'>piQ</div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin:4px 0px 8px 0px; border-top: 2px solid #e2e8f0;'>", unsafe_allow_html=True)

        piq_scroll = st.container(height=380)
        with piq_scroll:
            for rank, (author, piq) in enumerate(sorted_leaderboard, start=1):
                book_addr = book_dict.get(author, "None")
                r_c1, r_c2, r_c3, r_c4 = st.columns([0.8, 3.2, 4.5, 1.5], vertical_alignment="center")
                r_c1.markdown(f"**{rank}**")
                r_c2.markdown(f"**{author}**")
                r_c3.markdown(f"`{book_addr}`")
                r_c4.markdown(f"**{safe_float(piq, 0.0):.2f}**")
    else:
        st.info("No piQ tokens minted yet.")

with side_col2:
    st.markdown("### pi-Index (piX) Leaderboard [Top Papers]")
    conn_pi = get_db_connection()
    try:
        cur_pi = conn_pi.cursor()
        cur_pi.execute(
            """SELECT title, author_name, final_score, logic_score, 
                      c1, c2, c3, c4, c5, c6, c7, c8, 
                      piq_minted, tx_hash, zk_proof, mdar_adherence_score, 
                      rrid_valid_count, reproducibility_score, eval_hash, filename,
                      consensus_data, evidence_report, scilem_score
               FROM papers_assessment ORDER BY final_score DESC LIMIT 20"""
        )
        top_papers = cur_pi.fetchall()
    finally:
        conn_pi.close()
    
    if top_papers:
        h_c1, h_c2, h_c3, h_c4 = st.columns([0.8, 4.2, 2.5, 2.5])
        h_c1.markdown("<div style='color:#64748b; font-size:12px; font-weight:700; text-transform:uppercase;'>#</div>", unsafe_allow_html=True)
        h_c2.markdown("<div style='color:#64748b; font-size:12px; font-weight:700; text-transform:uppercase;'>Manuscript Title</div>", unsafe_allow_html=True)
        h_c3.markdown("<div style='color:#64748b; font-size:12px; font-weight:700; text-transform:uppercase;'>Author</div>", unsafe_allow_html=True)
        h_c4.markdown("<div style='color:#64748b; font-size:12px; font-weight:700; text-transform:uppercase;'>Score / Action</div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin:4px 0px 8px 0px; border-top: 2px solid #e2e8f0;'>", unsafe_allow_html=True)

        pix_scroll = st.container(height=380)
        with pix_scroll:
            for rank, tp in enumerate(top_papers, start=1):
                (
                    p_title, p_author, p_filename, p_score, p_logic,
                    p_c1, p_c2, p_c3, p_c4, p_c5, p_c6, p_c7, p_c8,
                    p_piq, p_tx, p_zk, p_mdar, p_rrid, p_repro, p_hash,
                    p_consensus, p_report, p_scilem
                ) = tp
                clean_auth = clean_author_name(p_author)
                r_c1, r_c2, r_c3, r_c4 = st.columns([0.8, 4.2, 2.5, 2.5], vertical_alignment="center")
                r_c1.markdown(f"**{rank}**")
                r_c2.markdown(f"**{p_title}**")
                r_c3.markdown(f"*{clean_auth}*")
                with r_c4:
                    if st.button("View Dossier", key=f"pix_row_dossier_{rank}_{p_hash}", use_container_width=True):
                        item_dossier = {
                            "title": p_title,
                            "author_name": p_author,
                            "score": safe_float(p_score, 0.0),
                            "logic_integrity": safe_float(p_logic, 75.0),
                            "scores_dict": {
                                "C1_Semantic_Originality": safe_float(p_c1, 0), "C2_Methodological_Rigor_SciScore": safe_float(p_c2, 0),
                                "C3_Interdisciplinary_Entropy": safe_float(p_c3, 0), "C4_Societal_Impact": safe_float(p_c4, 0),
                                "C5_Open_Science_Repro": safe_float(p_c5, 0), "C6_Literature_Integration": safe_float(p_c6, 0),
                                "C7_Empirical_Density": safe_float(p_c7, 0), "C8_Future_Actionability_FAIR": safe_float(p_c8, 0)
                            },
                            "used_weights": [1.0]*8,
                            "eval_hash": p_hash,
                            "piq": safe_float(p_piq, 0.0),
                            "tx_hash": p_tx,
                            "zk_proof": p_zk,
                            "h_idx": safe_float(p_mdar, 0.0),
                            "i10_idx": int(safe_float(p_rrid, 0)),
                            "repro_score": safe_float(p_repro, 0.0),
                            "filename": p_filename or "N/A",
                            "warnings": [],
                            "consensus_raw": json.loads(p_consensus) if p_consensus else {},
                            "evidence_report_text": p_report or "",
                            "scilem_rating": safe_float(p_scilem, 50.0)
                        }
                        more_details_dialog(item_dossier)
    else:
        st.info("No assessments recorded for Pi-Index leaderboard yet.")

st.markdown("---")

st.markdown("### Latest Assessed Papers")
conn_recent = get_db_connection()
try:
    cur_recent = conn_recent.cursor()
    cur_recent.execute(
        """SELECT p.title, p.author_name, p.filename, p.final_score, p.logic_score, 
                  p.c1, p.c2, p.c3, p.c4, p.c5, p.c6, p.c7, p.c8, 
                  p.piq_minted, p.tx_hash, p.zk_proof, p.mdar_adherence_score, 
                  p.rrid_valid_count, p.reproducibility_score, p.eval_hash, p.timestamp,
                  b.block_height, b.block_hash,
                  p.consensus_data, p.evidence_report, p.scilem_score
           FROM papers_assessment p
           LEFT JOIN blockchain_por_weights b ON p.eval_hash = b.eval_hash
           ORDER BY p.timestamp DESC LIMIT 20"""
    )
    merged_papers = cur_recent.fetchall()
finally:
    conn_recent.close()

if merged_papers:
    st.markdown("<p style='font-size:13px; color:#64748b; margin-bottom:10px;'>Scroll to view more records. Click <b>View Dossier</b> on any manuscript card to open its complete research integrity record:</p>", unsafe_allow_html=True)
    
    h_c1, h_c2, h_c3, h_c4 = st.columns([1.5, 4.5, 2.0, 2.0])
    h_c1.markdown("<div style='color:#64748b; font-size:12px; font-weight:700; text-transform:uppercase;'>Block</div>", unsafe_allow_html=True)
    h_c2.markdown("<div style='color:#64748b; font-size:12px; font-weight:700; text-transform:uppercase;'>Manuscript & Author</div>", unsafe_allow_html=True)
    h_c3.markdown("<div style='color:#64748b; font-size:12px; font-weight:700; text-transform:uppercase;'>Score / piQ</div>", unsafe_allow_html=True)
    h_c4.markdown("<div style='color:#64748b; font-size:12px; font-weight:700; text-transform:uppercase;'>Action</div>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:4px 0px 8px 0px; border-top: 2px solid #e2e8f0;'>", unsafe_allow_html=True)

    recent_scroll_container = st.container(height=420)
    with recent_scroll_container:
        for idx, mp in enumerate(merged_papers):
            (
                m_title, m_author, m_filename, m_score, m_logic,
                m_c1, m_c2, m_c3, m_c4, m_c5, m_c6, m_c7, m_c8,
                m_piq, m_tx, m_zk, m_mdar, m_rrid, m_repro, m_hash, m_time,
                m_block_height, m_block_hash,
                m_consensus, m_report, m_scilem
            ) = mp
            
            bh = m_block_height if m_block_height is not None else "Pending"
            clean_auth = clean_author_name(m_author)
            
            r_col1, r_col2, r_col3, r_col4 = st.columns([1.5, 4.5, 2.0, 2.0], vertical_alignment="center")
            with r_col1:
                st.markdown(f"**Block {bh}**")
            with r_col2:
                st.markdown(f"**{m_title}**")
                st.markdown(f"*{clean_auth}*")
            with r_col3:
                st.markdown(f"`{safe_float(m_score, 0.0):.2f}` / `{safe_float(m_piq, 0.0):.2f}`")
            with r_col4:
                if st.button("View Dossier", key=f"native_row_dossier_{idx}_{m_hash}", use_container_width=True):
                    item_dossier = {
                        "title": m_title,
                        "author_name": m_author,
                        "score": safe_float(m_score, 0.0),
                        "logic_integrity": safe_float(m_logic, 75.0),
                        "scores_dict": {
                            "C1_Semantic_Originality": safe_float(m_c1, 0), "C2_Methodological_Rigor_SciScore": safe_float(m_c2, 0),
                            "C3_Interdisciplinary_Entropy": safe_float(m_c3, 0), "C4_Societal_Impact": safe_float(m_c4, 0),
                            "C5_Open_Science_Repro": safe_float(m_c5, 0), "C6_Literature_Integration": safe_float(m_c6, 0),
                            "C7_Empirical_Density": safe_float(m_c7, 0), "C8_Future_Actionability_FAIR": safe_float(m_c8, 0)
                        },
                        "used_weights": [1.0]*8,
                        "eval_hash": m_hash,
                        "piq": safe_float(m_piq, 0.0),
                        "tx_hash": m_tx,
                        "zk_proof": m_zk,
                        "h_idx": safe_float(m_mdar, 0.0),
                        "i10_idx": int(safe_float(m_rrid, 0)),
                        "repro_score": safe_float(m_repro, 0.0),
                        "filename": m_filename or "N/A",
                        "warnings": [],
                        "consensus_raw": json.loads(m_consensus) if m_consensus else {},
                        "evidence_report_text": m_report or "",
                        "scilem_rating": safe_float(m_scilem, 50.0)
                    }
                    more_details_dialog(item_dossier)
else:
    st.info("No paper assessments recorded on ledger yet.")

st.markdown("---")

exp_head_col1, exp_head_col2 = st.columns([12, 1], vertical_alignment="center")
with exp_head_col1:
    st.markdown("### Proof-of-Research Blockchain Explorer", unsafe_allow_html=True)
with exp_head_col2:
    with st.popover("ⓘ", help="View Extra Ledger Info"):
        st.markdown(
            "**Proof-of-Research (PoR) Validation:** Anchors assessment outcomes on the Sepolia testnet, "
            "sealing the block index, criteria weights, and unalterable state hashes (`formulas_hash`) "
            "into a cryptographically verified SHA-256 block."
        )
        conn_pop = get_db_connection()
        try:
            cur_pop = conn_pop.cursor()
            cur_pop.execute(
                "SELECT por_proof, block_hash, formulas_hash FROM blockchain_por_weights ORDER BY block_height DESC LIMIT 1"
            )
            p_data = cur_pop.fetchone()
        except Exception:
            p_data = None
        finally:
            conn_pop.close()
            
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
        cursor.execute(
            "SELECT block_height, w1, w2, w3, w4, w5, w6, w7, w8, model_used,"
            " eval_hash, block_hash, por_proof, formulas_hash FROM"
            " blockchain_por_weights ORDER BY block_height DESC LIMIT 1"
        )
        epoch_data = cursor.fetchone()
    except Exception:
        epoch_data = None

    if epoch_data:
        explore_col1, explore_col2 = st.columns([3, 1], vertical_alignment="bottom")
        with explore_col1:
            search_query = st.text_input(
                "Search Ledger",
                placeholder="Enter Evaluation Hash, Block Hash, Paper Name, Author Name, or Book Address...",
                label_visibility="collapsed",
                key="pidyne_ledger_search_query"
            )
        with explore_col2:
            search_btn = st.button("Verify Ledger Record", key="pidyne_verify_record_btn", use_container_width=True)

        if search_btn and search_query:
            try:
                q_term = f"%{search_query.strip()}%"
                cursor.execute(
                    """SELECT p.title, p.author_name, p.filename, p.final_score, p.logic_score, 
                              p.c1, p.c2, p.c3, p.c4, p.c5, p.c6, p.c7, p.c8, 
                              p.piq_minted, p.tx_hash, p.zk_proof, p.mdar_adherence_score, 
                              p.rrid_valid_count, p.reproducibility_score, p.eval_hash, p.timestamp,
                              b.block_height, b.block_hash, b.por_proof, b.formulas_hash, p.eth_book,
                              p.consensus_data, p.evidence_report, p.scilem_score
                       FROM papers_assessment p
                       LEFT JOIN blockchain_por_weights b ON p.eval_hash = b.eval_hash
                       WHERE b.block_hash LIKE ? OR p.eval_hash LIKE ? OR p.title LIKE ? OR p.author_name LIKE ? OR p.eth_book LIKE ?
                       LIMIT 5""",
                    (q_term, q_term, q_term, q_term, q_term)
                )
                matched_records = cursor.fetchall()
                if matched_records:
                    st.success(f"Found {len(matched_records)} matching record(s) on ledger.")
                    for m_idx, mr in enumerate(matched_records):
                        (
                            m_title, m_author, m_filename, m_score, m_logic,
                            m_c1, m_c2, m_c3, m_c4, m_c5, m_c6, m_c7, m_c8,
                            m_piq, m_tx, m_zk, m_mdar, m_rrid, m_repro, m_hash, m_time,
                            m_block_height, m_block_hash, m_por, m_form, m_book_addr,
                            m_consensus, m_report, m_scilem
                        ) = mr

                        m_author_clean = clean_author_name(m_author)
                        m_book = m_book_addr if m_book_addr else ("0x" + hashlib.sha256(m_author_clean.encode()).hexdigest()[:40])
                        m_tx_url = safe_get_sepolia_url(m_tx)
                        
                        tx_disp_val = m_tx if m_tx and str(m_tx).strip() not in ["None", ""] else "Not Connected / No Book / Missing PK"

                        with st.expander(
                            f"[{m_idx+1}] {m_title[:65]}... — *{m_author_clean}* (Score:"
                            f" **{safe_float(m_score, 0.0):.2f}** | {m_time[:16]})",
                            expanded=True,
                        ):
                            st.write(f"**File Name:** {m_filename if m_filename else 'N/A'}")
                            st.write(f"**Evaluation Hash (Paper Address):** `{m_hash}`")
                            st.write(f"**Unique Book Address:** `{m_book}`")
                            st.write(f"**piQ Minted:** `{m_piq}`")
                            st.markdown(f"**zk-SNARK Proof:** `{m_zk}`", unsafe_allow_html=True)
                            
                            if m_tx_url:
                                st.markdown(f"**Tx Hash:** [`{tx_disp_val}`]({m_tx_url})")
                            else:
                                st.write(f"**Tx Hash:** `{tx_disp_val}`")

                            st.markdown(f"**Executable Reproducibility Score:** `{safe_float(m_repro, 0.0) * 100:.1f}%`", unsafe_allow_html=True)
                            st.markdown(f"**SciScore MDAR Adherence:** `{safe_float(m_mdar, 0.0) * 100:.1f}%` | **Valid RRIDs:** `{m_rrid}`", unsafe_allow_html=True)

                            if st.button("View Full Multi-LLM & Scilem Dossier", key=f"search_det_{m_idx}_{m_hash}"):
                                search_item = {
                                    "title": m_title,
                                    "author_name": m_author,
                                    "score": safe_float(m_score, 0.0),
                                    "logic_integrity": safe_float(m_logic, 75.0),
                                    "scores_dict": {
                                        "C1_Semantic_Originality": safe_float(m_c1, 0), "C2_Methodological_Rigor_SciScore": safe_float(m_c2, 0),
                                        "C3_Interdisciplinary_Entropy": safe_float(m_c3, 0), "C4_Societal_Impact": safe_float(m_c4, 0),
                                        "C5_Open_Science_Repro": safe_float(m_c5, 0), "C6_Literature_Integration": safe_float(m_c6, 0),
                                        "C7_Empirical_Density": safe_float(m_c7, 0), "C8_Future_Actionability_FAIR": safe_float(m_c8, 0)
                                    },
                                    "used_weights": [1.0]*8,
                                    "eval_hash": m_hash,
                                    "piq": safe_float(m_piq, 0.0),
                                    "tx_hash": m_tx,
                                    "zk_proof": m_zk,
                                    "h_idx": safe_float(m_mdar, 0.0),
                                    "i10_idx": int(safe_float(m_rrid, 0)),
                                    "repro_score": safe_float(m_repro, 0.0),
                                    "filename": m_filename or "N/A",
                                    "warnings": [],
                                    "consensus_raw": json.loads(m_consensus) if m_consensus else {},
                                    "evidence_report_text": m_report or "",
                                    "scilem_rating": safe_float(m_scilem, 50.0)
                                }
                                more_details_dialog(search_item)
                else:
                    st.error(
                        "No records matching that evaluation hash, block hash, paper name, author name, or book address were found on the ledger."
                    )
            except Exception as e:
                st.error(f"Error reading database: {str(e)}")

finally:
    conn.close()

@st.dialog("The Pi-Index Framework: Next-Gen Architecture & CoARA Compliance Workflow", width="large")
def framework_workflow_dialog():
    st.markdown(
        "Pi-Index filters noise and yields quantitative results strictly aligned with **Responsible Research Assessment (RRA)** and **CoARA** (Coalition for Advancing Research Assessment) guidelines.\n\n"
        "### Architecture Flowchart & Whitepaper DOI\n\n"
        "Read the foundational framework whitepaper and preprints via [Ali Vafadar Yengejeh's ResearchGate Profile](https://www.researchgate.net/profile/Ali-Vafadar-Yengejeh).\n\n"
        "The enhanced system architecture flow below details the decentralized intake, ZK double-blind reviewer assignment, SciScore deterministic parsing, Item Response Theory (IRT) calibration, and smart contract slashing mechanisms."
    )

    st.graphviz_chart("""
    digraph PiIndexSystemOverview {
        rankdir=TB;
        compound=true;
        fontname="Helvetica,Arial,sans-serif";
        node [fontname="Helvetica,Arial,sans-serif", style=filled, margin=0.2];
        edge [fontname="Helvetica,Arial,sans-serif", fontsize=10];

        node [shape=box, fillcolor="#f8f9fa", color="#2c3e50", penwidth=1.5];

        subgraph cluster_intake {
            label = "1. Unified Multi-Source Intake & ZK-Identity Registry (ZIP-600)";
            style = rounded;
            color = "#34495e";
            fillcolor = "#ecf0f1";

            Auth [label="Researcher Authentication\n• ORCID iD / W3C DID Verification\n• ZK-Email Institutional Proof", fillcolor="#aed6f1"];
            Intake [label="Multi-Source Ingestion Engine\n• Local Binary PDFs Extraction\n• Unpaywall DOI Resolver\n• OpenAlex Topic API Search", fillcolor="#aed6f1"];
            TempDisk [label="Temp Disk State Management\n• Streamlit Render Protection\n• Buffered Binary Writes", fillcolor="#aed6f1", style="dashed,filled"];
            ZKBlind [label="ZK Double-Blind Assignment\n• Merkle Tree Non-Membership Proofs\n• Anonymous Author Shielding", fillcolor="#aed6f1"];
            
            Auth -> Intake -> TempDisk -> ZKBlind;
        }

        subgraph cluster_eval {
            label = "2. Core Evaluation & Adversarial Analysis Pipeline (CoARA/RRA)";
            style = rounded;
            color = "#27ae60";
            fillcolor = "#e8f8f5";

            PyMuPDF [label="PyMuPDF Layout Sort\n• Spatial Reading Extraction\n• Mathematical Integrity Safeguard", fillcolor="#a3e4d7", style="dashed,filled"];
            SciParser [label="Deterministic SciScore API\n• MDAR Reporting Adherence\n• Valid RRIDs Count Extraction", fillcolor="#a3e4d7"];
            Retry [label="Multi-LLM Consensus Engine\n• Llama, Mistral, Qwen, Gemini & Scilem Analysis\n• Synthesized Evidence Report", fillcolor="#a3e4d7", style="dashed,filled"];
            IRTCalib [label="Item Response Theory Calibration\n• Counterfactual Stress Testing\n• Variance & Difficulty Mapping", fillcolor="#a3e4d7"];
            Criteria [label="8 Transparent Criteria Rubrics\n• C1 Originality to C8 FAIR Actionability\n• Formulaic Score Computation", fillcolor="#a3e4d7"];
            Logic [label="Adversarial Logic Integrity Matrix\n• Premise Validity & Evidence Strength\n• AI Hallucination & Laundering Penalty", fillcolor="#a3e4d7"];
            
            PyMuPDF -> SciParser -> Retry -> IRTCalib -> Criteria -> Logic;
        }

        subgraph cluster_blockchain {
            label = "3. Blockchain Consensus, Cryptographic Proofs & Slashing Tokenomics";
            style = rounded;
            color = "#8e44ad";
            fillcolor = "#f4ecf7";

            PoR [label="Proof-of-Research (PoR) Validation\n• Dynamic Epoch Weight Shifting\n• Formulas Hash Stamping & SHA-256 Block", fillcolor="#d7bde2"];
            Slashing [label="Anti-Laundering Slashing Guard\n• Smart Contract piQ Burn for Fraud\n• Stake Penalty Enforcement", fillcolor="#f5b7b1"];
            Mint [label="Soulbound Token Minting\n• Author-Specific Book Address (eth_book)\n• Shared Paper Address (eval_hash) & Tx Hash", fillcolor="#d7bde2"];
            
            PoR -> Slashing -> Mint;
        }

        subgraph cluster_outputs {
            label = "4. User Interface, Cartography & Institutional Policy Support";
            style = rounded;
            color = "#d35400";
            fillcolor = "#fef5e7";

            Dossier [label="CoARA & DORA-Aligned Dossier\n• Markdown Research Integrity Report\n• AI Defense Rebuttal Strategy", fillcolor="#f8c471"];
            Cartography [label="Global Map of Science\n• Ledger PyVis Network Cartography\n• Author & Topic Bubble Filtering", fillcolor="#f8c471"];
            PidyneBrain [label="Pidyne LSTM Meta-Learning\n• PyTorch Temporal Weight Prediction\n• Calibration Drift & Epoch Forecasting", fillcolor="#f8c471"];
        }

        ZKBlind -> PyMuPDF [lhead=cluster_eval, label="Processed Manuscript Text"];
        Logic -> PoR [lhead=cluster_blockchain, label="Audited Score & Hashes"];
        Mint -> Dossier [lhead=cluster_outputs, label="Ledger Seal & Tokens"];
        Mint -> Cartography;
        Mint -> PidyneBrain;
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
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.9em; padding-bottom: 5px;'>Framework Author: Ali Vafadar Yengejeh | Universita degli Studi di Milano-Bicocca</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

col_pad1, col_center, col_pad2 = st.columns([1, 2, 1])
with col_center:
    if st.button("The Pi-Index Framework Workflow", use_container_width=True):
        framework_workflow_dialog()
