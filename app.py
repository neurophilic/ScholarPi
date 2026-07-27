import os
import re
import json
import time
import hashlib
import tempfile
import shutil
import colorsys
import logging
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

from config import BASE_DIR, EPOCH_BLOCK_SIZE, PIQ_CONTRACT_ADDRESS, REGISTRY_CONTRACT_ADDRESS
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

# System Action Log Monitor
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
        share = piq / len(alist)
        for a in alist:
            author_piq[a] = author_piq.get(a, 0.0) + share
            author_book[a] = eth_book if eth_book and w3.is_address(eth_book) else "Unbound / Escrow"
    return author_piq, author_book

def rbot(topic_key):
    return f"<span class='scilem-trigger' data-query='{topic_key}' title='Ask Scilem' style='cursor: pointer !important; opacity:0.8;'>[?]</span>"

custom_ui_code = """
<style>
.stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a, 
.stMarkdown h4 a, .stMarkdown h5 a, .stMarkdown h6 a,
[data-testid="stHeaderActionElements"] {
    display: none !important;
}

[data-testid="stSidebar"] {
    overflow-y: auto !important;
}

[data-testid="stChatMessage"]:has(div:contains("👤")) {
    flex-direction: row-reverse !important;
    background-color: #e8f0fe !important;
    border-radius: 10px 0 10px 10px !important;
    text-align: right !important;
    margin-left: 20px !important;
}

.scilem-trigger {
    font-size: 0.9em;
    margin-left: 4px;
    font-weight: bold;
    color: #2563eb;
    vertical-align: middle;
    display: inline-block;
    transition: transform 0.15s ease-in-out;
}
.scilem-trigger:hover {
    transform: scale(1.2);
}

#scilem-drag-handle {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    color: white;
    padding: 12px 16px;
    font-weight: 700;
    font-size: 15px;
    cursor: grab;
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
    margin: -1rem -1rem 0 -1rem;
    user-select: none;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    height: 48px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
#scilem-drag-handle:active {
    cursor: grabbing;
}

button[kind="secondaryFormSubmit"], button[kind="primaryFormSubmit"] {
    padding: 0.35rem 0.75rem !important;
    font-size: 14px !important;
    white-space: nowrap !important;
}

iframe {
    border: none !important;
    border-radius: 8px !important;
    outline: none !important;
    box-shadow: none !important;
    margin: 0 !important;
    padding: 0 !important;
}

.pyvis-map-wrapper iframe {
    width: 100% !important;
    height: 600px !important;
    display: block !important;
}
</style>

<script>
const parentDoc = window.parent.document;

parentDoc.addEventListener('click', function(e) {
    let handle = e.target.closest('#scilem-drag-handle');
    if (handle) {
        let block = handle.closest('[data-draggable="true"]');
        if (block && !window._wasDragging) {
            let isMin = block.getAttribute('data-minimized') === 'true';
            block.setAttribute('data-minimized', isMin ? 'false' : 'true');
            let children = Array.from(block.children);
            children.forEach(child => {
                if (child !== handle && !child.contains(handle)) {
                    child.style.display = isMin ? 'block' : 'none';
                }
            });
        }
        e.preventDefault();
        e.stopPropagation();
        return;
    }

    let trigger = e.target.closest('.scilem-trigger');
    if (!trigger) return; 
    e.preventDefault();
    e.stopPropagation();

    let query = trigger.getAttribute('data-query');
    if (!query) return;

    let chatBlock = parentDoc.querySelector('[data-draggable="true"]');
    if (!chatBlock) return;

    let inputField = chatBlock.querySelector('input[type="text"]');
    if (inputField) {
        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
        nativeInputValueSetter.call(inputField, "Explain: " + query);
        inputField.dispatchEvent(new Event('input', { bubbles: true }));
    }
}, true);

function initUI() {
    const handle = parentDoc.getElementById('scilem-drag-handle');
    if (handle) {
        let block = handle.closest('[data-testid="stVerticalBlock"]');
        if (block && block.getAttribute('data-draggable') !== 'true') {
            if (!block.innerText.includes("Live System Monitor")) {
                block.setAttribute('data-draggable', 'true');
                block.setAttribute('data-minimized', 'true');
                
                let children = Array.from(block.children);
                children.forEach(child => {
                    if (child !== handle && !child.contains(handle)) {
                        child.style.display = 'none';
                    }
                });
                
                let isDragging = false;
                let startX, startY, initialX, initialY;
                window._wasDragging = false;

                handle.addEventListener('mousedown', function(e) {
                    isDragging = true;
                    window._wasDragging = false;
                    startX = e.clientX;
                    startY = e.clientY;
                    const rect = block.getBoundingClientRect();
                    initialX = rect.left;
                    initialY = rect.top;
                    
                    block.style.position = 'fixed';
                    block.style.left = initialX + 'px';
                    block.style.top = initialY + 'px';
                    block.style.bottom = 'auto';
                    block.style.right = 'auto';
                    block.style.width = '380px';
                    block.style.backgroundColor = '#ffffff';
                    block.style.border = '1px solid #d0d7de';
                    block.style.borderRadius = '12px';
                    block.style.boxShadow = '0 10px 40px rgba(0,0,0,0.3)';
                    block.style.zIndex = '999999';
                    block.style.padding = '1rem';
                    block.style.transition = 'none'; 
                });

                parentDoc.addEventListener('mousemove', function(e) {
                    if (!isDragging) return;
                    let dx = e.clientX - startX;
                    let dy = e.clientY - startY;
                    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
                        window._wasDragging = true;
                    }
                    block.style.left = (initialX + dx) + 'px';
                    block.style.top = (initialY + dy) + 'px';
                });

                parentDoc.addEventListener('mouseup', function() { isDragging = false; });
            }
        }
    }
}
setInterval(initUI, 800);
</script>
"""
components.html(custom_ui_code, height=0, width=0)

st.sidebar.title("System Access")

if "initialized" not in st.session_state:
    st.session_state["initialized"] = True

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

conn_main = get_db_connection()
try:
    cur = conn_main.cursor()
    cur.execute("SELECT ip_address FROM auto_ip_tracking WHERE ip_address=?", (client_ip,))
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO auto_ip_tracking (ip_address, first_seen) VALUES (?, ?)",
            (client_ip, datetime.now().isoformat()),
        )
        conn_main.commit()

    cur.execute("SELECT COUNT(*) FROM papers_assessment")
    total_analyzed_count = cur.fetchone()[0]
finally:
    conn_main.close()

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
            "content": "**Welcome! I am Scilem.** Ask any research question or click indicators across criteria."
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
    st.sidebar.markdown("### 1. Authenticate Web3 Wallet")
    user_wallet = st.sidebar.text_input("Ethereum Wallet Address (EIP-4361)", placeholder="0x...")
    
    if st.sidebar.button("Connect Wallet"):
        if w3.is_address(user_wallet):
            with st.sidebar.status("Connecting to Identity Registry..."):
                clean_wallet = w3.to_checksum_address(user_wallet)
                st.session_state.orcid_id = clean_wallet
                st.session_state.orcid_name = "Verified Decentralized Identity"
                st.session_state.is_authenticated = True
                st.session_state.auth_method = "Web3"
                add_log(f"Identity Authenticated via SIWE: {clean_wallet}")
                st.rerun()
        else:
            st.sidebar.error("Invalid Ethereum Address format.")

    st.sidebar.markdown("### 2. Authenticate Academic ID")
    manual_id = st.sidebar.text_input("Enter ORCID iD or W3C DID", placeholder="0000-0000-0000-0000")
    if st.sidebar.button("Connect ID"):
        if validate_orcid_did(manual_id):
            st.session_state.academic_id = manual_id.strip()
            st.session_state.orcid_name = "Verified Academic Researcher"
            st.session_state.is_authenticated = True
            st.session_state.auth_method = "Academic ID"
            add_log(f"Identity Authenticated via Academic ID: {manual_id.strip()}")
            st.rerun()
        else:
            st.sidebar.error("Invalid ORCID or DID format.")

    st.sidebar.markdown("---")
    st.sidebar.info("Notice: Please connect your Web3 Ethereum Wallet or Academic ID above to unlock personal features.")
else:
    st.sidebar.success("Securely Connected")
    
    conn_hist = get_db_connection()
    total_user_piq = 0.0
    try:
        cur_h = conn_hist.cursor()
        if st.session_state.auth_method == "Web3":
            cur_h.execute("SELECT piq_minted FROM papers_assessment WHERE eth_book = ?", (st.session_state.orcid_id,))
        else:
            cur_h.execute("SELECT piq_minted FROM papers_assessment WHERE eth_book = ?", (st.session_state.academic_id,))
        piq_rows = cur_h.fetchall()
        total_user_piq = sum(float(r[0]) for r in piq_rows if r[0])
    finally:
        conn_hist.close()
        
    auth_disp = st.session_state.orcid_id if st.session_state.auth_method == "Web3" else st.session_state.academic_id
    
    st.sidebar.markdown(
        f"**Researcher:** {st.session_state.orcid_name}\n\n"
        f"**Connected ID:** `{auth_disp[:12]}...`\n\n"
        f"**TOTAL piQ AWARDED:** `{total_user_piq:.2f} piQ`"
    )

    if st.sidebar.button("Disconnect Session"):
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

scilem_container = st.sidebar.container()
with scilem_container:
    st.markdown("""
    <div id='scilem-drag-handle'>
        <div style="display: flex; align-items: center; justify-content: center; width: 100%;">
            <span>Scilem Assistant</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    floating_chat_container = st.container(height=240)
    with floating_chat_container:
        for idx, message in enumerate(st.session_state.scilem_messages):
            msg_avatar = "🧠" if message["role"] == "assistant" else "👤"
            with st.chat_message(message["role"], avatar=msg_avatar):
                st.markdown(message["content"])

    with st.form(key="scilem_floating_form", clear_on_submit=False):
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
        if st.button("Reset Scilem (Owner)"):
            msg = reset_scilem()
            st.session_state.scilem_messages = [
                {
                    "role": "assistant", 
                    "content": "**Scilem has been reset.** Neural weights cleared to baseline."
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
            score = float(final_score) if final_score else 50.0
            for rs in raw_subfields:
                if rs and rs.lower() not in exclude_terms:
                    s = refine_science_field(rs)
                    if s not in topic_aggregates:
                        topic_aggregates[s] = {"weight_sum": 0.0, "frequency": 0}
                    topic_aggregates[s]["weight_sum"] += score
                    topic_aggregates[s]["frequency"] += 1
        except Exception:
            continue

    if not topic_aggregates:
        topic_aggregates["Computer Science > Algorithms & Software Engineering"] = {
            "weight_sum": 50.0,
            "frequency": 1,
        }

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
            "barnesHut": {{ 
                "gravitationalConstant": {repulsion}, 
                "centralGravity": {central_grav}, 
                "springLength": {spring_len}, 
                "springConstant": 0.005,
                "damping": 1.0,
                "avoidOverlap": 2.0 
            }}, 
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
            shadow={"enabled": True, "color": "rgba(0,0,0,0.5)", "size": 6, "x": 3, "y": 3},
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
        canvas {{ background: radial-gradient(circle at 50% 50%, #ffffff 0%, #f0f2f5 100%); width: 100% !important; height: 600px !important; }}
        #mynetwork, .vis-network, .card-body {{ border: none !important; width: 100% !important; height: 600px !important; }}
    </style>
    </head>
    """
    html_string = html_string.replace("</head>", gradient_injection)

    table_html = "<style>.table-big { width: 100%; font-size: 13px; border-collapse: collapse; margin-top: 10px; } .table-big th { background-color: #2c3e50; color: white; padding: 6px; } .table-big td { padding: 6px; border-bottom: 1px solid #ecf0f1; } .color-box { width: 20px; height: 20px; border-radius: 4px; display: inline-block; } </style>"
    table_html += "<div class='legend-container'><table class='table-big'><thead><tr><th style='text-align: center;'>Color</th><th>Science Field</th><th style='text-align: center;'>Freq</th><th style='text-align: center;'>Avg Weight</th></tr></thead><tbody>"
    for topic, metrics in sorted(topic_aggregates.items(), key=lambda x: x[1]["frequency"], reverse=True):
        avg_w = metrics["weight_sum"] / metrics["frequency"]
        table_html += f"<tr><td style='text-align: center;'><div class='color-box' style='background-color:{color_map[topic]};'></div></td><td><b>{topic}</b></td><td style='text-align: center;'>{metrics['frequency']}</td><td style='text-align: center;'>{avg_w:.1f}</td></tr>"
    table_html += "</tbody></table></div>"

    return html_string, table_html

@st.dialog("Evaluation Metrics, SciScore Reproducibility & Adversarial Logic Engine", width="large")
def evaluation_metrics_dialog():
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT block_height, w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights ORDER BY block_height DESC LIMIT 1")
        top_epoch_data = cur.fetchone()
    finally:
        conn.close()

    if top_epoch_data:
        _, tw1, tw2, tw3, tw4, tw5, tw6, tw7, tw8 = top_epoch_data
    else:
        tw1, tw2, tw3, tw4, tw5, tw6, tw7, tw8 = [1.0]*8

    st.markdown("**Adversarial Logic Gap ($\Delta_{Logic}$):** Evaluates reasoning structure and penalizes claims unsupported by evidence.")
    st.markdown(r"$$ L_i = \left( (\mathcal{P}_{valid} \cdot \mathcal{E}_{strength}) \cdot \exp\left(-\left(2 \cdot \max(0, \mathcal{C}_{reach} - \mathcal{E}_{strength}) + 1.5 \cdot \lambda_{jumps}\right)\right) \right) \times \frac{1}{1 + e^{-\Delta Premise}} + \lambda \cdot \text{vapri} $$")

    criteria_list = [
        ("C1: Originality", "c1", tw1, "1", "Semantic distance from literature corpus penalized by generative AI heuristics.", r"$$ C_1 = \varpi_1 \cdot \mathcal{D}_{semantic}(P_{target}, P_{corpus}) \times (1 - \lambda_{laundering}) + \text{vapri} $$"),
        ("C2: Methodological Rigor", "c2", tw2, "2", "Adherence to MDAR standards and valid RRIDs via SciScore.", r"$$ C_2 = \varpi_2 \cdot \mathcal{I}_{blinding} + \varpi_2 \cdot \mathcal{I}_{randomization} + \varpi_2 \cdot \mathcal{I}_{power\_calc} + \varpi_2 \cdot \left(\frac{N_{RRID\_valid}}{N_{RRID\_expected} + \epsilon}\right) $$"),
        ("C3: Interdisciplinary Synergy", "c3", tw3, "3", "Measures cross-disciplinary integration and entropy across domains.", r"$$ C_3 = \varpi_3 \cdot -\sum_{i=1}^{k} p_i \ln(p_i) $$"),
        ("C4: Societal Impact", "c4", tw4, "4", "Evaluates open infrastructure contributions and societal relevance.", r"$$ C_4 = \varpi_4 \cdot \Theta\left[ \sum_{v \in \mathcal{V}} \omega_v U_v(\tau, \mathbf{x}) \right] $$"),
        ("C5: Open Science", "c5", tw5, "5", "Evaluates open data, open code, and reproducibility.", r"$$ C_5 = \varpi_5 \cdot (\beta_1 \cdot \mathcal{V}_{data} + \beta_2 \cdot \mathcal{V}_{code} + \beta_3 \cdot \mathcal{Z}_{container}) $$"),
        ("C6: Literature Integration", "c6", tw6, "6", "Evaluates citation polarity and integration with foundational literature.", r"$$ C_6 = \varpi_6 \cdot \frac{1}{\mathcal{N}} \sum_{i=1}^{\mathcal{N}} \text{Polarity}(x_i) \cdot \text{PR}(x_i) $$"),
        ("C7: Empirical Density", "c7", tw7, "7", "Assesses sample strength and baseline empirical variance.", r"$$ C_7 = \varpi_7 \cdot \tanh \left( \frac{n_{\text{valid}} \cdot \text{Cohort Strength}}{\text{Baseline Variance}} \right) $$"),
        ("C8: Future Actionability", "c8", tw8, "8", "Evaluates future research actionability and FAIR adherence.", r"$$ C_8 = \varpi_8 \cdot \frac{1}{\mathcal{Z}} \int_{\mathcal{X}} \text{FAIR\_Score}(\mathbf{x}) \, d\mu(\mathbf{x}) $$"),
    ]

    for title, q_key, weight_val, sym, desc, formula in criteria_list:
        with st.expander(f"{title} ( $\varpi_{sym}$ = `{weight_val:.6f}` ):", expanded=(title.startswith("C1"))):
            st.markdown(f"{desc} {rbot(q_key)}", unsafe_allow_html=True)
            st.markdown(formula)

col_t1, col_t2 = st.columns([4, 2], vertical_alignment="center")
with col_t1:
    st.markdown("<h1 style='margin-bottom:0;'>Pi-Index Assessment Engine</h1>", unsafe_allow_html=True)
with col_t2:
    if st.button("Evaluation Metrics, SciScore & Logic Engine", use_container_width=True):
        evaluation_metrics_dialog()

st.markdown("")

with st.container(border=True):
    selected_uploaded_files = []
    uploaded_files = st.file_uploader(
        "Upload Local PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state['reset_token']}",
    )
    if uploaded_files:
        st.markdown("**Tick local files to include:**")
        for i, file in enumerate(uploaded_files):
            if st.checkbox(
                f"Local File: {file.name}",
                value=True,
                key=f"up_chk_{i}_{st.session_state['reset_token']}",
            ):
                selected_uploaded_files.append(file)

    research_scope = ""
    doi_input = ""
    include_doi = False

    unified_query = st.text_input(
        "Research Scope, DOI, or OpenAlex Topic",
        placeholder="Enter research topic, DOI (e.g. 10.1038/...), or keyword...",
        key=f"unified_query_{st.session_state['reset_token']}",
    )
    
    if unified_query.strip():
        q_str = unified_query.strip()
        if re.match(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$", q_str) or q_str.startswith("10.") or "doi.org" in q_str:
            doi_input = q_str
            include_doi = True
            st.caption("Detected as DOI. Will resolve via Unpaywall.")
        else:
            research_scope = q_str
            if st.button("Search OpenAlex Papers for this Topic", key=f"unified_alex_btn_{st.session_state['reset_token']}"):
                st.session_state.alex_visible_count = 10
                with st.spinner("Querying OpenAlex..."):
                    alex_results = search_openalex_topics(q_str, limit=50)
                    if alex_results:
                        st.session_state["alex_search_results"] = alex_results
                        add_log(f"Harvested {len(alex_results)} Open Access records.")
                        st.success(f"Harvested {len(alex_results)} papers.")
                    else:
                        add_log("Failed to find records via OpenAlex.")
                        st.warning("No Open Access papers found matching criteria.")

    selected_alex_papers = []
    if "alex_search_results" in st.session_state and st.session_state["alex_search_results"]:
        st.markdown("---")
        col_res_header, col_close_btn = st.columns([5, 1])
        with col_res_header:
            st.markdown("#### OpenAlex Harvested Results")
        with col_close_btn:
            if st.button("Close", key=f"close_alex_{st.session_state['reset_token']}"):
                del st.session_state["alex_search_results"]
                st.rerun()

        visible_results = st.session_state["alex_search_results"][: st.session_state.get("alex_visible_count", 10)]
        for idx, p in enumerate(visible_results):
            if st.checkbox(f"OpenAlex: {p['title']} — *{clean_author_name(p['authors'])}*", key=f"alex_chk_{idx}_{st.session_state['reset_token']}"):
                selected_alex_papers.append(p)

    stake_amount = st.checkbox(
        "Stake 0.01 piQ to Process (Returned on Valid Assessment)",
        value=True,
        key=f"stake_chk_{st.session_state['reset_token']}",
    )

    if st.session_state["is_running"]:
        col_run, col_stop = st.columns([4, 1])
        with col_run:
            st.button("Working...", type="primary", use_container_width=True, disabled=True)
        with col_stop:
            if st.button("Stop", type="secondary", use_container_width=True):
                st.session_state["is_running"] = False
                st.session_state["cancel_requested"] = True
                add_log("Pipeline operation interrupted by user.")
                st.rerun()

        progress_bar, status_text = st.progress(0), st.empty()
        scope_val = st.session_state.get("snap_scope", "")
        snap_files = st.session_state.get("snap_files", [])
        snap_alex = st.session_state.get("snap_alex", [])
        include_doi_snap = st.session_state.get("snap_include_doi", False)
        doi_snap = st.session_state.get("snap_doi", "")

        try:
            if snap_files and not st.session_state["cancel_requested"]:
                total_files = len(snap_files)
                for i, (fname, fpath) in enumerate(snap_files):
                    if st.session_state["cancel_requested"]:
                        break
                    status_text.text(f"Analyzing uploaded file {i+1} of {total_files}: {fname}...")
                    with open(fpath, "rb") as in_f:
                        raw_bytes = in_f.read()
                        
                    (
                        title, author_name, score, logic_integrity, drift, rec,
                        fields, subfields, scores_dict, eval_hash, piq, tx_hash,
                        zk_proof, used_weights, mdar_score, rrid_count, repro_score, is_cached, warnings_list,
                        consensus_raw, evidence_report_text, scilem_rating
                    ) = process_single_pdf(
                        raw_bytes, fname, scope_val, current_user, valid_book_address, current_email, "None",
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
                    progress_bar.progress((i + 1) / total_files)

            status_text.success("Pipeline processing complete.")
            time.sleep(1)
        finally:
            st.session_state["is_running"] = False
            st.session_state["cancel_requested"] = False
            st.session_state["reset_token"] += 1
            st.session_state["assessment_update_token"] = time.time()
            st.rerun()

    else:
        if st.button("Run Assessment Pipeline", type="primary", use_container_width=True):
            if not stake_amount:
                st.error("You must agree to the piQ micro-stake to execute the assessment pipeline.")
            elif not selected_uploaded_files and not (include_doi and doi_input.strip()) and not selected_alex_papers:
                st.warning("Please tick at least one source to assess.")
            else:
                saved_files = []
                for f in selected_uploaded_files:
                    f_path = os.path.join(st.session_state["session_temp_dir"], f.name)
                    with open(f_path, "wb") as out_f:
                        out_f.write(f.getvalue())
                    saved_files.append((f.name, f_path))
                    
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
    st.subheader(f"{item['title']} by {clean_author_name(item['author_name'])}")

    st.markdown("### Multi-LLM & Scilem Consensus Extractions")
    consensus_raw = item.get("consensus_raw", {})
    if consensus_raw and isinstance(consensus_raw, dict):
        llm_cols = st.columns(2)
        target_llms = ["llama", "mistral", "qwen", "gemini", "scilem"]
        for idx, llm_key in enumerate(target_llms):
            col = llm_cols[idx % 2]
            with col:
                data = consensus_raw.get(llm_key, {})
                with st.container(border=True):
                    st.markdown(f"**Model: {llm_key.upper()}**")
                    if data.get("api_failed", False):
                        st.markdown(f"**Status:** Limit / Offline")
                        st.markdown(f"**Opinion:** {data.get('opinion', 'N/A')}")
                    else:
                        st.markdown(f"**Extracted Title:** `{data.get('title', 'N/A')}`")
                        st.markdown(f"**Extracted Authors:** `{data.get('authors', 'N/A')}`")
                        st.markdown(f"**Criteria Opinion:** {data.get('opinion', 'N/A')}")

    st.markdown("---")
    st.markdown("### Synthesized Evidence Report")
    st.markdown(item.get("evidence_report_text", "No report available."))

@st.dialog("AI Peer Review Defense Strategy", width="medium")
def defense_strategy_dialog(scores_dict):
    st.markdown(generate_rebuttal_strategy(scores_dict))

# Results Section
if st.session_state["evaluated_papers_buffer"]:
    st.markdown("### Active Session Assessment Results")
    for item_idx, item in enumerate(st.session_state["evaluated_papers_buffer"]):
        with st.container(border=True):
            st.markdown(f"**{item['title']}** — *{clean_author_name(item['author_name'])}*")
            st.markdown(f"**Score: {item['score']:.2f} | piQ: {item['piq']}**")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("More Details", key=f"det_{item_idx}_{item['eval_hash']}"):
                    more_details_dialog(item)
            with c2:
                if st.button("Suggest Defense", key=f"def_{item_idx}_{item['eval_hash']}"):
                    defense_strategy_dialog(item["scores_dict"])

# Analytics Section
top_analytics_col1, top_analytics_col2 = st.columns(2)

with top_analytics_col1:
    st.markdown("### Pidyne Forecast")
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights ORDER BY block_height ASC")
        historical_rows = cur.fetchall()
    finally:
        conn.close()

    if len(historical_rows) >= 2:
        df_history = pd.DataFrame(historical_rows, columns=["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"])
        st.line_chart(df_history)
    else:
        st.info("Assess more manuscripts to unlock Pidyne blockchain forecasting.")

with top_analytics_col2:
    st.markdown("### Global Map of Science")
    interactive_html_top, table_html_top = render_bubble_chart_clean("All Authors")
    if interactive_html_top:
        components.html(interactive_html_top, height=600, scrolling=False)

st.markdown("---")
st.markdown("### Pi Quotient (piQ) Leaderboard")
piq_dict, book_dict = get_author_piq_dict()
if piq_dict:
    st.dataframe(pd.DataFrame(list(piq_dict.items()), columns=["Author", "Total piQ Minted"]))
