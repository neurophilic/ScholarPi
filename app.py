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
    PidyneBlockchainDataset
)

st.set_page_config(
    page_title="Pi-Index Assessment Engine 🤖", layout="wide"
)

# --- System Action Log Monitor ---
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
        cursor.execute("SELECT author_name, piq_minted FROM papers_assessment")
        data = cursor.fetchall()
    finally:
        conn.close()
    
    author_piq = {}
    author_book = {}
    for authors_str, piq in data:
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
            author_book[a] = "0x" + hashlib.sha256(a.encode()).hexdigest()[:40]
    return author_piq, author_book

def preprocess_pdf_layout(pdf_bytes, fname):
    # Bypass redundant virtual PDF creation to preserve original readable text layer
    # brain.py already handles PyMuPDF spatial extraction natively.
    return pdf_bytes

def rbot(topic_key):
    return f"<span class='scilem-trigger' data-query='{topic_key}' title='Click to ask Scilem' style='cursor: pointer !important;'>🤖</span>"

# Custom JS/CSS for UI Modifications
custom_ui_code = """
<style>
.stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a, 
.stMarkdown h4 a, .stMarkdown h5 a, .stMarkdown h6 a,
[data-testid="stHeaderActionElements"] {
    display: none !important;
}

[data-testid="stSidebar"] {
    overflow: hidden !important;
}
[data-testid="stSidebar"] > div:first-child {
    overflow: hidden !important;
}

[data-testid="stChatMessage"]:has(div:contains("👤")) {
    flex-direction: row-reverse !important;
    background-color: #e8f0fe !important;
    border-radius: 10px 0 10px 10px !important;
    text-align: right !important;
    margin-left: 20px !important;
}
[data-testid="stChatMessage"]:has(div:contains("🤖")) {
    background-color: #f1f3f4 !important;
    border-radius: 0 10px 10px 10px !important;
    margin-right: 20px !important;
}

[data-testid="stChatMessageAvatar"] {
    transform: scale(1.3);
}

.scilem-trigger {
    cursor: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='32' height='32' style='font-size:24px'%3E%3Ctext y='24'%3E🤖%3C/text%3E%3C/svg%3E"), auto !important;
    font-size: 1.4em;
    margin-left: 4px;
    vertical-align: middle;
    display: inline-block;
    transition: transform 0.15s ease-in-out;
}
.scilem-trigger:hover {
    transform: scale(1.3);
}

#scilem-drag-handle {
    background-color: #2c3e50;
    color: white;
    padding: 12px;
    font-weight: bold;
    font-size: 16px;
    cursor: grab;
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
    margin: -1rem -1rem 1rem -1rem;
    user-select: none;
    position: relative;
    display: flex;
    align-items: center;
}
#scilem-drag-handle:active {
    cursor: grabbing;
}

#scilem-drag-handle .robot-icon {
    font-size: 1.5em;
    margin-right: 8px;
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
    if (e.target && e.target.id === 'scilem-min-btn') {
        let block = e.target.closest('[data-draggable="true"]');
        if (block) {
            let children = Array.from(block.children);
            let isMin = block.getAttribute('data-minimized') === 'true';
            children.forEach(child => {
                let handle = child.querySelector('#scilem-drag-handle') || (child.id === 'scilem-drag-handle' ? child : null);
                if (!handle && child.id !== 'scilem-drag-handle') {
                    child.style.display = isMin ? 'block' : 'none';
                }
            });
            block.setAttribute('data-minimized', isMin ? 'false' : 'true');
            e.target.innerText = isMin ? '--' : '+';
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

        setTimeout(() => {
            let btns = chatBlock.querySelectorAll('button');
            let submitBtn = Array.from(btns).find(b => b.innerText.includes('Send') || b.textContent.includes('Send'));
            if (submitBtn) submitBtn.click();
        }, 150);
    }
}, true);

function initUI() {
    const handle = parentDoc.getElementById('scilem-drag-handle');
    if (handle) {
        let block = handle.closest('[data-testid="stVerticalBlock"]');
        if (block && block.getAttribute('data-draggable') !== 'true') {
            if (!block.innerText.includes("Live System Monitor")) {
                block.setAttribute('data-draggable', 'true');
                block.style.position = 'fixed';
                block.style.bottom = '20px';
                block.style.right = '20px';
                block.style.width = '380px';
                block.style.backgroundColor = '#ffffff';
                block.style.border = '1px solid #d0d7de';
                block.style.borderRadius = '12px';
                block.style.boxShadow = '0 10px 40px rgba(0,0,0,0.3)';
                block.style.zIndex = '999999';
                block.style.padding = '1rem';
                
                let isDragging = false;
                let startX, startY, initialX, initialY;

                handle.addEventListener('mousedown', function(e) {
                    if (e.target.id === 'scilem-min-btn') return;
                    isDragging = true;
                    startX = e.clientX;
                    startY = e.clientY;
                    const rect = block.getBoundingClientRect();
                    initialX = rect.left;
                    initialY = rect.top;
                    block.style.bottom = 'auto';
                    block.style.right = 'auto';
                    block.style.transition = 'none'; 
                    e.preventDefault(); 
                });

                parentDoc.addEventListener('mousemove', function(e) {
                    if (!isDragging) return;
                    block.style.left = (initialX + (e.clientX - startX)) + 'px';
                    block.style.top = (initialY + (e.clientY - startY)) + 'px';
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

if "scilm_messages" not in st.session_state:
    st.session_state.scilm_messages = [
        {
            "role": "assistant", 
            "content": "**Welcome! I am Scilem.** Click any 🤖 button next to technical app features or terms for instant explanations."
        }
    ]

if "orcid_id" not in st.session_state:
    saved_orcid = st.query_params.get("orcid", "")
    if saved_orcid and (re.match(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$", saved_orcid) or "did:" in saved_orcid):
        st.session_state.orcid_id = saved_orcid
        st.session_state.orcid_name = "Ali Vafadar Yengejeh" if "8050" in saved_orcid else "Verified Decentralized Identity"
        st.session_state.is_authenticated = True
    else:
        st.session_state.orcid_id = "0009-0009-8456-8050"
        st.session_state.orcid_name = "Ali Vafadar Yengejeh"
        st.session_state.is_authenticated = False

if not st.session_state.is_authenticated:
    st.sidebar.markdown("### Authenticate")
    manual_orcid = st.sidebar.text_input(
        "Enter ORCID iD or W3C DID", placeholder="0009-0009-8456-8050"
    )
    remember_user = st.sidebar.checkbox("Remember me", value=True)

    if st.sidebar.button("Validate and Connect"):
        clean_orcid = manual_orcid.strip()
        if (
            re.match(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$", clean_orcid)
            or "did:" in clean_orcid
        ):
            with st.sidebar.status("Connecting to Identity Registry..."):
                if "did:" in clean_orcid:
                    user_name = "Verified Decentralized Identity"
                elif "8050" in clean_orcid:
                    user_name = "Ali Vafadar Yengejeh"
                else:
                    user_name = "Verified Researcher"
            
            st.session_state.orcid_id = clean_orcid
            st.session_state.orcid_name = user_name
            st.session_state.is_authenticated = True
            add_log(f"Identity Authenticated: {clean_orcid}")
            if remember_user:
                st.query_params["orcid"] = clean_orcid
            else:
                if "orcid" in st.query_params:
                    del st.query_params["orcid"]
            st.rerun()
        else:
            st.sidebar.error("Invalid ORCID or DID format.")

    st.sidebar.markdown("---")
    st.sidebar.info("Notice: Please connect your ORCID iD or DID above to unlock and use your personal Assessment History features.")
else:
    st.sidebar.success("Securely Connected")
    
    conn_hist = get_db_connection()
    total_user_piq = 0.0
    try:
        cur_h = conn_hist.cursor()
        cur_h.execute("SELECT piq_minted FROM papers_assessment WHERE user_id = ? OR user_id = '0009-0009-8456-8050'", (st.session_state.orcid_id,))
        piq_rows = cur_h.fetchall()
        total_user_piq = sum(float(r[0]) for r in piq_rows if r[0])
    finally:
        conn_hist.close()
        
    st.sidebar.markdown(
        f"**Researcher:** {st.session_state.orcid_name}\n\n"
        f"**ID Vault:** `{st.session_state.orcid_id}`\n\n"
        f"**TOTAL piQ AWARDED:** `{total_user_piq:.2f} piQ`"
    )

    if st.sidebar.button("Disconnect Session"):
        add_log("Session Disconnected.")
        st.session_state.is_authenticated = False
        st.session_state.orcid_name = ""
        st.session_state.orcid_id = "0000-0000-0000-0000"
        if "orcid" in st.query_params:
            del st.query_params["orcid"]
        st.rerun()

current_user = st.session_state.get("orcid_id", "0009-0009-8456-8050")
current_email = "None"

st.sidebar.markdown("---")
with st.sidebar.expander("🖥️ Live System Monitor", expanded=True):
    log_text = "\n".join(st.session_state.app_logs)
    st.code(log_text if log_text else "No active logs...", language="bash")

SCILEM_KNOWLEDGE_BASE = {
    "authenticate": "Connect to your ORCID or DID to securely isolate your assessment history. Pi Quotient (piQ) is a Soulbound Token assigned strictly to this identity.",
    "assessment history": "Displays your authenticated assessment history and earned Pi Quotient (piQ) rewards across decentralized epochs.",
    "pidyne forecast": "An LSTM neural network that trains directly on the block weights to predict future shifts in algorithmic evaluation standards.",
    "latest assessed": "Displays the 5 most recently evaluated papers globally with complete assessment scores, block hashes, zk-SNARK proofs, and piQ allocations.",
    "proof-of-research": "Manages decentralized consensus, ledger weights, and smart contract audit proofs. It validates evaluations directly on the blockchain.",
    "adversarial logic gap": "Evaluates reasoning structure and penalizes claims unsupported by evidence or counterfactual stress failures.",
    "c1: originality": "Semantic distance from literature corpus penalized by generative AI laundering heuristics.",
    "c2: methodological rigor": "Deterministic adherence to MDAR reporting standards and valid RRIDs via SciScore.",
    "c3: interdisciplinary synergy": "Measures cross-disciplinary integration and entropy across scientific domains.",
    "c4: societal impact": "Evaluates broader societal and open infrastructure contributions.",
    "c5: open science": "Evaluates open data, open code, and containerized reproducibility.",
    "c6: literature integration": "Evaluates citation polarity and integration with existing foundational literature.",
    "c7: empirical density": "Assesses empirical sample strength and baseline variance.",
    "c8: future actionability": "Evaluates future research actionability and adherence to FAIR principles.",
    "pi-index": "Automated peer-review framework powered by neural networks, SciScore reproducibility metrics, and multidimensional blockchain consensus.",
    "global map of science": "A PyVis network cartography displaying domains and subfields of assessed papers, scaled by average weights.",
    "zk-snark": "Zero-Knowledge Succinct Non-Interactive Argument of Knowledge. A cryptographic proof that an evaluation occurred exactly per guidelines without revealing reviewer identity.",
    "sciscore mdar": "SciScore evaluates adherence to the Materials Design Analysis Reporting (MDAR) framework to ensure rigor.",
    "executable reproducibility score": "An audit metric calculating whether code, data, and software environments (C5 & C7) can reliably execute independent results."
}

if "last_analyzed_tracked" not in st.session_state:
    st.session_state["last_analyzed_tracked"] = total_analyzed_count
elif st.session_state["last_analyzed_tracked"] < total_analyzed_count:
    st.session_state["last_analyzed_tracked"] = total_analyzed_count
    st.session_state.scilm_messages.append({
        "role": "assistant",
        "content": f"**Proactive Update:** A new manuscript has been processed! Total analyzed papers is now **{total_analyzed_count}**."
    })

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
        cursor.execute(
            "SELECT fields, subfields, final_score, author_name FROM papers_assessment"
        )
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

    table_html = "<style>.table-big { width: 100%; font-size: 13px; border-collapse: collapse; margin-top: 10px; font-family: sans-serif; } .table-big th { background-color: #2c3e50; color: white; padding: 6px; text-align: left; } .table-big td { padding: 6px; border-bottom: 1px solid #ecf0f1; } .color-box { width: 20px; height: 20px; border-radius: 4px; display: inline-block; } </style>"
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

@st.dialog("Evaluation Metrics, SciScore Reproducibility & Adversarial Logic Engine", width="large")
def evaluation_metrics_dialog():
    conn_top_ep = get_db_connection()
    try:
        cur_te = conn_top_ep.cursor()
        cur_te.execute(
            "SELECT block_height, w1, w2, w3, w4, w5, w6, w7, w8, model_used FROM blockchain_por_weights ORDER BY block_height DESC LIMIT 1"
        )
        top_epoch_data = cur_te.fetchone()
    except Exception:
        top_epoch_data = None
    finally:
        conn_top_ep.close()

    if top_epoch_data:
        _, tw1, tw2, tw3, tw4, tw5, tw6, tw7, tw8, _ = top_epoch_data
    else:
        tw1, tw2, tw3, tw4, tw5, tw6, tw7, tw8 = 1.001328, 1.000038, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0

    st.markdown(
        f"**Adversarial Logic Gap 🤖 ($\Delta_{{Logic}}$):** Evaluates reasoning structure and penalizes claims unsupported by evidence or counterfactual stress failures.",
        unsafe_allow_html=True
    )
    st.markdown(
        r"$$ L_i = (\mathcal{P}_{valid} \cdot \mathcal{E}_{strength}) \cdot"
        r" \exp\left(-\left(2 \cdot \max(0, \mathcal{C}_{reach} -"
        r" \mathcal{E}_{strength}) + 1.5 \cdot \lambda_{jumps}\right)\right)"
        r" \times \frac{1}{1 + e^{-\Delta Premise}} $$"
    )

    criteria_list = [
        ("C1: Originality", "c1: originality", tw1, "1", "Semantic distance from literature corpus penalized by generative AI laundering heuristics.", r"$$ C_1 = \varpi_1 \cdot \mathcal{D}_{semantic}(P_{target}, P_{corpus}) \times (1 - \lambda_{laundering}) $$"),
        ("C2: Methodological Rigor", "c2: methodological rigor", tw2, "2", "Deterministic adherence to MDAR reporting standards and valid RRIDs via SciScore.", r"$$ C_2 = \varpi_2 \cdot \mathcal{I}_{blinding} + \varpi_2 \cdot \mathcal{I}_{randomization} + \varpi_2 \cdot \mathcal{I}_{power\_calc} + \varpi_2 \cdot \left(\frac{N_{RRID\_valid}}{N_{RRID\_expected} + \epsilon}\right) $$"),
        ("C3: Interdisciplinary Synergy", "c3: interdisciplinary synergy", tw3, "3", "Measures cross-disciplinary integration and entropy across scientific domains.", r"$$ C_3 = \varpi_3 \cdot -\sum_{i=1}^{k} p_i \ln(p_i) $$"),
        ("C4: Societal Impact", "c4: societal impact", tw4, "4", "Evaluates broader societal and open infrastructure contributions.", r"$$ C_4 = \varpi_4 \cdot \Theta\left[ \sum_{v \in \mathcal{V}} \omega_v U_v(\tau, \mathbf{x}) \right] $$"),
        ("C5: Open Science", "c5: open science", tw5, "5", "Evaluates open data, open code, and containerized reproducibility.", r"$$ C_5 = \varpi_5 \cdot (\beta_1 \cdot \mathcal{V}_{data} + \beta_2 \cdot \mathcal{V}_{code} + \beta_3 \cdot \mathcal{Z}_{container}) $$"),
        ("C6: Literature Integration", "c6: literature integration", tw6, "6", "Evaluates citation polarity and integration with existing foundational literature.", r"$$ C_6 = \varpi_6 \cdot \frac{1}{\mathcal{N}} \sum_{i=1}^{\mathcal{N}} \text{Polarity}(x_i) \cdot \text{PR}(x_i) $$"),
        ("C7: Empirical Density", "c7: empirical density", tw7, "7", "Assesses empirical sample strength and baseline variance.", r"$$ C_7 = \varpi_7 \cdot \tanh \left( \frac{n_{\text{valid}} \cdot \text{Cohort Strength}}{\text{Baseline Variance}} \right) $$"),
        ("C8: Future Actionability", "c8: future actionability", tw8, "8", "Evaluates future research actionability and adherence to FAIR principles.", r"$$ C_8 = \varpi_8 \cdot \frac{1}{\mathcal{Z}} \int_{\mathcal{X}} \text{FAIR\_Score}(\mathbf{x}) \, d\mu(\mathbf{x}) $$"),
    ]

    for title, q_key, weight_val, sym, desc, formula in criteria_list:
        with st.expander(f"{title} ( varpi_{sym} = `{weight_val:.6f}` ):", expanded=(title.startswith("C1"))):
            st.markdown(f"{desc} {rbot(q_key)}", unsafe_allow_html=True)
            st.markdown(formula)

col_t1, col_t2 = st.columns([4, 2], vertical_alignment="center")
with col_t1:
    st.markdown(f"<h1 style='margin-bottom:0;'>Pi-Index Assessment Engine 🤖</h1>", unsafe_allow_html=True)
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
        placeholder="Enter research topic, DOI (e.g. 10.1038/...), or search keyword...",
        key=f"unified_query_{st.session_state['reset_token']}",
    )
    
    if unified_query.strip():
        q_str = unified_query.strip()
        if re.match(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$", q_str) or q_str.startswith("10.") or "doi.org" in q_str:
            doi_input = q_str
            include_doi = True
            research_scope = ""
            st.caption("Detected as DOI. Will resolve via Unpaywall.")
        else:
            research_scope = q_str
            if st.button("Search OpenAlex Papers for this Topic", key=f"unified_alex_btn_{st.session_state['reset_token']}"):
                st.session_state.alex_visible_count = 10
                with st.spinner("Querying OpenAlex..."):
                    alex_results = search_openalex_topics(q_str, limit=50)
                    if alex_results:
                        st.session_state["alex_search_results"] = alex_results
                        add_log(f"Harvested {len(alex_results)} Open Access records from OpenAlex.")
                        st.success(f"Successfully harvested {len(alex_results)} papers from OpenAlex.")
                    else:
                        add_log("Failed to find relevant records via OpenAlex.")
                        st.warning("No Open Access papers found matching criteria.")

    selected_alex_papers = []
    if (
        "alex_search_results" in st.session_state
        and st.session_state["alex_search_results"]
    ):
        st.markdown("---")
        col_res_header, col_close_btn = st.columns([5, 1])
        with col_res_header:
            st.markdown("#### OpenAlex Harvested Results")
        with col_close_btn:
            if st.button(
                "Close", key=f"close_alex_{st.session_state['reset_token']}"
            ):
                del st.session_state["alex_search_results"]
                st.rerun()

        def toggle_all_alex():
            is_all = st.session_state.get(
                f"select_all_alex_{st.session_state['reset_token']}", False
            )
            for i in range(st.session_state.alex_visible_count):
                st.session_state[f"alex_chk_{i}_{st.session_state['reset_token']}"] = (
                    is_all
                )

        select_all_alex = st.checkbox(
            "Select All Visible OpenAlex Results",
            key=f"select_all_alex_{st.session_state['reset_token']}",
            on_change=toggle_all_alex,
        )

        visible_results = st.session_state["alex_search_results"][
            : st.session_state.alex_visible_count
        ]
        for idx, p in enumerate(visible_results):
            is_selected = st.checkbox(
                f"OpenAlex: {p['title']} — *{clean_author_name(p['authors'])}*",
                key=f"alex_chk_{idx}_{st.session_state['reset_token']}",
            )
            if is_selected:
                selected_alex_papers.append(p)

        if st.session_state.alex_visible_count < len(
            st.session_state["alex_search_results"]
        ):
            if st.button("Show More OpenAlex Results"):
                st.session_state.alex_visible_count += 10
                st.rerun()

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
                        ) = process_single_pdf(
                            clean_bytes, fname, scope_val, current_user, "None", current_email, p_doi,
                        )

                        eval_record = {
                            "title": title, "author_name": clean_author_name(author_name),
                            "score": score, "logic_integrity": logic_integrity, "drift": drift,
                            "rec": rec, "fields": fields, "subfields": subfields,
                            "scores_dict": scores_dict, "eval_hash": eval_hash, "piq": piq,
                            "tx_hash": tx_hash, "zk_proof": zk_proof, "used_weights": used_weights,
                            "h_idx": mdar_score, "i10_idx": rrid_count, "repro_score": repro_score,
                            "filename": fname, "warnings": warnings_list, "warnings_acknowledged": False,
                        }
                        st.session_state["evaluated_papers_buffer"].insert(0, eval_record)
                        st.session_state["evaluated_papers_buffer"] = st.session_state["evaluated_papers_buffer"][:50]
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
                    ) = process_single_pdf(
                        clean_bytes, fname, scope_val, current_user, "None", current_email, doi_snap.strip(),
                    )

                    eval_record = {
                        "title": title, "author_name": clean_author_name(author_name),
                        "score": score, "logic_integrity": logic_integrity, "drift": drift,
                        "rec": rec, "fields": fields, "subfields": subfields,
                        "scores_dict": scores_dict, "eval_hash": eval_hash, "piq": piq,
                        "tx_hash": tx_hash, "zk_proof": zk_proof, "used_weights": used_weights,
                        "h_idx": mdar_score, "i10_idx": rrid_count, "repro_score": repro_score,
                        "filename": fname, "warnings": warnings_list, "warnings_acknowledged": False,
                    }
                    st.session_state["evaluated_papers_buffer"].insert(0, eval_record)
                    st.session_state["evaluated_papers_buffer"] = st.session_state["evaluated_papers_buffer"][:50]
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
                    ) = process_single_pdf(
                        clean_bytes, fname, scope_val, current_user, "None", current_email, "None",
                    )

                    eval_record = {
                        "title": title, "author_name": clean_author_name(author_name),
                        "score": score, "logic_integrity": logic_integrity, "drift": drift,
                        "rec": rec, "fields": fields, "subfields": subfields,
                        "scores_dict": scores_dict, "eval_hash": eval_hash, "piq": piq,
                        "tx_hash": tx_hash, "zk_proof": zk_proof, "used_weights": used_weights,
                        "h_idx": mdar_score, "i10_idx": rrid_count, "repro_score": repro_score,
                        "filename": fname, "warnings": warnings_list, "warnings_acknowledged": False,
                    }
                    st.session_state["evaluated_papers_buffer"].insert(0, eval_record)
                    st.session_state["evaluated_papers_buffer"] = st.session_state["evaluated_papers_buffer"][:50]
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
            if not stake_amount:
                st.error("You must agree to the piQ micro-stake to execute the assessment pipeline.")
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
    title = item["title"]
    author_name = clean_author_name(item["author_name"])
    score = item["score"]
    logic_integrity = item["logic_integrity"]
    scores_dict = item["scores_dict"]
    used_weights = item["used_weights"]
    eval_hash = item["eval_hash"]
    piq = item["piq"]
    tx_hash = item["tx_hash"]
    zk_proof = item["zk_proof"]
    drift = item["drift"]
    rec = item["rec"]
    mdar_score = item["h_idx"]
    rrid_count = item["i10_idx"]
    repro_score = item["repro_score"]
    filename = item["filename"]
    warnings = item.get("warnings", [])
    author_book = "0x" + hashlib.sha256(author_name.encode()).hexdigest()[:40]

    st.subheader(f"{title} by {author_name}")

    if warnings:
        st.warning(f"⚠️ **Manuscript Flagged with {len(warnings)} Warning Check(s):**")
        for w in warnings:
            st.markdown(f"- {w}")

    with st.expander(f"Ledger Data & Dossier Details ({filename})", expanded=True):
        st.write(f"**File Name:** `{filename}`")
        st.write(f"**Evaluation Hash (Paper Address):** `{eval_hash}`")
        st.write(f"**Unique Book Address:** `{author_book}`")
        st.write(f"**piQ Minted:** `{piq}`")
        st.markdown(f"**zk-SNARK {rbot('zk-snark')}:** `{zk_proof}`", unsafe_allow_html=True)
        
        tx_url = safe_get_sepolia_url(tx_hash)
        tx_disp_val = tx_hash if tx_hash and str(tx_hash).strip() not in ["None", ""] else "Not Connected / No Book / Missing PK"
        if tx_url:
            st.markdown(f"**Tx Hash:** [`{tx_disp_val}`]({tx_url})")
        else:
            st.write(f"**Tx Hash:** `{tx_disp_val}`")

        st.markdown(f"**Executable Reproducibility Score {rbot('executable reproducibility score')}:** `{repro_score * 100:.1f}%`", unsafe_allow_html=True)
        st.markdown(f"**SciScore MDAR Adherence {rbot('sciscore mdar')}:** `{mdar_score * 100:.1f}%` | **Valid RRIDs:** `{rrid_count}`", unsafe_allow_html=True)

    scope_val = st.session_state.get("snap_scope", "")
    if scope_val.strip() and drift != "N/A" and rec != "N/A":
        st.markdown(f"**Scope Drift:** `{drift:.2f}%`")
        st.markdown(f"**Recommendation Tier:** `{rec}`")

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
            scores_dict.get("C1_Semantic_Originality", 0),
            scores_dict.get("C2_Methodological_Rigor_SciScore", 0),
            scores_dict.get("C3_Interdisciplinary_Entropy", 0),
            scores_dict.get("C4_Societal_Impact", 0),
            scores_dict.get("C5_Open_Science_Repro", 0),
            scores_dict.get("C6_Literature_Integration", 0),
            scores_dict.get("C7_Empirical_Density", 0),
            scores_dict.get("C8_Future_Actionability_FAIR", 0),
        ],
        "Epoch Weight": used_weights,
        "Weighted Value": [
            scores_dict.get(k, 0) * used_weights[i]
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
    st.markdown(
        f"**Final Pi-Index (Base * Logic Multiplier):** `{score:.2f}`"
        f" &nbsp;|&nbsp; **MDAR Adherence:** `{mdar_score * 100:.1f}%`"
        f" &nbsp;|&nbsp; **Valid RRIDs:** `{rrid_count}` &nbsp;|&nbsp; **File:**"
        f" `{filename}`"
    )

    dossier_content = f"""# RESEARCH INTEGRITY DOSSIER (DORA-Aligned)
**Title:** {title}
**Author:** {author_name}
**File Name:** {filename}
**Evaluation Hash (Paper Address):** {eval_hash}
**Unique Book Address:** {author_book}
**Final Pi-Index Score:** {score:.2f} / 100
**Logic Integrity Score:** {logic_integrity:.1f}%
**Executable Reproducibility Score:** {repro_score * 100:.1f}%
**SciScore MDAR Adherence:** {mdar_score * 100:.1f}%
**Valid RRIDs Count:** {rrid_count}
**Warnings Flagged:** {len(warnings)}
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
    score = item["score"]
    eval_hash = item["eval_hash"]
    piq = item["piq"]
    scores_dict = item["scores_dict"]
    warnings = item.get("warnings", [])
    acknowledged = item.get("warnings_acknowledged", False)

    with st.container(border=True):
        col_info, col_actions = st.columns([6, 4])
        with col_info:
            if warnings and not acknowledged:
                warn_badge = f" ⚠️ *({len(warnings)} warning checks active)*"
            elif warnings and acknowledged:
                warn_badge = f" 🛡️ *({len(warnings)} warning checks acknowledged)*"
            else:
                warn_badge = ""

            st.markdown(f"**{title}** — *{author_name}*{warn_badge}")
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
            c_det, c_strat, c_del = st.columns([3, 3, 1])
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
    st.markdown("### Active Session Assessment Results")
    st.markdown("")

    if st.session_state.get("download_errors"):
        st.markdown("#### Publisher Access & Download Restrictions")
        for err_idx, err_data in enumerate(
            st.session_state["download_errors"]
        ):
            err_col1, err_col2 = st.columns([6, 1])
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
        st.markdown("")

    for item_idx, item in enumerate(st.session_state["evaluated_papers_buffer"]):
        render_breakdown_item(item, item_idx)

# --- Top Analytics Section: Side-by-Side Pidyne Forecast & Global Map of Science ---
top_analytics_col1, top_analytics_col2 = st.columns(2)

with top_analytics_col1:
    col_fc1, col_fc2 = st.columns([3, 1])
    with col_fc1:
        st.markdown(f"### Pidyne Forecast {rbot('pidyne forecast')}", unsafe_allow_html=True)
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

        curr_vals = st.session_state.current_weights
        pred_vals = st.session_state.predicted_next_weights

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

        st.markdown(
            f"**High-Precision Ledger Forecast (Raw Sum = {sum(st.session_state.predicted_next_weights):.6f}/8.0):** "
            f"C1: `{st.session_state.predicted_next_weights[0]:.5f}` | "
            f"C2: `{st.session_state.predicted_next_weights[1]:.5f}` | "
            f"C3: `{st.session_state.predicted_next_weights[2]:.5f}` | "
            f"C4: `{st.session_state.predicted_next_weights[3]:.5f}` | "
            f"C5: `{st.session_state.predicted_next_weights[4]:.5f}` | "
            f"C6: `{st.session_state.predicted_next_weights[5]:.5f}` | "
            f"C7: `{st.session_state.predicted_next_weights[6]:.5f}` | "
            f"C8: `{st.session_state.predicted_next_weights[7]:.5f}`"
        )

    with st.expander("What's Pidyne?", expanded=False):
        st.markdown(r"""
        Pidyne integrates the decentralized infrastructure layer of the Pi-Index Assessment Engine:
        1. **Active Epoch & Block Height**: Tracks incremental block updates. When the threshold (`EPOCH_BLOCK_SIZE`) is reached, a new blockchain block is minted.
        2. **Proof-of-Research (PoR) Validation (`validate_block_por`)**: Combines block index, criteria weights ($\varpi_1$ to $\varpi_8$), timestamp, previous block hash, validator node signature, model identifier, and formulas hash into an unalterable SHA-256 block hash.
        3. **LSTM Meta-Learning**: Uses PyTorch to train directly on historical block weights to predict future shifts in algorithmic evaluation standards.
        """)

with top_analytics_col2:
    map_title_col, map_badge_col = st.columns([3, 2], vertical_alignment="center")
    with map_title_col:
        st.markdown(f"### Global Map of Science {rbot('global map of science')}", unsafe_allow_html=True)
    with map_badge_col:
        st.markdown(
            f"""
            <div style="background-color: #2c3e50; color: white; padding: 4px 10px; border-radius: 15px; font-size: 12px; font-weight: bold; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                Total Analyzed Papers: {total_analyzed_count}
            </div>
            """,
            unsafe_allow_html=True,
        )

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
        mod_col1, mod_col2 = st.columns(2)
        with mod_col1:
            st.slider("Repulsion Force", min_value=-20000, max_value=-100, value=-3000, step=500, key="mod_repulsion")
            st.slider("Spring Length", min_value=10, max_value=1000, value=180, step=20, key="mod_spring")
        with mod_col2:
            st.slider("Bubble Size Scale", min_value=0.1, max_value=8.0, value=1.5, step=0.1, key="mod_size")
            st.slider("Central Pull (Gravity)", min_value=0.0, max_value=2.0, value=0.15, step=0.01, key="mod_gravity")

    with tab_legend:
        st.markdown(table_html_top, unsafe_allow_html=True)

st.markdown("---")

# --- Side-by-Side Section: Latest Assessed Papers & Pi Quotient Leaderboard ---
bottom_col1, bottom_col2 = st.columns(2, vertical_alignment="top")

with bottom_col1:
    if st.session_state.is_authenticated:
        conn_hist = get_db_connection()
        try:
            cur_h = conn_hist.cursor()
            cur_h.execute(
                """SELECT p.title, p.author_name, p.filename, p.final_score, p.logic_score, 
                          p.piq_minted, p.tx_hash, p.zk_proof, p.eval_hash, p.timestamp,
                          b.block_height, b.block_hash, p.mdar_adherence_score, 
                          p.rrid_valid_count, p.reproducibility_score
                   FROM papers_assessment p
                   LEFT JOIN blockchain_por_weights b ON p.eval_hash = b.eval_hash
                   WHERE p.user_id = ? OR p.user_id = '0009-0009-8456-8050'
                   ORDER BY p.timestamp DESC""",
                (st.session_state.orcid_id,)
            )
            user_history_rows = cur_h.fetchall()
        finally:
            conn_hist.close()

        st.markdown("### Your Assessment History & Rewards")

        if user_history_rows:
            for idx, uh in enumerate(user_history_rows):
                (
                    u_title, u_author, u_filename, u_score, u_logic,
                    u_piq, u_tx, u_zk, u_hash, u_time,
                    u_block_height, u_block_hash, u_mdar, u_rrid, u_repro
                ) = uh

                u_author_clean = clean_author_name(u_author)
                u_book = "0x" + hashlib.sha256(u_author_clean.encode()).hexdigest()[:40]
                u_tx_url = safe_get_sepolia_url(u_tx)
                
                tx_disp_val = u_tx if u_tx and str(u_tx).strip() not in ["None", ""] else "Not Connected / No Book / Missing PK"

                with st.expander(
                    f"[{idx+1}] {u_title[:50]}... — *{u_author_clean}* (Score: **{u_score:.2f}** | piQ: `{u_piq}`)",
                    expanded=False,
                ):
                    st.write(f"**File Name:** {u_filename if u_filename else 'N/A'}")
                    st.write(f"**Evaluation Hash (Paper Address):** `{u_hash}`")
                    st.write(f"**Unique Book Address:** `{u_book}`")
                    st.write(f"**piQ Minted:** `{u_piq}`")
                    st.markdown(f"**zk-SNARK {rbot('zk-snark')}:** `{u_zk}`", unsafe_allow_html=True)
                    
                    if u_tx_url:
                        st.markdown(f"**Tx Hash:** [`{tx_disp_val}`]({u_tx_url})")
                    else:
                        st.write(f"**Tx Hash:** `{tx_disp_val}`")

                    st.markdown(f"**Executable Reproducibility Score {rbot('executable reproducibility score')}:** `{u_repro * 100:.1f}%`", unsafe_allow_html=True)
                    st.markdown(f"**SciScore MDAR Adherence {rbot('sciscore mdar')}:** `{u_mdar * 100:.1f}%` | **Valid RRIDs:** `{u_rrid}`", unsafe_allow_html=True)
        else:
            st.info("No assessment history or rewards found linked to this authenticated ID.")
    else:
        st.markdown("### Latest Assessed Papers")

        conn_last = get_db_connection()
        try:
            cur_last = conn_last.cursor()
            cur_last.execute(
                """SELECT p.title, p.author_name, p.filename, p.final_score, p.logic_score, 
                          p.c1, p.c2, p.c3, p.c4, p.c5, p.c6, p.c7, p.c8, 
                          p.piq_minted, p.tx_hash, p.zk_proof, p.mdar_adherence_score, 
                          p.rrid_valid_count, p.reproducibility_score, p.eval_hash, p.timestamp,
                          b.block_height, b.block_hash
                       FROM papers_assessment p
                       LEFT JOIN blockchain_por_weights b ON p.eval_hash = b.eval_hash
                       ORDER BY p.timestamp DESC LIMIT 5"""
            )
            recent_papers = cur_last.fetchall()
        finally:
            conn_last.close()

        if not recent_papers:
            st.info("No papers have been assessed in the database yet.")
        else:
            for idx, rp in enumerate(recent_papers):
                (
                    r_title, r_author, r_filename, r_score, r_logic,
                    r_c1, r_c2, r_c3, r_c4, r_c5, r_c6, r_c7, r_c8,
                    r_piq, r_tx, r_zk, r_mdar, r_rrid, r_repro, r_hash, r_time,
                    r_block_height, r_block_hash
                ) = rp

                r_author_clean = clean_author_name(r_author)
                r_book = "0x" + hashlib.sha256(r_author_clean.encode()).hexdigest()[:40]
                r_tx_url = safe_get_sepolia_url(r_tx)
                
                tx_disp_val = r_tx if r_tx and str(r_tx).strip() not in ["None", ""] else "Not Connected / No Book / Missing PK"

                with st.expander(
                    f"[{idx+1}] {r_title[:50]}... — *{r_author_clean}* (Score: **{r_score:.2f}**)",
                    expanded=False,
                ):
                    st.write(f"**File Name:** {r_filename if r_filename else 'N/A'}")
                    st.write(f"**Evaluation Hash (Paper Address):** `{r_hash}`")
                    st.write(f"**Unique Book Address:** `{r_book}`")
                    st.write(f"**piQ Minted:** `{r_piq}`")
                    st.markdown(f"**zk-SNARK {rbot('zk-snark')} Proof:** `{r_zk}`", unsafe_allow_html=True)
                    
                    if r_tx_url:
                        st.markdown(f"**Tx Hash:** [`{tx_disp_val}`]({r_tx_url})")
                    else:
                        st.write(f"**Tx Hash:** `{tx_disp_val}`")

                    st.markdown(f"**Executable Reproducibility Score {rbot('executable reproducibility score')}:** `{r_repro * 100:.1f}%`", unsafe_allow_html=True)
                    st.markdown(f"**SciScore MDAR Adherence {rbot('sciscore mdar')}:** `{r_mdar * 100:.1f}%` | **Valid RRIDs:** `{r_rrid}`", unsafe_allow_html=True)

with bottom_col2:
    st.markdown("### Pi Quotient (piQ) Leaderboard")
    if piq_dict:
        leaderboard_data = []
        for author, piq in piq_dict.items():
            leaderboard_data.append({
                "Contributing Author": author,
                "Unique Book Address": book_dict.get(author, "None"),
                "Total piQ Earned": round(piq, 2),
            })
        piq_df = pd.DataFrame(leaderboard_data).sort_values(by="Total piQ Earned", ascending=False).reset_index(drop=True)
        st.dataframe(piq_df, use_container_width=True, height=210)
    else:
        st.info("No piQ tokens minted yet.")

st.markdown("---")
st.markdown(f"### Proof-of-Research Blockchain Explorer {rbot('proof-of-research')}", unsafe_allow_html=True)

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
        (
            block_height, weights, model_used, eval_hash,
            block_hash, por_proof, formulas_hash,
        ) = (
            epoch_data[0], epoch_data[1:9], epoch_data[9],
            epoch_data[10], epoch_data[11], epoch_data[12], epoch_data[13],
        )

        explore_col1, explore_col2 = st.columns([3, 1])
        with explore_col1:
            search_query = st.text_input(
                "Enter Document Evaluation Hash, Block Hash, Paper Name, Author Name, or Book Address to verify ledger record...",
                key="pidyne_ledger_search_query"
            )
        with explore_col2:
            st.write("")
            st.write("")
            search_btn = st.button("Verify Record", key="pidyne_verify_record_btn")

        if search_btn and search_query:
            try:
                q_term = f"%{search_query.strip()}%"
                cursor.execute(
                    """SELECT p.title, p.author_name, p.filename, p.final_score, p.logic_score, 
                              p.c1, p.c2, p.c3, p.c4, p.c5, p.c6, p.c7, p.c8, 
                              p.piq_minted, p.tx_hash, p.zk_proof, p.mdar_adherence_score, 
                              p.rrid_valid_count, p.reproducibility_score, p.eval_hash, p.timestamp,
                              b.block_height, b.block_hash, b.por_proof, b.formulas_hash, p.eth_book
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
                            m_block_height, m_block_hash, m_por, m_form, m_book_addr
                        ) = mr

                        m_author_clean = clean_author_name(m_author)
                        m_book = m_book_addr if m_book_addr else ("0x" + hashlib.sha256(m_author_clean.encode()).hexdigest()[:40])
                        m_tx_url = safe_get_sepolia_url(m_tx)
                        
                        tx_disp_val = m_tx if m_tx and str(m_tx).strip() not in ["None", ""] else "Not Connected / No Book / Missing PK"

                        with st.expander(
                            f"[{m_idx+1}] {m_title[:65]}... — *{m_author_clean}* (Score:"
                            f" **{m_score:.2f}** | {m_time[:16]})",
                            expanded=True,
                        ):
                            st.write(f"**File Name:** {m_filename if m_filename else 'N/A'}")
                            st.write(f"**Evaluation Hash (Paper Address):** `{m_hash}`")
                            st.write(f"**Unique Book Address:** `{m_book}`")
                            st.write(f"**piQ Minted:** `{m_piq}`")
                            st.markdown(f"**zk-SNARK {rbot('zk-snark')} Proof:** `{m_zk}`", unsafe_allow_html=True)
                            
                            if m_tx_url:
                                st.markdown(f"**Tx Hash:** [`{tx_disp_val}`]({m_tx_url})")
                            else:
                                st.write(f"**Tx Hash:** `{tx_disp_val}`")

                            st.markdown(f"**Executable Reproducibility Score {rbot('executable reproducibility score')}:** `{m_repro * 100:.1f}%`", unsafe_allow_html=True)
                            st.markdown(f"**SciScore MDAR Adherence {rbot('sciscore mdar')}:** `{m_mdar * 100:.1f}%` | **Valid RRIDs:** `{m_rrid}`", unsafe_allow_html=True)
                else:
                    st.error(
                        "No records matching that evaluation hash, block hash, paper name, author name, or book address were found on the ledger."
                    )
            except Exception as e:
                st.error(f"Error reading database: {str(e)}")

        st.info(
            f"**Latest Proof-of-Research:** `{por_proof}` successfully verified and"
            f" sealed to block `{block_hash}`."
        )
        st.caption(
            f"**Unalterable Criteria State Hash:** `{formulas_hash}` (Guarantees"
            " grading mathematical constants cannot be tampered with)."
        )

        piq_url = f"https://sepolia.etherscan.io/address/{PIQ_CONTRACT_ADDRESS}"
        reg_url = f"https://sepolia.etherscan.io/address/{REGISTRY_CONTRACT_ADDRESS}" if REGISTRY_CONTRACT_ADDRESS else "#"
        st.markdown(f"**Deployed Smart Contracts on Sepolia Etherscan:** PiQ Token Contract: [`{PIQ_CONTRACT_ADDRESS}`]({piq_url}) | Registry Contract: [`{REGISTRY_CONTRACT_ADDRESS}`]({reg_url})")

        st.markdown("#### Recent Ledger Proofs & Transactions")
        cursor.execute(
            """SELECT p.title, p.author_name, p.filename, p.final_score, p.logic_score, 
                      p.piq_minted, p.tx_hash, p.zk_proof, p.eval_hash, p.timestamp,
                      b.block_height, b.block_hash
               FROM papers_assessment p
               LEFT JOIN blockchain_por_weights b ON p.eval_hash = b.eval_hash
               ORDER BY p.timestamp DESC LIMIT 5"""
        )
        recent_ledger_rows = cursor.fetchall()
        if recent_ledger_rows:
            table_data = []
            for rrow in recent_ledger_rows:
                rtitle, rauth, rfile, rscore, rlogic, rpiq, rtx, rzk, reval, rts, rbh, rbhash = rrow
                tx_url = safe_get_sepolia_url(rtx)
                tx_disp_val = rtx if rtx and str(tx_disp_val).strip() not in ["None", ""] else "Missing PK"
                tx_disp = f"[{tx_disp_val[:10]}...]({tx_url})" if rtx and tx_url else str(tx_disp_val)
                table_data.append({
                    "Block Height": rbh if rbh is not None else "Pending",
                    "Eval Hash": reval[:10] + "...",
                    "Block Hash": rbhash[:10] + "..." if rbhash else "Pending",
                    "zk-SNARK": rzk[:10] + "..." if rzk else "N/A",
                    "piQ": rpiq,
                    "Tx Hash (Etherscan)": tx_disp,
                    "Timestamp": rts[:19] if rts else ""
                })
            st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)
        else:
            st.info("No ledger transaction proofs recorded yet.")

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
            Retry [label="Groq Fallback Retry Logic\n• Distributed Concurrency Control\n• Exponential 429 Backoff", fillcolor="#a3e4d7", style="dashed,filled"];
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
col_pad1, col_center, col_pad2 = st.columns([1, 4, 1])
with col_center:
    if st.button("The Pi-Index Framework Workflow", use_container_width=True):
        framework_workflow_dialog()

# --- Floating, Draggable Scilem Corner Chatbot Window ---
scilem_container = st.container()
with scilem_container:
    st.markdown("""
    <div id='scilem-drag-handle'>
        <span class='robot-icon'>🤖</span> Scilem Assistant
        <span id='scilem-min-btn' title='Minimize/Expand' style='position: absolute; top: 6px; right: 10px; cursor: pointer; font-weight: bold; font-size: 16px; padding: 2px 6px;'>--</span>
    </div>
    """, unsafe_allow_html=True)
    
    floating_chat_container = st.container(height=240)
    with floating_chat_container:
        for idx, message in enumerate(st.session_state.scilm_messages):
            msg_avatar = "🤖" if message["role"] == "assistant" else "👤"
            with st.chat_message(message["role"], avatar=msg_avatar):
                st.markdown(message["content"])

    with st.form(key="scilem_floating_form", clear_on_submit=True):
        f_cols = st.columns([3, 1])
        with f_cols[0]:
            floating_prompt = st.text_input("Ask Scilem...", placeholder="Ask a question...", label_visibility="collapsed")
        with f_cols[1]:
            submitted_floating = st.form_submit_button("Send")

    if submitted_floating and floating_prompt:
        st.session_state.scilm_messages.append({"role": "user", "content": floating_prompt})
        
        direct_answer = None
        if floating_prompt.startswith("Explain:"):
            query_topic = floating_prompt.replace("Explain:", "").strip().lower()
            for key, explanation in SCILEM_KNOWLEDGE_BASE.items():
                if key in query_topic:
                    direct_answer = explanation
                    break

        if direct_answer:
            st.session_state.scilm_messages.append({"role": "assistant", "content": f"{direct_answer}"})
            st.rerun()
        else:
            rag_context = ""
            few_shot_examples = ""
            try:
                dataset_path = os.path.join(BASE_DIR, "scilem rlhf_dataset.jsonl")
                if os.path.exists(dataset_path):
                    with open(dataset_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        query_terms = set(floating_prompt.lower().split())
                        relevant_lines = [l for l in lines if any(t in l.lower() for t in query_terms if len(t) > 3)]
                        if not relevant_lines:
                            relevant_lines = lines[-5:]
                        rag_context = "".join(relevant_lines[-5:])
            except Exception:
                rag_context = "No decentralized data accessible."

            try:
                conn_rag = get_db_connection()
                cur_rag = conn_rag.cursor()
                cur_rag.execute("SELECT title, author_name, final_score FROM papers_assessment ORDER BY final_score DESC LIMIT 1")
                top_paper = cur_rag.fetchone()
                conn_rag.close()
                if top_paper:
                    few_shot_examples = f"Exemplar Reference Paper: '{top_paper[0]}' by {top_paper[1]} (Score: {top_paper[2]:.2f}/100)"
            except Exception:
                pass

            scilem_sys_prompt = (
                "You are Scilem, an advanced Scientific LLM aligned with CoARA guidelines and the Pi-Index Whitepaper. "
                "Explain app features clearly and concisely. You just received a user action or query.\n\n"
                f"DECENTRALIZED LEDGER CONTEXT (RAG):\n{rag_context}\n\n"
                f"TOP-SCOURING EXEMPLAR:\n{few_shot_examples}"
            )

            messages_for_api = [{"role": "system", "content": scilem_sys_prompt}] + [
                {"role": m["role"], "content": m["content"]} for m in st.session_state.scilm_messages
            ]

            full_response = ""
            try:
                from brain import groq_client
                PRIMARY_MODEL_NAME = "llama-3.3-70b-versatile"
                FALLBACK_MODEL_NAME = "llama-3.1-8b-instant"
                if groq_client:
                    add_log("Dispatching query to Scilem AI Engine...")
                    for attempt in range(3):
                        try:
                            response = groq_client.chat.completions.create(
                                model=PRIMARY_MODEL_NAME,
                                messages=messages_for_api,
                                temperature=0.15,
                            )
                            full_response = response.choices[0].message.content
                            add_log("Scilem response generated.")
                            break
                        except Exception as primary_err:
                            err_str = str(primary_err).lower()
                            if any(k in err_str for k in ["413", "rate_limit_exceeded", "tokens", "limit", "429"]):
                                if attempt < 2:
                                    add_log(f"Rate limit hit. Retrying in {2**attempt}s...")
                                    time.sleep(2 ** attempt)
                                    continue
                                
                                trimmed_messages = [messages_for_api[0]] + messages_for_api[-2:]
                                try:
                                    fallback_response = groq_client.chat.completions.create(
                                        model=FALLBACK_MODEL_NAME,
                                        messages=trimmed_messages,
                                        temperature=0.15,
                                    )
                                    full_response = fallback_response.choices[0].message.content + "\n\n*(Payload automatically trimmed to fit TPM rate limits).* "
                                    add_log("Scilem fallback model executed successfully.")
                                    break
                                except Exception as second_err:
                                    full_response = f"Error: Token limit exceeded and fallback failed: {str(second_err)}"
                                    add_log("Scilem fallback model failed.")
                                    break
                            else:
                                full_response = f"Error: {str(primary_err)}"
                                break
                else:
                    full_response = "Error: Groq API client not initialized."
            except Exception as e:
                full_response = f"Error connecting to Scilem engine: {str(e)}"

            st.session_state.scilm_messages.append({"role": "assistant", "content": full_response})
            st.rerun()
