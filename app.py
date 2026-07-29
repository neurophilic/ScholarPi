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

st.set_page_config(page_title="Pi-Index Assessment Engine", layout="wide")

if "app_logs" not in st.session_state:
    st.session_state.app_logs = deque(maxlen=50)

def add_log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {msg}"
    st.session_state.app_logs.appendleft(log_entry)
    logging.info(log_entry)

def get_tx_url(tx):
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
    finally: conn.close()
    
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

def rbot(topic_key):
    return f"<span class='scilem-trigger' data-query='{topic_key}' title='Ask Scilem' style='cursor: pointer !important; opacity:0.8;'>[?]</span>"

if "web3_wallet" not in st.session_state: st.session_state.web3_wallet = None
if "orcid_profile" not in st.session_state: st.session_state.orcid_profile = None
if "researcher_name" not in st.session_state: st.session_state.researcher_name = "Anonymous Researcher"

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
                if w3.eth.account.recover_message(encode_defunct(text=decoded_msg), signature=raw_signature).lower() == clean_wallet.lower():
                    authenticated = True
            except Exception: pass
        if authenticated:
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
        response = requests.post("https://orcid.org/oauth/token", data={"client_id": ORCID_CLIENT_ID, "client_secret": ORCID_CLIENT_SECRET, "grant_type": "authorization_code", "code": auth_code, "redirect_uri": ORCID_REDIRECT_URI}, headers={"Accept": "application/json"})
        if response.status_code == 200:
            orcid_data = response.json()
            if orcid_data.get("orcid"):
                st.session_state.orcid_profile = orcid_data.get("orcid")
                st.session_state.researcher_name = orcid_data.get("name") or f"ORCID Scholar"
                st.toast(f"ORCID Linked: {st.session_state.researcher_name}")
    except Exception: pass
    st.query_params.clear()
    st.rerun()

custom_ui_code = """
<style>
h1, h2, h3, h4, h5, h6 { color: #0f172a !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; font-weight: 600 !important; }
hr { border-color: #e2e8f0 !important; margin: 1.5rem 0 !important; }
[data-testid="stHeaderActionElements"] { display: none !important; }
[data-testid="stSidebar"] { background-color: #f8fafc !important; border-right: 1px solid #e2e8f0 !important; overflow-y: auto !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px !important; border: 1px solid #e2e8f0 !important; background-color: #ffffff !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); padding: 0.5rem !important; }
button[kind="primary"] { background-color: #000080 !important; border-color: #000080 !important; color: #ffffff !important; }
button[kind="secondary"] { background-color: #dc2626 !important; border-color: #dc2626 !important; color: #ffffff !important; }
.unified-auth-btn { width: 100%; background-color: #0f172a; color: white; border: 1px solid #1e293b; padding: 10px 14px; border-radius: 8px; font-weight: 600; font-size: 14px; cursor: pointer; display: flex; align-items: center; justify-content: center; text-decoration: none; box-sizing: border-box; }
.unified-auth-btn:hover { background-color: #1e293b; color: white; }
.auth-status-txt { margin-top: 4px; font-size: 11px; color: #dc2626; font-weight: 500; text-align: center; }
.vis-gradient-canvas { background: radial-gradient(circle at 50% 50%, #ffffff 0%, #f0f2f5 100%); width: 100% !important; height: 600px !important; }
.color-box { width: 14px; height: 14px; border-radius: 3px; display: inline-block; vertical-align: middle; margin-right: 8px; }
</style>
"""
components.html(custom_ui_code, height=0, width=0)

st.sidebar.title("System Access & Sync")
has_web3 = bool(st.session_state.web3_wallet and w3.is_address(st.session_state.web3_wallet))
has_orcid = bool(st.session_state.orcid_profile)
orcid_auth_url = f"https://orcid.org/oauth/authorize?client_id={ORCID_CLIENT_ID}&response_type=code&scope=/authenticate&redirect_uri={ORCID_REDIRECT_URI}&state={st.session_state.web3_wallet if has_web3 else 'none'}"

mm_button_html = f"""<button id="connect-mm-btn" class="unified-auth-btn" type="button">Connect MetaMask Web3</button><div id="mm-status" class="auth-status-txt"></div>
<script>
const mmBtn = document.getElementById('connect-mm-btn');
if (mmBtn) {{
    mmBtn.addEventListener('click', async () => {{
        const provider = window.ethereum || (window.parent && window.parent.ethereum);
        if (!provider) {{ document.getElementById('mm-status').innerText = "MetaMask not detected!"; return; }}
        try {{
            const accounts = await provider.request({{ method: 'eth_requestAccounts' }});
            const account = accounts[0];
            const message = `ScholarPi wants you to sign in with your Ethereum account:\\n${{account}}\\n\\nNonce: ${{Math.floor(Math.random() * 100000000)}}`;
            const hexMessage = '0x' + Array.from(new TextEncoder().encode(message)).map(b => b.toString(16).padStart(2, '0')).join('');
            const signature = await provider.request({{ method: 'personal_sign', params: [hexMessage, account] }}).catch(() => null);
            const targetUrl = new URL(window.top.location.href.split('?')[0]);
            targetUrl.searchParams.set("siwe_address", account);
            if (signature) {{ targetUrl.searchParams.set("siwe_signature", signature); targetUrl.searchParams.set("siwe_message", encodeURIComponent(message)); }}
            window.open(targetUrl.href, '_blank');
        }} catch (err) {{ document.getElementById('mm-status').innerText = "Rejected."; }}
    }});
}}
</script>"""

with st.sidebar:
    if not has_web3: components.html(mm_button_html, height=100)
    else: st.success(f"Web3 Linked: `{st.session_state.web3_wallet[:6]}...{st.session_state.web3_wallet[-4:]}`")
    if not has_orcid: components.html(f'<a href="{orcid_auth_url}" target="_blank" class="unified-auth-btn">Link ORCID Account</a>', height=60)
    else: st.success(f"ORCID Linked: `{st.session_state.orcid_profile}`")

client_ip = "127.0.0.1"
try:
    headers = st.context.headers
    client_ip = (headers.get("X-Forwarded-For") or headers.get("X-Real-Ip") or "127.0.0.1").split(",")[0].strip()
except Exception: pass

conn_ip = get_db_connection()
try:
    cur_ip = conn_ip.cursor()
    cur_ip.execute("SELECT ip_address FROM auto_ip_tracking WHERE ip_address=?", (client_ip,))
    if not cur_ip.fetchone():
        cur_ip.execute("INSERT INTO auto_ip_tracking (ip_address, first_seen) VALUES (?, ?)", (client_ip, datetime.now().isoformat()))
        conn_ip.commit()
finally: conn_ip.close()

conn_cnt = get_db_connection()
try: total_analyzed_count = conn_cnt.cursor().execute("SELECT COUNT(*) FROM papers_assessment").fetchone()[0]
finally: conn_cnt.close()

if "state_restored" not in st.session_state:
    restore_state_from_web3()
    st.session_state["state_restored"] = True

if "assessment_update_token" not in st.session_state: st.session_state["assessment_update_token"] = time.time()
if "reset_token" not in st.session_state: st.session_state["reset_token"] = 0
if "evaluated_papers_buffer" not in st.session_state: st.session_state["evaluated_papers_buffer"] = []
if "download_errors" not in st.session_state: st.session_state["download_errors"] = []
if "is_running" not in st.session_state: st.session_state["is_running"] = False
if "cancel_requested" not in st.session_state: st.session_state["cancel_requested"] = False
if "session_temp_dir" not in st.session_state: st.session_state["session_temp_dir"] = tempfile.mkdtemp()
if "scilem_messages" not in st.session_state: st.session_state.scilem_messages = [{"role": "assistant", "content": "**Welcome! I am Scilem.** Ask any research question."}]

@st.cache_data(ttl=3600)
def build_science_map(target_author, repulsion=-3000, spring_len=180, size_scale=1.5, central_grav=0.15, _db_token=0):
    conn = get_db_connection()
    try: data = conn.cursor().execute("SELECT fields, subfields, final_score, author_name FROM papers_assessment").fetchall()
    finally: conn.close()

    html_string, table_data = "", []
    if not data: return html_string, table_data

    topic_aggregates = {}
    for fields_json, subfields_json, final_score, author_str in data:
        if target_author and target_author != "All Authors" and target_author not in clean_author_name(author_str): continue
        try:
            for rs in [s.title().strip() for s in json.loads(subfields_json)]:
                if rs and rs.lower() not in {"general", "unspecified"}:
                    topic_aggregates.setdefault(rs, {"weight_sum": 0.0, "frequency": 0})
                    topic_aggregates[rs]["weight_sum"] += safe_float(final_score, 50.0)
                    topic_aggregates[rs]["frequency"] += 1
        except Exception: continue

    if not topic_aggregates: topic_aggregates["General Science > Core Research"] = {"weight_sum": 50.0, "frequency": 1}
    if len(topic_aggregates) > 15: topic_aggregates = dict(sorted(topic_aggregates.items(), key=lambda x: (x[1]["frequency"], x[1]["weight_sum"]), reverse=True)[:15])

    unique_topics = list(topic_aggregates.keys())
    major_fields_dict = {}
    for topic in unique_topics:
        major = topic.split('>')[0].strip() if '>' in topic else topic
        major_fields_dict.setdefault(major, []).append(topic)

    color_map = {}
    major_keys = sorted(list(major_fields_dict.keys()))
    for i, major in enumerate(major_keys):
        h = i / max(1, len(major_keys))
        for j, topic in enumerate(sorted(major_fields_dict[major])):
            rgb = colorsys.hsv_to_rgb(h, 0.6 + (0.3 * (j / max(1, len(major_fields_dict[major]) - 1))), 0.9)
            color_map[topic] = "#%02x%02x%02x" % tuple(int(x * 255) for x in rgb)

    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#2c3e50", notebook=False)
    net.set_options(f"""{{ "physics": {{ "barnesHut": {{ "gravitationalConstant": {repulsion}, "centralGravity": {central_grav}, "springLength": {spring_len}, "springConstant": 0.005, "avoidOverlap": 2.0 }}, "stabilization": {{ "enabled": true, "iterations": 2500 }} }} }}""")

    for topic, metrics in topic_aggregates.items():
        avg_weight = metrics["weight_sum"] / metrics["frequency"]
        net.add_node(n_id=topic, label=" ", title=f"Field: {topic} | Freq: {metrics['frequency']} | Avg Score: {avg_weight:.1f}", size=max(35, (25 + (avg_weight * 3.0)) * size_scale), shape="dot", color={"background": color_map[topic], "border": "#1a1a1a"})

    for i, t1 in enumerate(unique_topics):
        for j, t2 in enumerate(unique_topics):
            if i < j and t1.split(">")[0].strip() == t2.split(">")[0].strip():
                net.add_edge(t1, t2, color="rgba(150,150,150,0.2)")

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".html")
    os.close(tmp_fd)
    try:
        net.save_graph(tmp_name)
        with open(tmp_name, "r", encoding="utf-8") as f: html_string = f.read()
    finally:
        if os.path.exists(tmp_name): os.remove(tmp_name)

    html_string = html_string.replace("</head>", f"<!-- reload: {time.time()} --></head>").replace("<canvas", "<canvas class='vis-gradient-canvas'")
    for topic, metrics in sorted(topic_aggregates.items(), key=lambda x: x[1]["frequency"], reverse=True):
        table_data.append({"Color": color_map[topic], "Science Field": topic, "Frequency": metrics["frequency"], "Avg Weight": round(metrics["weight_sum"] / metrics["frequency"], 1)})
    return html_string, table_data

def get_criteria_info(weights):
    tw1, tw2, tw3, tw4, tw5, tw6, tw7, tw8 = weights
    return [
        ("C1", "Originality", "c1", tw1, "1", "Semantic distance from literature corpus.", r"$$ C_1 = \varpi_1 \cdot \mathcal{D}_{semantic} $$"),
        ("C2", "Methodological Rigor", "c2", tw2, "2", "MDAR reporting standards and valid RRIDs.", r"$$ C_2 = \varpi_2 \cdot \mathcal{I}_{mdar} $$"),
        ("C3", "Interdisciplinary Synergy", "c3", tw3, "3", "Cross-disciplinary integration and entropy.", r"$$ C_3 = \varpi_3 \cdot \text{Entropy} $$"),
        ("C4", "Societal Impact", "c4", tw4, "4", "Broader societal and open infrastructure.", r"$$ C_4 = \varpi_4 \cdot \Theta $$"),
        ("C5", "Open Science", "c5", tw5, "5", "Open data, code, and reproducibility.", r"$$ C_5 = \varpi_5 \cdot \text{Repro} $$"),
        ("C6", "Literature Integration", "c6", tw6, "6", "Citation polarity and foundational integration.", r"$$ C_6 = \varpi_6 \cdot \text{Polarity} $$"),
        ("C7", "Empirical Density", "c7", tw7, "7", "Empirical sample strength and variance.", r"$$ C_7 = \varpi_7 \cdot \text{Density} $$"),
        ("C8", "Future Actionability", "c8", tw8, "8", "FAIR principles adherence.", r"$$ C_8 = \varpi_8 \cdot \text{FAIR} $$"),
    ]

@st.dialog("Criterion Details", width="medium")
def show_criterion_metrics(c_id, title, q_key, weight_val, sym, desc, formula):
    st.markdown(f"### {c_id}: {title}")
    st.markdown(rf"**Epoch Weight ($\varpi_{sym}$):** `{weight_val:.6f}`")
    st.markdown(desc)
    st.markdown(formula)

col1, col2 = st.columns([4, 2], vertical_alignment="center")
with col1: st.markdown("<h1 style='margin-bottom:0;'>Pi-Index Assessment Engine</h1>", unsafe_allow_html=True)
with col2: st.markdown(f"<div style='float: right; background-color: #0f172a; color: white; padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 600;'>Total Analyzed Papers: <span style='color: #60a5fa;'>{total_analyzed_count}</span></div>", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("### Assess a Manuscript")
    free_evals = st.session_state.get("free_evals_used", 0)
    stake_amount = True if free_evals == 0 else st.checkbox("Stake 0.1 piQ to Process (Returned on Valid Assessment)", value=True, key=f"stake_{st.session_state['reset_token']}")

    tab_local, tab_doi, tab_search = st.tabs(["📄 Local Upload", "🔗 DOI Lookup", "🌐 Open Source Search"])
    selected_files = []
    with tab_local:
        up_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True, key=f"up_{st.session_state['reset_token']}")
        if up_files:
            for i, f in enumerate(up_files):
                if st.checkbox(f"Local: {f.name}", value=True, key=f"chk_{i}_{st.session_state['reset_token']}"): selected_files.append(f)

    doi_input = ""
    include_doi = False
    with tab_doi:
        doi_input = st.text_input("Enter DOI", placeholder="10.1000/xyz123", key=f"doi_{st.session_state['reset_token']}")
        include_doi = st.checkbox("Include DOI in assessment", value=False, key=f"doi_chk_{st.session_state['reset_token']}", disabled=not doi_input.strip())

    selected_search = []
    with tab_search:
        q_term = st.text_input("Search OpenAlex Topics", key=f"srch_{st.session_state['reset_token']}")
        if st.button("Search Open Source", key=f"srch_btn_{st.session_state['reset_token']}") and q_term:
            with st.spinner("Querying OpenAlex..."): st.session_state["search_results"] = search_openalex_topics(q_term, limit=5)
        if "search_results" in st.session_state and st.session_state["search_results"]:
            for i, res in enumerate(st.session_state["search_results"]):
                if st.checkbox(f"{res['title']} ({res['authors']})", key=f"sres_{i}_{st.session_state['reset_token']}"): selected_search.append(res)

    if st.session_state["is_running"]:
        if st.button("Stop Pipeline", type="secondary", use_container_width=True):
            st.session_state["is_running"] = False
            st.session_state["cancel_requested"] = True
            st.rerun()

        with st.status("Executing Assessment Pipeline...", expanded=True) as status:
            try:
                for item in selected_search:
                    if st.session_state["cancel_requested"]: break
                    pdf_bytes = download_pdf_from_url(item['pdf_url']) or (create_virtual_pdf_from_text(fetch_core_text_by_doi(item['doi']), item['title']) if item['doi'] else None)
                    if pdf_bytes:
                        res = assess_manuscript(pdf_bytes, "Search.pdf", "", st.session_state.orcid_profile or st.session_state.web3_wallet or "Anonymous", valid_book_address, provided_doi=item['doi'])
                        if res and len(res) >= 22:
                            st.session_state["evaluated_papers_buffer"].insert(0, {"title": res[0], "author_name": clean_author_name(res[1]), "score": res[2], "logic_integrity": res[3], "fields": res[6], "subfields": res[7], "scores_dict": res[8], "eval_hash": res[9], "piq": res[10], "tx_hash": res[11], "zk_proof": res[12], "used_weights": res[13], "h_idx": res[14], "i10_idx": res[15], "repro_score": res[16], "warnings": res[18], "consensus_raw": res[19], "evidence_report_text": res[20]})
                            st.session_state["free_evals_used"] += 1
                if include_doi and doi_input.strip() and not st.session_state["cancel_requested"]:
                    meta = fetch_doi_metadata(doi_input)
                    pdf_bytes = download_pdf_from_url(meta.get("pdf_url")) if meta else None
                    if pdf_bytes:
                        res = assess_manuscript(pdf_bytes, "DOI.pdf", "", st.session_state.orcid_profile or st.session_state.web3_wallet or "Anonymous", valid_book_address, provided_doi=doi_input.strip())
                        if res and len(res) >= 22:
                            st.session_state["evaluated_papers_buffer"].insert(0, {"title": res[0], "author_name": clean_author_name(res[1]), "score": res[2], "logic_integrity": res[3], "fields": res[6], "subfields": res[7], "scores_dict": res[8], "eval_hash": res[9], "piq": res[10], "tx_hash": res[11], "zk_proof": res[12], "used_weights": res[13], "h_idx": res[14], "i10_idx": res[15], "repro_score": res[16], "warnings": res[18], "consensus_raw": res[19], "evidence_report_text": res[20]})
                            st.session_state["free_evals_used"] += 1
                for fname, fpath in st.session_state.get("snap_files", []):
                    if st.session_state["cancel_requested"]: break
                    with open(fpath, "rb") as f:
                        res = assess_manuscript(f.read(), fname, "", st.session_state.orcid_profile or st.session_state.web3_wallet or "Anonymous", valid_book_address)
                        if res and len(res) >= 22:
                            st.session_state["evaluated_papers_buffer"].insert(0, {"title": res[0], "author_name": clean_author_name(res[1]), "score": res[2], "logic_integrity": res[3], "fields": res[6], "subfields": res[7], "scores_dict": res[8], "eval_hash": res[9], "piq": res[10], "tx_hash": res[11], "zk_proof": res[12], "used_weights": res[13], "h_idx": res[14], "i10_idx": res[15], "repro_score": res[16], "warnings": res[18], "consensus_raw": res[19], "evidence_report_text": res[20]})
                            st.session_state["free_evals_used"] += 1
                status.update(label="Complete!", state="complete")
            finally:
                st.session_state["is_running"] = False
                st.session_state["cancel_requested"] = False
                st.session_state["reset_token"] += 1
                st.session_state["assessment_update_token"] = time.time()
                st.rerun()
    else:
        if st.button("Run Assessment Pipeline", type="primary", use_container_width=True):
            if free_evals >= 1 and not has_web3: st.error("Please connect your Web3 Wallet in the sidebar to stake 0.1 piQ.")
            elif free_evals >= 1 and not stake_amount: st.error("You must agree to stake 0.1 piQ.")
            elif not selected_files and not (include_doi and doi_input.strip()) and not selected_search: st.warning("Please select at least one source.")
            else:
                saved = []
                for f in selected_files:
                    fp = os.path.join(st.session_state["session_temp_dir"], f.name)
                    with open(fp, "wb") as out: out.write(f.getvalue())
                    saved.append((f.name, fp))
                st.session_state["snap_files"] = saved
                st.session_state["snap_search"] = selected_search
                st.session_state["snap_doi"] = doi_input
                st.session_state["snap_include_doi"] = include_doi
                st.session_state["is_running"] = True
                st.rerun()

@st.dialog("Detailed Research Integrity Dossier", width="large")
def show_dossier(item):
    st.subheader(f"{item.get('title')} by {clean_author_name(item.get('author_name'))}")
    st.write(f"**Evaluation Hash:** `{item.get('eval_hash')}`")
    st.write(f"**piQ Minted:** `{item.get('piq')}`")
    st.markdown(f"**zk-SNARK Proof:** `{item.get('zk_proof')}`")
    if item.get('evidence_report_text'): st.markdown(item.get('evidence_report_text'))

if st.session_state["evaluated_papers_buffer"]:
    st.markdown("### Assessment Results")
    for idx, item in enumerate(st.session_state["evaluated_papers_buffer"]):
        with st.container(border=True):
            cols = [st.columns([6, 4])[0], st.columns([6, 4])[1]]
            with cols[0]: st.markdown(f"**{item['title']}** — *{clean_author_name(item['author_name'])}*\nScore: **{item['score']:.2f}** | piQ: `{item['piq']}`")
            with cols[1]:
                if st.button("More Details", key=f"det_{idx}_{item['eval_hash']}", use_container_width=True): show_dossier(item)

col_a, col_b = st.columns(2, gap="large")
with col_a:
    st.markdown("### Pidyne Forecast")
    conn_pb = get_db_connection()
    try: hist_rows = conn_pb.cursor().execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights ORDER BY block_height ASC").fetchall()
    finally: conn_pb.close()

    if len(hist_rows) >= 2:
        df_hist = pd.DataFrame(hist_rows, columns=["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"])
        for col in df_hist.columns: df_hist[col] = 1.0 + (df_hist[col] - 1.0) * 1500.0
        st.altair_chart(alt.Chart(df_hist.reset_index().melt('index', var_name='Criterion', value_name='Weight')).mark_line(point=True).encode(x='index:O', y='Weight:Q', color='Criterion:N').properties(height=350), use_container_width=True)
        crit_info = get_criteria_info([1.0]*8)
        for idx, c_data in enumerate(crit_info[:4]):
            if st.button(f"{c_data[0]}: {c_data[3]:.5f}", key=f"crit_{c_data[0]}", use_container_width=True): show_criterion_metrics(*c_data)
    else: st.info("Assess at least 1 manuscript to generate historical training blocks.")

with col_b:
    st.markdown("### Global Map of Science")
    conn_m = get_db_connection()
    try: authors = sorted(list(set(clean_author_name(r[0]) for r in conn_m.cursor().execute("SELECT DISTINCT author_name FROM papers_assessment").fetchall() if r[0])))
    finally: conn_m.close()

    sel_auth = st.selectbox("Filter Map by Author:", ["All Authors"] + authors, key=f"auth_filter_{st.session_state['assessment_update_token']}")
    html_map, table_data = build_science_map(None if sel_auth == "All Authors" else sel_auth, _db_token=st.session_state['assessment_update_token'])
    
    if html_map:
        st.markdown("<div class='pyvis-map-wrapper'>", unsafe_allow_html=True)
        components.html(html_map, height=600, scrolling=False)
        st.markdown("</div>", unsafe_allow_html=True)
    
    tab_leg, tab_mod = st.tabs(["Legend", "Modulators"])
    with tab_leg:
        if table_data: st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)
        else: st.info("No topic data found.")
    with tab_mod:
        st.slider("Repulsion", -20000, -100, -3000, key="mod_repulsion")
        st.slider("Spring Length", 10, 1000, 180, key="mod_spring")

st.markdown("---")
col_s1, col_s2 = st.columns(2, gap="large")
with col_s1:
    st.markdown("### Pi Quotient (piQ) Leaderboard")
    piq_d, book_d = get_author_piq_dict()
    if piq_d:
        df_piq = pd.DataFrame(sorted(piq_d.items(), key=lambda x: x[1], reverse=True)[:20], columns=["Author", "piQ Mined"])
        df_piq["Book Address"] = [book_d.get(a, "None") for a in df_piq["Author"]]
        df_piq.index = np.arange(1, len(df_piq) + 1)
        st.dataframe(df_piq, use_container_width=True)
    else: st.info("No piQ minted yet.")

with col_s2:
    st.markdown("### pi-Index (piX) Leaderboard [Top Papers]")
    conn_pi = get_db_connection()
    try: top_papers = conn_pi.cursor().execute("SELECT title, author_name, final_score, eval_hash FROM papers_assessment ORDER BY final_score DESC LIMIT 10").fetchall()
    finally: conn_pi.close()
    if top_papers:
        for r, tp in enumerate(top_papers, 1):
            col1, col2 = st.columns([5, 1], vertical_alignment="center")
            col1.markdown(f"**#{r} {tp[0]}** — *{clean_author_name(tp[1])}* (Score: **{tp[2]:.2f}**)")
            if col2.button("Dossier", key=f"top_{r}_{tp[3]}"):
                conn_d = get_db_connection()
                row = conn_d.cursor().execute("SELECT * FROM papers_assessment WHERE eval_hash = ?", (tp[3],)).fetchone()
                conn_d.close()
                if row: show_dossier({"title": row[2], "author_name": row[17], "score": row[18], "eval_hash": row[0], "piq": row[21], "zk_proof": row[23], "evidence_report_text": row[33]})
    else: st.info("No leaderboard entries yet.")
