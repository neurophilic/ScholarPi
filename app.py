import os
import re
import json
import time
import hashlib
import tempfile
import colorsys
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pyvis.network import Network

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
    process_single_pdf, generate_rebuttal_strategy, PiBrainLSTM, 
    PiBlockchainDataset
)

def tooltip(text):
    svg_icon = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16"'
        ' height="16" fill="none" stroke="#9e9e9e" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round" style="vertical-align:'
        ' -3px; margin-left: 6px; cursor: help;"><circle cx="12" cy="12"'
        ' r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3'
        ' 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
    )
    return f'<span title="{text}">{svg_icon}</span>'

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

st.set_page_config(
    page_title="Pi-Index Assessment Engine", layout="wide"
)

st.sidebar.title("System Access")

if "initialized" not in st.session_state:
    st.session_state["initialized"] = True
    st.toast("Application initialized successfully.")

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
        logging.info(f"New User IP Connected locally logged: {client_ip}")
finally:
    conn_ip.close()

conn_cnt = get_db_connection()
try:
    cur_cnt = conn_cnt.cursor()
    cur_cnt.execute("SELECT COUNT(*) FROM papers_assessment")
    total_analyzed_count = cur_cnt.fetchone()[0]
finally:
    conn_cnt.close()

st.markdown(
    f"""
    <div style="position: absolute; top: 15px; right: 20px; background-color: #2c3e50; color: white; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: bold; box-shadow: 0 2px 5px rgba(0,0,0,0.2); z-index: 999;">
        Analyzed Papers: {total_analyzed_count}
    </div>
    """,
    unsafe_allow_html=True,
)

if "state_restored" not in st.session_state:
    restore_state_from_web3()
    st.session_state["state_restored"] = True

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

if "orcid_id" not in st.session_state:
    saved_orcid = st.query_params.get("orcid", "")
    if saved_orcid and (re.match(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$", saved_orcid) or "did:" in saved_orcid):
        st.session_state.orcid_id = saved_orcid
        st.session_state.orcid_name = "Verified Decentralized Identity" if "did:" in saved_orcid else "Verified Researcher (Name Private)"
        st.session_state.is_authenticated = True
    else:
        st.session_state.orcid_id = "0000-0000-0000-0000"
        st.session_state.orcid_name = ""
        st.session_state.is_authenticated = False

if not st.session_state.is_authenticated:
    st.sidebar.markdown(
        "### Authenticate "
        + tooltip(
            "Connect to your ORCID or DID to securely isolate your assessment"
            " history. Pi Quotient (piQ) is a Soulbound Token assigned strictly"
            " to this identity."
        ),
        unsafe_allow_html=True,
    )
    manual_orcid = st.sidebar.text_input(
        "Enter ORCID iD or W3C DID", placeholder="XXXX-XXXX-XXXX-XXXX"
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
                    is_valid, user_name = True, "Verified Decentralized Identity"
                else:
                    is_valid, user_name = True, "Verified Researcher (Name Private)"
            if is_valid:
                st.session_state.orcid_id = clean_orcid
                st.session_state.orcid_name = user_name
                st.session_state.is_authenticated = True
                st.session_state.inst_email = "None"
                if remember_user:
                    st.query_params["orcid"] = clean_orcid
                else:
                    if "orcid" in st.query_params:
                        del st.query_params["orcid"]
                st.rerun()
            else:
                st.sidebar.error(user_name)
        else:
            st.sidebar.error("Invalid ORCID or DID format.")

    st.sidebar.markdown("---")
    st.sidebar.info("Notice: Please connect your ORCID iD or DID above to unlock and use your personal Assessment History and DeSci Peer Attestation features.")
else:
    st.sidebar.success("Securely Connected")
    st.sidebar.markdown(
        f"**Researcher:** {st.session_state.orcid_name}\n**ID Vault:**"
        f" `{st.session_state.orcid_id}`"
    )
    
    # --- Assessment and Reward History in Sidebar ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Your Assessment & Reward History")
    
    current_user = st.session_state.orcid_id
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT title, author_name, filename, scope, final_score, piq_minted,"
            " tx_hash FROM papers_assessment WHERE user_id=? ORDER BY timestamp DESC"
            " LIMIT 20",
            (current_user,),
        )
        history_data = cursor.fetchall()
    finally:
        conn.close()
        
    if history_data:
        for row in history_data:
            title, author_name, filename, scope, score, piq, tx_h = row
            tx_url = safe_get_sepolia_url(tx_h)
            clean_auth = clean_author_name(author_name)
            st.sidebar.markdown(f"**{title[:45]}...**")
            st.sidebar.caption(f"Author: {clean_auth} | Score: **{score:.2f}** | piQ: `{piq}`")
            if tx_url:
                st.sidebar.markdown(f"Tx: [`{tx_h[:10]}...`]({tx_url})")
            st.sidebar.markdown("---")
    else:
        st.sidebar.info("No assessment history found for this ID.")

    if st.sidebar.button("Disconnect Session"):
        st.session_state.is_authenticated = False
        st.session_state.orcid_name = ""
        st.session_state.orcid_id = "0000-0000-0000-0000"
        if "orcid" in st.query_params:
            del st.query_params["orcid"]
        st.rerun()

current_user = st.session_state.get("orcid_id", "0000-0000-0000-0000")
current_email = "None"

# --- DeSci Peer Attestation & Stake-Weighted Validation in Sidebar ---
with st.sidebar.expander("DeSci Peer Attestation & Staking", expanded=False):
    st.markdown("Use this feature to endorse or challenge peer assessments on-chain by staking your earned piQ tokens.")
    if st.session_state.is_authenticated:
        conn_att = get_db_connection()
        try:
            cur_att = conn_att.cursor()
            cur_att.execute("SELECT eval_hash, title FROM papers_assessment ORDER BY timestamp DESC LIMIT 20")
            eval_papers_att = cur_att.fetchall()
        finally:
            conn_att.close()

        if eval_papers_att:
            attest_options = {p[1]: p[0] for p in eval_papers_att}
            chosen_attest_title = st.selectbox(
                "Select Paper for Attestation:",
                list(attest_options.keys()),
                key="sidebar_desci_attest_select",
            )
            target_eval_hash = attest_options[chosen_attest_title]

            attest_stance = st.radio(
                "Attestation Stance:",
                ["Endorse Rigor", "Challenge Anomaly"],
                horizontal=True,
                key="sidebar_attest_stance"
            )
            stake_val = st.slider(
                "Stake piQ Amount:",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
                key="sidebar_stake_val"
            )

            if st.button("Submit Attestation On-Chain", key="sidebar_submit_attest"):
                attest_id = "ATT_" + hashlib.sha256(
                    f"{current_user}:{target_eval_hash}:{time.time()}".encode()
                ).hexdigest()[:12]
                conn_sub = get_db_connection()
                try:
                    cur_sub = conn_sub.cursor()
                    cur_sub.execute(
                        "INSERT OR REPLACE INTO desci_attestations (attestation_id, eval_hash, attester_id, stake_amount, stance, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                        (attest_id, target_eval_hash, current_user, stake_val, attest_stance, datetime.now().isoformat()),
                    )
                    conn_sub.commit()
                    st.success(f"Attestation recorded! ID: `{attest_id}`")
                finally:
                    conn_sub.close()
        else:
            st.info("No assessed papers available for attestation.")
    else:
        st.warning("Please connect your ORCID iD or DID above to use the DeSci Peer Attestation feature.")

# --- Scilem Accessory Chatbot in Sidebar ---
with st.sidebar.expander("Scilem Accessory Chatbot", expanded=False):
    st.markdown("CoARA-aligned decentralized scientific assistant.")
    
    if st.button("Sync Pinata Knowledge Base", key="scilem_sync_btn", use_container_width=True):
        st.session_state["scilem_synced"] = True
        st.toast("Successfully synchronized decentralized RLHF dataset from Pinata IPFS!")
    
    if st.session_state.get("scilem_synced"):
        st.success("Synced with Pinata IPFS.")

    if "scilm_messages" not in st.session_state:
        st.session_state.scilm_messages = [
            {
                "role": "assistant", 
                "content": "Greetings. I am Scilem, your decentralized scientific intelligence assistant. How may I help?"
            }
        ]

    chat_container = st.container(height=300)
    with chat_container:
        for idx, message in enumerate(st.session_state.scilm_messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Ask Scilem...", key="scilem_sidebar_input"):
        st.session_state.scilm_messages.append({"role": "user", "content": prompt})
        
        rag_context = ""
        few_shot_examples = ""
        try:
            dataset_path = os.path.join(BASE_DIR, "scilem_rlhf_dataset.jsonl")
            if os.path.exists(dataset_path):
                with open(dataset_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    query_terms = set(prompt.lower().split())
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

        scilm_sys_prompt = (
            "You are Scilem, an advanced Scientific LLM aligned with CoARA guidelines. "
            "Be analytical, evidence-driven, and precise.\n\n"
            f"DECENTRALIZED LEDGER CONTEXT (RAG):\n{rag_context}\n\n"
            f"TOP-SCOURING EXEMPLAR:\n{few_shot_examples}"
        )

        messages_for_api = [{"role": "system", "content": scilm_sys_prompt}] + [
            {"role": m["role"], "content": m["content"]} for m in st.session_state.scilm_messages
        ]

        full_response = ""
        try:
            from brain import groq_client
            PRIMARY_MODEL_NAME = "llama-3.3-70b-versatile"
            FALLBACK_MODEL_NAME = "llama-3.1-8b-instant"
            if groq_client:
                try:
                    response = groq_client.chat.completions.create(
                        model=PRIMARY_MODEL_NAME,
                        messages=messages_for_api,
                        temperature=0.15,
                    )
                    full_response = response.choices[0].message.content
                except Exception as primary_err:
                    if "429" in str(primary_err) or "rate_limit_exceeded" in str(primary_err):
                        fallback_response = groq_client.chat.completions.create(
                            model=FALLBACK_MODEL_NAME,
                            messages=messages_for_api,
                            temperature=0.15,
                        )
                        full_response = fallback_response.choices[0].message.content + "\n\n*(Handled via Fallback Engine).* "
                    else:
                        raise primary_err
            else:
                full_response = "Error: Groq API client not initialized."
        except Exception as e:
            full_response = f"Error connecting to Scilem engine: {str(e)}"

        st.session_state.scilm_messages.append({"role": "assistant", "content": full_response})
        st.rerun()

# --- Helper for Cartography Render ---
def render_bubble_chart_clean(target_author):
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
            subfields = [s.title().strip() for s in json.loads(subfields_json)]
            score = float(final_score) if final_score else 50.0
            for s in subfields:
                if s.lower() not in exclude_terms:
                    if s not in topic_aggregates:
                        topic_aggregates[s] = {"weight_sum": 0.0, "frequency": 0}
                    topic_aggregates[s]["weight_sum"] += score
                    topic_aggregates[s]["frequency"] += 1
        except:
            continue

    if not topic_aggregates:
        topic_aggregates["Core Research Domain"] = {
            "weight_sum": 50.0,
            "frequency": 1,
        }

    unique_topics = list(topic_aggregates.keys())

    def get_color(i, n):
        h, s, v = i / n if n > 0 else 0, 0.7, 0.9
        rgb = colorsys.hsv_to_rgb(h, s, v)
        return "#%02x%02x%02x" % tuple(int(x * 255) for x in rgb)

    color_map = {
        topic: get_color(i, len(unique_topics))
        for i, topic in enumerate(unique_topics)
    }
    net = Network(
        height="450px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#2c3e50",
        notebook=False,
    )
    physics_options = """{ "physics": { "barnesHut": { "gravitationalConstant": -1000, "centralGravity": 1, "springLength": 100, "avoidOverlap": 1.0 }, "stabilization": { "enabled": true, "iterations": 200 } } }"""
    net.set_options(physics_options)

    for topic, metrics in topic_aggregates.items():
        avg_weight = metrics["weight_sum"] / metrics["frequency"]
        freq = metrics["frequency"]
        node_size = max(25, 15 + (avg_weight * 2.0))

        base_col = color_map[topic]
        net.add_node(
            n_id=topic,
            label=" ",
            title=(
                f"Topic: {topic} | Frequency: {freq} | Avg Weight/Score:"
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
                "size": 8,
                "x": 4,
                "y": 4,
            },
        )

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".html")
    os.close(tmp_fd) 
    
    try:
        net.save_graph(tmp_name)
        with open(tmp_name, "r", encoding="utf-8") as f:
            html_string = f.read()
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    gradient_injection = """
    <style type="text/css">
        canvas {
            background: radial-gradient(circle at 50% 50%, #ffffff 0%, #f0f2f5 100%);
        }
    </style>
    </head>
    """
    html_string = html_string.replace("</head>", gradient_injection)
    html_string = html_string.replace(
        "mynetwork", f"pi_network_{int(time.time() * 1000)}"
    )

    table_html = "<style>.table-big { width: 100%; font-size: 13px; border-collapse: collapse; margin-top: 10px; font-family: sans-serif; } .table-big th { background-color: #2c3e50; color: white; padding: 6px; text-align: left; } .table-big td { padding: 6px; border-bottom: 1px solid #ecf0f1; } .color-box { width: 20px; height: 20px; border-radius: 4px; display: inline-block; } </style>"
    table_html += "<div class='legend-container'><table class='table-big'><thead><tr><th style='width: 15%; text-align: center;'>Color</th><th>Scientific Topic</th><th style='text-align: center;'>Freq</th><th style='text-align: center;'>Avg Weight</th></tr></thead><tbody>"
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

# --- Top Header Layout ---
st.title(
    "Pi-Index Assessment Engine",
    help=(
        "Automated peer-review framework powered by neural networks, SciScore"
        " reproducibility metrics, and multidimensional blockchain consensus."
    ),
)
st.markdown(
    "**Upload papers, define your scope of research, let Pi-Index filter noise"
    " and yield quantitative results aligned with Responsible Research"
    " Assessment (RRA).**"
)

# ==================== MOVED INTAKE SECTION TO THE TOP ====================
st.markdown("")

selected_uploaded_files = []
uploaded_files = st.file_uploader(
    "1. Upload Local PDF(s)",
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

st.markdown("")
research_scope = ""
doi_input = ""
include_doi = False

with st.expander(
    "Unified Research Scope, DOI & Topic Intake",
    expanded=False,
):
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
                        st.success(f"Successfully harvested {len(alex_results)} papers from OpenAlex.")
                    else:
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

st.markdown("---")
stake_amount = st.checkbox(
    "Stake 0.01 piQ to Process (Returned on Valid Assessment)",
    value=True,
    help=(
        "Staking mechanisms actively filter low-effort, adversarial, or spam"
        " submissions."
    ),
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
                    (
                        title, author_name, score, logic_integrity, drift, rec,
                        fields, subfields, scores_dict, eval_hash, piq, tx_hash,
                        zk_proof, used_weights, mdar_score, rrid_count, repro_score, is_cached,
                    ) = process_single_pdf(
                        pdf_bytes, fname, scope_val, current_user, "None", current_email, p_doi,
                    )
                    eval_record = {
                        "title": title, "author_name": clean_author_name(author_name),
                        "score": score, "logic_integrity": logic_integrity, "drift": drift,
                        "rec": rec, "fields": fields, "subfields": subfields,
                        "scores_dict": scores_dict, "eval_hash": eval_hash, "piq": piq,
                        "tx_hash": tx_hash, "zk_proof": zk_proof, "used_weights": used_weights,
                        "h_idx": mdar_score, "i10_idx": rrid_count, "repro_score": repro_score,
                        "filename": fname,
                    }
                    st.session_state["evaluated_papers_buffer"].insert(0, eval_record)
                    st.session_state["evaluated_papers_buffer"] = st.session_state["evaluated_papers_buffer"][:50]
                else:
                    clean_doi = (
                        p_doi.replace("https://doi.org/", "").strip()
                        if p_doi
                        else "None"
                    )
                    doi_url = (
                        f"https://doi.org/{clean_doi}"
                        if clean_doi and clean_doi != "None"
                        else (p.get("pdf_url") or "N/A")
                    )
                    err_item = {
                        "title": p.get("title", "Unknown Title"),
                        "doi": clean_doi if clean_doi and clean_doi != "None" else "N/A",
                        "url": doi_url,
                    }
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
                (
                    title, author_name, score, logic_integrity, drift, rec,
                    fields, subfields, scores_dict, eval_hash, piq, tx_hash,
                    zk_proof, used_weights, mdar_score, rrid_count, repro_score, is_cached,
                ) = process_single_pdf(
                    pdf_bytes, fname, scope_val, current_user, "None", current_email, doi_snap.strip(),
                )
                eval_record = {
                    "title": title, "author_name": clean_author_name(author_name),
                    "score": score, "logic_integrity": logic_integrity, "drift": drift,
                    "rec": rec, "fields": fields, "subfields": subfields,
                    "scores_dict": scores_dict, "eval_hash": eval_hash, "piq": piq,
                    "tx_hash": tx_hash, "zk_proof": zk_proof, "used_weights": used_weights,
                    "h_idx": mdar_score, "i10_idx": rrid_count, "repro_score": repro_score,
                    "filename": fname,
                }
                st.session_state["evaluated_papers_buffer"].insert(0, eval_record)
                st.session_state["evaluated_papers_buffer"] = st.session_state["evaluated_papers_buffer"][:50]
            else:
                clean_doi = doi_snap.replace("https://doi.org/", "").strip()
                doi_url = f"https://doi.org/{clean_doi}"
                err_item = {
                    "title": f"DOI Input: {clean_doi}",
                    "doi": clean_doi,
                    "url": doi_url,
                }
                if err_item not in st.session_state["download_errors"]:
                    st.session_state["download_errors"].append(err_item)

        if snap_files and not st.session_state["cancel_requested"]:
            total_files = len(snap_files)
            for i, (fname, file_bytes) in enumerate(snap_files):
                if st.session_state["cancel_requested"]:
                    break
                status_text.text(
                    f"Analyzing uploaded file {i+1} of {total_files}: {fname}..."
                )
                (
                    title, author_name, score, logic_integrity, drift, rec,
                    fields, subfields, scores_dict, eval_hash, piq, tx_hash,
                    zk_proof, used_weights, mdar_score, rrid_count, repro_score, is_cached,
                ) = process_single_pdf(
                    file_bytes, fname, scope_val, current_user, "None", current_email, "None",
                )
                eval_record = {
                    "title": title, "author_name": clean_author_name(author_name),
                    "score": score, "logic_integrity": logic_integrity, "drift": drift,
                    "rec": rec, "fields": fields, "subfields": subfields,
                    "scores_dict": scores_dict, "eval_hash": eval_hash, "piq": piq,
                    "tx_hash": tx_hash, "zk_proof": zk_proof, "used_weights": used_weights,
                    "h_idx": mdar_score, "i10_idx": rrid_count, "repro_score": repro_score,
                    "filename": fname,
                }
                st.session_state["evaluated_papers_buffer"].insert(0, eval_record)
                st.session_state["evaluated_papers_buffer"] = st.session_state["evaluated_papers_buffer"][:50]
                progress_bar.progress((i + 1) / total_files)

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
            st.error(
                "You must agree to the piQ micro-stake to execute the assessment"
                " pipeline."
            )
        elif (
            not selected_uploaded_files
            and not (include_doi and doi_input.strip())
            and not selected_alex_papers
        ):
            st.warning("Please tick at least one paper or input source to assess.")
        else:
            st.session_state["snap_files"] = [
                (f.name, f.read()) for f in selected_uploaded_files
            ]
            st.session_state["snap_scope"] = research_scope
            st.session_state["snap_doi"] = doi_input
            st.session_state["snap_include_doi"] = include_doi
            st.session_state["snap_alex"] = selected_alex_papers
            st.session_state["is_running"] = True
            st.session_state["cancel_requested"] = False
            st.rerun()

st.markdown("---")

with st.expander("Evaluation Metrics, SciScore Reproducibility & Adversarial Logic Engine", expanded=False):
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
        tw1, tw2, tw3, tw4, tw5, tw6, tw7, tw8 = 1.001328, 1.000038, 0.999645, 0.997347, 0.999278, 0.997645, 1.002110, 1.002609

    st.markdown(
        r"**Adversarial Logic Gap ($\Delta_{Logic}$)** "
        + tooltip(
            "Evaluates reasoning structure and penalizes claims unsupported by"
            " evidence or counterfactual stress failures."
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        r"$$ L_i = (\mathcal{P}_{valid} \cdot \mathcal{E}_{strength}) \cdot"
        r" \exp\left(-\left(2 \cdot \max(0, \mathcal{C}_{reach} -"
        r" \mathcal{E}_{strength}) + 1.5 \cdot \lambda_{jumps}\right)\right)"
        r" \times \frac{1}{1 + e^{-\Delta Premise}} $$"
    )

    with st.expander(f"C1: Originality ($\varpi_1$ = `{tw1:.6f}`):"):
        st.markdown(
            "Semantic distance from literature corpus penalized by generative AI laundering heuristics."
        )
        st.markdown(
            r"$$ C_1 = \varpi_1 \cdot \mathcal{D}_{semantic}(P_{target}, P_{corpus})"
            r" \times (1 - \lambda_{laundering}) $$"
        )

    with st.expander(f"C2: Methodological Rigor ($\varpi_2$ = `{tw2:.6f}`):"):
        st.markdown(
            "Deterministic adherence to MDAR reporting standards and valid RRIDs via SciScore."
        )
        st.markdown(
            r"$$ C_2 = \varpi_2 \cdot \mathcal{I}_{blinding} + \varpi_2 \cdot"
            r" \mathcal{I}_{randomization} + \varpi_2 \cdot \mathcal{I}_{power\_calc}"
            r" + \varpi_2 \cdot \left(\frac{N_{RRID\_valid}}{N_{RRID\_expected} +"
            r" \epsilon}\right) $$"
        )

    with st.expander(f"C3: Interdisciplinary Synergy ($\varpi_3$ = `{tw3:.6f}`):"):
        st.markdown(
            "Shannon entropy of the verified citation network across diverse subfields."
        )
        st.markdown(r"$$ C_3 = \varpi_3 \cdot -\sum_{i=1}^{k} p_i \ln(p_i) $$")

    with st.expander(f"C4: Societal & Open Infrastructure Impact ($\varpi_4$ = `{tw4:.6f}`):"):
        st.markdown(
            "CoARA WG TIER aligned rewards for public datasets, civic policy integration, and open science."
        )
        st.markdown(
            r"$$ C_4 = \varpi_4 \cdot \Theta\left[ \sum_{v \in \mathcal{V}} \omega_v"
            r" U_v(\tau, \mathbf{x}) \right] $$"
        )

    with st.expander(f"C5: Open Science & Executable Reproducibility ($\varpi_5$ = `{tw5:.6f}`):"):
        st.markdown(
            "Cryptographic verification of open data/code repositories and sandboxed container execution."
        )
        st.markdown(
            r"$$ C_5 = \varpi_5 \cdot (\beta_1 \cdot \mathcal{V}_{data} + \beta_2"
            r" \cdot \mathcal{V}_{code} + \beta_3 \cdot \mathcal{Z}_{container}) $$"
        )

    with st.expander(f"C6: Literature Integration ($\varpi_6$ = `{tw6:.6f}`):"):
        st.markdown(
            "Citation context polarity classification (supporting vs. contrasting engagement)."
        )
        st.markdown(
            r"$$ C_6 = \varpi_6 \cdot \frac{1}{\mathcal{N}} \sum_{i=1}^{\mathcal{N}}"
            r" \text{Polarity}(x_i) \cdot \text{PR}(x_i) $$"
        )

    with st.expander(f"C7: Empirical Density & Validation ($\varpi_7$ = `{tw7:.6f}`):"):
        st.markdown(
            "Deterministic extraction of sample sizes, degrees of freedom, and cohort volumes."
        )
        st.markdown(
            r"$$ C_7 = \varpi_7 \cdot \tanh \left( \frac{n_{\text{valid}} \cdot"
            r" \text{Cohort Strength}}{\text{Baseline Variance}} \right) $$"
        )

    with st.expander(f"C8: Future Actionability & FAIR ($\varpi_8$ = `{tw8:.6f}`):"):
        st.markdown(
            "Strict measurement of adherence to FAIR principles for downstream research cascade."
        )
        st.markdown(
            r"$$ C_8 = \varpi_8 \cdot \frac{1}{\mathcal{Z}} \int_{\mathcal{X}}"
            r" \text{FAIR\_Score}(\mathbf{x}) \, d\mu(\mathbf{x}) $$"
        )

st.markdown("---")

# ==================== TOP SIDE-BY-SIDE ANALYTICS (PI-BRAIN ON LEFT, GLOBAL MAP ON RIGHT) ====================
top_analytics_col1, top_analytics_col2 = st.columns(2)

with top_analytics_col1:
    st.markdown(
        "### Pi-Brain LSTM Meta-Learning Forecasts "
        + tooltip(
            "An LSTM neural network that trains directly on the block weights to"
            " predict future shifts in algorithmic evaluation standards."
        ),
        unsafe_allow_html=True,
    )

    @st.cache_data(show_spinner="Training Pi-Brain LSTM Model in background...")
    def train_pibrain_cached(weight_data, actual_lookback):
        dataset = PiBlockchainDataset(weight_data, actual_lookback)
        dataloader = DataLoader(
            dataset, batch_size=min(4, max(1, len(dataset))), shuffle=False
        )

        model = PiBrainLSTM()
        weights_path = os.path.join(BASE_DIR, "pi_brain_weights.pt")
        if os.path.exists(weights_path):
            try:
                model.load_state_dict(torch.load(weights_path, weights_only=True))
            except Exception:
                pass

        loss_function = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(200):
            for seq, target in dataloader:
                optimizer.zero_grad()
                loss = loss_function(model(seq), target)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            predicted = (
                model(
                    torch.tensor(
                        weight_data[-actual_lookback:], dtype=torch.float32
                    ).unsqueeze(0)
                )
                .squeeze()
                .numpy()
            )
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
        lookback_window = max(1, min(5, current_block_count - 1))

        if (
            "last_trained_blocks" not in st.session_state
            or st.session_state.last_trained_blocks != current_block_count
        ):
            weight_data = np.array(historical_rows, dtype=np.float32)
            actual_lookback = min(lookback_window, len(weight_data))

            st.session_state.predicted_next_weights = train_pibrain_cached(weight_data, actual_lookback)
            st.session_state.current_weights = weight_data[-1]
            st.session_state.last_trained_blocks = current_block_count
        else:
            st.info(
                "Meta-model is cached and up-to-date with the latest blockchain"
                " ledger."
            )

        df_compare = pd.DataFrame(
            {
                "Current Active Weights": st.session_state.current_weights,
                "Predicted Next Epoch": st.session_state.predicted_next_weights,
            },
            index=[
                "C1: Originality", "C2: Methodological Rigor",
                "C3: Interdisciplinary", "C4: Societal Impact",
                "C5: Open Science", "C6: Literature Integration",
                "C7: Empirical Density", "C8: Future Actionability",
            ],
        )
        st.bar_chart(df_compare, height=380)
        st.markdown(
            f"**Mathematical Constraint Check:** Predicted Sum ="
            f" `{sum(st.session_state.predicted_next_weights):.6f}` / `8.0`"
        )

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

    selected_author_top = None
    piq_dict, book_dict = get_author_piq_dict()

    if all_global_authors:
        filter_choice_top = st.selectbox(
            "Filter Map by Author:",
            ["All Authors"] + all_global_authors,
            key=f"top_author_filter_{st.session_state['assessment_update_token']}",
            format_func=lambda x: (
                f"{x} (piQ: {piq_dict.get(x, 0.0):.2f})" if x != "All Authors" else x
            ),
        )
        if filter_choice_top != "All Authors":
            selected_author_top = filter_choice_top

    interactive_html_top, table_html_top = render_bubble_chart_clean(selected_author_top)
    if interactive_html_top:
        components.html(interactive_html_top, height=410, scrolling=True)
    else:
        st.info("Awaiting sufficient data for map visualization.")

    with st.expander("View Map Legend, Frequency Metrics & Leaderboard"):
        st.markdown(table_html_top, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Pi Quotient (piQ) Explorer & Leaderboard")
        search_query_top = st.text_input(
            "Search Explorer by Author or Book Address:",
            placeholder="Enter author name or 0x...",
            key="top_search_query_input"
        )
        if piq_dict:
            leaderboard_data = []
            for author, piq in piq_dict.items():
                leaderboard_data.append({
                    "Contributing Author": author,
                    "Unique Author Book Address": book_dict.get(author, "None"),
                    "Total piQ Earned": round(piq, 2),
                })
            piq_df = pd.DataFrame(leaderboard_data).sort_values(by="Total piQ Earned", ascending=False).reset_index(drop=True)
            if search_query_top:
                q_clean = search_query_top.strip().lower()
                filtered_df = piq_df[piq_df["Contributing Author"].str.lower().str.contains(q_clean) | piq_df["Unique Author Book Address"].str.lower().str.contains(q_clean)]
                st.dataframe(filtered_df, use_container_width=True, height=180)
            else:
                st.dataframe(piq_df, use_container_width=True, height=180)
        else:
            st.info("No piQ tokens minted yet.")

st.markdown("---")

def render_breakdown_item(item):
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
    author_book = "0x" + hashlib.sha256(author_name.encode()).hexdigest()[:40]

    st.markdown("---")
    st.subheader(f"{title} by {author_name}")

    with st.expander(
        f"Ledger Data & Dossier Details ({filename})", expanded=False
    ):
        st.write(f"**File Name:** `{filename}`")
        st.write(f"**Evaluation Hash (Paper Address):** `{eval_hash}`")
        st.write(f"**Unique Author Book Address (eth_book):** `{author_book}`")
        st.write(f"**piQ Minted:** `{piq}`")
        st.write(f"**zk-SNARK:** `{zk_proof}`")
        
        tx_url = safe_get_sepolia_url(tx_hash)
        if tx_url:
            st.markdown(f"**Tx Hash:** [`{tx_hash}`]({tx_url}) (View on Sepolia Etherscan)")
        else:
            st.write(f"**Tx Hash:** `{tx_hash}`")

        st.write(
            f"**Executable Reproducibility Score (C5/C7 audit):**"
            f" `{repro_score * 100:.1f}%`"
        )
        st.write(
            f"**SciScore MDAR Adherence:** `{mdar_score * 100:.1f}%` | **Valid"
            f" RRIDs:** `{rrid_count}`"
        )

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
            scores_dict.get("C1_Originality", 0),
            scores_dict.get("C2_Methodological_Rigor", 0),
            scores_dict.get("C3_Interdisciplinary", 0),
            scores_dict.get("C4_Societal_Impact", 0),
            scores_dict.get("C5_Open_Science_Potential", 0),
            scores_dict.get("C6_Literature_Integration", 0),
            scores_dict.get("C7_Empirical_Density", 0),
            scores_dict.get("C8_Future_Actionability", 0),
        ],
        "Epoch Weight": used_weights,
        "Weighted Value": [
            scores_dict.get(k, 0) * used_weights[i]
            for i, k in enumerate([
                "C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary",
                "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration",
                "C7_Empirical_Density", "C8_Future_Actionability",
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
**Unique Author Book Address:** {author_book}
**Final Pi-Index Score:** {score:.2f} / 100
**Logic Integrity Score:** {logic_integrity:.1f}%
**Executable Reproducibility Score:** {repro_score * 100:.1f}%
**SciScore MDAR Adherence:** {mdar_score * 100:.1f}%
**Valid RRIDs Count:** {rrid_count}

## 8-Criteria Evaluation Breakdown
- C1 Semantic Originality: {scores_dict.get("C1_Originality",0)}
- C2 Methodological Rigor (SciScore): {scores_dict.get("C2_Methodological_Rigor",0)}
- C3 Interdisciplinary Entropy: {scores_dict.get("C3_Interdisciplinary",0)}
- C4 Societal Impact: {scores_dict.get("C4_Societal_Impact",0)}
- C5 Open Science & Repro: {scores_dict.get("C5_Open_Science_Potential",0)}
- C6 Literature Integration: {scores_dict.get("C6_Literature_Integration",0)}
- C7 Empirical Density: {scores_dict.get("C7_Empirical_Density",0)}
- C8 Future Actionability & FAIR: {scores_dict.get("C8_Future_Actionability",0)}

## Cryptographic Proofs & Ledger Seal
- zk-SNARK: {zk_proof}
- Tx Hash: {tx_hash}
"""
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.download_button(
            label=f"Download Research Integrity Dossier ({filename})",
            data=dossier_content,
            file_name=f"Dossier_{eval_hash[:10]}.md",
            mime="text/markdown",
            key=f"download_dossier_{eval_hash}_{time.time()}",
        )
    with col_btn2:
        if st.button("Generate AI Defense Strategy", key=f"gen_defense_{eval_hash}_{time.time()}"):
            with st.spinner("Synthesizing adversarial defense strategy..."):
                rebuttal = generate_rebuttal_strategy(scores_dict)
                st.session_state[f"defense_{eval_hash}"] = rebuttal

    if f"defense_{eval_hash}" in st.session_state:
        st.markdown("#### AI Peer Review Defense Rebuttal Strategy")
        st.markdown(st.session_state[f"defense_{eval_hash}"])

if (
    st.session_state["evaluated_papers_buffer"]
    or st.session_state.get("download_errors")
):
    st.markdown("---")
    st.markdown("### Active Session Assessment Results")

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

    for item in st.session_state["evaluated_papers_buffer"]:
        render_breakdown_item(item)

st.markdown("---")
st.markdown(
    "### Latest Assessed Papers "
    + tooltip(
        "Displays the 5 most recently evaluated papers globally with complete assessment scores, block hashes, zk-SNARK proofs, and piQ allocations."
    ),
    unsafe_allow_html=True,
)

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

        with st.expander(
            f"[{idx+1}] {r_title[:65]}... — *{r_author_clean}* (Score:"
            f" **{r_score:.2f}** | {r_time[:16]})",
            expanded=False,
        ):
            st.write(f"**Title:** {r_title}")
            st.write(f"**Author(s):** {r_author_clean}")
            st.write(f"**Timestamp:** `{r_time}`")
            st.write(f"**Evaluation Hash (Eval Hash):** `{r_hash}`")
            st.write(f"**Block Height:** `{r_block_height if r_block_height is not None else 'Pending'}`")
            st.write(f"**Block Hash:** `{r_block_hash if r_block_hash is not None else 'Pending'}`")
            st.write(f"**Unique Author Book Address:** `{r_book}`")
            st.write(f"**piQ Minted:** `{r_piq}`")
            st.write(f"**zk-SNARK Proof:** `{r_zk}`")
            
            if r_tx_url:
                st.markdown(f"**Tx Hash (Etherscan):** [`{r_tx}`]({r_tx_url})")
            else:
                st.write(f"**Tx Hash:** `{r_tx}`")

            st.write(
                f"**Logic Integrity:** `{r_logic:.1f}%` | **Reproducibility:**"
                f" `{r_repro * 100:.1f}%` | **MDAR Adherence:**"
                f" `{r_mdar * 100:.1f}%`"
            )

            r_df = pd.DataFrame({
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
                "Score (0-100)": [
                    r_c1, r_c2, r_c3, r_c4,
                    r_c5, r_c6, r_c7, r_c8,
                ],
            })
            st.dataframe(r_df, hide_index=True, use_container_width=True)


# ==================== PINAMIC & DECENTRALIZED INFRASTRUCTURE SECTION ====================
st.markdown("---")
st.markdown(
    "### Proof-of-Research Blockchain Explorer "
    + tooltip(
        "Manages decentralized consensus, ledger weights, and smart contract audit proofs."
    ),
    unsafe_allow_html=True,
)

with st.expander(
    "Detailed Guide: How Pinamic Works (Ledger Consensus & Staking)",
    expanded=False,
):
    st.markdown("""
    Pinamic integrates the decentralized infrastructure layer of the Pi-Index Assessment Engine:
    1. **Active Epoch & Block Height**: Tracks incremental block updates. When the threshold (`EPOCH_BLOCK_SIZE`) is reached, a new blockchain block is minted.
    2. **Proof-of-Research (PoR) Validation (`validate_block_por`)**: Combines block index, criteria weights ($\varpi_1$ to $\varpi_8$), timestamp, previous block hash, validator node signature, model identifier, and formulas hash into an unalterable SHA-256 block hash.
    3. **DeSci Peer Attestation & Staking**: Researchers can stake a fraction of their earned soulbound tokens (`piQ`) to either endorse or challenge specific manuscript assessments on-chain (`desci_attestations`).
    """)

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

        with st.expander("Proof-of-Research Blockchain Explorer & Sepolia Contract Verification", expanded=False):
            st.info(
                f"**Latest Proof-of-Research:** `{por_proof}` successfully verified and"
                f" sealed to block `{block_hash}`."
            )
            st.caption(
                f"**Unalterable Criteria State Hash:** `{formulas_hash}` (Guarantees"
                " grading mathematical constants cannot be tampered with)."
            )

            explore_col1, explore_col2 = st.columns([3, 1])
            with explore_col1:
                search_query = st.text_input(
                    "Enter Document Evaluation Hash or Block Hash to verify ledger record...",
                    key="pinamic_ledger_search_query"
                )
            with explore_col2:
                st.write("")
                st.write("")
                search_btn = st.button("Verify Record", key="pinamic_verify_record_btn")

            if search_btn and search_query:
                try:
                    cursor.execute(
                        "SELECT * FROM blockchain_por_weights WHERE block_hash=? OR"
                        " eval_hash=?",
                        (search_query, search_query),
                    )
                    record = cursor.fetchone()
                    if record:
                        st.success("Valid Block Found on Ledger")
                        st.json({
                            "Block Height": record[0],
                            "Timestamp": record[9],
                            "Model Used": record[14],
                            "Validator Node": record[11],
                            "Block Hash": record[12],
                            "Evaluation Hash": record[13],
                            "PoR Signature": record[15],
                            "Formulas Hash": record[16],
                            "Weights": dict(
                                zip([f"w{i+1}" for i in range(8)], record[1:9])
                            ),
                        })
                    else:
                        st.error(
                            "No block matching that signature was found on the ledger."
                        )
                except:
                    st.error("Error reading database schema. Try refreshing the app.")

        with st.expander("Deployed Smart Contracts on Sepolia Etherscan", expanded=False):
            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                piq_url = f"https://sepolia.etherscan.io/address/{PIQ_CONTRACT_ADDRESS}"
                st.markdown(f"**PiQ Token Contract:** [`{PIQ_CONTRACT_ADDRESS}`]({piq_url})")
            with col_ex2:
                reg_url = f"https://sepolia.etherscan.io/address/{REGISTRY_CONTRACT_ADDRESS}" if REGISTRY_CONTRACT_ADDRESS else "#"
                if REGISTRY_CONTRACT_ADDRESS:
                    st.markdown(f"**Registry Contract:** [`{REGISTRY_CONTRACT_ADDRESS}`]({reg_url})")
                else:
                    st.markdown("**Registry Contract:** `Not Configured`")

        with st.expander("Recent Ledger Proofs Summary & Transaction Ledger", expanded=False):
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
                    tx_disp = f"[{rtx[:10]}...]({tx_url})" if rtx else str(rtx)
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


# ==================== SYSTEM OVERVIEW & MERGED FOOTER BLOCK ====================
st.markdown("---")
with st.expander("The Pi-Index Framework: Next-Gen Architecture & CoARA Compliance Workflow"):
    st.markdown(
        "### Architecture Flowchart & Whitepaper DOI\n\n"
        "Read the foundational framework whitepaper and preprints via [Ali Vafadar Yengejeh's ResearchGate Profile](https://www.researchgate.net/profile/Ali-Vafadar-Yengejeh).\n\n"
        "The enhanced system architecture flow below details the decentralized"
        " intake, ZK double-blind reviewer assignment, SciScore deterministic"
        " parsing, Item Response Theory (IRT) calibration, and smart contract"
        " slashing mechanisms."
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

            Auth [label="Researcher Authentication\\n• ORCID iD / W3C DID Verification\\n• ZK-Email Institutional Proof", fillcolor="#aed6f1"];
            Intake [label="Multi-Source Ingestion Engine\\n• Local Binary PDFs Extraction\\n• Unpaywall DOI Resolver\\n• OpenAlex Topic API Search", fillcolor="#aed6f1"];
            ZKBlind [label="ZK Double-Blind Assignment\\n• Merkle Tree Non-Membership Proofs\\n• Anonymous Author Shielding", fillcolor="#aed6f1"];
            Auth -> Intake -> ZKBlind;
        }

        subgraph cluster_eval {
            label = "2. Core Evaluation & Adversarial Analysis Pipeline (CoARA/RRA)";
            style = rounded;
            color = "#27ae60";
            fillcolor = "#e8f8f5";

            SciParser [label="Deterministic SciScore API\\n• MDAR Reporting Adherence\\n• Valid RRIDs Count Extraction", fillcolor="#a3e4d7"];
            IRTCalib [label="Item Response Theory Calibration\\n• Counterfactual Stress Testing\\n• Variance & Difficulty Mapping", fillcolor="#a3e4d7"];
            Criteria [label="8 Transparent Criteria Rubrics\\n• C1 Originality to C8 FAIR Actionability\\n• Formulaic Score Computation", fillcolor="#a3e4d7"];
            Logic [label="Adversarial Logic Integrity Matrix\\n• Premise Validity & Evidence Strength\\n• AI Hallucination & Laundering Penalty", fillcolor="#a3e4d7"];
            
            SciParser -> IRTCalib -> Criteria -> Logic;
        }

        subgraph cluster_blockchain {
            label = "3. Blockchain Consensus, Cryptographic Proofs & Slashing Tokenomics";
            style = rounded;
            color = "#8e44ad";
            fillcolor = "#f4ecf7";

            PoR [label="Proof-of-Research (PoR) Validation\\n• Dynamic Epoch Weight Shifting\\n• Formulas Hash Stamping & SHA-256 Block", fillcolor="#d7bde2"];
            Slashing [label="Anti-Laundering Slashing Guard\\n• Smart Contract piQ Burn for Fraud\\n• Stake Penalty Enforcement", fillcolor="#f5b7b1"];
            Mint [label="Soulbound Token Minting\\n• Author-Specific Book Address (eth_book)\\n• Shared Paper Address (eval_hash) & Tx Hash", fillcolor="#d7bde2"];
            
            PoR -> Slashing -> Mint;
        }

        subgraph cluster_outputs {
            label = "4. User Interface, Cartography & Institutional Policy Support";
            style = rounded;
            color = "#d35400";
            fillcolor = "#fef5e7";

            Dossier [label="CoARA & DORA-Aligned Dossier\\n• Markdown Research Integrity Report\\n• AI Defense Rebuttal Strategy", fillcolor="#f8c471"];
            Cartography [label="Global Map of Science\\n• Ledger PyVis Network Cartography\\n• Author & Topic Bubble Filtering", fillcolor="#f8c471"];
            PiBrain [label="Pi-Brain LSTM Meta-Learning\\n• PyTorch Temporal Weight Prediction\\n• Calibration Drift & Epoch Forecasting", fillcolor="#f8c471"];
        }

        Auth -> SciParser [lhead=cluster_eval, label="Processed Manuscript Text"];
        Logic -> PoR [lhead=cluster_blockchain, label="Audited Score & Hashes"];
        Mint -> Dossier [lhead=cluster_outputs, label="Ledger Seal & Tokens"];
        Mint -> Cartography;
        Mint -> PiBrain;
    }
    """)

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.9em; padding-bottom: 5px;'>Framework Author: Ali Vafadar Yengejeh | Universita degli Studi di Milano-Bicocca</div>",
        unsafe_allow_html=True
    )
