import os
import time
import traceback
import streamlit as st
from database import get_db_connection
from integrations import fetch_doi_metadata, fetch_semantic_scholar_pdf, download_pdf_from_url, fetch_core_text_by_doi, create_virtual_pdf_from_text, clean_author_name
from brain import process_single_pdf
from shared_ui import add_log, safe_float, more_details_dialog, defense_strategy_dialog

def preprocess_pdf_layout(pdf_bytes, fname):
    return pdf_bytes

@st.dialog("The Pi-Index Framework Workflow", width="large")
def framework_workflow_dialog():
    st.markdown("Pi-Index filters noise and yields quantitative results strictly aligned with **Responsible Research Assessment (RRA)** and **CoARA** guidelines...")

conn_cnt = get_db_connection()
total_analyzed_count = conn_cnt.execute("SELECT COUNT(*) FROM papers_assessment").fetchone()[0]
conn_cnt.close()

top_title_col, top_badge_col = st.columns([4, 2], vertical_alignment="center")
top_title_col.markdown("<h1 style='margin-bottom:0;'>📄 Manuscript Intake & Processing</h1>", unsafe_allow_html=True)
top_badge_col.markdown(f"<div style='float: right; background-color: #0f172a; color: white; padding: 6px 16px; border-radius: 20px; font-weight: 600;'>Total Analyzed: <span style='color: #60a5fa;'>{total_analyzed_count}</span></div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

has_web3 = bool(st.session_state.web3_wallet)
current_user = st.session_state.orcid_profile if st.session_state.orcid_profile else (st.session_state.web3_wallet if has_web3 else "Anonymous")
valid_book_address = st.session_state.web3_wallet if has_web3 else "0x0000000000000000000000000000000000000000"

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

    intake_tab_local, intake_tab_doi = st.tabs(["📄 Local Upload", "🔗 DOI Lookup"])
    selected_uploaded_files = []
    
    with intake_tab_local:
        uploaded_files = st.file_uploader("Upload Local PDF(s)", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            for f in uploaded_files:
                if st.checkbox(f"Local File: {f.name}", value=True): selected_uploaded_files.append(f)

    with intake_tab_doi:
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
                # Execution of DOIs
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

                # Execution of Local Files
                snap_files = st.session_state.get("snap_files", [])
                for i, (fname, fpath) in enumerate(snap_files):
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
            if free_evals_used >= 1 and (not has_web3 or not stake_amount):
                st.error("Free trial limit reached. Connect Web3 and stake 0.1 piQ.")
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

# --- Results Buffer Rendering ---
if st.session_state["evaluated_papers_buffer"] or st.session_state.get("download_errors"):
    st.markdown("### Assessment Results")
    for err_idx, err in enumerate(st.session_state.get("download_errors", [])):
        st.warning(f"**Failed DOI:** `{err['doi']}` (Publisher restricts direct access)")

    for item_idx, item in enumerate(st.session_state["evaluated_papers_buffer"]):
        with st.container(border=True):
            c_info, c_actions = st.columns([6, 4])
            c_info.markdown(f"**{item['title']}** — *{item['author_name']}*\n\n**Score: {item['score']:.2f} | piQ: {item['piq']}**")
            ac1, ac2, ac3 = c_actions.columns([3, 3, 1])
            if ac1.button("More Details", key=f"det_{item['eval_hash']}_{item_idx}", use_container_width=True): more_details_dialog(item)
            if ac2.button("Suggest Defense", key=f"strat_{item['eval_hash']}_{item_idx}", use_container_width=True): defense_strategy_dialog(item['scores_dict'])
            if ac3.button("❌", key=f"del_{item['eval_hash']}_{item_idx}"):
                st.session_state["evaluated_papers_buffer"].pop(item_idx)
                st.rerun()

st.markdown("---")
if st.button("The Pi-Index Framework Workflow", use_container_width=True):
    framework_workflow_dialog()
