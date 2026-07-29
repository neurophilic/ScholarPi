import os
import sqlite3
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

from config import BASE_DIR
from database import get_db_connection
from integrations import (
    fetch_doi_metadata, download_pdf_from_url, 
    fetch_semantic_scholar_pdf, fetch_core_text_by_doi, 
    create_virtual_pdf_from_text
)

st.set_page_config(page_title="Pi-Index Intake", layout="wide")

# Custom UI
custom_ui_code = """
<style>
h1, h2, h3, h4 { color: #0f172a !important; font-family: -apple-system, sans-serif !important; font-weight: 600 !important; }
button[kind="primary"] { background-color: #000080 !important; border-color: #000080 !important; color: #ffffff !important; }
</style>
"""
components.html(custom_ui_code, height=0, width=0)

# Initialize Shared Queue
def init_queue():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT,
            source_val TEXT,
            file_name TEXT,
            raw_bytes BLOB,
            status TEXT,
            timestamp DATETIME
        )
    """)
    conn.commit()
    conn.close()

init_queue()

st.title("📄 Manuscript Intake & Preprocessing Engine")
st.markdown("Upload local files or resolve open-access DOIs. Manuscripts are sent to the Pidyne Brain queue.")

intake_tab_local, intake_tab_doi = st.tabs(["📄 Local Upload", "🔗 DOI Lookup"])

with intake_tab_local:
    uploaded_files = st.file_uploader("Upload Local PDF(s)", type=["pdf"], accept_multiple_files=True)
    if st.button("Queue Uploaded Papers", type="primary") and uploaded_files:
        conn = get_db_connection()
        cur = conn.cursor()
        for f in uploaded_files:
            file_bytes = f.getvalue()
            cur.execute(
                "INSERT INTO ingestion_queue (source_type, source_val, file_name, raw_bytes, status, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                ("local", "upload", f.name, file_bytes, "pending", datetime.now().isoformat())
            )
        conn.commit()
        conn.close()
        st.success(f"Successfully queued {len(uploaded_files)} manuscript(s) for Pidyne AI evaluation!")

with intake_tab_doi:
    doi_input = st.text_input("Enter a DOI", placeholder="10.1000/xyz123 or https://doi.org/10.1000/xyz123")
    if st.button("Resolve & Queue DOI", type="primary") and doi_input.strip():
        with st.status(f"Resolving DOI: {doi_input}..."):
            metadata = fetch_doi_metadata(doi_input)
            fname = f"DOI_{doi_input.replace('/', '_')}.pdf"
            pdf_bytes = None
            
            if metadata and metadata.get("pdf_url"):
                pdf_bytes = download_pdf_from_url(metadata["pdf_url"])
            if not pdf_bytes:
                s2_url = fetch_semantic_scholar_pdf(doi_input)
                if s2_url:
                    pdf_bytes = download_pdf_from_url(s2_url)
            if not pdf_bytes:
                core_text = fetch_core_text_by_doi(doi_input)
                if core_text:
                    pdf_bytes = create_virtual_pdf_from_text(core_text, title="DOI Target Text")

            if pdf_bytes:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO ingestion_queue (source_type, source_val, file_name, raw_bytes, status, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                    ("doi", doi_input, fname, pdf_bytes, "pending", datetime.now().isoformat())
                )
                conn.commit()
                conn.close()
                st.success("DOI resolved and securely added to the processing queue!")
            else:
                st.error("Publisher access blocks direct binary extraction for this standalone DOI.")