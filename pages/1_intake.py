
import streamlit as st
from database import get_db_connection
from integrations import fetch_doi_metadata, download_pdf_from_url, fetch_semantic_scholar_pdf, fetch_core_text_by_doi, create_virtual_pdf_from_text
from datetime import datetime

st.title("📄 Manuscript Intake & Preprocessing")
st.markdown("Upload local files or resolve open-access DOIs. Manuscripts are sent to the Pidyne queue.")

tab_local, tab_doi = st.tabs(["📄 Local Upload", "🔗 DOI Lookup"])
with tab_local:
    uploaded_files = st.file_uploader("Upload Local PDF(s)", type=["pdf"], accept_multiple_files=True)
    if st.button("Queue Uploaded Papers", type="primary") and uploaded_files:
        conn = get_db_connection()
        cur = conn.cursor()
        for f in uploaded_files:
            cur.execute("INSERT INTO ingestion_queue (source_type, source_val, file_name, raw_bytes, status, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                ("local", "upload", f.name, f.getvalue(), "pending", datetime.now().isoformat()))
        conn.commit()
        conn.close()
        st.success(f"Successfully queued {len(uploaded_files)} manuscript(s)!")

with tab_doi:
    doi_input = st.text_input("Enter a DOI", placeholder="10.1000/xyz123")
    if st.button("Resolve & Queue DOI", type="primary") and doi_input.strip():
        with st.status(f"Resolving {doi_input}..."):
            metadata = fetch_doi_metadata(doi_input)
            fname = f"DOI_{doi_input.replace('/', '_')}.pdf"
            pdf_bytes = metadata.get("pdf_url") and download_pdf_from_url(metadata["pdf_url"])
            if not pdf_bytes: pdf_bytes = fetch_semantic_scholar_pdf(doi_input) and download_pdf_from_url(fetch_semantic_scholar_pdf(doi_input))
            if not pdf_bytes: pdf_bytes = fetch_core_text_by_doi(doi_input) and create_virtual_pdf_from_text(fetch_core_text_by_doi(doi_input))

            if pdf_bytes:
                conn = get_db_connection()
                conn.execute("INSERT INTO ingestion_queue (source_type, source_val, file_name, raw_bytes, status, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                    ("doi", doi_input, fname, pdf_bytes, "pending", datetime.now().isoformat()))
                conn.commit()
                conn.close()
                st.success("DOI resolved and queued!")
            else:
                st.error("Publisher blocks direct binary extraction.")
