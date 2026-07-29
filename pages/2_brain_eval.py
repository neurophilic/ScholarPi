import streamlit as st
import time
from database import get_db_connection
from brain import process_single_pdf, evaluate_scilem_analysis_report

st.title("🧠 Pidyne Brain & Adversarial LLM Jury")

if "scilem_messages" not in st.session_state:
    st.session_state.scilem_messages = [{"role": "assistant", "content": "**Welcome! I am Scilem.** Ask any research question."}]

with st.sidebar.expander("🧠 Scilem Assistant", expanded=True):
    for message in st.session_state.scilem_messages:
        st.chat_message(message["role"]).markdown(message["content"])
    prompt = st.text_input("Ask Scilem...", key="scilem_input")
    if st.button("Send") and prompt:
        st.session_state.scilem_messages.append({"role": "user", "content": prompt})
        st.session_state.scilem_messages.append({"role": "assistant", "content": evaluate_scilem_analysis_report(prompt)})
        st.rerun()

conn = get_db_connection()
cur = conn.cursor()
cur.execute("SELECT id, file_name, source_type, source_val, timestamp FROM ingestion_queue WHERE status='pending'")
pending = cur.fetchall()

if not pending:
    st.info("No pending manuscripts in queue. Submit via the Intake Engine.")
else:
    for pid, fname, src_type, src_val, ts in pending:
        with st.container(border=True):
            cols = st.columns([4, 1])
            cols[0].markdown(f"**{fname}** (Source: {src_type} | Queued: {ts})")
            if cols[1].button("Assess", key=f"run_{pid}", type="primary"):
                cur.execute("SELECT raw_bytes FROM ingestion_queue WHERE id=?", (pid,))
                raw_bytes = cur.fetchone()[0]
                with st.status(f"Evaluating {fname}...", expanded=True) as status:
                    res = process_single_pdf(raw_bytes, fname, "", "Anonymous", "None", "None", src_val if src_type == 'doi' else "None")
                    if res:
                        cur.execute("UPDATE ingestion_queue SET status='completed' WHERE id=?", (pid,))
                        conn.commit()
                        status.update(label="Complete!", state="complete")
                        st.success(f"Score: {res[2]:.2f} | piQ Minted: {res[10]}")
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        status.update(label="Failed.", state="error")
conn.close()
