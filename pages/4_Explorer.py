import json
import streamlit as st
import pandas as pd
from database import get_db_connection
from shared_ui import more_details_dialog, safe_float

st.title("⛓️ Proof-of-Research Blockchain Explorer")

conn = get_db_connection()
search_q = st.text_input("Search Ledger by Eval Hash, Block Hash, Title, or Author:")

if search_q.strip():
    q_term = f"%{search_q.strip()}%"
    rows = conn.execute("""
        SELECT p.title, p.author_name, p.filename, p.final_score, p.logic_score, 
               p.c1, p.c2, p.c3, p.c4, p.c5, p.c6, p.c7, p.c8, 
               p.piq_minted, p.tx_hash, p.zk_proof, p.mdar_adherence_score, 
               p.rrid_valid_count, p.reproducibility_score, p.eval_hash,
               p.consensus_data, p.evidence_report, p.scilem_score
        FROM papers_assessment p
        LEFT JOIN blockchain_por_weights b ON p.eval_hash = b.eval_hash
        WHERE b.block_hash LIKE ? OR p.eval_hash LIKE ? OR p.title LIKE ? OR p.author_name LIKE ?
        LIMIT 10
    """, (q_term, q_term, q_term, q_term)).fetchall()
    
    if rows:
        for r in rows:
            with st.expander(f"{r[0]} - {r[1]} (Score: {r[3]:.2f})"):
                st.write(f"**Hash:** `{r[19]}`")
                if st.button("Full Dossier", key=r[19]):
                    more_details_dialog({
                        "title": r[0], "author_name": r[1], "score": r[3], "logic_integrity": r[4], 
                        "scores_dict": {"C1": r[5], "C2": r[6], "C3": r[7], "C4": r[8], "C5": r[9], "C6": r[10], "C7": r[11], "C8": r[12]},
                        "eval_hash": r[19], "piq": r[13], "tx_hash": r[14], "zk_proof": r[15],
                        "h_idx": r[16], "i10_idx": r[17], "repro_score": r[18], "filename": r[2], 
                        "consensus_raw": json.loads(r[20]) if r[20] else {}, "evidence_report_text": r[21], "scilem_rating": r[22]
                    })
    else:
        st.error("No matching ledger records found.")
else:
    st.markdown("### Latest Assessed Papers")
    df = pd.read_sql_query("SELECT title as Title, author_name as Author, final_score as Score, eval_hash as Hash FROM papers_assessment ORDER BY timestamp DESC LIMIT 20", conn)
    st.dataframe(df, use_container_width=True, hide_index=True)

conn.close()
