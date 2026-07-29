import streamlit as st
import pandas as pd
from database import get_db_connection
from config import PIQ_CONTRACT_ADDRESS, REGISTRY_CONTRACT_ADDRESS

st.title("⛓️ Proof-of-Research Blockchain Explorer")
st.markdown(f"**Smart Contracts on Sepolia:** PiQ Token: `{PIQ_CONTRACT_ADDRESS}` | Registry: `{REGISTRY_CONTRACT_ADDRESS}`")

search = st.text_input("Search Ledger by Eval Hash, Block Hash, or Author Name:")
conn = get_db_connection()

if search.strip():
    q = f"%{search.strip()}%"
    df = pd.read_sql_query("SELECT p.title, p.author_name, p.final_score, p.piq_minted, p.eval_hash, b.block_hash FROM papers_assessment p LEFT JOIN blockchain_por_weights b ON p.eval_hash = b.eval_hash WHERE b.block_hash LIKE ? OR p.eval_hash LIKE ? OR p.title LIKE ? OR p.author_name LIKE ?", conn, params=(q, q, q, q))
    st.dataframe(df, use_container_width=True, hide_index=True) if not df.empty else st.error("No matching records found.")
else:
    st.markdown("### Latest PoR Blocks Mined")
    df = pd.read_sql_query("SELECT block_height, block_hash, eval_hash, timestamp, por_proof FROM blockchain_por_weights ORDER BY block_height DESC LIMIT 20", conn)
    st.dataframe(df, use_container_width=True, hide_index=True)

conn.close()
