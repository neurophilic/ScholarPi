import streamlit as st
import pandas as pd
from database import get_db_connection
from integrations import clean_author_name, is_likely_institution

st.title("🌐 Global Map of Science & Analytics Dashboard")

col1, col2 = st.columns(2, gap="large")
conn = get_db_connection()

with col1:
    st.markdown("### Pi Quotient (piQ) Leaderboard")
    data = pd.read_sql_query("SELECT author_name, piq_minted FROM papers_assessment", conn)
    author_piq = {}
    for _, row in data.iterrows():
        clean_authors = clean_author_name(row["author_name"])
        if clean_authors and clean_authors.lower() not in ["unidentified", "unknown"] and not is_likely_institution(clean_authors):
            for a in [x.strip() for x in clean_authors.split(",") if x.strip()]:
                author_piq[a] = author_piq.get(a, 0.0) + (float(row["piq_minted"] or 0.0) / len(clean_authors.split(",")))

    if author_piq:
        df_piq = pd.DataFrame(sorted(author_piq.items(), key=lambda x: x[1], reverse=True)[:20], columns=["Author", "Total piQ"])
        st.dataframe(df_piq, use_container_width=True, hide_index=True)
    else:
        st.info("No piQ tokens minted yet.")

with col2:
    st.markdown("### pi-Index (piX) Leaderboard [Top Papers]")
    df_pix = pd.read_sql_query("SELECT title as 'Manuscript Title', author_name as 'Author', final_score as 'Score' FROM papers_assessment ORDER BY final_score DESC LIMIT 20", conn)
    st.dataframe(df_pix, use_container_width=True, hide_index=True) if not df_pix.empty else st.info("No assessments recorded yet.")

conn.close()
