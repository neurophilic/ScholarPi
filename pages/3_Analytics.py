import tempfile
import os
import colorsys
import time
import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pyvis.network import Network

from database import get_db_connection
from integrations import clean_author_name, is_likely_institution
from shared_ui import safe_float

st.title("🌐 Analytics & Map of Science")

def refine_science_field(s):
    s_l = s.lower()
    if "blockchain" in s_l: return "Computer Science > Blockchain"
    if "biology" in s_l: return "Life Sciences > Biology"
    return f"Applied Technical Research ({s.title()})"

def render_bubble_chart():
    conn = get_db_connection()
    data = conn.execute("SELECT fields, subfields, final_score FROM papers_assessment").fetchall()
    conn.close()
    
    if not data: return "<p>No data</p>"
    
    net = Network(height="500px", width="100%", bgcolor="#ffffff", notebook=False)
    topic_aggs = {}
    
    for _, sub_j, score in data:
        try:
            for raw_s in json.loads(sub_j):
                t = refine_science_field(raw_s)
                if t not in topic_aggs: topic_aggs[t] = {"wt": 0.0, "freq": 0}
                topic_aggs[t]["wt"] += safe_float(score, 50.0)
                topic_aggs[t]["freq"] += 1
        except Exception: pass
        
    for t, m in topic_aggs.items():
        net.add_node(t, label=" ", title=t, size=max(20, m["freq"] * 10))
        
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".html")
    os.close(tmp_fd)
    net.save_graph(tmp_name)
    with open(tmp_name, "r", encoding="utf-8") as f:
        html = f.read()
    os.remove(tmp_name)
    return html

st.markdown("### Topic Distribution Map")
components.html(render_bubble_chart(), height=500)

col1, col2 = st.columns(2, gap="large")
conn = get_db_connection()

with col1:
    st.markdown("### piQ Leaderboard")
    data = pd.read_sql_query("SELECT author_name, piq_minted FROM papers_assessment", conn)
    author_piq = {}
    for _, row in data.iterrows():
        ca = clean_author_name(row["author_name"])
        if ca and ca.lower() not in ["unidentified", "unknown"] and not is_likely_institution(ca):
            for a in [x.strip() for x in ca.split(",")]: author_piq[a] = author_piq.get(a, 0.0) + float(row["piq_minted"] or 0)
    if author_piq:
        st.dataframe(pd.DataFrame(sorted(author_piq.items(), key=lambda x: x[1], reverse=True)[:20], columns=["Author", "piQ"]), hide_index=True)

with col2:
    st.markdown("### piX Top Papers")
    df_pix = pd.read_sql_query("SELECT title, author_name, final_score FROM papers_assessment ORDER BY final_score DESC LIMIT 20", conn)
    st.dataframe(df_pix, hide_index=True)

conn.close()
