import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from config import BASE_DIR
from database import get_db_connection
from brain import PidyneLSTM, PidyneBlockchainDataset
from shared_ui import safe_float, rbot

st.title("🧠 Pidyne Brain & Epoch Forecasting")

def get_criteria_info(weights):
    tw1, tw2, tw3, tw4, tw5, tw6, tw7, tw8 = weights
    return [
        ("C1", "Originality", "c1", tw1, "1", "Semantic distance from literature.", ""),
        ("C2", "Methodological Rigor", "c2", tw2, "2", "Adherence to MDAR and valid RRIDs.", ""),
        ("C3", "Interdisciplinary Synergy", "c3", tw3, "3", "Cross-disciplinary integration.", ""),
        ("C4", "Societal Impact", "c4", tw4, "4", "Societal and open infrastructure contributions.", ""),
        ("C5", "Open Science", "c5", tw5, "5", "Open data, open code, containerized rep.", ""),
        ("C6", "Literature Integration", "c6", tw6, "6", "Integration with foundational literature.", ""),
        ("C7", "Empirical Density", "c7", tw7, "7", "Empirical sample strength.", ""),
        ("C8", "Future Actionability", "c8", tw8, "8", "Actionability and adherence to FAIR.", ""),
    ]

@st.dialog("Criterion Details", width="medium")
def criterion_details_dialog(c_id, title, q_key, weight_val, sym, desc, formula):
    st.markdown(f"### {c_id}: {title}")
    st.markdown(rf"**Current Epoch Weight:** `{weight_val:.6f}`")
    st.markdown(f"{desc} {rbot(q_key)}", unsafe_allow_html=True)

lookback = st.selectbox("Lookback Window", ["1 Epoch", "3 Epochs", "5 Epochs"], index=1)
actual_lookback = int(lookback.split()[0])

conn_pb = get_db_connection()
historical_rows = conn_pb.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights ORDER BY block_height ASC").fetchall()
conn_pb.close()

if len(historical_rows) < 2:
    st.warning("Not enough blockchain data to train meta-model. Need at least 2 blocks.")
else:
    @st.cache_data(show_spinner="Training LSTM Model...")
    def train_pidyne_cached(weight_data, lookback):
        dataset = PidyneBlockchainDataset(weight_data, lookback)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        model = PidyneLSTM()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for _ in range(300):
            for seq, target in dataloader:
                optimizer.zero_grad()
                loss = nn.MSELoss()(model(seq), target)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            raw_pred = model(torch.tensor(weight_data[-lookback:], dtype=torch.float32).unsqueeze(0)).squeeze().numpy()
            predicted = weight_data[-1] + (raw_pred - weight_data[-1]) * 20.0
            return np.clip(predicted, 0.01, 7.9) * (8.0 / np.sum(np.clip(predicted, 0.01, 7.9)))

    weight_data = np.array([[safe_float(v, 1.0) for v in r] for r in historical_rows], dtype=np.float32)
    next_weights = train_pidyne_cached(weight_data, max(1, min(actual_lookback, len(historical_rows)-1)))

    df_hist = pd.DataFrame(historical_rows[-(actual_lookback + 1):], columns=["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"])
    df_hist.index.name = "Block"
    df_melted = df_hist.reset_index().melt('Block', var_name='Criterion', value_name='Weight')
    
    st.altair_chart(alt.Chart(df_melted).mark_line(point=True).encode(x='Block:O', y='Weight:Q', color='Criterion:N'), use_container_width=True)

    st.markdown("### Ledger Forecast Criteria")
    crit_info = get_criteria_info(next_weights)
    cols = st.columns(4)
    for i, c_data in enumerate(crit_info):
        if cols[i%4].button(f"{c_data[0]}: {c_data[3]:.4f}", use_container_width=True):
            criterion_details_dialog(*c_data)
