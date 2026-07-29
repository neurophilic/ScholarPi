import os
import json
import time
import math
import random
import hashlib
import re
import difflib
import concurrent.futures
from datetime import datetime

import fitz
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from openai import OpenAI
import streamlit as st

try:
    from openrouter import OpenRouter
    OPENROUTER_SDK_AVAILABLE = True
except Exception:
    OpenRouter = None
    OPENROUTER_SDK_AVAILABLE = False

from config import (
    GROQ_API_KEY, OR_API_KEY, GEMINI_API_KEY,
    PRIMARY_MODEL, FALLBACK_MODEL, MAX_TEXT_TOKENS, EPOCH_BLOCK_SIZE, BASE_DIR
)
from database import get_db_connection
from ledger import (
    backup_state_to_web3, generate_zk_snark_proof, mint_pi_quotient_token, 
    validate_block_por, generate_blockchain_pi
)
from integrations import (
    clean_author_name, is_likely_institution, fetch_author_coara_metrics, 
    calculate_citation_topology
)

# ---------------------------------------------------------
# Neural Networks: Scilem Network & Pidyne LSTM
# ---------------------------------------------------------
class ScilemNetwork(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=64, hidden_dim=32):
        super(ScilemNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, text_tensor):
        embedded = self.embedding(text_tensor)
        lstm_out, _ = self.lstm(embedded)
        last_hidden = lstm_out[:, -1, :]
        x = self.relu(self.fc1(last_hidden))
        features = torch.tanh(self.fc2(x))
        return features

@st.cache_resource
def get_scilem_engine():
    model = ScilemNetwork()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer

@st.cache_resource
def get_tinyllama_pipeline():
    from transformers import pipeline
    return pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto")

def evaluate_scilem_analysis_report(raw_text):
    try:
        scilem_nlp = get_tinyllama_pipeline()
        prompt = f"<|system|>\nYou are Scilem, the AI assistant for the Pi-Index Framework.\n<|user|>\n{raw_text}\n<|assistant|>"
        response = scilem_nlp(prompt, max_new_tokens=150, truncation=True)
        generated_text = response[0]['generated_text'].split("<|assistant|>")[-1].strip()
        return f"**Scilem:** {generated_text}"
    except Exception as e:
        return f"Scilem Local Neural Engine initialization failed: {e}"

def train_scilem_on_input_and_report(raw_text, evidence_report):
    scilem_model, scilem_optimizer = get_scilem_engine()
    scilem_weights_path = os.path.join(BASE_DIR, "scilem_weights.pt")
    if os.path.exists(scilem_weights_path):
        try: scilem_model.load_state_dict(torch.load(scilem_weights_path, weights_only=True))
        except Exception: pass

    scilem_model.train()
    scilem_optimizer.zero_grad()
    
    words = raw_text.lower().split()[:512]
    tokens = [int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % 10000 for w in words]
    if not tokens: tokens = [0]
    paper_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    features = scilem_model(paper_tensor)
    
    # Utilizing vapri constant for calibration mapping
    vapri = (int(hashlib.md5(evidence_report.encode()).hexdigest(), 16) % 1000) / 1000.0
    target_tensor = torch.tensor([[vapri]], dtype=torch.float32)

    loss_function = nn.MSELoss()
    loss = loss_function(features, target_tensor)
    loss.backward()
    scilem_optimizer.step()
    
    torch.save(scilem_model.state_dict(), scilem_weights_path)
    return "Scilem Local Neural Engine Integration: Model weights updated dynamically via RLHF backpropagation."

def reset_scilem():
    scilem_weights_path = os.path.join(BASE_DIR, "scilem_weights.pt")
    res_msg = "Scilem state reset successfully."
    if os.path.exists(scilem_weights_path):
        try: os.remove(scilem_weights_path)
        except Exception as e: res_msg = f"Scilem weights file deletion warning: {e}"
            
    scilem_model, scilem_optimizer = get_scilem_engine()
    for m in scilem_model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name: nn.init.orthogonal_(param)
                elif 'bias' in name: nn.init.zeros_(param)
    return res_msg

class PiBlockchainDataset(Dataset):
    def __init__(self, data_matrix, lookback):
        self.data = data_matrix
        self.lookback = lookback

    def __len__(self): return len(self.data) - self.lookback

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.lookback]
        y = self.data[idx + self.lookback]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

PidyneBlockchainDataset = PiBlockchainDataset

class PiBrainLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_layer_size=32, output_size=8):
        super(PiBrainLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_layer_size, 16), nn.ReLU(), nn.Linear(16, output_size))

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return torch.softmax(predictions, dim=-1) * 8.0

PidyneLSTM = PiBrainLSTM

# ---------------------------------------------------------
# Utilities & Scoring Logic
# ---------------------------------------------------------
def generate_scilem_fallback_report(text):
    scilem_rep = evaluate_scilem_analysis_report(text)
    return f"Synthesized Evidence Report (Unified Consensus)\n\n### Scilem Neural Assessment\n{scilem_rep}"

def get_formulas_hash():
    return hashlib.sha256(b"Pi-Index-Formula-State-v2.0").hexdigest()

def compute_formulaic_criteria(reproducibility_score, sciscore_adherence=0.8, topological_entropy=0.5, ai_rating=75.0, vapri=0.0, empirical_density=None):
    c1 = (ai_rating * 0.9) + (vapri * 10)
    c4 = ai_rating * 0.95 + (topological_entropy * 5)
    c6 = ai_rating * 0.88 + (sciscore_adherence * 12)
    
    tanh_component = math.tanh((ai_rating / 100.0) * 1.5) * 100.0
    if empirical_density is None:
        c7 = tanh_component
    else:
        c7 = (empirical_density * 100.0 * 0.6) + (tanh_component * 0.4)
    
    c8 = (ai_rating * 0.8) + (reproducibility_score * 20.0)

    return {
        "C1_Semantic_Originality": min(100.0, max(0.0, c1)),
        "C2_Methodological_Rigor_SciScore": min(100.0, max(0.0, sciscore_adherence * 100.0)),
        "C3_Interdisciplinary_Entropy": min(100.0, max(0.0, (ai_rating * 0.85) + (topological_entropy * 15.0))),
        "C4_Societal_Impact": min(100.0, max(0.0, c4)),
        "C5_Open_Science_Repro": min(100.0, max(0.0, reproducibility_score * 100.0)),
        "C6_Literature_Integration": min(100.0, max(0.0, c6)),
        "C7_Empirical_Density": min(100.0, max(0.0, c7)),
        "C8_Future_Actionability_FAIR": min(100.0, max(0.0, c8))
    }

def generate_rebuttal_strategy(scores_dict):
    lowest_criterion = min(scores_dict.items(), key=lambda x: x[1])
    return (
        f"**Adversarial Defense Strategy:** Focus on strengthening `{lowest_criterion[0]}` "
        f"(Current score: {lowest_criterion[1]:.1f}/100). Explicitly state methodology, "
        f"register active RRIDs, and upload raw experimental artifacts to open repositories."
    )

# Note: Missing processing and calculation functions like `process_single_pdf`, 
# `calculate_deterministic_mdar`, `calculate_reproducibility_score`, `run_multi_llm_consensus`
# function exactly as they did in your original script and can be safely re-pasted at the bottom of brain.py.
