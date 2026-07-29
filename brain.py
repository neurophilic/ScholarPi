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

def extract_with_scilem(paper_text):
    scilem_model, scilem_optimizer = get_scilem_engine()
    scilem_weights_path = os.path.join(BASE_DIR, "scilem_weights.pt")
    if os.path.exists(scilem_weights_path):
        try:
            scilem_model.load_state_dict(torch.load(scilem_weights_path, weights_only=True))
        except Exception:
            pass

    scilem_model.eval()
    words = paper_text.lower().split()[:512]
    tokens = [int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % 10000 for w in words]
    if not tokens: tokens = [0]
    paper_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        feat_val = scilem_model(paper_tensor).item()

    scilem_numeric_score = 50.0 + (feat_val * 40.0)

    lines = [l.strip() for l in paper_text.split("\n") if l.strip()]
    cand_title = lines[0] if lines else "Scilem Neural Extraction"
    cand_author = "Independent Research Scholar"
    for line in lines[1:10]:
        if any(kw in line.lower() for kw in ["by", "author", "university", "department", "@"]):
            cand_author = line
            break

    return "scilem", {
        "title": cand_title[:120],
        "authors": clean_author_name(cand_author)[:80],
        "opinion": f"Scilem Neural Engine Analysis: LSTM feature score = {feat_val:.4f}.",
        "references": [],
        "api_failed": False,
        "is_heuristic_fallback": True,
        "scilem_score": scilem_numeric_score,
    }

def train_scilem_on_input_and_report(raw_text, evidence_report):
    scilem_model, scilem_optimizer = get_scilem_engine()
    scilem_weights_path = os.path.join(BASE_DIR, "scilem_weights.pt")
    if os.path.exists(scilem_weights_path):
        try:
            scilem_model.load_state_dict(torch.load(scilem_weights_path, weights_only=True))
        except Exception:
            pass

    scilem_model.train()
    scilem_optimizer.zero_grad()
    
    words = raw_text.lower().split()[:512]
    tokens = [int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % 10000 for w in words]
    if not tokens: tokens = [0]
    paper_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    features = scilem_model(paper_tensor)
    vapri = (int(hashlib.md5(evidence_report.encode()).hexdigest(), 16) % 1000) / 1000.0
    target_tensor = torch.tensor([[vapri]], dtype=torch.float32)

    loss = nn.MSELoss()(features, target_tensor)
    loss.backward()
    scilem_optimizer.step()
    
    torch.save(scilem_model.state_dict(), scilem_weights_path)
    return "Scilem model weights updated dynamically."

def reset_scilem():
    scilem_weights_path = os.path.join(BASE_DIR, "scilem_weights.pt")
    if os.path.exists(scilem_weights_path):
        try: os.remove(scilem_weights_path)
        except Exception: pass
    return "Scilem state reset successfully."

class PiBlockchainDataset(Dataset):
    def __init__(self, data_matrix, lookback):
        self.data = data_matrix
        self.lookback = lookback
    def __len__(self):
        return len(self.data) - self.lookback
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx : idx + self.lookback], dtype=torch.float32), torch.tensor(self.data[idx + self.lookback], dtype=torch.float32)

PidyneBlockchainDataset = PiBlockchainDataset

class PiBrainLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_layer_size=32, output_size=8):
        super(PiBrainLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_layer_size, 16), nn.ReLU(), nn.Linear(16, output_size))
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return torch.softmax(self.linear(lstm_out[:, -1, :]), dim=-1) * 8.0

PidyneLSTM = PiBrainLSTM

def query_llm_json(provider_name, model_name, api_key, base_url, prompt):
    if not api_key or not str(api_key).strip():
        return provider_name, {"title": "N/A", "authors": "Unconfigured Key", "opinion": "API key missing.", "references": [], "api_failed": True}
    try:
        client = OpenAI(api_key=api_key.strip(), base_url=base_url)
        response = client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0.1
        )
        data = json.loads(response.choices[0].message.content)
        data["api_failed"] = False
        return provider_name, data
    except Exception as e:
        return provider_name, {"title": "N/A", "authors": "N/A", "opinion": f"Error: {str(e)}", "references": [], "api_failed": True}

def run_multi_llm_consensus(paper_text):
    prompt = f"Analyze manuscript excerpts and respond strictly in JSON with keys: 'title', 'authors', 'opinion', 'references'.\n\n{paper_text[:3000]}"
    results = {}
    if GROQ_API_KEY: results["llama"] = query_llm_json("llama", PRIMARY_MODEL, GROQ_API_KEY, "https://api.groq.com/openai/v1", prompt)[1]
    if GEMINI_API_KEY: results["gemini"] = query_llm_json("gemini", "gemini-2.0-flash", GEMINI_API_KEY, "https://generativelanguage.googleapis.com/v1beta/openai/", prompt)[1]
    results["scilem"] = extract_with_scilem(paper_text)[1]
    return results

def generate_rebuttal_strategy(scores_dict):
    lowest_criterion = min(scores_dict.items(), key=lambda x: x[1])
    return f"**Adversarial Defense Strategy:** Focus on strengthening `{lowest_criterion[0]}` (Current score: {lowest_criterion[1]:.1f}/100)."

def process_single_pdf(file_bytes, filename, scope, user_id, book_address="None", email="None", provided_doi="None", force_proceed=False):
    active_weights = [1.0] * 8
    warnings_list = []
    if not file_bytes:
        return ("Download Failed", "Scholar", 0.0, 75.0, "N/A", "N/A", ["Domain"], ["Subdomain"], {}, "Failed", 0.0, "None", "None", active_weights, 0.0, 0, 0.0, False, ["Empty payload"], {}, "", 75.0)

    file_hash = hashlib.sha256(file_bytes).hexdigest()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            pdf_meta_author = doc.metadata.get("author", "").strip()
            full_text = "\n".join([page.get_text("text", sort=True) for page in doc])
        except Exception:
            full_text = ""

        scores_dict = {
            "C1_Semantic_Originality": 75.0, "C2_Methodological_Rigor_SciScore": 80.0, 
            "C3_Interdisciplinary_Entropy": 70.0, "C4_Societal_Impact": 75.0, 
            "C5_Open_Science_Repro": 80.0, "C6_Literature_Integration": 75.0, 
            "C7_Empirical_Density": 75.0, "C8_Future_Actionability_FAIR": 80.0
        }
        final_score = sum(scores_dict.values()) / 8.0
        piq_minted = round((final_score / 100.0) * 10.0, 2)
        zk_proof = generate_zk_snark_proof(file_hash, final_score, 75.0, "None")
        tx_hash = mint_pi_quotient_token(book_address, piq_minted, file_hash, zk_proof) if book_address != "0x0000000000000000000000000000000000000000" and piq_minted > 0 else "Simulated_Ledger_Record"

        cursor.execute(
            """INSERT OR REPLACE INTO papers_assessment (
                eval_hash, user_id, title, filename, scope, c1, c2, c3, c4, c5, c6, c7, c8, 
                logic_score, scope_alignment, subfields, fields, author_name, final_score, 
                timestamp, eth_book, piq_minted, tx_hash, zk_proof, did, zk_email_proof, 
                gaming_penalty, mdar_adherence_score, rrid_valid_count, credit_taxonomy_roles, 
                reproducibility_score, doi, consensus_data, evidence_report, scilem_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                file_hash, user_id, filename.replace(".pdf", ""), filename, scope, *scores_dict.values(),
                75.0, 0.0, json.dumps(["Core Research Domain"]),
                json.dumps(["Computer Science"]), pdf_meta_author or "Scholar", final_score,
                datetime.now().isoformat(), book_address, piq_minted,
                tx_hash, zk_proof, user_id, "None", 0.0,
                0.8, 1, json.dumps(["Data Curation"]), 0.8,
                provided_doi, "{}", "Evidence Report", 75.0
            ),
        )
        conn.commit()
    finally:
        conn.close()
    
    backup_state_to_web3()
    return (filename.replace(".pdf", ""), pdf_meta_author or "Scholar", final_score, 75.0, "N/A", "N/A", ["Computer Science"], ["Core Research Domain"], scores_dict, file_hash, piq_minted, tx_hash, zk_proof, active_weights, 0.8, 1, 0.8, False, warnings_list, {}, "Evidence Report", 75.0)
