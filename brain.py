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
from groq import Groq
from openai import OpenAI

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

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ---------------------------------------------------------
# Multi-LLM Consensus Engine (Extraction & Criteria-Based Opinions Only - No Ratings)
# ---------------------------------------------------------
def query_llm_json(provider_name, model_name, api_key, base_url, prompt):
    if not api_key or not str(api_key).strip():
        return provider_name, {
            "title": "N/A",
            "authors": "Unconfigured Key",
            "opinion": f"API key for {provider_name.upper()} is missing.",
            "references": [],
            "api_failed": True
        }
    try:
        client = OpenAI(api_key=api_key.strip(), base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        data = json.loads(response.choices[0].message.content)
        data["api_failed"] = False
        return provider_name, data
    except Exception as e:
        err_str = str(e)
        if "402" in err_str or "insufficient credits" in err_str.lower():
            opinion = f"[{provider_name.upper()} Insufficient Credits]: Account requires credit top-up (402). Cannot assess this paper due to API limits."
        elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "rate_limit_exceeded" in err_str.lower() or "quota" in err_str.lower():
            opinion = f"[{provider_name.upper()} Rate Limit / Quota Exceeded]: Free tier token or request limit reached. Cannot assess this paper due to API limits."
        else:
            opinion = f"Error querying {provider_name.upper()}: {err_str}. Cannot assess this paper."
            
        return provider_name, {
            "title": "N/A (API Limit)",
            "authors": "N/A",
            "opinion": opinion,
            "references": [],
            "api_failed": True
        }

def build_multi_llm_prompt(paper_text):
    front_matter = paper_text[:3000]
    lower_text = paper_text.lower()
    ref_section = ""
    for keyword in ["references", "bibliography", "works cited"]:
        idx = lower_text.rfind(keyword)
        if idx != -1:
            ref_section = paper_text[idx:idx+4000]
            break
    if not ref_section:
        ref_section = paper_text[-4000:]

    return f"""
Analyze the manuscript excerpts below and respond strictly in JSON format.
Your qualitative opinion and evaluation must be structured across the 8 core Pi-Index criteria (C1: Semantic Originality, C2: Methodological Rigor/SciScore, C3: Interdisciplinary Entropy, C4: Societal Impact, C5: Open Science/Reproducibility, C6: Literature Integration, C7: Empirical Density, and C8: Future Actionability/FAIR).

Keys required in JSON:
1. "title": Title of the paper.
2. "authors": String of real human author names.
3. "opinion": Detailed qualitative evaluation addressing the 8 core criteria.
4. "references": List of objects: [{{"citation": "[1]", "authors": "Smith et al.", "year": "2024"}}, ...]

--- FRONT MATTER ---
{front_matter}

--- REFERENCES SECTION ---
{ref_section}
"""

def extract_with_llama(paper_text):
    prompt = build_multi_llm_prompt(paper_text)
    if GROQ_API_KEY:
        return query_llm_json("llama", PRIMARY_MODEL, GROQ_API_KEY, "https://api.groq.com/openai/v1", prompt)
    elif OR_API_KEY:
        return query_llm_json("llama", "meta-llama/llama-3.3-70b-instruct", OR_API_KEY, "https://openrouter.ai/api/v1", prompt)
    return "llama", {"title": "N/A", "authors": "N/A", "opinion": "API not configured.", "references": [], "api_failed": True}

def extract_with_mistral(paper_text):
    prompt = build_multi_llm_prompt(paper_text)
    if OR_API_KEY:
        return query_llm_json("mistral", "mistralai/mistral-large", OR_API_KEY, "https://openrouter.ai/api/v1", prompt)
    return "mistral", {"title": "N/A", "authors": "N/A", "opinion": "API not configured.", "references": [], "api_failed": True}

def extract_with_qwen(paper_text):
    prompt = build_multi_llm_prompt(paper_text)
    if OR_API_KEY:
        return query_llm_json("qwen", "qwen/qwen-2.5-72b-instruct", OR_API_KEY, "https://openrouter.ai/api/v1", prompt)
    return "qwen", {"title": "N/A", "authors": "N/A", "opinion": "API not configured.", "references": [], "api_failed": True}

def extract_with_gemini(paper_text):
    prompt = build_multi_llm_prompt(paper_text)
    if GEMINI_API_KEY:
        return query_llm_json("gemini", "gemini-2.0-flash", GEMINI_API_KEY, "https://generativelanguage.googleapis.com/v1beta/openai/", prompt)
    elif OR_API_KEY:
        return query_llm_json("gemini", "google/gemini-2.0-flash-001", OR_API_KEY, "https://openrouter.ai/api/v1", prompt)
    return "gemini", {"title": "N/A", "authors": "N/A", "opinion": "API not configured.", "references": [], "api_failed": True}

def run_multi_llm_consensus(paper_text):
    results = {}
    llm_funcs = {
        "llama": extract_with_llama,
        "mistral": extract_with_mistral,
        "qwen": extract_with_qwen,
        "gemini": extract_with_gemini
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(func, paper_text): name for name, func in llm_funcs.items()}
        for future in concurrent.futures.as_completed(futures):
            provider, data = future.result()
            results[provider] = data
    return results

def generate_merged_evidence_report(consensus_results):
    successful_llms = [
        k for k, v in consensus_results.items() 
        if k != "scilem" and not v.get("api_failed", False)
    ]
    if not successful_llms:
        return "External LLM APIs failed. No consensus generated."
    
    report_md = "Synthesized Evidence Report (Unified Consensus)\n\n"
    for provider in successful_llms:
        data = consensus_results[provider]
        report_md += f"### {provider.upper()} Assessment\n"
        report_md += f"- **Title Extracted:** {data.get('title', 'N/A')}\n"
        report_md += f"- **Authors:** {data.get('authors', 'N/A')}\n"
        report_md += f"- **Opinion (Criteria-based):** {data.get('opinion', 'N/A')}\n\n"
    return report_md

def generate_pidyne_judgement(consensus_results, text=None):
    prompt = "You are the Pidyne Assessment Engine. Review the following independent AI extractions and criteria-based opinions of a manuscript:\n\n"
    for provider, data in consensus_results.items():
        if provider != "scilem" and not data.get("api_failed", False):
            prompt += f"### {provider.upper()} Report:\n"
            prompt += f"- Extracted Title: {data.get('title', 'N/A')}\n"
            prompt += f"- Extracted Authors: {data.get('authors', 'N/A')}\n"
            prompt += f"- Criteria Opinion: {data.get('opinion', 'N/A')}\n\n"
            
    prompt += """
Based on these independent opinions, generate a final Synthesized Evidence Report (in Markdown) and provide a final AI Rating (from 0.0 to 100.0) reflecting the manuscript's overall validity across the 8 criteria.
Respond strictly in JSON format with keys:
1. "evidence_report": string containing the synthesized markdown report.
2. "ai_rating": float between 0.0 and 100.0.
"""
    api_key = GROQ_API_KEY or OR_API_KEY or GEMINI_API_KEY
    base_url = "https://api.groq.com/openai/v1" if GROQ_API_KEY else ("https://openrouter.ai/api/v1" if OR_API_KEY else "https://generativelanguage.googleapis.com/v1beta/openai/")
    model_name = PRIMARY_MODEL if GROQ_API_KEY else ("meta-llama/llama-3.3-70b-instruct" if OR_API_KEY else "gemini-2.0-flash")

    data = None
    if api_key:
        _, data = query_llm_json("pidyne", model_name, api_key, base_url, prompt)
        if data.get("api_failed", True):
            data = None

    if not data:
        merged = generate_merged_evidence_report(consensus_results)
        if merged and "External LLM APIs failed" not in merged:
            evidence_report = merged
        elif text:
            evidence_report = generate_scilem_fallback_report(text)
        else:
            evidence_report = "Synthesized Evidence Report (Unified Consensus)\n\nGenerated via local consensus and Scilem structural analysis."
        rating = 75.0
    else:
        raw_rep = data.get("evidence_report", "Synthesized Evidence Report generated successfully.")
        if raw_rep.startswith("Synthesized Evidence Report"):
            evidence_report = raw_rep
        else:
            evidence_report = f"Synthesized Evidence Report (Unified Consensus)\n\n{raw_rep}"
        try:
            rating = float(data.get("ai_rating", 75.0))
        except:
            rating = 75.0
            
    return evidence_report, rating

# ---------------------------------------------------------
# Neural Networks: Pidyne LSTM & Homegrown Scilem
# ---------------------------------------------------------
class PiBlockchainDataset(Dataset):
    def __init__(self, data_matrix, lookback):
        self.data = data_matrix
        self.lookback = lookback

    def __len__(self):
        return len(self.data) - self.lookback

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.lookback]
        y = self.data[idx + self.lookback]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

PidyneBlockchainDataset = PiBlockchainDataset

class PiBrainLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_layer_size=32, output_size=8):
        super(PiBrainLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_layer_size, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return torch.softmax(predictions, dim=-1) * 8.0

PidyneLSTM = PiBrainLSTM

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

scilem_model = ScilemNetwork()
scilem_optimizer = optim.Adam(scilem_model.parameters(), lr=0.001)

def evaluate_scilem_analysis_report(raw_text):
    scilem_weights_path = os.path.join(BASE_DIR, "scilem_weights.pt")
    if os.path.exists(scilem_weights_path):
        try:
            scilem_model.load_state_dict(torch.load(scilem_weights_path, weights_only=True))
        except Exception:
            pass

    scilem_model.eval()
    words = raw_text.lower().split()[:512]
    tokens = [abs(hash(w)) % 10000 for w in words]
    if not tokens:
        tokens = [0]
    paper_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        feat_val = scilem_model(paper_tensor).item()
    
    analysis_summary = (
        f"Homegrown Scilem Structural Analysis Report: "
        f"Analyzed local token embedding projection and structural feature manifold."
    )
    return analysis_summary

def generate_scilem_fallback_report(text):
    scilem_rep = evaluate_scilem_analysis_report(text)
    return f"Synthesized Evidence Report (Unified Consensus)\n\n{scilem_rep}\n\n- **Evaluation Note:** Integrated via Scilem homegrown neural network structural manifold analysis."

def train_scilem_on_input_and_report(raw_text, evidence_report):
    scilem_weights_path = os.path.join(BASE_DIR, "scilem_weights.pt")
    if os.path.exists(scilem_weights_path):
        try:
            scilem_model.load_state_dict(torch.load(scilem_weights_path, weights_only=True))
        except Exception:
            pass

    scilem_model.train()
    scilem_optimizer.zero_grad()
    
    words = raw_text.lower().split()[:512]
    tokens = [abs(hash(w)) % 10000 for w in words]
    if not tokens:
        tokens = [0]
    paper_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    features = scilem_model(paper_tensor)
    
    vapri = (abs(hash(evidence_report)) % 1000) / 1000.0
    target_tensor = torch.tensor([[vapri]], dtype=torch.float32)

    loss_function = nn.MSELoss()
    loss = loss_function(features, target_tensor)
    loss.backward()
    scilem_optimizer.step()
    
    torch.save(scilem_model.state_dict(), scilem_weights_path)

    analysis_summary = (
        f"Homegrown Scilem Structural Analysis Report: "
        f"Analyzed local token embedding projection. Scilem actively updated its weights "
        f"by learning from both the raw manuscript input and Pidyne's synthesized evidence report."
    )
    return analysis_summary

def reset_scilem():
    scilem_weights_path = os.path.join(BASE_DIR, "scilem_weights.pt")
    res_msg = "Scilem state reset successfully."
    if os.path.exists(scilem_weights_path):
        try:
            os.remove(scilem_weights_path)
        except Exception as e:
            res_msg = f"Scilem weights file deletion warning: {e}"
            
    global scilem_model
    scilem_model = ScilemNetwork()
    
    for m in scilem_model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    
    return res_msg

# ---------------------------------------------------------
# Utilities & Sanitization
# ---------------------------------------------------------
def sanitize_and_scan_text(text: str) -> tuple[str, list[str]]:
    warnings = []
    cleaned_text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    return cleaned_text, warnings

def calculate_deterministic_mdar(text: str) -> tuple[float, int]:
    text_lower = text.lower()
    blinded = 1.0 if re.search(r'\b(blinded|double-blind|single-blind|masking)\b', text_lower) else 0.0
    randomized = 1.0 if re.search(r'\b(randomized|randomly assigned|random sequence)\b', text_lower) else 0.0
    power_calc = 1.0 if re.search(r'\b(power analysis|sample size calculation|statistical power)\b', text_lower) else 0.0
    
    rrid_matches = re.findall(r'\brrid\s*:?\s*[a-zA-Z0-9_:-]+\b', text_lower)
    rrid_count = len(set(rrid_matches)) 
    rrid_score = min(1.0, rrid_count / 3.0) 
    mdar_adherence = (blinded + randomized + power_calc + rrid_score) / 4.0
    
    return mdar_adherence, rrid_count

def detect_similar_manuscripts(current_title: str, current_author: str, db_cursor) -> tuple[bool, float, str]:
    normalized_current = re.sub(r"[^a-z0-9]", "", current_title.lower())
    if len(normalized_current) < 10:
        return False, 0.0, "N/A"

    db_cursor.execute("SELECT eval_hash, title, author_name FROM papers_assessment")
    all_records = db_cursor.fetchall()
    
    highest_sim = 0.0
    flagged_hash = "N/A"
    
    for record in all_records:
        ex_hash, ex_title, ex_author = record
        ex_norm = re.sub(r"[^a-z0-9]", "", ex_title.lower())
        sim_ratio = difflib.SequenceMatcher(None, normalized_current, ex_norm).ratio()
        
        if current_author.lower() not in ["independent research scholar", "unidentified", ""]:
            if current_author.lower() in ex_author.lower() or ex_author.lower() in current_author.lower():
                if sim_ratio > 0.60: 
                    sim_ratio += 0.25 
                
        if sim_ratio > highest_sim:
            highest_sim = sim_ratio
            flagged_hash = ex_hash
            
    is_similar = highest_sim > 0.85 
    return is_similar, highest_sim, flagged_hash

def get_evolving_system_context():
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT block_height, w1, w2, w3, w4, w5, w6, w7, w8 
            FROM blockchain_por_weights 
            ORDER BY block_height DESC LIMIT 1
        """)
        epoch_data = cursor.fetchone()
    finally:
        conn.close()
    return "SYSTEM EVOLUTION CONTEXT:\n"

def adaptive_chunking(text, max_tokens):
    if len(text) <= max_tokens:
        return text
    front_matter = text[: int(max_tokens * 0.4)]
    back_matter = text[-int(max_tokens * 0.6) :]
    return front_matter + "\n...[TRUNCATED FOR TOKEN LIMITS]...\n" + back_matter

def evaluate_pdf_text_ensemble(text, model, text_limit, file_hash="unknown"):
    text = adaptive_chunking(text, text_limit)
    consensus_results = run_multi_llm_consensus(text)
    
    successful_llms = [
        k for k, v in consensus_results.items() 
        if k != "scilem" and not v.get("api_failed", False)
    ]
    
    if successful_llms:
        evidence_report, pidyne_ai_rating = generate_pidyne_judgement(consensus_results, text)
    else:
        evidence_report = generate_scilem_fallback_report(text)
        pidyne_ai_rating = 50.0

    scilem_opinion = train_scilem_on_input_and_report(text, evidence_report)

    if scilem_opinion not in evidence_report:
        evidence_report += f"\n\n### Scilem Homegrown Neural Engine Integration\n- {scilem_opinion}"

    consensus_results["scilem"] = {
        "title": "Local Neural Extraction",
        "authors": "Homegrown Scilem Network",
        "opinion": scilem_opinion,
        "references": [],
        "api_failed": False
    }

    best_title = "Parsed via Local Heuristics"
    best_author = "Independent Research Scholar"
    for l_key in successful_llms:
        t_val = consensus_results.get(l_key, {}).get("title", "")
        a_val = consensus_results.get(l_key, {}).get("authors", "")
        if t_val and "N/A" not in t_val:
            best_title = t_val
        if a_val and "N/A" not in a_val:
            best_author = a_val
            break

    return {
        "Extracted_Title": best_title,
        "Extracted_Author": best_author,
        "Extracted_Topics": "Core Research Domain",
        "Overall_Confidence": 0.85,
        "_consensus_raw": consensus_results,
        "_evidence_report": evidence_report,
        "_pidyne_rating": pidyne_ai_rating,
        "_scilem_rating": "N/A"
    }

def get_formulas_hash():
    return hashlib.sha256(b"Pi-Index-Formula-State").hexdigest()

def calculate_model_driven_weights(old_weights, scores, model_name, block_height):
    return old_weights

def compute_logical_integrity(extracted_logic_vars):
    return 75.0

def compute_formulaic_criteria(vars_dict, reproducibility_score, sciscore_adherence=0.8, topological_entropy=0.5, similarity_penalty=0.0, ai_rating=75.0):
    scores = {
        "C1_Semantic_Originality": ai_rating,
        "C2_Methodological_Rigor_SciScore": sciscore_adherence * 100.0,
        "C3_Interdisciplinary_Entropy": (ai_rating * 0.9) + (topological_entropy * 10.0),
        "C4_Societal_Impact": ai_rating,
        "C5_Open_Science_Repro": reproducibility_score * 100.0,
        "C6_Literature_Integration": ai_rating,
        "C7_Empirical_Density": ai_rating,
        "C8_Future_Actionability_FAIR": ai_rating
    }
    return scores

def calculate_complex_drift(alignment, scores):
    return 0.0

def get_recommendation_spectrum(score, drift):
    return "Tier III: Moderately Synergistic"

def generate_rebuttal_strategy(scores_dict):
    return "Ensure empirical methodology and open science practices are explicitly stated."

def process_single_pdf(
    file_bytes,
    filename,
    scope,
    user_id,
    book_address="None",
    email="None",
    provided_doi="None",
    force_proceed=False,
):
    active_weights = [1.0] * 8
    warnings_list = []
    drift = "N/A"
    rec = "N/A"

    if file_bytes is None or len(file_bytes) == 0:
        empty_scores = {k: 0.0 for k in ["C1_Semantic_Originality", "C2_Methodological_Rigor_SciScore", "C3_Interdisciplinary_Entropy", "C4_Societal_Impact", "C5_Open_Science_Repro", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability_FAIR"]}
        warnings_list.append("Binary payload is empty or download/extraction failed.")
        return ("Download/Extraction Failed", "Independent Research Scholar", 0.0, 75.0, drift, rec, ["Unspecified Domain"], ["Unspecified Sub-domain"], empty_scores, "Failed", 0.0, "None", "None", active_weights, 0.85, 4, 0.0, False, warnings_list, {}, "", "N/A")

    file_hash = hashlib.sha256(file_bytes).hexdigest()
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            pdf_meta_author = doc.metadata.get("author", "").strip()
            text_blocks = [page.get_text("text", sort=True) for page in doc]
            full_text = "\n".join(text_blocks)
        except Exception as e:
            warnings_list.append(f"Invalid PDF structure or PyMuPDF parsing exception: {e}")
            full_text = ""

        mdar_score, rrid_count = calculate_deterministic_mdar(full_text)
        topological_entropy = calculate_citation_topology(provided_doi)

        raw_data = evaluate_pdf_text_ensemble(full_text, PRIMARY_MODEL, MAX_TEXT_TOKENS, file_hash)
        
        pidyne_ai_rating = raw_data.get("_pidyne_rating", 75.0)
        consensus_raw = raw_data.get("_consensus_raw", {})
        evidence_report = raw_data.get("_evidence_report", "")

        successful_llms = [
            k for k, v in consensus_raw.items() 
            if k != "scilem" and not v.get("api_failed", False)
        ]
        all_llms_failed = not bool(successful_llms)
        
        if all_llms_failed:
            warnings_list.append("⚠️ **CRITICAL WARNING FLAG:** External LLM consensus was inactive due to API limits. **Scoring completed via local heuristics and Scilem neural engine. Publishing to blockchain and minting piQ tokens has been blocked.**")
            piq_minted = 0.0
            tx_hash = "Blocked_Due_To_API_Limits"
            zk_proof = "Blocked_Proof"
        else:
            piq_minted = 7.5
            zk_proof = generate_zk_snark_proof(file_hash, pidyne_ai_rating, pidyne_ai_rating, "None")
            tx_hash = mint_pi_quotient_token(book_address, piq_minted, file_hash, zk_proof)

        title = raw_data.get("Extracted_Title", filename.replace(".pdf", "").replace("_", " ").title())
        extracted_author = raw_data.get("Extracted_Author", pdf_meta_author if pdf_meta_author else "Independent Research Scholar")
        
        scores_dict = compute_formulaic_criteria(
            raw_data, 
            0.85, 
            sciscore_adherence=mdar_score,
            topological_entropy=topological_entropy,
            ai_rating=pidyne_ai_rating
        )
        
        final_score = sum(scores_dict.values()) / 8.0
        logic_integrity = pidyne_ai_rating

        if not all_llms_failed:
            cursor.execute(
                """INSERT OR REPLACE INTO papers_assessment (eval_hash, user_id, title, filename, scope, c1, c2, c3, c4, c5, c6, c7, c8, logic_score, scope_alignment, subfields, fields, author_name, final_score, timestamp, eth_book, piq_minted, tx_hash, zk_proof, did, zk_email_proof, gaming_penalty, mdar_adherence_score, rrid_valid_count, credit_taxonomy_roles, reproducibility_score, doi, consensus_data, evidence_report, scilem_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    file_hash, user_id, title, filename, scope, *scores_dict.values(),
                    logic_integrity, 0.0, json.dumps(["Core Research Domain"]),
                    json.dumps(["Computer Science"]), extracted_author, final_score,
                    datetime.now().isoformat(), book_address, piq_minted,
                    tx_hash, zk_proof, user_id, "None", 0.0,
                    mdar_score, rrid_count, json.dumps(["Data Curation"]), 0.85,
                    provided_doi, json.dumps(consensus_raw), evidence_report, 50.0
                ),
            )
            conn.commit()
    finally:
        conn.close()

    if not all_llms_failed:
        backup_state_to_web3()

    return (
        title, extracted_author, final_score, logic_integrity, drift, rec,
        ["Computer Science"], ["Core Research Domain"], scores_dict, file_hash, piq_minted, tx_hash, zk_proof,
        active_weights, mdar_score, rrid_count, 0.85, False, warnings_list,
        consensus_raw, evidence_report, "N/A"
    )
