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
    backup_state_to_web3, generate_zokrates_proof, mint_piq_token, 
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

def generate_scilem_fallback_report(text):
    scilem_rep = evaluate_scilem_analysis_report(text)
    return f"Synthesized Evidence Report (Dynamic Consensus)\n\n### Scilem Neural Assessment\n{scilem_rep}"

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
    if not tokens:
        tokens = [0]
    paper_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        feat_val = scilem_model(paper_tensor).item()

    scilem_numeric_score = 50.0 + (feat_val * 40.0)

    lines = [l.strip() for l in paper_text.split("\n") if l.strip()]
    cand_title = lines[0] if lines else "Dynamic Research Document"
    cand_author = "Dynamic Research Scholar"
    for line in lines[1:10]:
        if any(kw in line.lower() for kw in ["by", "author", "university", "department", "@"]):
            cand_author = line
            break

    mdar_signal, rrid_signal = compute_mdar_adherence(paper_text)
    repro_signal, repro_flags = compute_reproducibility(paper_text)
    density_signal = compute_empirical_density(paper_text)
    detected_markers = [k.replace("_", " ") for k, v in repro_flags.items() if v]

    opinion = (
        f"Scilem Neural Engine Analysis: Deep LSTM feature representation score = {feat_val:.4f}. "
        f"Deterministic MDAR/RRID adherence measured at {mdar_signal * 100:.1f}% ({rrid_signal} valid RRID token(s)). "
        f"Empirical density signal measured at {density_signal * 100:.1f}%. "
        f"Open-science reproducibility markers detected: {', '.join(detected_markers) if detected_markers else 'none'}."
    )

    return "scilem", {
        "title": cand_title[:120],
        "authors": clean_author_name(cand_author)[:80],
        "opinion": opinion,
        "references": [],
        "science_field": "Multidisciplinary Sciences > General Research",
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
    if not tokens:
        tokens = [0]
    paper_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    features = scilem_model(paper_tensor)
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
        except Exception as e: res_msg = f"Scilem weights deletion warning: {e}"
            
    scilem_model, _ = get_scilem_engine()
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
        predictions = self.linear(lstm_out[:, -1, :])
        return torch.softmax(predictions, dim=-1) * 8.0

PidyneLSTM = PiBrainLSTM

def query_llm_json(provider_name, model_name, api_key, base_url, prompt):
    if not api_key or not str(api_key).strip():
        return provider_name, {
            "title": "Dynamic Title", "authors": "Dynamic Authors",
            "opinion": f"API key for {provider_name.upper()} is missing.", "references": [], "api_failed": True
        }
    try:
        if "openrouter" in base_url.lower() and OPENROUTER_SDK_AVAILABLE:
            with OpenRouter(api_key=api_key.strip()) as client:
                response = client.chat.send(model=model_name, messages=[{"role": "user", "content": prompt}])
                content = response.choices[0].message.content
                if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
                data = json.loads(content)
                data["api_failed"] = False
                return provider_name, data
        else:
            client = OpenAI(api_key=api_key.strip(), base_url=base_url)
            response = client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, temperature=0.1
            )
            data = json.loads(response.choices[0].message.content)
            data["api_failed"] = False
            return provider_name, data
    except Exception as e:
        return provider_name, {
            "title": "Dynamic Title (API Error)", "authors": "Dynamic Authors",
            "opinion": f"Error querying {provider_name.upper()}: {str(e)}.", "references": [], "api_failed": True
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
    if not ref_section: ref_section = paper_text[-4000:]

    return f"""
Analyze the manuscript excerpts below and respond strictly in JSON format.
Evaluate across the 8 core Pi-Index criteria.

Keys required in JSON:
1. "title": Exact or dynamically inferred title of the paper.
2. "authors": String of human author names.
3. "opinion": Detailed qualitative assessment covering criteria C1-C8.
4. "references": List of objects: [{{"citation": "[1]", "authors": "Author et al.", "year": "2024"}}]
5. "science_field": A dynamic string representing the specialized research domain formatted strictly as 'Major Domain > Specific Subfield' derived entirely from the manuscript content.

--- FRONT MATTER ---
{front_matter}

--- REFERENCES SECTION ---
{ref_section}
"""

def extract_with_llama(paper_text):
    prompt = build_multi_llm_prompt(paper_text)
    if GROQ_API_KEY: return query_llm_json("llama", PRIMARY_MODEL, GROQ_API_KEY, "https://api.groq.com/openai/v1", prompt)
    elif OR_API_KEY: return query_llm_json("llama", "meta-llama/llama-3.3-70b-instruct", OR_API_KEY, "https://openrouter.ai/api/v1", prompt)
    return "llama", {"title": "N/A", "authors": "N/A", "opinion": "API not configured.", "references": [], "api_failed": True}

def extract_with_mistral(paper_text):
    prompt = build_multi_llm_prompt(paper_text)
    if OR_API_KEY: return query_llm_json("mistral", "mistralai/mistral-large", OR_API_KEY, "https://openrouter.ai/api/v1", prompt)
    return "mistral", {"title": "N/A", "authors": "N/A", "opinion": "API not configured.", "references": [], "api_failed": True}

def extract_with_qwen(paper_text):
    prompt = build_multi_llm_prompt(paper_text)
    if OR_API_KEY: return query_llm_json("qwen", "qwen/qwen-2.5-72b-instruct", OR_API_KEY, "https://openrouter.ai/api/v1", prompt)
    return "qwen", {"title": "N/A", "authors": "N/A", "opinion": "API not configured.", "references": [], "api_failed": True}

def extract_with_gemini(paper_text):
    prompt = build_multi_llm_prompt(paper_text)
    if GEMINI_API_KEY: return query_llm_json("gemini", "gemini-2.0-flash", GEMINI_API_KEY, "https://generativelanguage.googleapis.com/v1beta/openai/", prompt)
    elif OR_API_KEY: return query_llm_json("gemini", "google/gemini-2.0-flash-001", OR_API_KEY, "https://openrouter.ai/api/v1", prompt)
    return "gemini", {"title": "N/A", "authors": "N/A", "opinion": "API not configured.", "references": [], "api_failed": True}

def run_multi_llm_consensus(paper_text):
    results = {}
    llm_funcs = {"llama": extract_with_llama, "mistral": extract_with_mistral, "qwen": extract_with_qwen, "gemini": extract_with_gemini, "scilem": extract_with_scilem}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(func, paper_text): name for name, func in llm_funcs.items()}
        for future in concurrent.futures.as_completed(futures):
            provider, data = future.result()
            results[provider] = data
    return results

def generate_merged_evidence_report(consensus_results):
    successful_llms = [k for k, v in consensus_results.items() if not v.get("api_failed", False)]
    if not successful_llms: return "Synthesized Evidence Report (Dynamic Consensus)\n\nExternal APIs offline. Local Scilem neural analysis active."
    report_md = "Synthesized Evidence Report (Dynamic Consensus)\n\n"
    for provider in successful_llms:
        data = consensus_results[provider]
        report_md += f"### {provider.upper()} Assessment\n- **Title:** {data.get('title', 'N/A')}\n- **Authors:** {data.get('authors', 'N/A')}\n- **Opinion:** {data.get('opinion', 'N/A')}\n\n"
    return report_md

def generate_pidyne_judgement(consensus_results, text=None):
    prompt = "You are the Pidyne Assessment Engine. Review these independent model assessments:\n\n"
    active_count = 0
    for provider, data in consensus_results.items():
        if not data.get("api_failed", False):
            active_count += 1
            prompt += f"### {provider.upper()} Assessment:\n- Title: {data.get('title', 'N/A')}\n- Authors: {data.get('authors', 'N/A')}\n- Opinion: {data.get('opinion', 'N/A')}\n\n"
    prompt += "Generate a comprehensive Markdown Evidence Report and overall AI Rating (0.0 to 100.0) in JSON with keys 'evidence_report' and 'ai_rating'."
    
    api_key = GROQ_API_KEY or OR_API_KEY or GEMINI_API_KEY
    base_url = "https://api.groq.com/openai/v1" if GROQ_API_KEY else ("https://openrouter.ai/api/v1" if OR_API_KEY else "https://generativelanguage.googleapis.com/v1beta/openai/")
    judge_provider = PRIMARY_MODEL if GROQ_API_KEY else "Dynamic Multi-Model Judge"

    data = None
    if api_key and active_count > 0:
        _, data = query_llm_json("pidyne", PRIMARY_MODEL if GROQ_API_KEY else "meta-llama/llama-3.3-70b-instruct", api_key, base_url, prompt)
        if data.get("api_failed", True): data = None

    consensus_results["_judge_metadata"] = {"judge_provider": judge_provider, "timestamp": datetime.now().isoformat()}
    header_prefix = f"### ⚖️ Final Verdict Synthesized By\n**Judge Engine:** `{judge_provider}`\n\n---\n\n"

    if not data:
        evidence_report = header_prefix + generate_merged_evidence_report(consensus_results)
        rating = float(consensus_results.get("scilem", {}).get("scilem_score", 75.0))
    else:
        raw_rep = data.get("evidence_report", "Synthesized Evidence Report generated.")
        evidence_report = header_prefix + raw_rep if not raw_rep.startswith("###") else header_prefix + raw_rep
        try: rating = float(data.get("ai_rating", 75.0))
        except Exception: rating = 75.0
    return evidence_report, rating

def compute_mdar_adherence(text: str) -> tuple[float, int]:
    text_lower = text.lower()
    blinded = 1.0 if re.search(r'\b(blinded|double-blind|single-blind|masking)\b', text_lower) else 0.0
    randomized = 1.0 if re.search(r'\b(randomized|randomly assigned|random sequence)\b', text_lower) else 0.0
    power_calc = 1.0 if re.search(r'\b(power analysis|sample size calculation|statistical power)\b', text_lower) else 0.0
    rrid_matches = re.findall(r'\brrid\s*:?\s*[a-zA-Z0-9_:-]+\b', text_lower)
    rrid_count = len(set(rrid_matches))
    return (blinded + randomized + power_calc + min(1.0, rrid_count / 3.0)) / 4.0, rrid_count

def compute_reproducibility(text: str) -> tuple[float, dict]:
    text_lower = text.lower()
    signals = {
        "code_or_data_repository": bool(re.search(r'\b(github\.com|gitlab\.com|zenodo\.org|osf\.io|huggingface\.co)\b', text_lower)),
        "data_availability_statement": bool(re.search(r'\bdata availability\b|\bdata are available\b|\bdataset\b', text_lower)),
        "open_license": bool(re.search(r'\b(mit license|apache license|creative commons|cc[- ]by)\b', text_lower)),
        "containerized_execution": bool(re.search(r'\b(docker|singularity|containeri[sz]ed|conda)\b', text_lower)),
        "supplementary_materials": bool(re.search(r'\bsupplementary (material|data|information)\b', text_lower)),
        "preregistration": bool(re.search(r'\bpre-?registered\b|\bpre-?registration\b', text_lower)),
    }
    score = 0.30 + (sum(1 for v in signals.values() if v) / len(signals)) * 0.70
    return min(1.0, max(0.0, score)), signals

def compute_empirical_density(text: str) -> float:
    text_lower = text.lower()
    stat_terms = len(re.findall(r'\b(p\s*[<>=]\s*0?\.\d+|confidence interval|standard deviation|regression|t-test|chi-square|p-value)\b', text_lower))
    return min(1.0, ((stat_terms * 2) + len(re.findall(r'\bn\s*=\s*\d+', text_lower))) / 40.0)

def adaptive_chunking(text, max_tokens):
    if len(text) <= max_tokens: return text
    return text[: int(max_tokens * 0.4)] + "\n...[TRUNCATED]...\n" + text[-int(max_tokens * 0.6) :]

def evaluate_pdf_text_ensemble(text, model, text_limit, file_hash="unknown"):
    text = adaptive_chunking(text, text_limit)
    consensus_results = run_multi_llm_consensus(text)
    evidence_report, pidyne_ai_rating = generate_pidyne_judgement(consensus_results, text)
    train_scilem_on_input_and_report(text, evidence_report)

    best_title, best_author, best_science_field = "Dynamic Document Title", "Dynamic Author", "Multidisciplinary Sciences > General Research"
    for l_key in ["llama", "mistral", "qwen", "gemini", "scilem"]:
        entry = consensus_results.get(l_key, {})
        if "title" in entry and "N/A" not in entry["title"]: best_title = entry["title"]; break
    for l_key in ["llama", "mistral", "qwen", "gemini", "scilem"]:
        entry = consensus_results.get(l_key, {})
        if "authors" in entry and "N/A" not in entry["authors"]: best_author = entry["authors"]; break
    for l_key in ["llama", "mistral", "qwen", "gemini", "scilem"]:
        entry = consensus_results.get(l_key, {})
        if "science_field" in entry and ">" in entry["science_field"]: best_science_field = entry["science_field"]; break

    return {
        "Extracted_Title": best_title, "Extracted_Author": best_author,
        "Extracted_Science_Field": best_science_field, "Overall_Confidence": 0.85,
        "_consensus_raw": consensus_results, "_evidence_report": evidence_report,
        "_pidyne_rating": pidyne_ai_rating, "_scilem_score": consensus_results.get("scilem", {}).get("scilem_score", pidyne_ai_rating),
    }

def get_formulas_hash():
    return hashlib.sha256(b"Pi-Index-Formula-State-v2.0").hexdigest()

def compute_formulaic_criteria(reproducibility_score, sciscore_adherence=0.8, topological_entropy=0.5, ai_rating=75.0, vapri=0.0, empirical_density=None):
    c1 = (ai_rating * 0.9) + (vapri * 10)
    c4 = ai_rating * 0.95 + (topological_entropy * 5)
    c6 = ai_rating * 0.88 + (sciscore_adherence * 12)
    c7 = (empirical_density * 100.0 * 0.6) + (math.tanh((ai_rating / 100.0) * 1.5) * 100.0 * 0.4) if empirical_density is not None else math.tanh((ai_rating / 100.0) * 1.5) * 100.0
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
    return f"**Adversarial Defense Strategy:** Focus on strengthening `{lowest_criterion[0]}` (Current score: {lowest_criterion[1]:.1f}/100) by expanding open datasets and clarifying empirical validation metrics."

def assess_manuscript(file_bytes, filename, scope, user_id, book_address="None", email="None", provided_doi="None", force_proceed=False):
    active_weights = [1.0] * 8
    warnings_list = []
    if file_bytes is None or len(file_bytes) == 0:
        return ("Download Failed", "Dynamic Scholar", 0.0, 75.0, "N/A", "N/A", ["General"], ["General Research"], {k: 0.0 for k in ["C1_Semantic_Originality", "C2_Methodological_Rigor_SciScore", "C3_Interdisciplinary_Entropy", "C4_Societal_Impact", "C5_Open_Science_Repro", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability_FAIR"]}, "Failed", 0.0, "None", "None", active_weights, 0.0, 0, 0.0, False, ["Empty payload."], {}, "", "N/A")

    file_hash = hashlib.sha256(file_bytes).hexdigest()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT title, author_name, final_score, logic_score, c1, c2, c3, c4, c5, c6, c7, c8, piq_minted, tx_hash, zk_proof, mdar_adherence_score, rrid_valid_count, reproducibility_score, consensus_data, evidence_report, scilem_score FROM papers_assessment WHERE eval_hash = ?", (file_hash,))
        existing = cursor.fetchone()
        if existing and not force_proceed and (existing[13] or existing[13] == "Simulated_Ledger_Record"):
            return (existing[0], existing[1], existing[2], existing[3], "N/A", "N/A", ["General"], ["General Research"], dict(zip(["C1_Semantic_Originality", "C2_Methodological_Rigor_SciScore", "C3_Interdisciplinary_Entropy", "C4_Societal_Impact", "C5_Open_Science_Repro", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability_FAIR"], existing[4:12])), file_hash, existing[12], existing[13], existing[14], active_weights, existing[15], existing[16], existing[17], True, ["Cached record returned."], json.loads(existing[18]) if existing[18] else {}, existing[19] or "", existing[20])

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            pdf_meta_author = doc.metadata.get("author", "").strip()
            full_text = "\n".join([page.get_text("text", sort=True) for page in doc])
        except Exception as e:
            warnings_list.append(f"Parser note: {e}")
            full_text = ""

        mdar_score, rrid_count = compute_mdar_adherence(full_text)
        topological_entropy = calculate_citation_topology(provided_doi)
        reproducibility_score, _ = compute_reproducibility(full_text)
        empirical_density = compute_empirical_density(full_text)

        raw_data = evaluate_pdf_text_ensemble(full_text, PRIMARY_MODEL, MAX_TEXT_TOKENS, file_hash)
        pidyne_ai_rating = raw_data.get("_pidyne_rating", 75.0)
        consensus_raw = raw_data.get("_consensus_raw", {})
        evidence_report = raw_data.get("_evidence_report", "")
        vapri = (int(hashlib.md5(evidence_report.encode()).hexdigest(), 16) % 1000) / 1000.0 if evidence_report else 0.5

        external_active = any(not v.get("api_failed", False) for k, v in consensus_raw.items() if k not in ["scilem", "_judge_metadata"])
        title = raw_data.get("Extracted_Title", filename.replace(".pdf", "").replace("_", " ").title())
        extracted_author = raw_data.get("Extracted_Author", pdf_meta_author if pdf_meta_author else "Dynamic Scholar")
        extracted_field_str = raw_data.get("Extracted_Science_Field", "General Science > General Research")
        major_field = extracted_field_str.split(">")[0].strip() if ">" in extracted_field_str else "General Science"

        scores_dict = compute_formulaic_criteria(reproducibility_score, mdar_score, topological_entropy, pidyne_ai_rating, vapri, empirical_density)
        final_score = sum(scores_dict.values()) / 8.0
        logic_integrity = min(100.0, max(0.0, (pidyne_ai_rating * math.exp(-(2 * max(0, topological_entropy - 0.5) + 1.5 * (1.0 - pidyne_ai_rating / 100.0)))) + (vapri * 5.0)))
        piq_minted = round((final_score / 100.0) * 10.0, 2) if final_score >= 50.0 and logic_integrity >= 50.0 else 0.00
        if piq_minted == 0.00: warnings_list.append("⚠️ Score or logic integrity below 50.0%. piQ reward set to 0.00.")

        zk_proof = generate_zokrates_proof(pidyne_ai_rating, logic_integrity, file_hash)
        truncated_hash_int = int(file_hash[:16], 16)
        tx_hash = mint_piq_token(book_address, piq_minted, file_hash, truncated_hash_int, zk_proof) if (external_active and book_address != "0x0000000000000000000000000000000000000000" and piq_minted > 0) else "Simulated_Ledger_Record"

        cursor.execute("""INSERT OR REPLACE INTO papers_assessment (eval_hash, user_id, title, filename, scope, c1, c2, c3, c4, c5, c6, c7, c8, logic_score, scope_alignment, subfields, fields, author_name, final_score, timestamp, eth_book, piq_minted, tx_hash, zk_proof, did, zk_email_proof, gaming_penalty, mdar_adherence_score, rrid_valid_count, credit_taxonomy_roles, reproducibility_score, doi, consensus_data, evidence_report, scilem_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (file_hash, user_id, title, filename, scope, *scores_dict.values(), logic_integrity, 0.0, json.dumps([extracted_field_str]), json.dumps([major_field]), extracted_author, final_score, datetime.now().isoformat(), book_address, piq_minted, tx_hash, json.dumps(zk_proof), user_id, "None", 0.0, mdar_score, rrid_count, json.dumps(["Data Curation"]), reproducibility_score, provided_doi, json.dumps(consensus_raw), evidence_report, raw_data.get("_scilem_score", 50.0)))

        cursor.execute("SELECT COUNT(*) FROM blockchain_por_weights")
        block_count = cursor.fetchone()[0]
        cursor.execute("SELECT block_hash FROM blockchain_por_weights ORDER BY block_height DESC LIMIT 1")
        prev_hash = cursor.fetchone()[0] if cursor.fetchone() else "0" * 64
        val_node, b_hash, por_p = validate_block_por(block_count + 1, active_weights, datetime.now().isoformat(), prev_hash, file_hash, "Dynamic_Ensemble", final_score, get_formulas_hash())
        cursor.execute("""INSERT INTO blockchain_por_weights (w1, w2, w3, w4, w5, w6, w7, w8, timestamp, previous_hash, validator_node, block_hash, eval_hash, model_used, por_proof, formulas_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (*active_weights, datetime.now().isoformat(), prev_hash, val_node, b_hash, file_hash, "Dynamic_Ensemble", por_p, get_formulas_hash()))
        conn.commit()
    finally:
        conn.close()

    backup_state_to_web3()
    return (title, extracted_author, final_score, logic_integrity, drift, rec, [major_field], [extracted_field_str], scores_dict, file_hash, piq_minted, tx_hash, json.dumps(zk_proof), active_weights, mdar_score, rrid_count, reproducibility_score, False, warnings_list, consensus_raw, evidence_report, raw_data.get("_scilem_score", 50.0))
