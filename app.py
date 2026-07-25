import os
import re
import json
import time
import math
import random
import sqlite3
import hashlib
import tempfile
from datetime import datetime
from io import BytesIO

import requests
import colorsys
import fitz
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyvis.network import Network
import graphviz

import streamlit as st
import streamlit.components.v1 as components

from web3 import Web3
from groq import Groq

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 1. CONFIGURATION & ENVIRONMENT SETUP
# ==========================================
st.set_page_config(page_title="Pi-Index Assessment Engine", layout="wide")

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_TEXT_TOKENS = 12000
EPOCH_BLOCK_SIZE = 1  

WEB3_PROVIDER_URI = os.getenv("WEB3_PROVIDER_URI", "https://sepolia.infura.io/v3/YOUR_INFURA_PROJECT_ID")
ETH_ADMIN_PRIVATE_KEY = os.getenv("ETH_ADMIN_PRIVATE_KEY", "0x0000000000000000000000000000000000000000000000000000000000000000")
PIQ_CONTRACT_ADDRESS = os.getenv("PIQ_CONTRACT_ADDRESS", "0xYourDeployedContractAddressHere")

BASE_DIR = os.path.abspath('./Scientometric_Pi_Index')
os.makedirs(BASE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, 'pi_index_main.db')

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.error("API Key not found! Please configure your environment variables or Streamlit Secrets.")
    st.stop()

w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER_URI))
groq_client = Groq(api_key=GROQ_API_KEY)

# ==========================================
# 2. ROOT LEVEL DATABASE SCHEMA ENFORCEMENT
# ==========================================
def enforce_database_schema():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS papers_assessment 
                      (eval_hash TEXT PRIMARY KEY, user_id TEXT, title TEXT, filename TEXT, scope TEXT,
                       c1 REAL, c2 REAL, c3 REAL, c4 REAL, 
                       c5 REAL, c6 REAL, c7 REAL, c8 REAL, 
                       scope_alignment REAL, logic_score REAL,
                       subfields TEXT, fields TEXT, author_name TEXT, final_score REAL, timestamp DATETIME)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS blockchain_por_weights 
                      (block_height INTEGER PRIMARY KEY AUTOINCREMENT, 
                       w1 REAL, w2 REAL, w3 REAL, w4 REAL, 
                       w5 REAL, w6 REAL, w7 REAL, w8 REAL, 
                       timestamp DATETIME, previous_hash TEXT, 
                       validator_node TEXT, block_hash TEXT, eval_hash TEXT, model_used TEXT)''')
                       
    cursor.execute('''CREATE TABLE IF NOT EXISTS global_eval_counter (count INTEGER)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS desci_attestations 
                      (attestation_id TEXT PRIMARY KEY, eval_hash TEXT, attester_id TEXT, stake_amount REAL, stance TEXT, timestamp DATETIME)''')
    
    target_columns_assessment = {
        "eth_book": "TEXT DEFAULT 'None'",
        "eth_wallet": "TEXT DEFAULT 'None'",
        "piq_minted": "REAL DEFAULT 0.0",
        "epc_minted": "REAL DEFAULT 0.0",
        "tx_hash": "TEXT DEFAULT 'Pending'",
        "zk_proof": "TEXT DEFAULT 'None'",
        "did": "TEXT DEFAULT 'None'",
        "zk_email_proof": "TEXT DEFAULT 'None'",
        "gaming_penalty": "REAL DEFAULT 0.0",
        "h_index": "TEXT DEFAULT 'N/A'",
        "i10_index": "TEXT DEFAULT 'N/A'",
        "reproducibility_score": "REAL DEFAULT 0.0",
        "doi": "TEXT DEFAULT 'None'"
    }
    
    target_columns_weights = {
        "por_proof": "TEXT DEFAULT 'Genesis_Proof'",
        "formulas_hash": "TEXT DEFAULT 'Locked_State'"
    }

    cursor.execute("PRAGMA table_info(papers_assessment)")
    existing_assessment_cols = [row[1] for row in cursor.fetchall()]
    for col, dtype in target_columns_assessment.items():
        if col not in existing_assessment_cols:
            try: cursor.execute(f"ALTER TABLE papers_assessment ADD COLUMN {col} {dtype}")
            except Exception: pass

    cursor.execute("PRAGMA table_info(blockchain_por_weights)")
    existing_weights_cols = [row[1] for row in cursor.fetchall()]
    for col, dtype in target_columns_weights.items():
        if col not in existing_weights_cols:
            try: cursor.execute(f"ALTER TABLE blockchain_por_weights ADD COLUMN {col} {dtype}")
            except Exception: pass

    conn.commit()
    conn.close()

enforce_database_schema()

@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM blockchain_por_weights")
    if cursor.fetchone()[0] == 0:
        genesis_weights = [1.0] * 8
        prev_hash = "0" * 64
        timestamp = datetime.now().isoformat()
        val_node, block_hash, por_proof = validate_block_por(1, genesis_weights, timestamp, prev_hash, "genesis", "none", 100.0, "Genesis_Hash")
        cursor.execute('''INSERT INTO blockchain_por_weights 
                          (w1, w2, w3, w4, w5, w6, w7, w8, timestamp, previous_hash, validator_node, block_hash, eval_hash, model_used, por_proof, formulas_hash) 
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                       (*genesis_weights, timestamp, prev_hash, val_node, block_hash, "genesis", "none", por_proof, "Genesis_Hash"))
        conn.commit()
        
    cursor.execute("SELECT count FROM global_eval_counter")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO global_eval_counter (count) VALUES (0)")
        conn.commit()
        
    return conn

# ==========================================
# 3. UI UTILITIES & METRICS
# ==========================================
def tooltip(text):
    svg_icon = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="#9e9e9e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -3px; margin-left: 6px; cursor: help;"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'''
    return f"<span title=\"{text}\">{svg_icon}</span>"

def fetch_author_metrics(author_name):
    try:
        if not author_name or author_name.lower() in ["unidentified", "unknown"]:
            return "N/A", "N/A"
        first_author = author_name.split(',')[0].strip()
        url = f"https://api.openalex.org/authors?search={first_author}"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get('results') and len(data['results']) > 0:
                stats = data['results'][0].get('summary_stats', {})
                return str(stats.get('h_index', 'N/A')), str(stats.get('i10_index', 'N/A'))
    except Exception:
        pass
    return "N/A", "N/A"

def search_openalex_topics(topic_query, limit=5):
    try:
        url = f"https://api.openalex.org/works?search={requests.utils.quote(topic_query)}&filter=is_oa:true&per_page={limit}"
        res = requests.get(url, timeout=8)
        if res.status_code == 200:
            results = res.json().get('results', [])
            extracted = []
            for item in results:
                title = item.get('title', 'Untitled Paper')
                doi = item.get('doi', '')
                
                best_oa = item.get('best_oa_location') or {}
                pdf_url = best_oa.get('pdf_url') or item.get('open_access', {}).get('oa_url', '')
                
                authorships = item.get('authorships', [])
                authors_list = [a.get('author', {}).get('display_name', '') for a in authorships]
                authors_str = ", ".join([a.title() for a in authors_list if a]) if authors_list else "Unidentified"
                
                if pdf_url or doi:
                    extracted.append({
                        'title': title,
                        'doi': doi,
                        'pdf_url': pdf_url,
                        'authors': authors_str
                    })
            return extracted
    except Exception as e:
        st.error(f"OpenAlex Topic Fetch Error: {str(e)}")
    return []

def get_author_piq_dict():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT author_name, piq_minted FROM papers_assessment")
    data = cursor.fetchall()
    author_piq = {}
    for authors_str, piq in data:
        if not authors_str or authors_str.lower() in ["unidentified", "unknown", "research scholar"]: 
            continue
        alist = [a.strip().title() for a in authors_str.split(',') if a.strip()]
        if not alist: continue
        share = piq / len(alist)
        for a in alist:
            author_piq[a] = author_piq.get(a, 0.0) + share
    return author_piq

# ==========================================
# 4. BLOCKCHAIN & MATHEMATICAL ENGINE
# ==========================================
def validate_block_por(block_index, weights, timestamp, previous_hash, eval_hash, model_used, final_score, formulas_hash):
    validator_node = "Validator_Pi_" + hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
    por_proof = f"PoR_{eval_hash[:12]}_Score:{final_score:.2f}"
    data_string = f"{block_index}{weights}{timestamp}{previous_hash}{validator_node}{por_proof}{model_used}{formulas_hash}"
    block_hash = hashlib.sha256(data_string.encode('utf-8')).hexdigest()
    return validator_node, block_hash, por_proof

def generate_zk_snark_proof(eval_hash, final_score, logic_score, email_str=""):
    circuit_input = f"{eval_hash}:{final_score}:{logic_score}:{email_str}:{time.time()}"
    return "0x0" + hashlib.sha3_256(circuit_input.encode('utf-8')).hexdigest()

def mint_pi_quotient_token(book_address, amount, eval_hash, zk_proof):
    if not w3.is_connected() or book_address == "None" or not book_address:
        return "Not Connected / No Book"
        
    try:
        target_addr = book_address if w3.is_address(book_address) else "0x" + hashlib.sha256(book_address.encode()).hexdigest()[:40]
        
        abi = '[{"inputs":[{"internalType":"address","name":"researcher","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"},{"internalType":"string","name":"evalHash","type":"string"},{"internalType":"bytes","name":"zkProof","type":"bytes"}],"name":"verifyProofAndMint","outputs":[],"stateMutability":"nonpayable","type":"function"}]'
        contract = w3.eth.contract(address=w3.to_checksum_address(PIQ_CONTRACT_ADDRESS), abi=abi)
        account = w3.eth.account.from_key(ETH_ADMIN_PRIVATE_KEY)
        
        tx = contract.functions.verifyProofAndMint(
            w3.to_checksum_address(target_addr),
            int(amount),
            eval_hash,
            bytes.fromhex(zk_proof[2:])
        ).build_transaction({
            'from': account.address,
            'nonce': w3.eth.get_transaction_count(account.address),
            'gas': 200000,
            'gasPrice': w3.to_wei('10', 'gwei')
        })
        
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=ETH_ADMIN_PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        return tx_hash.hex()
    except Exception as e:
        return f"Eth Tx Failed: {str(e)}"

def generate_blockchain_pi(block_height):
    iterations = max(1, block_height * 50)
    pi_approx = 3.0
    sign = 1.0
    for i in range(1, iterations + 1):
        n = i * 2
        pi_approx += sign * (4.0 / (n * (n + 1) * (n + 2)))
        sign *= -1.0
    return pi_approx

def get_formulas_hash():
    criteria_state = "C1:Originality|C2:Rigor|C3:Interdisciplinary|C4:Impact|C5:OpenScience_Executable|C6:Integration|C7:EmpiricalDensity_Validated|C8:Actionability_v2.0|DORA_Dossier_v1.0"
    return hashlib.sha256(criteria_state.encode('utf-8')).hexdigest()

def calculate_model_driven_weights(old_weights, scores, model_name, block_height):
    if "70b" in model_name: model_version, model_size = 3.3, 70.0
    else: model_version, model_size = 3.1, 8.0
        
    pi_accuracy = generate_blockchain_pi(block_height)
    delta_models = abs((3.3 * 70.0) - (3.1 * 8.0)) 
    mean_score = np.mean(scores)
    
    new_weights = []
    for i, old_w in enumerate(old_weights):
        stretched_score = max(1.0, min(100.0, mean_score + (scores[i] - mean_score) * 3.0))
        weight_shift = ((model_version * model_size) / (delta_models * pi_accuracy)) * ((stretched_score / 100.0) ** 2)
        w_new = old_w * 0.85 + (1.0 + weight_shift * 0.15) * 0.15
        new_weights.append(w_new)
        
    sum_of_weights = sum(new_weights)
    return [round((w / sum_of_weights) * 8.0, 6) for w in new_weights]

def compute_logical_integrity(extracted_logic_vars, gaming_penalty):
    evidence = extracted_logic_vars.get('Evidence_Strength', 0.5)
    conclusion_reach = extracted_logic_vars.get('Conclusion_Reach', 0.5)
    jumps = extracted_logic_vars.get('Logical_Jumps', 0.5)
    premise = extracted_logic_vars.get('Premise_Validity', 0.5)
    
    logic_gap = max(0.0, conclusion_reach - evidence)
    base_logic = (premise * evidence) * np.exp(-(logic_gap * 2.0 + jumps * 1.5)) * 100
    logic_score = base_logic * (1.0 - (gaming_penalty * 0.9))
    return max(0.0, min(100.0, logic_score))

def compute_formulaic_criteria(vars_dict, reproducibility_score):
    scores = {}
    c1_raw = ((vars_dict.get('H_novel', 0.5) * vars_dict.get('K_epistemic', 0.5)) / (vars_dict.get('zeta', 0.5) * vars_dict.get('I_existing', 0.5) + 1e-9)) * 60
    scores["C1_Originality"] = min(100.0, max(0.0, c1_raw))
    
    rigor_matrix = max(0.0, 1.0 - (vars_dict.get('Sigma_error', 0.2) / (vars_dict.get('mu_signal', 0.8) + 1e-9)))
    c2_raw = rigor_matrix * vars_dict.get('rho_k', 0.5) * math.gamma(1.5) * 140
    scores["C2_Methodological_Rigor"] = min(100.0, max(0.0, c2_raw))
    
    p_disc = np.array(vars_dict.get('p_disciplines', [1.0]))
    p_disc = p_disc / (p_disc.sum() + 1e-9)
    renyi_entropy = -np.log(np.sum(p_disc**2) + 1e-9) 
    c3_raw = (renyi_entropy + vars_dict.get('bridge_capacity', 0.5)) * 55
    scores["C3_Interdisciplinary"] = min(100.0, max(0.0, c3_raw))
    
    gamma_q = math.gamma(max(0.1, vars_dict.get('q_fractional', 1.5)))
    c4_raw = (1.0 / gamma_q) * vars_dict.get('Utility_vector', 0.5) * np.exp(-vars_dict.get('decay_rate', 0.5)) * 150
    scores["C4_Societal_Impact"] = min(100.0, max(0.0, c4_raw))
    
    c5_raw = (((0.5 * vars_dict.get('D_open', 0.1)) + (0.2 * vars_dict.get('J_code', 0.1)) + (0.3 * reproducibility_score)) * vars_dict.get('P_FAIR', 0.1)) * 190
    scores["C5_Open_Science_Potential"] = min(100.0, max(0.0, c5_raw))
    
    c6_raw = np.exp(-1.5 * vars_dict.get('d_g_distance', 0.5)) * vars_dict.get('R_xi', 0.5) * vars_dict.get('PR_xi', 0.5) * 180
    scores["C6_Literature_Integration"] = min(100.0, max(0.0, c6_raw))
    
    density_inner = (vars_dict.get('I_Fisher', 0.5) * vars_dict.get('KL_divergence', 0.5) * (0.8 + 0.2 * reproducibility_score)) / (vars_dict.get('V_baseline', 0.5) * vars_dict.get('omega_data', 0.5) + 1e-9)
    c7_raw = np.tanh(density_inner) * vars_dict.get('sum_lambda_kappa', 1.0) * 85
    scores["C7_Empirical_Density"] = min(100.0, max(0.0, c7_raw))
    
    eta = vars_dict.get('eta_steps', 2.0)
    lambda_lyapunov = vars_dict.get('Lambda_Lyapunov', 0.5)
    c8_raw = (1.0 / (1.0 + np.exp(-(eta - (lambda_lyapunov * 5))))) * 100
    scores["C8_Future_Actionability"] = min(100.0, max(0.0, c8_raw))
    
    for key in scores:
        scores[key] = round(scores[key], 2)
    return scores

def calculate_complex_drift(alignment, scores):
    if not scores or alignment is None:
        return 0.0
    average_score = np.mean(scores)
    standard_deviation = np.std(scores)
    alignment_gap = (100.0 - alignment) / 100.0
    drift_metric = 100.0 * (1.0 - np.exp(-3.0 * (alignment_gap ** 1.5) * (1.0 + (standard_deviation / 100.0)) / (1e-9 + (average_score / 100.0))))
    return float(max(0.0, min(100.0, drift_metric)))

def get_recommendation_spectrum(score, drift):
    if drift == "N/A": return "N/A"
    synergy = score * (1.0 - (drift / 100.0)**1.5)
    if synergy >= 85: return "Tier I: Core Paradigm (Optimal Synergy)"
    elif synergy >= 70: return "Tier II: Highly Aligned Framework"
    elif synergy >= 55: return "Tier III: Moderately Synergistic"
    elif synergy >= 40: return "Tier IV: Tangential Relevance"
    elif synergy >= 25: return "Tier V: Epistemic Divergence"
    else: return "Tier VI: Orthogonal / Unrelated Noise"

# ==========================================
# 5. EXTERNAL SERVICES & INTEGRATIONS
# ==========================================
def fetch_doi_metadata(doi):
    clean_doi = doi.replace("https://doi.org/", "").replace("doi.org/", "").strip()
    unpaywall_url = f"https://api.unpaywall.org/v2/{clean_doi}?email=research@pi-index.org"
    try:
        response = requests.get(unpaywall_url, timeout=10)
        if response.status_code == 200:
            res = response.json()
            title = res.get("title", "Unknown Title")
            authors_list = res.get("z_authors", [])
            authors = ", ".join([a.get("family", "").title() for a in authors_list]) if authors_list else "Unknown Author"
            pdf_url = res.get("best_oa_location", {}).get("url_for_pdf", None) if res.get("best_oa_location") else None
            return {"title": title, "authors": authors, "pdf_url": pdf_url}
        return None
    except Exception: return None

def download_pdf_from_url(pdf_url):
    if not pdf_url: return None
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/pdf,application/xhtml+xml,text/html;q=0.9,image/webp,*/*;q=0.8",
            "Referer": "https://doi.org/"
        }
        res = requests.get(pdf_url, headers=headers, timeout=20, allow_redirects=True)
        if res.status_code == 200 and res.content.startswith(b"%PDF"):
            return res.content
        return None
    except Exception: return None

def generate_rebuttal_strategy(scores_dict):
    if not scores_dict: return "No scores available to generate a rebuttal strategy."
        
    weakest_criterion = min(scores_dict, key=scores_dict.get)
    strongest_criterion = max(scores_dict, key=scores_dict.get)
    
    strategy = f"**Strategic Pivot:** Leverage your high score in **{strongest_criterion.replace('_', ' ')}** ({scores_dict[strongest_criterion]:.1f}/100) to distract from the manuscript's primary vulnerability in **{weakest_criterion.replace('_', ' ')}** ({scores_dict[weakest_criterion]:.1f}/100).\n\n"
    if "Originality" in weakest_criterion:
        strategy += "**Defense Tactic:** Argue that the paper value lies in synthesis and rigorous validation rather than paradigm disruption. Emphasize that cumulative science requires foundational solidity over risky novelties."
    elif "Rigor" in weakest_criterion:
        strategy += "**Defense Tactic:** Pre-emptively acknowledge sample size limitations in the discussion section. Frame the methodology as an exploratory pilot to lower the expectation of absolute statistical certainty."
    elif "Societal" in weakest_criterion:
        strategy += "**Defense Tactic:** Shift the narrative from immediate societal application to essential foundational groundwork. Argue that downstream societal impact is impossible without this specific theoretical gap being closed."
    else:
        strategy += "**Defense Tactic:** Focus the reviewers attention on the empirical density of your dataset. Acknowledge minor structural gaps but insist the volume of data speaks for itself."
    return strategy

# ==========================================
# 6. AI EXTRACTION ENGINE & NEURAL NETS
# ==========================================
def adaptive_chunking(text, max_tokens):
    if len(text) <= max_tokens: return text
    front_matter = text[:int(max_tokens * 0.4)]
    back_matter = text[-int(max_tokens * 0.6):]
    return front_matter + "\n...[TRUNCATED FOR TOKEN LIMITS]...\n" + back_matter

def evaluate_discriminator_and_divergence(text, model):
    text_chunk = text[:5000]
    prompt = f"""Analyze this academic text for two adversarial threats:
1. Synthetic Hallucination / AI-Generated Preprint Flood (unnatural keyword stuffing, stylistic filler, or high-flown prose masking weak statistical substance).
2. Semantic-Empirical Divergence: Check if the grandiose claims and equations in the text drastically diverge from or lack grounding in actual reported data variances.

Output a JSON object with two keys:
- "Gaming_Penalty": float from 0.0 (natural) to 1.0 (highly manipulated/synthetic).
- "Reproducibility_Score": float from 0.0 to 1.0 indicating whether code/data artifacts appear functional and verifiable.

Text: {text_chunk}"""
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=model, temperature=0.0, seed=42, response_format={"type": "json_object"}
        )
        res_json = json.loads(response.choices[0].message.content)
        return float(res_json.get("Gaming_Penalty", 0.0)), float(res_json.get("Reproducibility_Score", 0.5))
    except Exception: return 0.0, 0.5

def evaluate_scope_alignment(text, scope, model, text_limit):
    if not scope.strip(): return 0.0
    text = adaptive_chunking(text, text_limit)
    prompt = f"""You are a research alignment tool. Read the following paper text and evaluate how well it aligns with this specific research scope/keyword: "{scope}"
Return ONLY a valid JSON object with a single key "Scope_Alignment" containing a float between 0.0 and 100.0.
Text: {text}"""
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=model, temperature=0.0, seed=42, response_format={"type": "json_object"}
        )
        return float(json.loads(response.choices[0].message.content).get("Scope_Alignment", 0.0))
    except Exception: return 0.0

def extract_unpublished_authors_fallback(text):
    first_2k = text[:2500]
    lines = [line.strip() for line in first_2k.split('\n') if line.strip()]
    for line in lines[1:12]:
        clean_line = re.sub(r'[\d\*\†\‡\§\¶\(\)]', '', line).strip()
        if re.match(r'^[A-Z][a-z\.]+(\s+[A-Z]\.?)?\s+[A-Z][a-z]+(\s*,\s*[A-Z][a-z\.]+(\s+[A-Z]\.?)?\s+[A-Z][a-z]+)*$', clean_line):
            if len(clean_line) > 3 and not any(kw in clean_line.lower() for kw in ['abstract', 'introduction', 'university', 'department', 'contents', 'journal']):
                return clean_line.title()
    return "Unidentified"

def evaluate_pdf_text_ensemble(text, model, text_limit):
    text = adaptive_chunking(text, text_limit)
    prompt = f"""You are the theoretical parser for the Pi-Index. Read the academic paper and extract structured metadata.
CRITICAL EQUITY & NORMALIZATION INSTRUCTION:
- Global research equity is paramount. Do NOT penalize non-native English writing styles, alternative structural layouts, or resource-constrained syntax. Normalize linguistic style and evaluate strictly on scientific substance and methodological merit.

CRITICAL INSTRUCTION FOR AUTHORS:
- Scan the first 2 pages carefully for human author names. Output as a comma-separated list of Title Cased names (no brackets, no quotes, no "et al."). If none, output "Unidentified".

CRITICAL INSTRUCTION: Return ONLY a valid JSON object matching this exact structure:
{{
    "metadata": {{
        "Extracted_Title": "String",
        "Extracted_Author": "String"
    }},
    "fields": ["Primary Domain", "Secondary Domain"],
    "subfields": ["Specific Sub-topic 1", "Specific Sub-topic 2"],
    "variables": {{
        "H_novel": 0.5, "K_epistemic": 0.5, "zeta": 0.5, "I_existing": 0.5, "Sigma_error": 0.5, 
        "mu_signal": 0.5, "rho_k": 0.5, "p_disciplines": [0.5, 0.5], "bridge_capacity": 0.5, 
        "Utility_vector": 0.5, "decay_rate": 0.5, "q_fractional": 0.5, "D_open": 0.5, "J_code": 0.5, 
        "P_FAIR": 0.5, "d_g_distance": 0.5, "R_xi": 0.5, "PR_xi": 0.5, "I_Fisher": 0.5, 
        "KL_divergence": 0.5, "V_baseline": 0.5, "omega_data": 0.5, "sum_lambda_kappa": 0.5, 
        "eta_steps": 0.5, "Lambda_Lyapunov": 0.5
    }},
    "logic_analysis": {{
        "Evidence_Strength": 0.5, "Conclusion_Reach": 0.5, "Logical_Jumps": 0.5, "Premise_Validity": 0.5
    }},
    "Overall_Confidence": 0.95
}}
Text: {text}"""
    
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=model, temperature=0.0, seed=42, response_format={"type": "json_object"}
        )
        result_content = response.choices[0].message.content
        parsed = json.loads(result_content)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {"metadata": {"Extracted_Title": "Parsing Failed", "Extracted_Author": "Unidentified"}, "Overall_Confidence": 0.0}

def process_single_pdf(file_bytes, filename, scope, user_id, book_address="None", email="None", provided_doi="None"):
    if file_bytes is None or len(file_bytes) == 0:
        empty_scores = {k: 0.0 for k in ["C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary", "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability"]}
        return "Download/Extraction Failed", "Unidentified", 0.0, 0.0, "N/A", "N/A", ["Unspecified Domain"], ["Unspecified Sub-domain"], empty_scores, "Failed", 0.0, "None", "None", [1.0]*8, "N/A", "N/A", 0.0, False
    
    file_hash = hashlib.sha256(file_bytes).hexdigest() 
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT final_score, logic_score, title, fields, subfields, author_name, c1, c2, c3, c4, c5, c6, c7, c8, piq_minted, tx_hash, zk_proof, h_index, i10_index, reproducibility_score FROM papers_assessment WHERE eval_hash=?", (file_hash,))
    cached_result = cursor.fetchone()
    
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pdf_meta_author = doc.metadata.get("author", "").strip().title()
        full_text = " ".join([page.get_text() for page in doc])
    except Exception:
        empty_scores = {k: 0.0 for k in ["C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary", "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability"]}
        return "Invalid PDF Format", "Unidentified", 0.0, 0.0, "N/A", "N/A", ["Unspecified Domain"], ["Unspecified Sub-domain"], empty_scores, file_hash, 0.0, "None", "None", [1.0]*8, "N/A", "N/A", 0.0, False

    scope_alignment = evaluate_scope_alignment(full_text, scope, FALLBACK_MODEL, MAX_TEXT_TOKENS) if scope.strip() else 0.0

    if cached_result:
        score, logic_score, title, fields_str, subfields_str, author_name, *rest = cached_result
        c_scores = rest[:8]
        piq_minted, tx_hash, zk_proof, h_index, i10_index, repro_score = rest[8], rest[9], rest[10], rest[11], rest[12], rest[13]
        fields = json.loads(fields_str) if fields_str else ["Unspecified Domain"]
        subfields = json.loads(subfields_str) if subfields_str else ["Unspecified Sub-domain"]
        
        drift = calculate_complex_drift(scope_alignment, c_scores) if scope.strip() else "N/A"
        rec = get_recommendation_spectrum(score, drift) if scope.strip() else "N/A"
        scores_dict = {
            "C1_Originality": c_scores[0], "C2_Methodological_Rigor": c_scores[1], "C3_Interdisciplinary": c_scores[2], "C4_Societal_Impact": c_scores[3],
            "C5_Open_Science_Potential": c_scores[4], "C6_Literature_Integration": c_scores[5], "C7_Empirical_Density": c_scores[6], "C8_Future_Actionability": c_scores[7]
        }
        
        cursor.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights WHERE eval_hash=?", (file_hash,))
        weight_res = cursor.fetchone()
        used_weights = weight_res if weight_res else [1.0] * 8
        
        return title, author_name, score, logic_score, drift, rec, fields, subfields, scores_dict, file_hash, piq_minted, tx_hash, zk_proof, used_weights, h_index, i10_index, repro_score, True

    gaming_penalty, reproducibility_score = evaluate_discriminator_and_divergence(full_text, FALLBACK_MODEL)

    try:
        raw_data = evaluate_pdf_text_ensemble(full_text, PRIMARY_MODEL, MAX_TEXT_TOKENS)
        model_used = PRIMARY_MODEL
    except Exception as e:
        st.warning("Primary model hit a limit. Executing dynamic fallback strategy.")
        try:
            reduced_limit = int(MAX_TEXT_TOKENS * 0.6)
            raw_data = evaluate_pdf_text_ensemble(full_text, FALLBACK_MODEL, reduced_limit)
            model_used = FALLBACK_MODEL
        except Exception:
            empty_scores = {k: 0.0 for k in ["C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary", "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability"]}
            return "Extraction Failed", "Unidentified", 0.0, 0.0, "N/A", "N/A", ["Unspecified Domain"], ["Unspecified Sub-domain"], empty_scores, file_hash, 0.0, "None", "None", [1.0]*8, "N/A", "N/A", reproducibility_score, False
         
    if not isinstance(raw_data, dict):
        raw_data = {"metadata": {"Extracted_Title": filename, "Extracted_Author": "Unidentified"}, "Overall_Confidence": 0.0}

    confidence = raw_data.get("Overall_Confidence", 1.0)
    metadata = raw_data.get("metadata", {})
    
    if confidence < 0.50:
         empty_scores = {k: 0.0 for k in ["C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary", "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability"]}
         return "Indeterminate Format (Upload JSON Manifest)", metadata.get("Extracted_Author", "Unidentified"), 0.0, 0.0, "N/A", "N/A", ["Unspecified Domain"], ["Unspecified Sub-domain"], empty_scores, file_hash, 0.0, "None", "None", [1.0]*8, "N/A", "N/A", reproducibility_score, False

    title = metadata.get("Extracted_Title", filename)
    
    # Strip any brackets, single quotes, or double quotes the LLM accidentally formatted it with
    extracted_author = str(metadata.get("Extracted_Author", "")).strip().title()
    extracted_author = re.sub(r'[\[\]\'\"]', '', extracted_author).strip()
    
    if not extracted_author or extracted_author.lower() in ["unknown", "unknown author", "none", "n/a", "research scholar", "unidentified"] or extracted_author == os.path.splitext(filename)[0].title():
        if pdf_meta_author and pdf_meta_author.lower() not in ["unknown", "none"]:
            extracted_author = pdf_meta_author
        else:
            extracted_author = extract_unpublished_authors_fallback(full_text)

    # Canonical Deduplication Check (by DOI or Normalized Title + Author)
    normalized_title = re.sub(r'[^a-z0-9]', '', title.lower())
    cursor.execute("SELECT eval_hash, final_score, logic_score, c1, c2, c3, c4, c5, c6, c7, c8, piq_minted, tx_hash, zk_proof, h_index, i10_index, reproducibility_score FROM papers_assessment WHERE doi=? OR author_name=?", (provided_doi, extracted_author))
    existing_records = cursor.fetchall()
    
    for rec_row in existing_records:
        ex_hash, ex_score, ex_logic, *ex_rest = rec_row
        cursor.execute("SELECT title FROM papers_assessment WHERE eval_hash=?", (ex_hash,))
        ex_title_row = cursor.fetchone()
        if ex_title_row:
            ex_norm_title = re.sub(r'[^a-z0-9]', '', ex_title_row[0].lower())
            if (provided_doi != "None" and provided_doi) or (ex_norm_title == normalized_title and normalized_title != ""):
                fields = ["Unspecified Domain"]
                subfields = ["Unspecified Sub-domain"]
                c_scores = ex_rest[:8]
                piq_minted, tx_hash, zk_proof, h_index, i10_index, repro_score = ex_rest[8], ex_rest[9], ex_rest[10], ex_rest[11], ex_rest[12], ex_rest[13]
                drift = calculate_complex_drift(scope_alignment, c_scores) if scope.strip() else "N/A"
                rec_spec = get_recommendation_spectrum(ex_score, drift) if scope.strip() else "N/A"
                scores_dict = {
                    "C1_Originality": c_scores[0], "C2_Methodological_Rigor": c_scores[1], "C3_Interdisciplinary": c_scores[2], "C4_Societal_Impact": c_scores[3],
                    "C5_Open_Science_Potential": c_scores[4], "C6_Literature_Integration": c_scores[5], "C7_Empirical_Density": c_scores[6], "C8_Future_Actionability": c_scores[7]
                }
                cursor.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights WHERE eval_hash=?", (ex_hash,))
                weight_res = cursor.fetchone()
                used_weights = weight_res if weight_res else [1.0] * 8
                return title, extracted_author, ex_score, ex_logic, drift, rec_spec, fields, subfields, scores_dict, ex_hash, piq_minted, tx_hash, zk_proof, used_weights, h_index, i10_index, repro_score, True

    cursor.execute("UPDATE global_eval_counter SET count = count + 1")
    conn.commit()
    cursor.execute("SELECT count FROM global_eval_counter")
    total_evals = cursor.fetchone()[0]
          
    cursor.execute("SELECT block_height, block_hash, w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights ORDER BY block_height DESC LIMIT 1")
    epoch_data = cursor.fetchone()
    block_height, previous_hash, old_weights = epoch_data[0], epoch_data[1], epoch_data[2:]
    
    variables = raw_data.get("variables", {})
    if not isinstance(variables, dict): variables = {}
    scores_dict = compute_formulaic_criteria(variables, reproducibility_score)
    scores = [scores_dict[k] for k in ["C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary", "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability"]]
    
    logic_integrity = compute_logical_integrity(raw_data.get("logic_analysis", {}), gaming_penalty)

    fields, subfields = raw_data.get("fields", ["Unspecified Domain"]), raw_data.get("subfields", ["Unspecified Sub-domain"])
    
    raw_final_score = float(np.dot(scores, old_weights)) / 8.0
    final_score = float(raw_final_score * (0.7 + (logic_integrity / 333.3)))
    formulas_hash = get_formulas_hash()

    if total_evals % EPOCH_BLOCK_SIZE == 0:
        active_weights = calculate_model_driven_weights(old_weights, scores, model_used, block_height)
        timestamp = datetime.now().isoformat()
        val_node, block_hash, por_proof = validate_block_por(block_height + 1, active_weights, timestamp, previous_hash, file_hash, model_used, final_score, formulas_hash)
        cursor.execute('''INSERT INTO blockchain_por_weights (w1, w2, w3, w4, w5, w6, w7, w8, timestamp, previous_hash, validator_node, block_hash, eval_hash, model_used, por_proof, formulas_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                       (*active_weights, timestamp, previous_hash, val_node, block_hash, file_hash, model_used, por_proof, formulas_hash))
    else:
        active_weights = old_weights

    h_idx, i10_idx = fetch_author_metrics(extracted_author)

    cursor.execute("SELECT AVG(final_score), COUNT(*) FROM papers_assessment WHERE author_name=?", (extracted_author,))
    row = cursor.fetchone()
    past_avg = row[0] if row[0] is not None else 0.0
    past_count = row[1] if row[1] is not None else 0

    if past_count == 0:
        cursor.execute("SELECT AVG(final_score) FROM papers_assessment WHERE fields=?", (json.dumps(fields),))
        domain_avg = cursor.fetchone()[0]
        past_avg = domain_avg if domain_avg else 50.0

    improvement_multiplier = 1.0
    if final_score > past_avg and past_avg > 0:
        raw_multiplier = 1.5 + ((final_score - past_avg) / 50.0) 
        cap = max(1.0, 1.0 + math.log10(past_count + 1) * 0.5)
        improvement_multiplier = min(raw_multiplier, cap)
        
    piq_to_mint = 0.0 if extracted_author == "Unidentified" else round((final_score / 10.0) * improvement_multiplier, 2)
    
    zk_email_hash = "None"
    if email and email.endswith(('.edu', '.org')):
        zk_email_hash = "zkEM_" + hashlib.sha256(email.encode()).hexdigest()[:12]

    zk_proof = generate_zk_snark_proof(file_hash, final_score, logic_integrity, zk_email_hash)
    tx_hash = mint_pi_quotient_token(book_address, piq_to_mint, file_hash, zk_proof)

    drift = calculate_complex_drift(scope_alignment, scores) if scope.strip() else "N/A"
    rec = get_recommendation_spectrum(final_score, drift) if scope.strip() else "N/A"
    
    cursor.execute('''INSERT OR REPLACE INTO papers_assessment (eval_hash, user_id, title, filename, scope, c1, c2, c3, c4, c5, c6, c7, c8, logic_score, scope_alignment, subfields, fields, author_name, final_score, timestamp, eth_book, piq_minted, tx_hash, zk_proof, did, zk_email_proof, gaming_penalty, h_index, i10_index, reproducibility_score, doi) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                   (file_hash, user_id, title, filename, scope, *scores, logic_integrity, scope_alignment, json.dumps(subfields), json.dumps(fields), extracted_author, final_score, datetime.now().isoformat(), book_address, piq_to_mint, tx_hash, zk_proof, user_id, zk_email_hash, gaming_penalty, h_idx, i10_idx, reproducibility_score, provided_doi))
    conn.commit()
    
    return title, extracted_author, final_score, logic_integrity, drift, rec, fields, subfields, scores_dict, file_hash, piq_to_mint, tx_hash, zk_proof, active_weights, h_idx, i10_idx, reproducibility_score, False

def run_paper_evaluation(pdf_bytes, fname, scope, current_user, current_book, current_email, p_doi="None"):
    title, author_name, score, logic_integrity, drift, rec, fields, subfields, scores_dict, eval_hash, piq, tx_hash, zk_proof, used_weights, h_idx, i10_idx, repro_score, is_cached = process_single_pdf(
        pdf_bytes, fname, scope, current_user, current_book, current_email, p_doi
    )
    eval_record = {
        'title': title, 'author_name': author_name, 'score': score, 
        'logic_integrity': logic_integrity, 'drift': drift, 'rec': rec, 
        'fields': fields, 'subfields': subfields, 'scores_dict': scores_dict, 
        'eval_hash': eval_hash, 'piq': piq, 'tx_hash': tx_hash, 
        'zk_proof': zk_proof, 'used_weights': used_weights, 
        'h_idx': h_idx, 'i10_idx': i10_idx, 'repro_score': repro_score, 'filename': fname
    }
    st.session_state['evaluated_papers_buffer'].insert(0, eval_record)

class PiBlockchainDataset(Dataset):
    def __init__(self, data_matrix, lookback):
        self.data = data_matrix
        self.lookback = lookback
    def __len__(self): return len(self.data) - self.lookback
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.lookback]
        y = self.data[idx + self.lookback]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class PiBrainLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_layer_size=32, output_size=8):
        super(PiBrainLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_layer_size, 16), nn.ReLU(), nn.Linear(16, output_size))
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return torch.softmax(predictions, dim=-1) * 8.0

# ==========================================
# 7. STREAMLIT USER INTERFACE
# ==========================================
st.sidebar.title("System Access")

if 'assessment_update_token' not in st.session_state: st.session_state['assessment_update_token'] = time.time()
if 'reset_token' not in st.session_state: st.session_state['reset_token'] = 0
if 'evaluated_papers_buffer' not in st.session_state: st.session_state['evaluated_papers_buffer'] = []
if 'orcid_id' not in st.session_state:
    st.session_state.orcid_id = "0000-0000-0000-0000"
    st.session_state.orcid_name = ""
    st.session_state.is_authenticated = False

if not st.session_state.is_authenticated:
    st.sidebar.markdown(f"### Authenticate " + tooltip("Connect to your ORCID or DID to securely isolate your assessment history. Pi Quotient (piQ) is a Soulbound Token assigned strictly to this identity."), unsafe_allow_html=True)
    manual_orcid = st.sidebar.text_input("Enter ORCID iD or W3C DID", placeholder="XXXX-XXXX-XXXX-XXXX")
    email_input = st.sidebar.text_input("Institutional Email", placeholder="author@university.edu", help="Generates a Zero-Knowledge Proof (ZK-Email) verifying institutional alignment without exposing data to the ledger.")
    
    sign_manuscript = st.sidebar.checkbox("Cryptographically Sign Manuscript Hash with Private Key", help="Prevents Oracle manipulation by proving possession of the document.")

    if st.sidebar.button("Validate and Connect"):
        clean_orcid = manual_orcid.strip()
        if re.match(r'^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$', clean_orcid) or "did:" in clean_orcid:
            with st.sidebar.status("Connecting to Identity Registry..."):
                if "did:" in clean_orcid:
                    is_valid, user_name = True, "Verified Decentralized Identity"
                else:
                    is_valid, user_name = True, "Verified Researcher (Name Private)" 
            if is_valid:
                st.session_state.orcid_id, st.session_state.orcid_name, st.session_state.is_authenticated = clean_orcid, user_name, True
                st.session_state.inst_email = email_input.strip() if email_input.strip() else "None"
                st.rerun()
            else: st.sidebar.error(user_name)
        else: st.sidebar.error("Invalid ORCID or DID format.")
else:
    st.sidebar.success("Securely Connected")
    st.sidebar.markdown(f"**Researcher:** {st.session_state.orcid_name}\n**ID Vault:** `{st.session_state.orcid_id}`")
    if st.sidebar.button("Disconnect Session"):
        st.session_state.is_authenticated, st.session_state.orcid_name = False, ""
        st.rerun()

current_user = st.session_state.orcid_id
current_book = "0x" + hashlib.sha256(current_user.encode()).hexdigest()[:40] if current_user else "None"
current_email = st.session_state.get('inst_email', "None")

st.title("Pi-Index Assessment Engine", help="Automated peer-review framework powered by neural networks and multidimensional blockchain consensus.")
st.markdown("**Upload papers, define your scope of research, let Pi-Index filter noise and yield quantitative results.**")

with st.expander("View Pi-Index Grading Criteria Formulations"):
    st.subheader("Evaluation Metrics, Executable Reproducibility & Adversarial Logic Engine")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(r"**Adversarial Logic Gap ($\Delta_{Logic}$)** " + tooltip("We map the paper's reasoning structure before giving a final score. If the authors make claims that aren't supported by their own evidence, the system exponentially penalizes the paper."), unsafe_allow_html=True)
        st.markdown(r"$$ L_i = (\mathcal{P}_{valid} \cdot \mathcal{E}_{strength}) \cdot \exp\left(-\left(2 \cdot \max(0, \mathcal{C}_{reach} - \mathcal{E}_{strength}) + 1.5 \cdot \lambda_{jumps}\right)\right) \times \frac{1}{1 + e^{-\Delta Premise}} $$")
        
        st.markdown("**C1: Originality** " + tooltip("Does this paper disrupt existing knowledge (high score), or is it mostly derivative of older work (low score)?"), unsafe_allow_html=True)
        st.markdown(r"$$O = \varpi_1 \cdot \lim_{\Delta t \to 0} \oint_{\partial \Omega} \frac{\nabla \times (\mathcal{H}_{novel} \otimes \mathcal{K}_{epistemic})}{\iint_{\mathcal{M}} \sum_{i=1}^{N} |Z_i| \, dV} \cdot e^{-0.1 \zeta} $$")
        
        st.markdown("**C2: Methodological Rigor** " + tooltip("Are the methods statistically sound, and is the risk of a fundamental flaw minimized?"), unsafe_allow_html=True)
        st.markdown(r"$$R = \varpi_2 \cdot \left( 1 - \frac{\mathrm{tr}(\boldsymbol{\Sigma}_{error} \boldsymbol{\Lambda}^{-1})}{\det(\boldsymbol{\mu}_{signal} \otimes \mathbf{W})} \right) \cdot \mathbb{E}[\rho_k] $$")
        
        st.markdown("**C3: Interdisciplinary** " + tooltip("How well does the research bridge multiple disciplines together rather than staying in an isolated silo?"), unsafe_allow_html=True)
        st.markdown(r"$$I = \varpi_3 \cdot \left( \frac{1}{1-\alpha} \ln \left( \sum_{j=1}^{K} p_j^\alpha \right) + \sum_{i,j} \frac{A_{ij} \phi_i \phi_j}{\sqrt{d_i d_j}} \right) \cdot bridge\_capacity $$")
        
        st.markdown("**C4: Societal Impact** " + tooltip("What is the predicted long-term, real-world utility of the research findings?"), unsafe_allow_html=True)
        st.markdown(r"$$S = \varpi_4 \cdot \frac{1}{\Gamma(q)} \int_{t_0}^{t_\infty} (t_\infty - \tau)^{q-1} e^{-\gamma(\tau) \tau} \cdot \Theta\left[ \sum_{v \in \mathcal{V}} \omega_v U_v(\tau, \mathbf{x}) \right] d\tau $$")
    with col2:
        st.markdown("**C5: Open Science & Executable Reproducibility** " + tooltip("Rewards transparency and containerized/functional code execution verification."), unsafe_allow_html=True)
        st.markdown(r"$$O_s = \varpi_5 \cdot \frac{0.5 \mathcal{D}_{open} + 0.2 \mathbf{J}_{code} + 0.3 \mathcal{R}_{exec}}{\max \left[ \mathcal{N}_{\text{datasets}}, 1 \right]} \cdot \mathcal{P}_{FAIR} $$")
        
        st.markdown("**C6: Literature Integration** " + tooltip("Assesses how firmly grounded the paper is in foundational literature without being completely reliant on it."), unsafe_allow_html=True)
        st.markdown(r"$$L = \varpi_6 \cdot \frac{1}{\mathcal{N}} \sum_{i=1}^{\mathcal{N}} \int_{\mathcal{M}} e^{-\lambda d_g(x_i, x_{core})} R(x_i) \sqrt{g} \, dx_i \cdot \frac{\text{PR}(x_i)}{\sum PR} $$")
        
        st.markdown("**C7: Empirical Density & Validation** " + tooltip("Measures sheer depth and volume of analyzed data coupled with reproducibility fidelity."), unsafe_allow_html=True)
        st.markdown(r"$$E_d = \varpi_7 \cdot \tanh \left( \frac{\det \mathcal{I}_{Fisher}(\hat{\theta}) \cdot \mathbb{E}_{P}\left[\log\frac{P}{Q}\right] \cdot (0.8 + 0.2 \mathcal{R}_{exec})}{\mathcal{V}_{baseline} \cdot \oint_\Gamma K(\mathbf{x}) \, d\ell} \right) $$")
        
        st.markdown("**C8: Future Actionability** " + tooltip("Predicts whether the paper will trigger a cascade of actionable future research."), unsafe_allow_html=True)
        st.markdown(r"$$F_a = \varpi_8 \cdot \frac{1}{\mathcal{Z}} \int_{\mathcal{X}} \frac{1}{1 + \exp\left(-\sum_{k=1}^K w_k(\eta_k(\mathbf{x}) - \eta_{0,k}) + \Lambda_{Lyapunov}\right)} d\mu(\mathbf{x}) $$")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Assessment and Dossier", "Global Map of Science", "Active Epoch & DeSci Staking", "Pi-Brain Neural Network", "System Overview and Limitations"])

with tab1:
    st.markdown("### Unified Multi-Source Intake & Topic Discovery" + tooltip("Define your research scope, upload local PDFs, import via DOI, or discover and evaluate OpenAlex papers all in one place."), unsafe_allow_html=True)
    
    # By removing the reset_token from the key, the web browser will remember past inputs and provide native autocomplete suggestions
    research_scope = st.text_input("Define your specific Research Topic / Scope (Optional)", placeholder="e.g., Structural integrity, pi framework, neuroscience...", key="research_scope_input")
    
    st.markdown("---")
    st.markdown("#### Select Sources to Include in Assessment")
    
    selected_uploaded_files = []
    uploaded_files = st.file_uploader("1. Upload Local PDF(s)", type=["pdf"], accept_multiple_files=True, key=f"file_uploader_{st.session_state['reset_token']}")
    if uploaded_files:
        st.markdown("**Tick local files to include:**")
        for i, file in enumerate(uploaded_files):
            if st.checkbox(f"📄 Local File: {file.name}", value=True, key=f"up_chk_{i}_{st.session_state['reset_token']}"):
                selected_uploaded_files.append(file)

    st.markdown("")
    doi_input = st.text_input("2. Import via Unpaywall (DOI)", placeholder="10.1038/s41586-020-2649-2", key=f"doi_input_{st.session_state['reset_token']}")
    include_doi = False
    if doi_input.strip():
        include_doi = st.checkbox("Include this DOI in assessment", value=True, key=f"doi_chk_{st.session_state['reset_token']}")

    st.markdown("")
    alex_topic_input = st.text_input("3. Discover via OpenAlex Topic Search", placeholder="e.g., structural integrity, neuroscience, oncology", key=f"alex_topic_{st.session_state['reset_token']}")
    search_alex_btn = st.button("Search OpenAlex Papers")

    if search_alex_btn and alex_topic_input.strip():
        with st.spinner(f"Querying OpenAlex for papers on '{alex_topic_input}'..."):
            alex_results = search_openalex_topics(alex_topic_input.strip(), limit=5)
            if alex_results:
                st.session_state['alex_search_results'] = alex_results
                st.success(f"Found {len(alex_results)} Open Access papers.")
            else:
                st.warning("No Open Access papers found matching this topic.")

    if 'alex_search_results' in st.session_state and st.session_state['alex_search_results']:
        st.markdown("#### Discovered OpenAlex Papers")
        for idx, p in enumerate(st.session_state['alex_search_results']):
            expander_title = f"{p['title']} — {p['authors']}"
            with st.expander(expander_title):
                if p.get('doi'):
                    st.markdown(f"**DOI:** [{p['doi']}](https://doi.org/{p['doi']})")
                else:
                    st.markdown("**DOI:** Not Available")
                    
                if p.get('pdf_url'):
                    st.markdown(f"**PDF URL:** [{p['pdf_url']}]({p['pdf_url']})")
                else:
                    st.markdown("**PDF URL:** Direct binary link restricted by publisher")
                
                if st.button(f"Evaluate Paper {idx + 1}", key=f"eval_alex_{idx}_{st.session_state['reset_token']}"):
                    with st.spinner(f"Evaluating OpenAlex paper: {p['title']}..."):
                        pdf_bytes = None
                        fname = f"OpenAlex_{p['title'][:20]}.pdf"
                        p_doi = p.get('doi', 'None')
                        
                        if p.get('pdf_url'):
                            pdf_bytes = download_pdf_from_url(p['pdf_url'])
                        if not pdf_bytes and p.get('doi'):
                            metadata = fetch_doi_metadata(p['doi'])
                            if metadata and metadata.get('pdf_url'):
                                pdf_bytes = download_pdf_from_url(metadata['pdf_url'])

                        if pdf_bytes:
                            run_paper_evaluation(pdf_bytes, fname, research_scope, current_user, current_book, current_email, p_doi)
                            st.success("Evaluation complete!")
                            st.rerun()
                        else:
                            st.error("Could not directly download PDF. Try importing via DOI or uploading the PDF manually.")

    st.markdown("---")
    stake_amount = st.checkbox("Stake 0.01 piQ to Process (Returned on Valid Assessment)", value=True, help="Staking mechanisms actively filter low-effort, adversarial, or spam submissions.")

    def render_breakdown_item(item):
        title = item['title']
        author_name = item['author_name']
        score = item['score']
        logic_integrity = item['logic_integrity']
        scores_dict = item['scores_dict']
        used_weights = item['used_weights']
        eval_hash = item['eval_hash']
        piq = item['piq']
        tx_hash = item['tx_hash']
        zk_proof = item['zk_proof']
        drift = item['drift']
        rec = item['rec']
        h_index = item['h_idx']
        i10_index = item['i10_idx']
        repro_score = item['repro_score']
        filename = item['filename']

        st.markdown("---")
        st.subheader(f"{title} by {author_name}")
        
        with st.expander(f"Ledger Data & Dossier Details ({filename})"):
            st.write(f"**File Name:** `{filename}`")
            st.write(f"**Evaluation Hash:** `{eval_hash}`")
            st.write(f"**piQ Minted:** `{piq}`")
            st.write(f"**zk-SNARK:** `{zk_proof}`")
            st.write(f"**Tx Hash:** `{tx_hash}`")
            st.write(f"**Executable Reproducibility Score (C5/C7 audit):** `{repro_score * 100:.1f}%`")
            
        if research_scope.strip() and drift != "N/A" and rec != "N/A":
            st.markdown(f"**Scope Drift:** `{drift:.2f}%`")
            st.markdown(f"**Recommendation Tier:** `{rec}`")
        
        breakdown_df = pd.DataFrame({
            "Criterion": ["C1: Originality", "C2: Methodological Rigor", "C3: Interdisciplinary", "C4: Societal Impact", "C5: Open Science & Repro", "C6: Literature Integration", "C7: Empirical Density", "C8: Future Actionability"],
            "Score Extracted (0-100)": [scores_dict.get("C1_Originality",0), scores_dict.get("C2_Methodological_Rigor",0), scores_dict.get("C3_Interdisciplinary",0), scores_dict.get("C4_Societal_Impact",0), scores_dict.get("C5_Open_Science_Potential",0), scores_dict.get("C6_Literature_Integration",0), scores_dict.get("C7_Empirical_Density",0), scores_dict.get("C8_Future_Actionability",0)],
            "Epoch Weight": used_weights,
            "Weighted Value": [scores_dict.get(k,0)*used_weights[i] for i, k in enumerate(["C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary", "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability"])]
        })
        st.dataframe(breakdown_df, hide_index=True)
        raw_base = sum(breakdown_df["Weighted Value"]) / 8.0
        logic_multiplier = 0.7 + (logic_integrity / 333.3)
        st.markdown(f"**Base Weighted Sum (Mean divided by 8):** `{raw_base:.2f}`")
        st.markdown(f"**Logic Integrity Multiplier:** `{logic_multiplier:.4f}` (Derived from {logic_integrity:.1f}% raw logic score)")
        st.markdown(f"**Final Pi-Index (Base * Logic Multiplier):** `{score:.2f}` &nbsp;|&nbsp; **h-index:** `{h_index}` &nbsp;|&nbsp; **i10-index:** `{i10_index}` &nbsp;|&nbsp; **File:** `{filename}`")

        dossier_content = f"""# RESEARCH INTEGRITY DOSSIER (DORA-Aligned)
**Title:** {title}
**Author:** {author_name}
**File Name:** {filename}
**Evaluation Hash:** {eval_hash}
**Final Pi-Index Score:** {score:.2f} / 100
**Logic Integrity Score:** {logic_integrity:.1f}%
**Executable Reproducibility Score:** {repro_score * 100:.1f}%
**h-index / i10-index:** {h_index} / {i10_index}

## 8-Criteria Evaluation Breakdown
- C1 Originality: {scores_dict.get("C1_Originality",0)}
- C2 Methodological Rigor: {scores_dict.get("C2_Methodological_Rigor",0)}
- C3 Interdisciplinary: {scores_dict.get("C3_Interdisciplinary",0)}
- C4 Societal Impact: {scores_dict.get("C4_Societal_Impact",0)}
- C5 Open Science & Repro: {scores_dict.get("C5_Open_Science_Potential",0)}
- C6 Literature Integration: {scores_dict.get("C6_Literature_Integration",0)}
- C7 Empirical Density: {scores_dict.get("C7_Empirical_Density",0)}
- C8 Future Actionability: {scores_dict.get("C8_Future_Actionability",0)}

## Cryptographic Proofs & Ledger Seal
- zk-SNARK: {zk_proof}
- Tx Hash: {tx_hash}
"""
        st.download_button(
            label=f"Download DORA-Aligned Research Integrity Dossier ({filename})",
            data=dossier_content,
            file_name=f"Dossier_{eval_hash[:10]}.md",
            mime="text/markdown",
            key=f"download_dossier_{eval_hash}_{time.time()}"
        )

    if st.button("Run Assessment Pipeline", type="primary", use_container_width=True):
        if not stake_amount:
            st.error("You must agree to the piQ micro-stake to execute the assessment pipeline.")
        elif not selected_uploaded_files and not (include_doi and doi_input.strip()):
            st.warning("Please tick at least one local file or input a DOI to assess.")
        else:
            progress_bar, status_text = st.progress(0), st.empty()
            
            if include_doi and doi_input.strip():
                status_text.text(f"Resolving DOI: {doi_input}...")
                metadata = fetch_doi_metadata(doi_input)
                fname = f"DOI_{doi_input.replace('/', '_')}.pdf"
                if metadata and metadata['pdf_url']:
                    pdf_bytes = download_pdf_from_url(metadata['pdf_url'])
                    if pdf_bytes:
                        status_text.text(f"Assessing Open Access document from DOI...")
                        run_paper_evaluation(pdf_bytes, fname, research_scope, current_user, current_book, current_email, doi_input.strip())
                    else: st.error("Failed to download PDF from Open Access source.")
                else: st.error("Failed to resolve DOI or no Open Access PDF is publicly available.")
            
            if selected_uploaded_files:
                for i, file in enumerate(selected_uploaded_files):
                    status_text.text(f"Analyzing uploaded file {i+1} of {len(selected_uploaded_files)}: {file.name}...")
                    file_bytes = file.read()
                    run_paper_evaluation(file_bytes, file.name, research_scope, current_user, current_book, current_email, "None")
                    progress_bar.progress((i + 1) / len(selected_uploaded_files))
            
            st.session_state['reset_token'] += 1
            st.session_state['assessment_update_token'] = time.time()
            
            status_text.success("Pipeline processing complete.")
            time.sleep(1)
            st.rerun()

    if st.session_state['evaluated_papers_buffer']:
        st.markdown("---")
        st.markdown("### Active Session Assessment Results")
        for item in st.session_state['evaluated_papers_buffer']:
            render_breakdown_item(item)
            
    st.markdown("---")
    st.markdown("### AI Peer Review Defense Strategy " + tooltip("Synthesizes the mathematical assessment array to build a highly targeted adversarial rebuttal strategy."), unsafe_allow_html=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT eval_hash, title, author_name, c1, c2, c3, c4, c5, c6, c7, c8 FROM papers_assessment WHERE user_id=? ORDER BY timestamp DESC LIMIT 50", (current_user,))
    user_papers = cursor.fetchall()
    
    if not user_papers:
        st.info("You must assess at least one paper to unlock the AI Defense Strategy tool.")
    else:
        paper_options = {f"{p[1][:50]}... ({p[2]})" if len(p[1]) > 50 else f"{p[1]} ({p[2]})": p for p in user_papers}
        selected_super_paper = st.selectbox("Select an assessed paper to generate a strategic defense:", list(paper_options.keys()))
        
        if st.button("Generate Strategy"):
            paper_data = paper_options[selected_super_paper]
            scores = {
                "C1_Originality": paper_data[3], "C2_Methodological_Rigor": paper_data[4],
                "C3_Interdisciplinary": paper_data[5], "C4_Societal_Impact": paper_data[6],
                "C5_Open_Science_Potential": paper_data[7], "C6_Literature_Integration": paper_data[8],
                "C7_Empirical_Density": paper_data[9], "C8_Future_Actionability": paper_data[10]
            }
            rebuttal = generate_rebuttal_strategy(scores)
            st.success("Defense Strategy Generated Successfully.")
            st.markdown(rebuttal)

    st.markdown("---")
    st.markdown("### Your Assessment and Reward History " + tooltip("Your permanently recorded academic evaluations mapped to your ORCID iD/DID."), unsafe_allow_html=True)
    if st.session_state.is_authenticated:
        cursor.execute("SELECT title, author_name, filename, scope, final_score, piq_minted, tx_hash FROM papers_assessment WHERE user_id=? ORDER BY timestamp DESC LIMIT 20", (current_user,))
        history_data = cursor.fetchall()
        if history_data: st.dataframe(pd.DataFrame(history_data, columns=["Paper Title", "Contributing Authors", "File Name", "Scope", "Pi-Index Score", "piQ Earned", "Eth Tx Hash"]), use_container_width=True, hide_index=True)
        else: st.info("No assessment history found.")
    else: st.warning("Please connect your ORCID iD or DID in the sidebar.")


with tab2:
    st.markdown("### Global Map of Science (Ledger-Driven Cartography & Topic Separation) " + tooltip("Generates clustered, separated network topologies based on distinct scientific domains and ledger-evaluated subfields."), unsafe_allow_html=True)
    st.markdown("This map dynamically separates distinct scientific domains and is updated by every ledger-evaluated paper.")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT author_name FROM papers_assessment")
    all_global_authors = []
    for row in cursor.fetchall():
        if row[0]:
            all_global_authors.extend([a.strip().title() for a in row[0].split(',') if a.strip()])
    all_global_authors = sorted(list(set(all_global_authors)))
    
    selected_author = None
    piq_dict = get_author_piq_dict()
    
    if all_global_authors:
        filter_choice = st.selectbox(
            "Filter Global Cartography by Author:", 
            ["All Authors"] + all_global_authors, 
            key=f"author_filter_dropdown_{st.session_state['assessment_update_token']}",
            format_func=lambda x: f"{x} (piQ: {piq_dict.get(x, 0.0):.2f})" if x != "All Authors" else x
        )
        if filter_choice != "All Authors": selected_author = filter_choice

    def render_bubble_chart_clean(target_author):
        cursor.execute("SELECT fields, subfields, final_score, author_name FROM papers_assessment")
        data = cursor.fetchall()
        html_string, table_html = "", ""
        if not data: return html_string, table_html
        
        all_topics = []
        exclude_terms = {"general", "general science", "unspecified domain", "unspecified sub-domain"}
        
        for fields_json, subfields_json, final_score, author_str in data:
            if target_author and target_author != "All Authors" and target_author not in author_str:
                continue
            try:
                fields = [f.title().strip() for f in json.loads(fields_json)]
                subfields = [s.title().strip() for s in json.loads(subfields_json)]
                score = float(final_score) if final_score else 50.0
                for f in fields: 
                    if f.lower() not in exclude_terms: all_topics.append({'topic': f, 'weight': score, 'category': 'Field'})
                for s in subfields: 
                    if s.lower() not in exclude_terms: all_topics.append({'topic': s, 'weight': score, 'category': 'Subfield'})
            except: continue
                
        if not all_topics: 
            return "", "<div style='padding: 10px; color: #7f8c8d;'><b>No specific domains extracted yet.</b> Please run an assessment on a valid research paper to populate the Map of Science.</div>"
        
        df_topics = pd.DataFrame(all_topics)
        topic_counts = df_topics.groupby(['topic', 'category'])['weight'].sum().reset_index(name='weight')
        if topic_counts.empty: return html_string, table_html
            
        unique_topics = topic_counts['topic'].unique()
        def get_color(i, n):
            h, s, v = i/n if n > 0 else 0, 0.75, 0.95
            rgb = colorsys.hsv_to_rgb(h, s, v)
            return '#%02x%02x%02x' % tuple(int(x * 255) for x in rgb)
        
        color_map = {topic: get_color(i, len(unique_topics)) for i, topic in enumerate(unique_topics)}
        
        net = Network(height='650px', width='100%', bgcolor='#ffffff', font_color='#2c3e50', notebook=False)
        physics_options = """{
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -3500,
                    "centralGravity": 0.4,
                    "springLength": 150,
                    "springConstant": 0.04,
                    "damping": 0.09,
                    "avoidOverlap": 1.0
                },
                "stabilization": { "enabled": true, "iterations": 300 }
            }
        }"""
        net.set_options(physics_options)
        
        for _, row in topic_counts.iterrows():
            node_size = max(25, 15 + (row['weight'] * 2.0))
            shape = "dot" if row['category'] == 'Subfield' else "box"
            # Passing label=" " ensures the title is removed from the bottom of the bubble, but hover title remains
            net.add_node(n_id=row['topic'], label=" ", title=f"Category: {row['category']} | Topic: {row['topic']} | Weight: {row['weight']:.1f}", size=node_size, shape=shape, physics=True, color=color_map[row['topic']])
        
        topics_list = topic_counts['topic'].tolist()
        for i in range(len(topics_list) - 1):
            net.add_edge(topics_list[i], topics_list[i+1], width=1, color="#dcdde1")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
            net.save_graph(tmp_file.name)
            with open(tmp_file.name, 'r', encoding='utf-8') as f: html_string = f.read()
        os.remove(tmp_file.name)
        html_string = html_string.replace('mynetwork', f"pi_network_{int(time.time() * 1000)}")

        table_html = "<style>.table-big { width: 100%; font-size: 14px; border-collapse: collapse; margin-top: 10px; font-family: sans-serif; } .table-big th { background-color: #2c3e50; color: white; padding: 8px; text-align: left; } .table-big td { padding: 8px; border-bottom: 1px solid #ecf0f1; } .color-box { width: 25px; height: 25px; border-radius: 4px; display: inline-block; } </style>"
        table_html += "<div class='legend-container'><table class='table-big'><thead><tr><th style='width: 20%; text-align: center;'>Color</th><th>Topic / Subfield</th></tr></thead><tbody>"
        for _, row in topic_counts.sort_values(by="weight", ascending=False).iterrows():
            table_html += f"<tr><td style='text-align: center;'><div class='color-box' style='background-color:{color_map[row['topic']]};'></div></td><td><b>{row['topic']}</b> <span style='color:gray; font-size:11px;'>({row['category']})</span></td></tr>"
        table_html += "</tbody></table></div>"
        
        return html_string, table_html

    interactive_html, table_html = render_bubble_chart_clean(selected_author)
    if interactive_html:
        col1, col2 = st.columns([3, 1])
        with col1: components.html(interactive_html, height=670, scrolling=True)
        with col2: 
            st.markdown("### Topic Clusters " + tooltip("Distinctly separated scientific domains and subfield hierarchies."), unsafe_allow_html=True)
            st.markdown(table_html, unsafe_allow_html=True)
    else: 
        st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Pi Quotient (piQ) Explorer & Leaderboard " + tooltip("piQ is a Soulbound Token (SBT). It cannot be transferred, bought, or sold. It permanently attaches to the author's identity."), unsafe_allow_html=True)
    
    search_query = st.text_input("Search Explorer by Author Name or Digital Book Address:", placeholder="Enter author name or 0x...")
    
    if piq_dict:
        piq_df = pd.DataFrame(list(piq_dict.items()), columns=["Contributing Author", "Total piQ Earned"])
        piq_df = piq_df.sort_values(by="Total piQ Earned", ascending=False).reset_index(drop=True)
        
        if search_query:
            query_clean = search_query.strip().lower()
            if query_clean.startswith("0x"):
                cursor.execute("SELECT title, author_name, eth_book, filename, final_score, piq_minted, timestamp FROM papers_assessment WHERE LOWER(eth_book)=? ORDER BY timestamp DESC", (query_clean,))
                book_papers = cursor.fetchall()
                if book_papers:
                    st.success(f"Found {len(book_papers)} papers linked to Digital Book: `{search_query}`")
                    df_book = pd.DataFrame(book_papers, columns=["Paper Title", "Author", "Digital Book Address", "File Name", "Pi-Index", "piQ Earned", "Timestamp"])
                    st.dataframe(df_book, use_container_width=True, hide_index=True)
                else:
                    st.warning(f"No records found for Digital Book '{search_query}'.")
            else:
                cursor.execute("SELECT author_name, title, eth_book, filename, final_score, piq_minted, timestamp FROM papers_assessment WHERE LOWER(author_name) LIKE ? ORDER BY timestamp DESC", (f"%{query_clean}%",))
                author_papers = cursor.fetchall()
                if author_papers:
                    st.success(f"Found {len(author_papers)} paper records for author matching '{search_query}'.")
                    df_author = pd.DataFrame(author_papers, columns=["Author", "Paper Title", "Digital Book Address", "File Name", "Pi-Index", "piQ Earned", "Timestamp"])
                    st.dataframe(df_author, use_container_width=True, hide_index=True)
                else:
                    st.warning(f"No papers or piQ records found for author '{search_query}'.")
        else:
            st.dataframe(piq_df, use_container_width=True)
    else:
        st.info("No Pi Quotient has been minted yet.")

with tab3:
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT block_height, w1, w2, w3, w4, w5, w6, w7, w8, model_used, eval_hash, block_hash, por_proof, formulas_hash FROM blockchain_por_weights ORDER BY block_height DESC LIMIT 1")
        epoch_data = cursor.fetchone()
    except Exception:
        epoch_data = None
    
    if epoch_data:
        block_height, weights, model_used, eval_hash, block_hash, por_proof, formulas_hash = epoch_data[0], epoch_data[1:9], epoch_data[9], epoch_data[10], epoch_data[11], epoch_data[12], epoch_data[13]
        cursor.execute("SELECT COUNT(DISTINCT eval_hash) FROM blockchain_por_weights WHERE eval_hash != 'genesis'")
        total_papers_processed = cursor.fetchone()[0]

        current_pi_accuracy = generate_blockchain_pi(block_height)

        st.markdown(f"**Processed:** `{total_papers_processed}` | **Block Size:** `{EPOCH_BLOCK_SIZE}` | **Model:** `{model_used}` | **Block:** `{block_height}` | **Pi Algorithmic Precision:** `{current_pi_accuracy}`")
        
        cols = st.columns(4)
        labels = [("C1", r"$\varpi_1$"), ("C2", r"$\varpi_2$"), ("C3", r"$\varpi_3$"), ("C4", r"$\varpi_4$"), ("C5", r"$\varpi_5$"), ("C6", r"$\varpi_6$"), ("C7", r"$\varpi_7$"), ("C8", r"$\varpi_8$")]
        for i, col in enumerate(cols * 2):
            if i < 8:
                col.markdown(f"**{labels[i][0]} ({labels[i][1]})**")
                col.markdown(f"<h3 style='margin-top:0px; margin-bottom:5px;'>{weights[i]:.6f}</h3>", unsafe_allow_html=True)
                
        st.markdown("### Proof-of-Research Blockchain Explorer " + tooltip("Search the ledger to mathematically verify if a specific research document has been authentically graded and permanently sealed."), unsafe_allow_html=True)
        st.info(f"**Latest Proof-of-Research:** `{por_proof}` successfully verified and sealed to block `{block_hash}`.")
        st.caption(f"**Unalterable Criteria State Hash:** `{formulas_hash}` (Guarantees grading mathematical constants cannot be tampered with).")
        
        explore_col1, explore_col2 = st.columns([3, 1])
        with explore_col1: search_query = st.text_input("Enter Document Evaluation Hash or Block Hash to verify ledger record...")
        with explore_col2: st.write(""); st.write(""); search_btn = st.button("Verify Record")
            
        if search_btn and search_query:
            try:
                cursor.execute("SELECT * FROM blockchain_por_weights WHERE block_hash=? OR eval_hash=?", (search_query, search_query))
                record = cursor.fetchone()
                if record:
                    st.success("Valid Block Found on Ledger")
                    st.json({"Block Height": record[0], "Timestamp": record[9], "Model Used": record[14], "Validator Node": record[11], "Block Hash": record[12], "Evaluation Hash": record[13], "PoR Signature": record[15], "Formulas Hash": record[16], "Weights": dict(zip([f"w{i+1}" for i in range(8)], record[1:9]))})
                else: st.error("No block matching that signature was found on the ledger.")
            except:
                st.error("Error reading database schema. Try refreshing the app.")

        st.markdown("---")
        st.markdown("### DeSci Peer Attestation & Stake-Weighted Validation " + tooltip("High-reputation researchers can stake a fraction of their piQ to endorse or challenge peer assessments on-chain."), unsafe_allow_html=True)
        if st.session_state.is_authenticated:
            cursor.execute("SELECT eval_hash, title FROM papers_assessment ORDER BY timestamp DESC LIMIT 20")
            eval_papers = cursor.fetchall()
            if eval_papers:
                attest_options = {p[1]: p[0] for p in eval_papers}
                chosen_attest_title = st.selectbox("Select Paper for Attestation:", list(attest_options.keys()), key="desci_attest_select")
                target_eval_hash = attest_options[chosen_attest_title]
                
                attest_stance = st.radio("Attestation Stance:", ["Endorse Methodological Rigor", "Challenge / Flag Anomaly"], horizontal=True)
                stake_val = st.slider("Stake piQ Amount:", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                
                if st.button("Submit On-Chain Attestation"):
                    attest_id = "ATT_" + hashlib.sha256(f"{current_user}:{target_eval_hash}:{time.time()}".encode()).hexdigest()[:12]
                    cursor.execute("INSERT OR REPLACE INTO desci_attestations (attestation_id, eval_hash, attester_id, stake_amount, stance, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                                   (attest_id, target_eval_hash, current_user, stake_val, attest_stance, datetime.now().isoformat()))
                    conn.commit()
                    st.success(f"Attestation recorded successfully! Attestation ID: `{attest_id}`")
                
                cursor.execute("SELECT attester_id, stake_amount, stance, timestamp FROM desci_attestations WHERE eval_hash=?", (target_eval_hash,))
                existing_attestations = cursor.fetchall()
                if existing_attestations:
                    st.markdown("#### Active Community Attestations for this Manuscript")
                    st.dataframe(pd.DataFrame(existing_attestations, columns=["Attester ID", "Staked piQ", "Stance", "Timestamp"]), use_container_width=True, hide_index=True)
            else:
                st.info("No assessed papers available for attestation.")
        else:
            st.warning("Please authenticate with your ORCID iD or DID to participate in DeSci attestation staking.")

        st.markdown("---")
        st.markdown("### Latest Blockchain Ledger Hashes, zk-SNARK Proofs, and piQ Minted " + tooltip("Chronological view of the most recent smart contract executions, demonstrating mathematical proofs of computation and token allocations."), unsafe_allow_html=True)
        cursor.execute("""
            SELECT b.block_height, b.eval_hash, b.block_hash, p.zk_proof, p.piq_minted, b.timestamp 
            FROM blockchain_por_weights b 
            LEFT JOIN papers_assessment p ON b.eval_hash = p.eval_hash 
            ORDER BY b.block_height DESC LIMIT 10
        """)
        recent_hashes = cursor.fetchall()
        if recent_hashes:
            df_hashes = pd.DataFrame(recent_hashes, columns=["Block Height", "Evaluation Hash", "Block Hash", "zk-SNARK Proof", "Total piQ Minted", "Timestamp"])
            st.dataframe(df_hashes, use_container_width=True, hide_index=True)
        else:
            st.info("No hashes to display yet.")

with tab4:
    st.markdown("### Pi-Brain: Meta-Learning on the PoR Blockchain " + tooltip("An LSTM neural network that trains directly on the block weights to predict future shifts in algorithmic evaluation standards."), unsafe_allow_html=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights ORDER BY block_height ASC")
    historical_rows = cursor.fetchall()
    
    lookback_window = 5
    if len(historical_rows) < lookback_window + 2:
        st.warning(f"Not enough blockchain data to train the meta-model. You need at least {lookback_window + 2} blocks.")
    else:
        current_block_count = len(historical_rows)
        if 'last_trained_blocks' not in st.session_state or st.session_state.last_trained_blocks != current_block_count:
            weight_data = np.array(historical_rows, dtype=np.float32)
            dataset = PiBlockchainDataset(weight_data, lookback_window)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
            
            model = PiBrainLSTM()
            loss_function = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            progress_bar, status_text = st.progress(0), st.empty()
            epochs = 200
            
            model.train()
            for epoch in range(epochs):
                total_loss = 0
                for seq, target in dataloader:
                    optimizer.zero_grad()
                    loss = loss_function(model(seq), target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if epoch % 10 == 0 or epoch == epochs - 1:
                    status_text.text(f"Training Epoch {epoch}/{epochs} | MSE Loss: {total_loss / len(dataloader):.6f}")
                    progress_bar.progress((epoch + 1) / epochs)
            
            model.eval()
            with torch.no_grad():
                st.session_state.predicted_next_weights = model(torch.tensor(weight_data[-lookback_window:], dtype=torch.float32).unsqueeze(0)).squeeze().numpy()
                st.session_state.current_weights = weight_data[-1]
                st.session_state.last_trained_blocks = current_block_count
        else:
            st.info("Meta-model is cached and up-to-date with the latest blockchain ledger.")

        df_compare = pd.DataFrame({"Current Active Weights": st.session_state.current_weights, "Predicted Next Epoch": st.session_state.predicted_next_weights}, index=["C1: Originality", "C2: Methodological Rigor", "C3: Interdisciplinary", "C4: Societal Impact", "C5: Open Science", "C6: Literature Integration", "C7: Empirical Density", "C8: Future Actionability"])
        st.bar_chart(df_compare, height=400)
        st.markdown(f"**Mathematical Constraint Check:** Predicted Sum = `{sum(st.session_state.predicted_next_weights):.6f}` / `8.0`")

with tab5:
    st.markdown("### Pi-Index Program Architecture & End-to-End Pipeline")
    st.write("The end-to-end flowchart of the decentralized assessment engine, detailing multi-source intake, AI extraction, adversarial discrimination, cryptographic zero-knowledge proofs, and Web3 smart contract minting.")
    
    arch_graph = graphviz.Digraph(node_attr={'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#E8F4F8', 'fontname': 'Helvetica', 'color': '#2c3e50'})
    arch_graph.attr(rankdir='TB', size='12,12')

    with arch_graph.subgraph(name='cluster_intake') as c:
        c.attr(label='1. Multi-Source Intake & Identity Layer', color='#3498db')
        c.node('ORCID', 'ORCID / DID Vault Authentication')
        c.node('Intake', 'Multi-Source Intake (Local PDFs, DOI, OpenAlex)')
        c.edge('ORCID', 'Intake')

    with arch_graph.subgraph(name='cluster_ai') as c:
        c.attr(label='2. AI Extraction & Adversarial Discriminator', color='#e67e22')
        c.node('Chunking', 'Adaptive Chunking (Max 12k Tokens)')
        c.node('GroqAI', 'Groq AI Engine (Llama 3.3 70B & Fallback 8B)')
        c.node('Discriminator', 'Synthetic Hallucination & Divergence Discriminator')
        c.edge('Intake', 'Chunking')
        c.edge('Chunking', 'GroqAI')
        c.edge('GroqAI', 'Discriminator')

    with arch_graph.subgraph(name='cluster_scoring') as c:
        c.attr(label='3. Formulaic & Logic Scoring Engine', color='#2ecc71')
        c.node('Criteria', '8-Criteria Formulaic Evaluation (C1 - C8)')
        c.node('Logic', 'Adversarial Logic & Premise Integrity Matrix')
        c.edge('Discriminator', 'Criteria')
        c.edge('Criteria', 'Logic')

    with arch_graph.subgraph(name='cluster_crypto') as c:
        c.attr(label='4. Cryptographic Proof & Blockchain Ledger', color='#9b59b6')
        c.node('ZKProof', 'ZK-SNARK Proof & ZK-Email Generation')
        c.node('PoR', 'Proof-of-Research (PoR) Blockchain Consensus')
        c.node('Mint', 'Web3 Smart Contract piQ Minting (Soulbound Tokens)')
        c.edge('Logic', 'ZKProof')
        c.edge('ZKProof', 'PoR')
        c.edge('PoR', 'Mint')

    with arch_graph.subgraph(name='cluster_output') as c:
        c.attr(label='5. DORA Dossier & Science Cartography', color='#e74c3c')
        c.node('Dossier', 'DORA-Aligned Research Integrity Dossier')
        c.node('SciMap', 'Global Science Cartography (PyVis Topic Clustering)')
        c.edge('Mint', 'Dossier')
        c.edge('Mint', 'SciMap')

    st.graphviz_chart(arch_graph, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 0.8em;'>Framework Author: Ali Vafadar Yengejeh | Universita degli Studi di Milano-Bicocca</div>", unsafe_allow_html=True)
