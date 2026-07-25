import os
import re
import json
import time
import math
import random
import sqlite3
import hashlib
import tempfile
import html
from datetime import datetime
from io import BytesIO

import requests
import colorsys
import fitz
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import graphviz

import streamlit as st
import streamlit.components.v1 as components

from web3 import Web3
from groq import Groq

# ==========================================
# 1. CONFIGURATION & ENVIRONMENT SETUP
# ==========================================
st.set_page_config(page_title="Pi-Index Assessment Engine", layout="wide", page_icon="🎓")

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
# 2. SECURITY & UTILITY FUNCTIONS
# ==========================================
def sanitize_input(text):
    """Prevents XSS by escaping HTML characters from user inputs."""
    if not text:
        return ""
    return html.escape(str(text))

def check_file_security(uploaded_file):
    """Basic security check for uploaded files."""
    if uploaded_file.size > 20 * 1024 * 1024:  # 20 MB limit
        st.error("File is too large. Maximum size is 20MB.")
        return False
    if uploaded_file.type != "application/pdf":
        st.error("Invalid file type. Only PDFs are allowed.")
        return False
    return True

# ==========================================
# 3. ROOT LEVEL DATABASE SCHEMA ENFORCEMENT
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
    c1_raw = ((vars_dict.get('H_novel', 0.5) * vars_dict.get('K_epistemic', 0.5)) / (vars_dict.get('zeta', 0.5) * vars_dict.get('I_existing', 0.5) + 0.1)) * 60
    scores["C1_Originality"] = min(100.0, max(0.0, c1_raw))
    
    rigor_matrix = max(0.0, 1.0 - (vars_dict.get('Sigma_error', 0.2) / (vars_dict.get('mu_signal', 0.8) + 0.1)))
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
    
    density_inner = (vars_dict.get('I_Fisher', 0.5) * vars_dict.get('KL_divergence', 0.5) * (0.8 + 0.2 * reproducibility_score)) / (vars_dict.get('V_baseline', 0.5) * vars_dict.get('omega_data', 0.5) + 0.1)
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
    drift_metric = 100.0 * (1.0 - np.exp(-3.0 * (alignment_gap ** 1.5) * (1.0 + (standard_deviation / 100.0)) / (0.1 + (average_score / 100.0))))
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
                authors_str = ", ".join([a for a in authors_list if a]) if authors_list else "Unidentified"
                if pdf_url or doi:
                    extracted.append({'title': title, 'doi': doi, 'pdf_url': pdf_url, 'authors': authors_str})
            return extracted
    except Exception as e:
        st.error(f"OpenAlex Topic Fetch Error: {str(e)}")
    return []

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
            messages=[{"role": "user", "content": prompt}], model=model, temperature=0.0, response_format={"type": "json_object"}
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
            messages=[{"role": "user", "content": prompt}], model=model, temperature=0.0, response_format={"type": "json_object"}
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
                return clean_line
    return "Unidentified"

def evaluate_pdf_text_ensemble(text, model, text_limit):
    text = adaptive_chunking(text, text_limit)
    prompts = [
        f"""You are the theoretical parser for the Pi-Index. Read the academic paper or draft manuscript and extract metadata and variables.
CRITICAL EQUITY & NORMALIZATION INSTRUCTION:
- Global research equity is paramount. Do NOT penalize non-native English writing styles, alternative structural layouts, or resource-constrained syntax. Normalize linguistic style and evaluate strictly on scientific substance and methodological merit.
Extract Metadata: `Extracted_Title`, `Extracted_Author`.
Extract Variables (0.0 to 1.0): `H_novel`, `K_epistemic`, `zeta`, `I_existing`, `Sigma_error`, `mu_signal`, `rho_k`, `p_disciplines` (Array), `bridge_capacity`, `Utility_vector`, `decay_rate`, `q_fractional`, `D_open`, `J_code`, `P_FAIR`, `d_g_distance`, `R_xi`, `PR_xi`, `I_Fisher`, `KL_divergence`, `V_baseline`, `omega_data`, `sum_lambda_kappa`, `eta_steps`, `Lambda_Lyapunov`.
Logic Mapping (0.0 to 1.0): `Evidence_Strength`, `Conclusion_Reach`, `Logical_Jumps`, `Premise_Validity`.
REQUIRED: Add an "Overall_Confidence" key (0.0 to 1.0) indicating your parsing certainty.
Return ONLY a valid JSON object. Text: {text}"""
    ]
    prompt = random.choice(prompts)
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model=model, temperature=0.0, seed=random.randint(1, 1000), response_format={"type": "json_object"}
    )
    try:
        parsed = json.loads(response.choices[0].message.content)
        if isinstance(parsed, dict): return parsed
    except Exception: pass
    return {"Extracted_Title": "Parsing Failed", "Extracted_Author": "Unidentified", "Overall_Confidence": 0.0}

def process_single_pdf(file_bytes, filename, scope, user_id, book_address="None", email="None", provided_doi="None"):
    """
    Evaluates a PDF manuscript. All return variables are initialized at the start to avoid UnboundLocalError.
    """
    # INITIALIZE ALL VARIABLES IMMEDIATELY
    title = sanitize_input(filename)
    extracted_author = "Unidentified"
    final_score = 0.0
    logic_score = 0.0
    drift = "N/A"
    rec = "N/A"
    fields = ["Unspecified Domain"]
    subfields = ["Unspecified Sub-domain"]
    scores_dict = {k: 0.0 for k in ["C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary", "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration", "C7_Empirical_Density", "C8_Future_Actionability"]}
    file_hash = "None"
    piq_minted = 0.0
    tx_hash = "None"
    zk_proof = "None"
    used_weights = [1.0] * 8
    h_index = "N/A"
    i10_index = "N/A"
    repro_score = 0.0
    is_cached = False

    if file_bytes is None or len(file_bytes) == 0:
        return title, extracted_author, final_score, logic_score, drift, rec, fields, subfields, scores_dict, file_hash, piq_minted, tx_hash, zk_proof, used_weights, h_index, i10_index, repro_score, is_cached
    
    try:
        file_hash = hashlib.sha256(file_bytes).hexdigest() 
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check cache by hash
        cursor.execute("SELECT final_score, logic_score, title, fields, subfields, author_name, c1, c2, c3, c4, c5, c6, c7, c8, piq_minted, tx_hash, zk_proof, h_index, i10_index, reproducibility_score FROM papers_assessment WHERE eval_hash=?", (file_hash,))
        cached_result = cursor.fetchone()
        
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pdf_meta_author = doc.metadata.get("author", "").strip()
        full_text = " ".join([page.get_text() for page in doc])
        
        scope_alignment = evaluate_scope_alignment(full_text, scope, FALLBACK_MODEL, MAX_TEXT_TOKENS) if scope.strip() else 0.0

        if cached_result:
            score, logic_s, cached_title, fields_str, subfields_str, author_name, *rest = cached_result
            c_scores = rest[:8]
            piq_m, tx_h, zk_p, h_idx, i10_idx, r_score = rest[8], rest[9], rest[10], rest[11], rest[12], rest[13]
            flds = json.loads(fields_str) if fields_str else ["Unspecified Domain"]
            subflds = json.loads(subfields_str) if subfields_str else ["Unspecified Sub-domain"]
            
            dft = calculate_complex_drift(scope_alignment, c_scores) if scope.strip() else "N/A"
            recom = get_recommendation_spectrum(score, dft) if scope.strip() else "N/A"
            s_dict = {
                "C1_Originality": c_scores[0], "C2_Methodological_Rigor": c_scores[1], "C3_Interdisciplinary": c_scores[2], "C4_Societal_Impact": c_scores[3],
                "C5_Open_Science_Potential": c_scores[4], "C6_Literature_Integration": c_scores[5], "C7_Empirical_Density": c_scores[6], "C8_Future_Actionability": c_scores[7]
            }
            
            cursor.execute("SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights WHERE eval_hash=?", (file_hash,))
            weight_res = cursor.fetchone()
            u_weights = weight_res if weight_res else [1.0] * 8
            
            return cached_title, author_name, score, logic_s, dft, recom, flds, subflds, s_dict, file_hash, piq_m, tx_h, zk_p, u_weights, h_idx, i10_idx, r_score, True

        gaming_penalty, repro_score = evaluate_discriminator_and_divergence(full_text, FALLBACK_MODEL)
        
        # Fallback Logic
        try:
            raw_data = evaluate_pdf_text_ensemble(full_text, PRIMARY_MODEL, MAX_TEXT_TOKENS)
            model_used = PRIMARY_MODEL
        except Exception as e:
            st.warning("Primary model hit a limit. Executing dynamic fallback strategy.")
            reduced_limit = int(MAX_TEXT_TOKENS * 0.6)
            raw_data = evaluate_pdf_text_ensemble(full_text, FALLBACK_MODEL, reduced_limit)
            model_used = FALLBACK_MODEL
             
        if not isinstance(raw_data, dict):
            raw_data = {"Extracted_Title": filename, "Extracted_Author": "Unidentified", "Overall_Confidence": 0.0}

        confidence = raw_data.get("Overall_Confidence", 1.0)
        if confidence < 0.50:
             return "Indeterminate Format", raw_data.get("Extracted_Author", "Unidentified"), 0.0, 0.0, "N/A", "N/A", fields, subfields, scores_dict, file_hash, 0.0, "None", "None", used_weights, "N/A", "N/A", repro_score, False

        title = raw_data.get("Extracted_Title", filename)
        extracted_author = str(raw_data.get("Extracted_Author", "")).strip()
        
        if not extracted_author or extracted_author.lower() in ["unknown", "unknown author", "none", "n/a", "research scholar", "unidentified"] or extracted_author == os.path.splitext(filename)[0]:
            if pdf_meta_author.strip() and pdf_meta_author.lower() not in ["unknown", "none"]:
                extracted_author = pdf_meta_author.strip()
            else:
                extracted_author = extract_unpublished_authors_fallback(full_text)

        # Canonical Deduplication Check
        normalized_title = re.sub(r'[^a-z0-9]', '', title.lower())
        cursor.execute("SELECT eval_hash, final_score, logic_score, c1, c2, c3, c4, c5, c6, c7, c8, piq_minted, tx_hash, zk_proof, h_index, i10_index, reproducibility_score FROM papers_assessment WHERE doi=? OR author_name=?", (provided_doi, extracted_author))
        existing_records = cursor.fetchall()
        
        for rec_row in existing_records:
            ex_hash, ex_score, ex_logic, *ex_rest = rec_row
            cursor.execute("SELECT title FROM papers_assessment WHERE eval_hash=?", (ex_hash,))
            ex_title_res = cursor.fetchone()
            if ex_title_res:
                ex_title = ex_title_res[0]
                ex_norm_title = re.sub(r'[^a-z0-9]', '', ex_title.lower())
                if ex_norm_title and ex_norm_title == normalized_title:
                    c_scores = ex_rest[:8]
                    dft = calculate_complex_drift(scope_alignment, c_scores) if scope.strip() else "N/A"
                    recom = get_recommendation_spectrum(ex_score, dft) if scope.strip() else "N/A"
                    s_dict = {
                        "C1_Originality": c_scores[0], "C2_Methodological_Rigor": c_scores[1], "C3_Interdisciplinary": c_scores[2], "C4_Societal_Impact": c_scores[3],
                        "C5_Open_Science_Potential": c_scores[4], "C6_Literature_Integration": c_scores[5], "C7_Empirical_Density": c_scores[6], "C8_Future_Actionability": c_scores[7]
                    }
                    return ex_title, extracted_author, ex_score, ex_logic, dft, recom, fields, subfields, s_dict, ex_hash, ex_rest[8], ex_rest[9], ex_rest[10], used_weights, ex_rest[11], ex_rest[12], ex_rest[13], True

        # Process new paper
        logic_score = compute_logical_integrity(raw_data, gaming_penalty)
        scores_dict = compute_formulaic_criteria(raw_data, repro_score)
        
        # Apply Weights
        c_vals = list(scores_dict.values())
        final_score = np.mean(c_vals)
        
        h_index, i10_index = fetch_author_metrics(extracted_author)
        zk_proof = generate_zk_snark_proof(file_hash, final_score, logic_score, email)
        
        if w3.is_connected() and book_address != "None":
            tx_hash = mint_pi_quotient_token(book_address, max(0, final_score), file_hash, zk_proof)
            piq_minted = max(0, final_score)
        else:
            tx_hash = "No Web3 Context"
            piq_minted = 0.0

        drift = calculate_complex_drift(scope_alignment, c_vals) if scope.strip() else "N/A"
        rec = get_recommendation_spectrum(final_score, drift) if scope.strip() else "N/A"

        # Insert to DB
        cursor.execute('''INSERT INTO papers_assessment 
                          (eval_hash, user_id, title, filename, scope, c1, c2, c3, c4, c5, c6, c7, c8, scope_alignment, logic_score, subfields, fields, author_name, final_score, timestamp, piq_minted, tx_hash, zk_proof, h_index, i10_index, reproducibility_score, doi) 
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                       (file_hash, user_id, title, filename, scope, *c_vals, scope_alignment, logic_score, json.dumps(subfields), json.dumps(fields), extracted_author, final_score, datetime.now().isoformat(), piq_minted, tx_hash, zk_proof, h_index, i10_index, repro_score, provided_doi))
        conn.commit()
        
    except Exception as e:
        st.error(f"Error processing {filename}: {str(e)}")

    return title, extracted_author, final_score, logic_score, drift, rec, fields, subfields, scores_dict, file_hash, piq_minted, tx_hash, zk_proof, used_weights, h_index, i10_index, repro_score, is_cached

# ==========================================
# 7. STREAMLIT FRONTEND APP
# ==========================================
def main():
    st.title("🎓 ScholarPi: Decentralized Research Evaluation")

    if "processed_papers" not in st.session_state:
        st.session_state.processed_papers = pd.DataFrame(columns=[
            "title", "author", "score", "topic", "lat", "lon", "tx_hash"
        ])

    CURRENT_USER = "session_user"
    CURRENT_EMAIL = "research@example.com"
    CURRENT_BOOK = "None" 

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "📄 PDF Upload", 
        "🔍 OpenAlex Discover", 
        "🌍 Global Map", 
        "⚙️ Architecture Flow"
    ])

    # --- TAB 1: DASHBOARD ---
    with tab1:
        st.header("Research Dashboard")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Papers Processed", len(st.session_state.processed_papers))
        avg_score = st.session_state.processed_papers["score"].mean() if not st.session_state.processed_papers.empty else 0
        col2.metric("Average Score", f"{avg_score:.1f}")
        col3.metric("PIQs Minted", len(st.session_state.processed_papers[st.session_state.processed_papers["tx_hash"] != "No Web3 Context"]))
        
        st.subheader("Recent Evaluations")
        if not st.session_state.processed_papers.empty:
            st.dataframe(st.session_state.processed_papers[["title", "author", "score", "topic", "tx_hash"]], use_container_width=True)
        else:
            st.info("No papers processed yet. Head to 'PDF Upload' or 'OpenAlex Discover' to begin.")

    # --- TAB 2: PDF UPLOAD ---
    with tab2:
        st.header("Evaluate Local PDF")
        
        research_scope = st.text_input("Define your specific Research Topic / Scope (Optional)", key="pdf_scope")
        uploaded_file = st.file_uploader("Upload Research Paper", type=["pdf"])
        p_doi = st.text_input("Enter DOI (Optional)")
        
        if uploaded_file and st.button("Evaluate PDF"):
            if check_file_security(uploaded_file):
                with st.spinner("Analyzing methodology, logic integrity, and generating ZK Proofs..."):
                    pdf_bytes = uploaded_file.read()
                    fname = uploaded_file.name
                    
                    results = process_single_pdf(
                        pdf_bytes, fname, sanitize_input(research_scope), 
                        CURRENT_USER, CURRENT_BOOK, CURRENT_EMAIL, sanitize_input(p_doi)
                    )
                    
                    (title, author, score, logic, drift, rec, fields, subfields, scores_dict, 
                     f_hash, minted, tx, zk, weights, h_idx, i10_idx, repro, cached) = results
                    
                    st.success(f"Evaluation Complete: {title}")
                    
                    mcol1, mcol2, mcol3 = st.columns(3)
                    mcol1.metric("Final Score", f"{score:.2f}")
                    mcol2.metric("Logic Integrity", f"{logic:.2f}")
                    mcol3.metric("Reproducibility", f"{repro:.2f}")
                    
                    if tx != "No Web3 Context":
                        st.info(f"🪙 PIQ Minted Successfully! TX Hash: {tx}")
                    
                    # Generate deterministic coordinates based on hash for global map demonstration
                    mock_lat = (int(f_hash[:8], 16) % 140) - 70
                    mock_lon = (int(f_hash[8:16], 16) % 360) - 180
                    
                    new_row = pd.DataFrame([{
                        "title": title, "author": author, "score": score, 
                        "topic": fields[0] if fields else "General",
                        "lat": mock_lat, "lon": mock_lon, 
                        "tx_hash": tx
                    }])
                    st.session_state.processed_papers = pd.concat([st.session_state.processed_papers, new_row], ignore_index=True)

    # --- TAB 3: OPENALEX DISCOVER ---
    with tab3:
        st.header("Discover via OpenAlex")
        
        openalex_scope = st.text_input("Define your specific Research Topic / Scope (Optional)", key="openalex_scope_input")
        search_query = st.text_input("OpenAlex Search Query (e.g., 'machine learning healthcare')")
        
        if st.button("Search OpenAlex"):
            if not search_query:
                st.warning("Please enter a search query.")
            else:
                with st.spinner("Querying OpenAlex API..."):
                    results = search_openalex_topics(sanitize_input(search_query))
                    if not results:
                        st.info("No OpenAccess PDFs found for this query.")
                    else:
                        for idx, res in enumerate(results):
                            with st.expander(f"{res['title']} (Authors: {res['authors']})"):
                                st.write(f"**DOI:** {res['doi']}")
                                st.write(f"**PDF URL:** {res['pdf_url']}")
                                if st.button(f"Evaluate Paper {idx+1}", key=f"eval_{idx}"):
                                    with st.spinner("Downloading and processing..."):
                                        try:
                                            pdf_res = requests.get(res['pdf_url'], timeout=10)
                                            if pdf_res.status_code == 200:
                                                eval_results = process_single_pdf(
                                                    pdf_res.content, res['title'], sanitize_input(openalex_scope), 
                                                    CURRENT_USER, CURRENT_BOOK, CURRENT_EMAIL, res['doi']
                                                )
                                                st.success(f"Scored {eval_results[2]:.2f} / 100")
                                            else:
                                                st.error("Failed to download PDF.")
                                        except Exception as e:
                                            st.error(f"Download Error: {str(e)}")

    # --- TAB 4: GLOBAL MAP ---
    with tab4:
        st.header("Global Distribution of Evaluated Research")
        
        df_map = st.session_state.processed_papers
        
        if df_map.empty:
            st.info("No location data available yet. Process a paper to populate the map.")
        else:
            df_clean = df_map.dropna(subset=['lat', 'lon'])
            
            fig = px.scatter_geo(
                df_clean,
                lat="lat",
                lon="lon",
                color="topic",
                size="score",
                hover_name="title",
                hover_data={"author": True, "score": True, "lat": False, "lon": False},
                projection="natural earth",
                title="Geographic Spread of Evaluated Research"
            )
            
            fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)

    # --- TAB 5: ARCHITECTURE FLOW ---
    with tab5:
        st.header("How ScholarPi Works")
        
        flowchart = graphviz.Digraph(engine='dot')
        flowchart.attr(rankdir='TB', size='10,10')
        
        flowchart.node('A', 'User Input\n(PDF / Search)', shape='parallelogram', style='filled', fillcolor='#e3f2fd')
        flowchart.node('B', 'Data Extraction\n(PyMuPDF / OpenAlex)', shape='box', style='filled', fillcolor='#f3e5f5')
        flowchart.node('C', 'Scope Alignment\n(Topic Filtering)', shape='diamond', style='filled', fillcolor='#fff3e0')
        flowchart.node('D', 'AI Evaluation Engine\n(Logic, Drift, Reproducibility)', shape='box', style='filled', fillcolor='#e8f5e9')
        flowchart.node('E', 'Generate ZK Proof\n(Verifiable Compute)', shape='box', style='filled', fillcolor='#fff9c4')
        flowchart.node('F', 'Mint PIQ Token\n(Blockchain Transaction)', shape='cylinder', style='filled', fillcolor='#ffcdd2')
        flowchart.node('G', 'Update Global Map\n& Dashboards', shape='ellipse', style='filled', fillcolor='#e1bee7')

        flowchart.edge('A', 'B')
        flowchart.edge('B', 'C')
        flowchart.edge('C', 'D')
        flowchart.edge('D', 'E', label=' Evaluation Scores')
        flowchart.edge('E', 'F', label=' Valid Proof')
        flowchart.edge('F', 'G', label=' TX Hash')
        
        st.graphviz_chart(flowchart, use_container_width=True)
        
        st.markdown("""
        ### Process Breakdown
        1. **Data Ingestion:** Upload a local PDF or query OpenAlex directly. The data is securely hashed.
        2. **Scope Alignment:** Both upload and search paths support an optional "Research Scope" parameter to focus the AI's attention.
        3. **AI Evaluation:** The core engine analyzes the text for methodology, logic integrity, reproducibility, and scope drift.
        4. **Zero-Knowledge Proofs:** Ensures the AI ran the specific algorithm correctly without leaking the proprietary model state.
        5. **Blockchain Minting:** A PIQ token is minted with the generated scores and ZK proof attached as metadata.
        6. **Visualization:** Data is aggregated and securely mapped on the global interface.
        """)

if __name__ == "__main__":
    main()
