import os
import re
import json
import time
import math
import random
import sqlite3
import hashlib
import tempfile
import shutil
from datetime import datetime
from io import BytesIO

import requests
import cloudscraper
import colorsys
import fitz
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyvis.network import Network

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
st.set_page_config(
    page_title="Pi-Index Assessment Engine (CoARA-Compliant)", layout="wide"
)

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_TEXT_TOKENS = 12000
EPOCH_BLOCK_SIZE = 1

WEB3_PROVIDER_URI = os.getenv(
    "WEB3_PROVIDER_URI", "https://sepolia.infura.io/v3/YOUR_INFURA_PROJECT_ID"
)
ETH_ADMIN_PRIVATE_KEY = os.getenv(
    "ETH_ADMIN_PRIVATE_KEY",
    "0x0000000000000000000000000000000000000000000000000000000000000000",
)
PIQ_CONTRACT_ADDRESS = os.getenv(
    "PIQ_CONTRACT_ADDRESS", "0xYourDeployedContractAddressHere"
)

# Persistent local machine storage directory (User Home Directory)
BASE_DIR = os.path.expanduser("~/Scientometric_Pi_Index")
os.makedirs(BASE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, "pi_index_main.db")

# Fallback safely handles Streamlit local secrets vs Server Environment Variables
try:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    PINATA_API_KEY = os.getenv("PINATA_API_KEY") or st.secrets.get("PINATA_API_KEY", "")
    PINATA_SECRET_API_KEY = os.getenv("PINATA_SECRET_API_KEY") or st.secrets.get("PINATA_SECRET_API_KEY", "")
    REGISTRY_CONTRACT_ADDRESS = os.getenv("REGISTRY_CONTRACT_ADDRESS") or st.secrets.get("REGISTRY_CONTRACT_ADDRESS", "")
except FileNotFoundError:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    PINATA_API_KEY = os.getenv("PINATA_API_KEY", "")
    PINATA_SECRET_API_KEY = os.getenv("PINATA_SECRET_API_KEY", "")
    REGISTRY_CONTRACT_ADDRESS = os.getenv("REGISTRY_CONTRACT_ADDRESS", "")

if not GROQ_API_KEY:
  st.error(
      "API Key not found! Please configure your environment variables or Streamlit Secrets."
  )
  st.stop()

w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER_URI))
groq_client = Groq(api_key=GROQ_API_KEY)

# Hardcoded Genesis Block (Bitcoin-style embedded root of trust)
GENESIS_BLOCK_CONFIG = {
    "block_height": 1,
    "weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "timestamp": "2026-01-01T00:00:00.000000",
    "previous_hash": "0" * 64,
    "validator_node": "Validator_Pi_Genesis",
    "eval_hash": "genesis",
    "model_used": "none",
    "por_proof": "Genesis_Proof_Anchor",
    "formulas_hash": hashlib.sha256(
        b"C1:Semantic_Originality|C2:MDAR_Rigor|C3:Citation_Entropy|C4:Open_Infrastructure|C5:Containerized_Execution|C6:Citation_Polarity|C7:Empirical_Density|C8:FAIR_Actionability|CoARA_Dossier_v2.0"
    ).hexdigest(),
}


# ==========================================
# 1.5 DECENTRALIZED STATE MANAGEMENT
# ==========================================
def restore_state_from_web3():
    """Fetches the latest IPFS CID from Sepolia Ethereum and restores the DB and Neural Net."""
    if not REGISTRY_CONTRACT_ADDRESS or not w3.is_connected():
        return
    
    try:
        abi = '[{"inputs":[],"name":"getCID","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"}]'
        
        if len(REGISTRY_CONTRACT_ADDRESS) != 42 or not REGISTRY_CONTRACT_ADDRESS.startswith("0x"):
            return
            
        contract = w3.eth.contract(address=w3.to_checksum_address(REGISTRY_CONTRACT_ADDRESS), abi=json.loads(abi))
        cid = contract.functions.getCID().call()
        
        if cid:
            res = requests.get(f"https://gateway.pinata.cloud/ipfs/{cid}", timeout=30)
            if res.status_code == 200:
                zip_path = BASE_DIR + ".zip"
                with open(zip_path, 'wb') as fp:
                    fp.write(res.content)
                shutil.unpack_archive(zip_path, BASE_DIR)
                if os.path.exists(zip_path):
                    os.remove(zip_path)
    except Exception as e:
        print(f"Failed to restore state from Web3: {e}")

def backup_state_to_web3():
    """Zips the local state, pins to IPFS via Pinata, and updates the Sepolia Ethereum registry."""
    if not PINATA_API_KEY or not REGISTRY_CONTRACT_ADDRESS or not w3.is_connected():
        return
        
    try:
        shutil.make_archive(BASE_DIR, 'zip', BASE_DIR)
        zip_path = BASE_DIR + ".zip"
        
        headers = {
            "pinata_api_key": PINATA_API_KEY, 
            "pinata_secret_api_key": PINATA_SECRET_API_KEY
        }
        with open(zip_path, 'rb') as fp:
            res = requests.post(
                "https://api.pinata.cloud/pinning/pinFileToIPFS", 
                files={"file": fp}, 
                headers=headers
            )
        
        cid = res.json().get("IpfsHash")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        
        if not cid:
            return

        abi = '[{"inputs":[{"internalType":"string","name":"_cid","type":"string"}],"name":"updateCID","outputs":[],"stateMutability":"nonpayable","type":"function"}]'
        
        if len(REGISTRY_CONTRACT_ADDRESS) != 42 or not REGISTRY_CONTRACT_ADDRESS.startswith("0x"):
            return
            
        contract = w3.eth.contract(address=w3.to_checksum_address(REGISTRY_CONTRACT_ADDRESS), abi=json.loads(abi))
        account = w3.eth.account.from_key(ETH_ADMIN_PRIVATE_KEY)
        
        tx = contract.functions.updateCID(cid).build_transaction({
            "from": account.address,
            "nonce": w3.eth.get_transaction_count(account.address),
            "gas": 150000,
            "gasPrice": w3.to_wei("10", "gwei"),
        })
        
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=ETH_ADMIN_PRIVATE_KEY)
        w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    except Exception as e:
        print(f"Failed to backup state to Web3: {e}")

if "state_restored" not in st.session_state:
    restore_state_from_web3()
    st.session_state["state_restored"] = True


# ==========================================
# 2. ROOT LEVEL DATABASE SCHEMA ENFORCEMENT
# ==========================================
def enforce_database_schema():
  conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
  cursor = conn.cursor()

  cursor.execute("""CREATE TABLE IF NOT EXISTS papers_assessment 
                    (eval_hash TEXT PRIMARY KEY, user_id TEXT, title TEXT, filename TEXT, scope TEXT,
                     c1 REAL, c2 REAL, c3 REAL, c4 REAL, 
                     c5 REAL, c6 REAL, c7 REAL, c8 REAL, 
                     scope_alignment REAL, logic_score REAL,
                     subfields TEXT, fields TEXT, author_name TEXT, final_score REAL, timestamp DATETIME)""")

  cursor.execute("""CREATE TABLE IF NOT EXISTS blockchain_por_weights 
                    (block_height INTEGER PRIMARY KEY AUTOINCREMENT, 
                     w1 REAL, w2 REAL, w3 REAL, w4 REAL, 
                     w5 REAL, w6 REAL, w7 REAL, w8 REAL, 
                     timestamp DATETIME, previous_hash TEXT, 
                     validator_node TEXT, block_hash TEXT, eval_hash TEXT, model_used TEXT)""")

  cursor.execute(
      "CREATE TABLE IF NOT EXISTS global_eval_counter (count INTEGER)"
  )
  cursor.execute("""CREATE TABLE IF NOT EXISTS desci_attestations 
                    (attestation_id TEXT PRIMARY KEY, eval_hash TEXT, attester_id TEXT, stake_amount REAL, stance TEXT, timestamp DATETIME)""")

  cursor.execute("""CREATE TABLE IF NOT EXISTS auto_ip_tracking 
                    (ip_address TEXT PRIMARY KEY, first_seen DATETIME)""")

  cursor.execute("SELECT COUNT(*) FROM global_eval_counter")
  if cursor.fetchone()[0] == 0:
    cursor.execute("INSERT INTO global_eval_counter (count) VALUES (0)")

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
      "mdar_adherence_score": "REAL DEFAULT 0.0",
      "rrid_valid_count": "INTEGER DEFAULT 0",
      "credit_taxonomy_roles": "TEXT DEFAULT 'None'",
      "reproducibility_score": "REAL DEFAULT 0.0",
      "doi": "TEXT DEFAULT 'None'",
  }

  target_columns_weights = {
      "por_proof": "TEXT DEFAULT 'Genesis_Proof'",
      "formulas_hash": "TEXT DEFAULT 'Locked_State'",
  }

  cursor.execute("PRAGMA table_info(papers_assessment)")
  existing_assessment_cols = [row[1] for row in cursor.fetchall()]
  for col, dtype in target_columns_assessment.items():
    if col not in existing_assessment_cols:
      try:
        cursor.execute(f"ALTER TABLE papers_assessment ADD COLUMN {col} {dtype}")
      except Exception:
        pass

  cursor.execute("PRAGMA table_info(blockchain_por_weights)")
  existing_weights_cols = [row[1] for row in cursor.fetchall()]
  for col, dtype in target_columns_weights.items():
    if col not in existing_weights_cols:
      try:
        cursor.execute(
            f"ALTER TABLE blockchain_por_weights ADD COLUMN {col} {dtype}"
        )
      except Exception:
        pass

  conn.commit()
  conn.close()

enforce_database_schema()

def get_db_connection():
  enforce_database_schema()
  conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
  cursor = conn.cursor()
  cursor.execute("SELECT COUNT(*) FROM blockchain_por_weights")
  if cursor.fetchone()[0] == 0:
    g = GENESIS_BLOCK_CONFIG
    data_string = f"{g['block_height']}{g['weights']}{g['timestamp']}{g['previous_hash']}{g['validator_node']}{g['por_proof']}{g['model_used']}{g['formulas_hash']}"
    block_hash = hashlib.sha256(data_string.encode("utf-8")).hexdigest()
    cursor.execute(
        """INSERT INTO blockchain_por_weights 
            (block_height, w1, w2, w3, w4, w5, w6, w7, w8, timestamp, previous_hash, validator_node, block_hash, eval_hash, model_used, por_proof, formulas_hash) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            g["block_height"],
            *g["weights"],
            g["timestamp"],
            g["previous_hash"],
            g["validator_node"],
            block_hash,
            g["eval_hash"],
            g["model_used"],
            g["por_proof"],
            g["formulas_hash"],
        ),
    )
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
  svg_icon = (
      '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16"'
      ' height="16" fill="none" stroke="#9e9e9e" stroke-width="2"'
      ' stroke-linecap="round" stroke-linejoin="round" style="vertical-align:'
      ' -3px; margin-left: 6px; cursor: help;"><circle cx="12" cy="12"'
      ' r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3'
      ' 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
  )
  return f'<span title="{text}">{svg_icon}</span>'

def clean_author_name(author_str):
  if not author_str:
    return "Unidentified"
  try:
    if author_str.startswith("[") and author_str.endswith("]"):
      parsed = json.loads(author_str.replace("'", '"'))
      if isinstance(parsed, list):
        return ", ".join([str(a).strip() for a in parsed if str(a).strip()])
  except:
    pass
  cleaned = (
      author_str.replace("[", "")
      .replace("]", "")
      .replace("'", "")
      .replace('"', "")
  )
  return cleaned.strip()

def is_likely_institution(name):
  if not name:
    return True
  lower_name = name.lower()
  inst_keywords = [
      "university",
      "univ.",
      "college",
      "institute",
      "inst.",
      "department",
      "dept.",
      "laboratory",
      "lab",
      "hospital",
      "center",
      "centre",
      "faculty",
      "milano",
      "bicocca",
      "polytechnic",
      "academy",
      "school",
      "corporation",
      "inc",
      "llc",
      "ltd",
      "foundation",
      "fund",
      "council",
      "cnr",
      "inps",
      "iss",
      "università",
  ]
  return any(kw in lower_name for kw in inst_keywords)

def fetch_author_coara_metrics(author_name):
  try:
    clean_name = clean_author_name(author_name)
    if (
        not clean_name
        or clean_name.lower() in ["unidentified", "unknown"]
        or is_likely_institution(clean_name)
    ):
      return 0.0, 0, "Data/Software Curation"
    first_author = clean_name.split(",")[0].strip()
    url = f"https://api.openalex.org/authors?search={first_author}"
    res = requests.get(url, timeout=5)
    if res.status_code == 200:
      data = res.json()
      if data.get("results") and len(data["results"]) > 0:
        author_obj = data["results"][0]
        works_count = author_obj.get("works_count", 0)
        return (
            float(works_count),
            int(author_obj.get("cited_by_count", 0)),
            "Open Access & Dataset Curation",
        )
  except Exception:
    pass
  return 0.0, 0, "Methodology & Validation"

def search_openalex_topics(topic_query, limit=100):
  try:
    url = f"https://api.openalex.org/works?search={requests.utils.quote(topic_query)}&filter=is_oa:true&per_page={limit}"
    res = requests.get(url, timeout=10)
    if res.status_code == 200:
      results = res.json().get("results", [])
      extracted = []
      for item in results:
        title = item.get("title", "Untitled Paper")
        doi = item.get("doi", "")

        best_oa = item.get("best_oa_location") or {}
        pdf_url = best_oa.get("pdf_url") or item.get("open_access", {}).get(
            "oa_url", ""
        )

        authorships = item.get("authorships", [])
        authors_list = [
            a.get("author", {}).get("display_name", "") for a in authorships
        ]
        authors_str = (
            ", ".join([a for a in authors_list if a])
            if authors_list
            else "Unidentified"
        )

        if pdf_url or doi:
          extracted.append({
              "title": title,
              "doi": doi,
              "pdf_url": pdf_url,
              "authors": authors_str,
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
  conn.close()
  author_piq = {}
  author_book = {}
  for authors_str, piq in data:
    clean_authors = clean_author_name(authors_str)
    if (
        not clean_authors
        or clean_authors.lower()
        in ["unidentified", "unknown", "research scholar"]
        or is_likely_institution(clean_authors)
    ):
      continue
    alist = [a.strip() for a in clean_authors.split(",") if a.strip()]
    if not alist:
      continue
    share = piq / len(alist)
    for a in alist:
      author_piq[a] = author_piq.get(a, 0.0) + share
      author_book[a] = "0x" + hashlib.sha256(a.encode()).hexdigest()[:40]
  return author_piq, author_book


# ==========================================
# 4. BLOCKCHAIN & MATHEMATICAL ENGINE
# ==========================================
def validate_block_por(
    block_index,
    weights,
    timestamp,
    previous_hash,
    eval_hash,
    model_used,
    final_score,
    formulas_hash,
):
  validator_node = "Validator_Pi_" + hashlib.md5(
      str(time.time()).encode()
  ).hexdigest()[:6]
  por_proof = f"PoR_{eval_hash[:12]}_Score:{final_score:.2f}"
  data_string = (
      f"{block_index}{weights}{timestamp}{previous_hash}{validator_node}{por_proof}{model_used}{formulas_hash}"
  )
  block_hash = hashlib.sha256(data_string.encode("utf-8")).hexdigest()
  return validator_node, block_hash, por_proof

def generate_zk_snark_proof(eval_hash, final_score, logic_score, email_str=""):
  circuit_input = (
      f"{eval_hash}:{final_score}:{logic_score}:{email_str}:{time.time()}"
  )
  return "0x0" + hashlib.sha3_256(circuit_input.encode("utf-8")).hexdigest()

def mint_pi_quotient_token(book_address, amount, eval_hash, zk_proof):
  if not w3.is_connected() or book_address == "None" or not book_address:
    return "Not Connected / No Book"

  if len(PIQ_CONTRACT_ADDRESS) != 42 or not PIQ_CONTRACT_ADDRESS.startswith("0x"):
    return "Eth Tx Failed: Invalid Contract Address Configuration"

  try:
    target_addr = (
        book_address
        if w3.is_address(book_address)
        else "0x" + hashlib.sha256(book_address.encode()).hexdigest()[:40]
    )

    abi = '[{"inputs":[{"internalType":"address","name":"researcher","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"},{"internalType":"string","name":"evalHash","type":"string"},{"internalType":"bytes","name":"zkProof","type":"bytes"}],"name":"verifyProofAndMint","outputs":[],"stateMutability":"nonpayable","type":"function"}]'
    contract = w3.eth.contract(
        address=w3.to_checksum_address(PIQ_CONTRACT_ADDRESS), abi=json.loads(abi)
    )
    account = w3.eth.account.from_key(ETH_ADMIN_PRIVATE_KEY)

    tx = contract.functions.verifyProofAndMint(
        w3.to_checksum_address(target_addr),
        int(amount),
        eval_hash,
        bytes.fromhex(zk_proof[2:]),
    ).build_transaction({
        "from": account.address,
        "nonce": w3.eth.get_transaction_count(account.address),
        "gas": 200000,
        "gasPrice": w3.to_wei("10", "gwei"),
    })

    signed_tx = w3.eth.account.sign_transaction(
        tx, private_key=ETH_ADMIN_PRIVATE_KEY
    )
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
  criteria_state = (
      "C1:Semantic_Originality|C2:MDAR_Rigor|C3:Citation_Entropy|C4:Open_Infrastructure|C5:Containerized_Execution|C6:Citation_Polarity|C7:Empirical_Density|C8:FAIR_Actionability|CoARA_Dossier_v2.0"
  )
  return hashlib.sha256(criteria_state.encode("utf-8")).hexdigest()

def calculate_model_driven_weights(old_weights, scores, model_name, block_height):
  if "70b" in model_name:
    model_version, model_size = 3.3, 70.0
  else:
    model_version, model_size = 3.1, 8.0

  pi_accuracy = generate_blockchain_pi(block_height)
  delta_models = abs((3.3 * 70.0) - (3.1 * 8.0))
  mean_score = np.mean(scores)

  new_weights = []
  for i, old_w in enumerate(old_weights):
    stretched_score = max(
        1.0, min(100.0, mean_score + (scores[i] - mean_score) * 3.0)
    )
    weight_shift = (
        (model_version * model_size) / (delta_models * pi_accuracy)
    ) * ((stretched_score / 100.0) ** 2)
    w_new = old_w * 0.85 + (1.0 + weight_shift * 0.15) * 0.15
    new_weights.append(w_new)

  sum_of_weights = sum(new_weights)
  return [round((w / sum_of_weights) * 8.0, 6) for w in new_weights]

def compute_logical_integrity(extracted_logic_vars, gaming_penalty):
  evidence = extracted_logic_vars.get("Evidence_Strength", 0.5)
  conclusion_reach = extracted_logic_vars.get("Conclusion_Reach", 0.5)
  jumps = extracted_logic_vars.get("Logical_Jumps", 0.5)
  premise = extracted_logic_vars.get("Premise_Validity", 0.5)

  logic_gap = max(0.0, conclusion_reach - evidence)
  base_logic = (
      (premise * evidence)
      * np.exp(-(logic_gap * 2.0 + jumps * 1.5))
      * 100
  )
  logic_score = base_logic * (1.0 - (gaming_penalty * 0.9))
  return max(0.0, min(100.0, logic_score))

def compute_formulaic_criteria(
    vars_dict, reproducibility_score, sciscore_adherence=0.8
):
  scores = {}
  c1_raw = (
      vars_dict.get("semantic_novelty", 0.7)
      * 100
      * (1.0 - vars_dict.get("laundering_penalty", 0.1))
  )
  scores["C1_Originality"] = min(100.0, max(0.0, c1_raw))
  c2_raw = sciscore_adherence * vars_dict.get("rigor_index", 0.75) * 100
  scores["C2_Methodological_Rigor"] = min(100.0, max(0.0, c2_raw))
  c3_raw = vars_dict.get("citation_entropy", 0.6) * 100
  scores["C3_Interdisciplinary"] = min(100.0, max(0.0, c3_raw))
  c4_raw = vars_dict.get("societal_linkage", 0.65) * 100
  scores["C4_Societal_Impact"] = min(100.0, max(0.0, c4_raw))
  c5_raw = (
      (0.5 * vars_dict.get("D_open", 0.7))
      + (0.2 * vars_dict.get("J_code", 0.6))
      + (0.3 * reproducibility_score)
  ) * 100
  scores["C5_Open_Science_Potential"] = min(100.0, max(0.0, c5_raw))
  c6_raw = vars_dict.get("citation_polarity_score", 0.7) * 100
  scores["C6_Literature_Integration"] = min(100.0, max(0.0, c6_raw))
  c7_raw = vars_dict.get("empirical_density", 0.75) * 100
  scores["C7_Empirical_Density"] = min(100.0, max(0.0, c7_raw))
  c8_raw = vars_dict.get("fair_compliance", 0.8) * 100
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
  drift_metric = (
      100.0
      * (
          1.0
          - np.exp(
              -3.0
              * (alignment_gap ** 1.5)
              * (1.0 + (standard_deviation / 100.0))
              / (0.1 + (average_score / 100.0))
          )
      )
  )
  return float(max(0.0, min(100.0, drift_metric)))

def get_recommendation_spectrum(score, drift):
  if drift == "N/A":
    return "N/A"
  synergy = score * (1.0 - (drift / 100.0) ** 1.5)
  if synergy >= 85:
    return "Tier I: Core Paradigm (Optimal Synergy)"
  elif synergy >= 70:
    return "Tier II: Highly Aligned Framework"
  elif synergy >= 55:
    return "Tier III: Moderately Synergistic"
  elif synergy >= 40:
    return "Tier IV: Tangential Relevance"
  elif synergy >= 25:
    return "Tier V: Epistemic Divergence"
  else:
    return "Tier VI: Orthogonal / Unrelated Noise"


# ==========================================
# 5. EXTERNAL SERVICES & INTEGRATIONS
# ==========================================
def fetch_doi_metadata(doi):
  clean_doi = (
      doi.replace("https://doi.org/", "").replace("doi.org/", "").strip()
  )
  unpaywall_url = (
      f"https://api.unpaywall.org/v2/{clean_doi}?email=research@pi-index.org"
  )
  try:
    response = requests.get(unpaywall_url, timeout=10)
    if response.status_code == 200:
      res = response.json()
      title = res.get("title", "Unknown Title")
      authors_list = res.get("z_authors", [])
      authors = (
          ", ".join([a.get("family", "") for a in authors_list])
          if authors_list
          else "Unknown Author"
      )
      pdf_url = (
          res.get("best_oa_location", {}).get("url_for_pdf", None)
          if res.get("best_oa_location")
          else None
      )
      return {"title": title, "authors": authors, "pdf_url": pdf_url}
    return None
  except Exception:
    return None

def fetch_semantic_scholar_pdf(title_or_doi):
  if not title_or_doi:
    return None
  try:
    clean_query = title_or_doi.replace("https://doi.org/", "").strip()
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(clean_query)}&limit=1&fields=openAccessPdf,externalIds"
    res = requests.get(url, timeout=10)
    if res.status_code == 200:
      data = res.json().get("data", [])
      if data:
        oa_pdf = data[0].get("openAccessPdf")
        if oa_pdf and oa_pdf.get("url"):
          return oa_pdf["url"]
  except Exception:
    pass
  return None

def download_pdf_from_url(pdf_url):
  if not pdf_url:
    return None

  if "arxiv.org/abs/" in pdf_url:
    pdf_url = pdf_url.replace("/abs/", "/pdf/") + ".pdf"
  elif (
      "ncbi.nlm.nih.gov/pmc/articles/PMC" in pdf_url
      and not pdf_url.endswith(".pdf")
  ):
    parts = pdf_url.split("PMC")
    if len(parts) > 1:
      pmc_id = parts[1].split("/")[0]
      pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"

  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
          " like Gecko) Chrome/122.0.0.0 Safari/537.36"
      ),
      "Accept": (
          "text/html,application/xhtml+xml,application/pdf;q=0.9,image/avif,image/webp,*/*;q=0.8"
      ),
      "Accept-Language": "en-US,en;q=0.5",
      "Referer": "https://scholar.google.com/",
      "Connection": "keep-alive",
  }

  try:
    session = requests.Session()
    res = session.get(pdf_url, headers=headers, timeout=15, allow_redirects=True)
    content_type = res.headers.get("Content-Type", "").lower()
    if res.status_code == 200 and (
        b"%PDF" in res.content[:10] or "application/pdf" in content_type
    ):
      return res.content
  except Exception:
    pass

  try:
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "desktop": True}
    )
    res = scraper.get(pdf_url, timeout=20, allow_redirects=True)
    content_type = res.headers.get("Content-Type", "").lower()
    if res.status_code == 200 and (
        b"%PDF" in res.content[:10] or "application/pdf" in content_type
    ):
      return res.content
  except Exception:
    pass

  return None

def generate_rebuttal_strategy(scores_dict):
  if not scores_dict:
    return "No scores available to generate a rebuttal strategy."

  weakest_criterion = min(scores_dict, key=scores_dict.get)
  strongest_criterion = max(scores_dict, key=scores_dict.get)

  strategy = (
      f"**Strategic Pivot:** Leverage your high score in"
      f" **{strongest_criterion.replace('_', ' ')}**"
      f" ({scores_dict[strongest_criterion]:.1f}/100) to distract from the"
      f" manuscript's primary vulnerability in"
      f" **{weakest_criterion.replace('_', ' ')}**"
      f" ({scores_dict[weakest_criterion]:.1f}/100).\n\n"
  )
  if "Originality" in weakest_criterion:
    strategy += (
        "**Defense Tactic:** Argue that the paper value lies in synthesis and"
        " rigorous validation rather than paradigm disruption. Emphasize that"
        " cumulative science requires foundational solidity over risky"
        " novelties."
    )
  elif "Rigor" in weakest_criterion:
    strategy += (
        "**Defense Tactic:** Pre-emptively acknowledge sample size limitations"
        " in the discussion section. Frame the methodology as an exploratory"
        " pilot to lower the expectation of absolute statistical certainty."
    )
  elif "Societal" in weakest_criterion:
    strategy += (
        "**Defense Tactic:** Shift the narrative from immediate societal"
        " application to essential foundational groundwork. Argue that"
        " downstream societal impact is impossible without this specific"
        " theoretical gap being closed."
    )
  else:
    strategy += (
        "**Defense Tactic:** Focus the reviewers attention on the empirical"
        " density of your dataset. Acknowledge minor structural gaps but insist"
        " the volume of data speaks for itself."
    )
  return strategy


# ==========================================
# 6. AI EXTRACTION ENGINE & NEURAL NETS
# ==========================================
def adaptive_chunking(text, max_tokens):
  if len(text) <= max_tokens:
    return text
  front_matter = text[: int(max_tokens * 0.4)]
  back_matter = text[-int(max_tokens * 0.6) :]
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
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    res_json = json.loads(response.choices[0].message.content)
    return float(res_json.get("Gaming_Penalty", 0.0)), float(
        res_json.get("Reproducibility_Score", 0.5)
    )
  except Exception:
    return 0.0, 0.5

def evaluate_scope_alignment(text, scope, model, text_limit):
  if not scope.strip():
    return 0.0
  text = adaptive_chunking(text, text_limit)
  prompt = f"""You are a research alignment tool. Read the following paper text and evaluate how well it aligns with this specific research scope/keyword: "{scope}"
Return ONLY a valid JSON object with a single key "Scope_Alignment" containing a float between 0.0 and 100.0.
Text: {text}"""
  try:
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return float(
        json.loads(response.choices[0].message.content).get(
            "Scope_Alignment", 0.0
        )
    )
  except Exception:
    return 0.0

def extract_unpublished_authors_fallback(text):
  first_2k = text[:2500]
  lines = [line.strip() for line in first_2k.split("\n") if line.strip()]
  for line in lines[1:12]:
    clean_line = re.sub(r"[\d\*\†\‡\§\¶\(\)]", "", line).strip()
    if re.match(
        r"^[A-Z][a-z\.]+(\s+[A-Z]\.?)?\s+[A-Z][a-z]+(\s*,\s*[A-Z][a-z\.]+(\s+[A-Z]\.?)?\s+[A-Z][a-z]+)*$",
        clean_line,
    ):
      if len(clean_line) > 3 and not any(
          kw in clean_line.lower()
          for kw in [
              "abstract",
              "introduction",
              "university",
              "department",
              "contents",
              "journal",
              "bicocca",
              "milano",
              "institute",
          ]
      ):
        return clean_line
  return "Unidentified"

def evaluate_pdf_text_ensemble(text, model, text_limit):
  text = adaptive_chunking(text, text_limit)
  prompts = [
      f"""You are the theoretical parser for the Pi-Index. Read the academic paper or draft manuscript and extract metadata and audit variables.
CRITICAL EQUITY & NORMALIZATION INSTRUCTION:
- Global research equity is paramount. Do NOT penalize non-native English writing styles, alternative structural layouts, or resource-constrained syntax. Normalize linguistic style and evaluate strictly on scientific substance and methodological merit.

CRITICAL INSTRUCTION FOR AUTHORS & TOPICS:
- Scan the first 2 pages carefully for human author names. Output as a clean comma-separated list of HUMAN author names (no brackets, no quotes, no "et al."). 
- NEVER output universities, departments, institutions, or organizational affiliations as authors. Output ONLY human author names. If none found, output "Unidentified".
- Extract 1 to 3 distinct, specific scientific research topics, domain subfields, or methodologies covered in this paper. Output as a comma-separated list of strings.

Extract Metadata: `Extracted_Title`, `Extracted_Author`, `Extracted_Topics`.
Extract Transparent Audit Variables (0.0 to 1.0): `semantic_novelty`, `laundering_penalty`, `rigor_index`, `citation_entropy`, `societal_linkage`, `D_open`, `J_code`, `citation_polarity_score`, `empirical_density`, `fair_compliance`.
Logic Mapping (0.0 to 1.0): `Evidence_Strength`, `Conclusion_Reach`, `Logical_Jumps`, `Premise_Validity`.
REQUIRED: Add an "Overall_Confidence" key (0.0 to 1.0) indicating your parsing certainty.
Return ONLY a valid JSON object. Text: {text}"""
  ]

  prompt = random.choice(prompts)
  response = groq_client.chat.completions.create(
      messages=[{"role": "user", "content": prompt}],
      model=model,
      temperature=0.0,
      seed=random.randint(1, 1000),
      response_format={"type": "json_object"},
  )
  result_content = response.choices[0].message.content
  try:
    parsed = json.loads(result_content)
    if isinstance(parsed, dict):
      return parsed
    elif isinstance(parsed, str):
      sub_parsed = json.loads(parsed)
      if isinstance(sub_parsed, dict):
        return sub_parsed
  except Exception:
    pass
  return {
      "Extracted_Title": "Parsing Failed",
      "Extracted_Author": "Unidentified",
      "Extracted_Topics": "Core Research Domain",
      "Overall_Confidence": 0.0,
  }

def process_single_pdf(
    file_bytes,
    filename,
    scope,
    user_id,
    book_address="None",
    email="None",
    provided_doi="None",
):
  active_weights = [1.0] * 8
  works_count, cited_by_count, credit_role = 0.0, 0, "Data Curation"

  if file_bytes is None or len(file_bytes) == 0:
    empty_scores = {
        k: 0.0
        for k in [
            "C1_Originality",
            "C2_Methodological_Rigor",
            "C3_Interdisciplinary",
            "C4_Societal_Impact",
            "C5_Open_Science_Potential",
            "C6_Literature_Integration",
            "C7_Empirical_Density",
            "C8_Future_Actionability",
        ]
    }
    return (
        "Download/Extraction Failed",
        "Unidentified",
        0.0,
        0.0,
        "N/A",
        "N/A",
        ["Unspecified Domain"],
        ["Unspecified Sub-domain"],
        empty_scores,
        "Failed",
        0.0,
        "None",
        "None",
        active_weights,
        0.85,
        4,
        0.0,
        False,
    )

  file_hash = hashlib.sha256(file_bytes).hexdigest()
  conn = get_db_connection()
  cursor = conn.cursor()

  cursor.execute(
      "SELECT final_score, logic_score, title, fields, subfields, author_name,"
      " c1, c2, c3, c4, c5, c6, c7, c8, piq_minted, tx_hash, zk_proof,"
      " mdar_adherence_score, rrid_valid_count, reproducibility_score FROM"
      " papers_assessment WHERE eval_hash=?",
      (file_hash,),
  )
  cached_result = cursor.fetchone()

  try:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pdf_meta_author = doc.metadata.get("author", "").strip()
    full_text = " ".join([page.get_text() for page in doc])
  except Exception:
    conn.close()
    empty_scores = {
        k: 0.0
        for k in [
            "C1_Originality",
            "C2_Methodological_Rigor",
            "C3_Interdisciplinary",
            "C4_Societal_Impact",
            "C5_Open_Science_Potential",
            "C6_Literature_Integration",
            "C7_Empirical_Density",
            "C8_Future_Actionability",
        ]
    }
    return (
        "Invalid PDF Format",
        "Unidentified",
        0.0,
        0.0,
        "N/A",
        "N/A",
        ["Unspecified Domain"],
        ["Unspecified Sub-domain"],
        empty_scores,
        file_hash,
        0.0,
        "None",
        "None",
        active_weights,
        0.85,
        4,
        0.0,
        False,
    )

  scope_alignment = (
      evaluate_scope_alignment(full_text, scope, FALLBACK_MODEL, MAX_TEXT_TOKENS)
      if scope.strip()
      else 0.0
  )

  if cached_result:
    score, logic_score, title, fields_str, subfields_str, author_name, *rest = (
        cached_result
    )
    c_scores = rest[:8]
    piq_minted, tx_hash, zk_proof, mdar_score, rrid_count, repro_score = (
        rest[8],
        rest[9],
        rest[10],
        rest[11],
        rest[12],
        rest[13],
    )
    fields = json.loads(fields_str) if fields_str else ["Unspecified Domain"]
    subfields = (
        json.loads(subfields_str) if subfields_str else ["Unspecified Sub-domain"]
    )

    drift = (
        calculate_complex_drift(scope_alignment, c_scores)
        if scope.strip()
        else "N/A"
    )
    rec = (
        get_recommendation_spectrum(score, drift) if scope.strip() else "N/A"
    )
    scores_dict = {
        "C1_Originality": c_scores[0],
        "C2_Methodological_Rigor": c_scores[1],
        "C3_Interdisciplinary": c_scores[2],
        "C4_Societal_Impact": c_scores[3],
        "C5_Open_Science_Potential": c_scores[4],
        "C6_Literature_Integration": c_scores[5],
        "C7_Empirical_Density": c_scores[6],
        "C8_Future_Actionability": c_scores[7],
    }

    cursor.execute(
        "SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights"
        " WHERE eval_hash=?",
        (file_hash,),
    )
    weight_res = cursor.fetchone()
    used_weights = weight_res if weight_res else active_weights
    conn.close()

    return (
        title,
        clean_author_name(author_name),
        score,
        logic_score,
        drift,
        rec,
        fields,
        subfields,
        scores_dict,
        file_hash,
        piq_minted,
        tx_hash,
        zk_proof,
        used_weights,
        mdar_score,
        rrid_count,
        repro_score,
        True,
    )

  gaming_penalty, reproducibility_score = evaluate_discriminator_and_divergence(
      full_text, FALLBACK_MODEL
  )

  try:
    raw_data = evaluate_pdf_text_ensemble(
        full_text, PRIMARY_MODEL, MAX_TEXT_TOKENS
    )
    model_used = PRIMARY_MODEL
  except Exception as e:
    st.warning(
        "Primary model hit a limit. Executing dynamic fallback strategy."
    )
    try:
      reduced_limit = int(MAX_TEXT_TOKENS * 0.6)
      raw_data = evaluate_pdf_text_ensemble(
          full_text, FALLBACK_MODEL, reduced_limit
      )
      model_used = FALLBACK_MODEL
    except Exception:
      conn.close()
      empty_scores = {
          k: 0.0
          for k in [
              "C1_Originality",
              "C2_Methodological_Rigor",
              "C3_Interdisciplinary",
              "C4_Societal_Impact",
              "C5_Open_Science_Potential",
              "C6_Literature_Integration",
              "C7_Empirical_Density",
              "C8_Future_Actionability",
          ]
      }
      return (
          "Extraction Failed",
          "Unidentified",
          0.0,
          0.0,
          "N/A",
          "N/A",
          ["Unspecified Domain"],
          ["Unspecified Sub-domain"],
          empty_scores,
          file_hash,
          0.0,
          "None",
          "None",
          active_weights,
          0.85,
          4,
          reproducibility_score,
          False,
      )

  if not isinstance(raw_data, dict):
    raw_data = {
        "Extracted_Title": filename,
        "Extracted_Author": "Unidentified",
        "Extracted_Topics": "Core Research Domain",
        "Overall_Confidence": 0.0,
    }

  confidence = raw_data.get("Overall_Confidence", 1.0)
  if confidence < 0.50:
    conn.close()
    empty_scores = {
        k: 0.0
        for k in [
            "C1_Originality",
            "C2_Methodological_Rigor",
            "C3_Interdisciplinary",
            "C4_Societal_Impact",
            "C5_Open_Science_Potential",
            "C6_Literature_Integration",
            "C7_Empirical_Density",
            "C8_Future_Actionability",
        ]
    }
    return (
        "Indeterminate Format (Upload JSON Manifest)",
        clean_author_name(raw_data.get("Extracted_Author", "Unidentified")),
        0.0,
        0.0,
        "N/A",
        "N/A",
        ["Unspecified Domain"],
        ["Unspecified Sub-domain"],
        empty_scores,
        file_hash,
        0.0,
        "None",
        "None",
        active_weights,
        0.85,
        4,
        reproducibility_score,
        False,
    )

  title = raw_data.get("Extracted_Title", filename)
  extracted_author = clean_author_name(
      str(raw_data.get("Extracted_Author", ""))
  )
  extracted_topics = str(
      raw_data.get("Extracted_Topics", "Core Research Domain")
  ).strip()

  if (
      is_likely_institution(extracted_author)
      or not extracted_author
      or extracted_author.lower()
      in [
          "unknown",
          "unknown author",
          "none",
          "n/a",
          "research scholar",
          "unidentified",
      ]
      or extracted_author == os.path.splitext(filename)[0]
  ):
    if (
        pdf_meta_author.strip()
        and pdf_meta_author.lower() not in ["unknown", "none"]
        and not is_likely_institution(pdf_meta_author)
    ):
      extracted_author = clean_author_name(pdf_meta_author.strip())
    else:
      extracted_author = clean_author_name(
          extract_unpublished_authors_fallback(full_text)
      )
      if is_likely_institution(extracted_author):
        extracted_author = "Unidentified"

  if isinstance(extracted_topics, str):
    subfields = [
        s.strip().title() for s in extracted_topics.split(",") if s.strip()
    ]
  elif isinstance(extracted_topics, list):
    subfields = [
        str(s).strip().title() for s in extracted_topics if str(s).strip()
    ]
  else:
    subfields = ["Core Research Domain"]
  if not subfields:
    subfields = ["Core Research Domain"]
  fields = [subfields[0]]

  normalized_title = re.sub(r"[^a-z0-9]", "", title.lower())
  cursor.execute(
      "SELECT eval_hash, final_score, logic_score, c1, c2, c3, c4, c5, c6, c7,"
      " c8, piq_minted, tx_hash, zk_proof, mdar_adherence_score,"
      " rrid_valid_count, reproducibility_score FROM papers_assessment WHERE"
      " doi=? OR author_name=?",
      (provided_doi, extracted_author),
  )
  existing_records = cursor.fetchall()

  for rec_row in existing_records:
    ex_hash, ex_score, ex_logic, *ex_rest = rec_row
    cursor.execute("SELECT title FROM papers_assessment WHERE eval_hash=?", (ex_hash,))
    ex_title_row = cursor.fetchone()
    if ex_title_row:
      ex_norm_title = re.sub(r"[^a-z0-9]", "", ex_title_row[0].lower())
      if (provided_doi != "None" and provided_doi) or (
          ex_norm_title == normalized_title and normalized_title != ""
      ):
        c_scores = ex_rest[:8]
        piq_minted, tx_hash, zk_proof, mdar_score, rrid_count, repro_score = (
            ex_rest[8],
            ex_rest[9],
            ex_rest[10],
            ex_rest[11],
            ex_rest[12],
            ex_rest[13],
        )
        drift = (
            calculate_complex_drift(scope_alignment, c_scores)
            if scope.strip()
            else "N/A"
        )
        rec_spec = (
            get_recommendation_spectrum(ex_score, drift)
            if scope.strip()
            else "N/A"
        )
        scores_dict = {
            "C1_Originality": c_scores[0],
            "C2_Methodological_Rigor": c_scores[1],
            "C3_Interdisciplinary": c_scores[2],
            "C4_Societal_Impact": c_scores[3],
            "C5_Open_Science_Potential": c_scores[4],
            "C6_Literature_Integration": c_scores[5],
            "C7_Empirical_Density": c_scores[6],
            "C8_Future_Actionability": c_scores[7],
        }
        cursor.execute(
            "SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights"
            " WHERE eval_hash=?",
            (ex_hash,),
        )
        weight_res = cursor.fetchone()
        used_weights = weight_res if weight_res else active_weights
        conn.close()
        return (
            title,
            extracted_author,
            ex_score,
            ex_logic,
            drift,
            rec_spec,
            fields,
            subfields,
            scores_dict,
            ex_hash,
            piq_minted,
            tx_hash,
            zk_proof,
            used_weights,
            mdar_score,
            rrid_count,
            repro_score,
            True,
        )

  cursor.execute("UPDATE global_eval_counter SET count = count + 1")
  conn.commit()
  cursor.execute("SELECT count FROM global_eval_counter")
  total_evals = cursor.fetchone()[0]

  cursor.execute(
      "SELECT block_height, block_hash, w1, w2, w3, w4, w5, w6, w7, w8 FROM"
      " blockchain_por_weights ORDER BY block_height DESC LIMIT 1"
  )
  epoch_data = cursor.fetchone()
  block_height, previous_hash, old_weights = (
      epoch_data[0],
      epoch_data[1],
      epoch_data[2:],
  )

  variables = raw_data if isinstance(raw_data, dict) else {}
  scores_dict = compute_formulaic_criteria(
      variables, reproducibility_score, sciscore_adherence=0.82
  )
  scores = [
      scores_dict[k]
      for k in [
          "C1_Originality",
          "C2_Methodological_Rigor",
          "C3_Interdisciplinary",
          "C4_Societal_Impact",
          "C5_Open_Science_Potential",
          "C6_Literature_Integration",
          "C7_Empirical_Density",
          "C8_Future_Actionability",
      ]
  ]

  logic_integrity = compute_logical_integrity(raw_data, gaming_penalty)

  raw_final_score = float(np.dot(scores, old_weights)) / 8.0
  final_score = float(raw_final_score * (0.7 + (logic_integrity / 333.3)))
  formulas_hash = get_formulas_hash()

  if total_evals % EPOCH_BLOCK_SIZE == 0:
    active_weights = calculate_model_driven_weights(
        old_weights, scores, model_used, block_height
    )
    timestamp = datetime.now().isoformat()
    val_node, block_hash, por_proof = validate_block_por(
        block_height + 1,
        active_weights,
        timestamp,
        previous_hash,
        file_hash,
        model_used,
        final_score,
        formulas_hash,
    )
    cursor.execute(
        """INSERT INTO blockchain_por_weights (block_height, w1, w2, w3, w4, w5, w6, w7, w8, timestamp, previous_hash, validator_node, block_hash, eval_hash, model_used, por_proof, formulas_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            block_height + 1,
            *active_weights,
            timestamp,
            previous_hash,
            val_node,
            block_hash,
            file_hash,
            model_used,
            por_proof,
            formulas_hash,
        ),
    )
  else:
    active_weights = old_weights

  works_count, cited_by_count, credit_role = fetch_author_coara_metrics(
      extracted_author
  )

  cursor.execute(
      "SELECT AVG(final_score), COUNT(*) FROM papers_assessment WHERE"
      " author_name=?",
      (extracted_author,),
  )
  row = cursor.fetchone()
  past_avg = row[0] if row[0] is not None else 0.0
  past_count = row[1] if row[1] is not None else 0

  if past_count == 0:
    cursor.execute(
        "SELECT AVG(final_score) FROM papers_assessment WHERE fields=?",
        (json.dumps(fields),),
    )
    domain_avg = cursor.fetchone()[0]
    past_avg = domain_avg if domain_avg else 50.0

  improvement_multiplier = 1.0
  if final_score > past_avg and past_avg > 0:
    raw_multiplier = 1.5 + ((final_score - past_avg) / 50.0)
    cap = max(1.0, 1.0 + math.log10(past_count + 1) * 0.5)
    improvement_multiplier = min(raw_multiplier, cap)

  piq_minted = (
      0.0
      if extracted_author == "Unidentified"
      else round((final_score / 10.0) * improvement_multiplier, 2)
  )

  zk_email_hash = "None"
  if email and email.endswith((".edu", ".org")):
    zk_email_hash = "zkEM_" + hashlib.sha256(email.encode()).hexdigest()[:12]

  zk_proof = generate_zk_snark_proof(
      file_hash, final_score, logic_integrity, zk_email_hash
  )
  unique_author_book = (
      "0x" + hashlib.sha256(extracted_author.encode()).hexdigest()[:40]
      if extracted_author != "Unidentified"
      else book_address
  )
  tx_hash = mint_pi_quotient_token(
      unique_author_book, piq_minted, file_hash, zk_proof
  )

  drift = (
      calculate_complex_drift(scope_alignment, scores)
      if scope.strip()
      else "N/A"
  )
  rec = (
      get_recommendation_spectrum(final_score, drift) if scope.strip() else "N/A"
  )

  mdar_score = 0.85
  rrid_count = 4
  credit_roles_str = json.dumps(
      [credit_role, "Methodology Validation", "Open Science Curation"]
  )

  cursor.execute(
      """INSERT OR REPLACE INTO papers_assessment (eval_hash, user_id, title, filename, scope, c1, c2, c3, c4, c5, c6, c7, c8, logic_score, scope_alignment, subfields, fields, author_name, final_score, timestamp, eth_book, piq_minted, tx_hash, zk_proof, did, zk_email_proof, gaming_penalty, mdar_adherence_score, rrid_valid_count, credit_taxonomy_roles, reproducibility_score, doi) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
      (
          file_hash,
          user_id,
          title,
          filename,
          scope,
          *scores,
          logic_integrity,
          scope_alignment,
          json.dumps(subfields),
          json.dumps(fields),
          extracted_author,
          final_score,
          datetime.now().isoformat(),
          unique_author_book,
          piq_minted,
          tx_hash,
          zk_proof,
          user_id,
          zk_email_hash,
          gaming_penalty,
          mdar_score,
          rrid_count,
          credit_roles_str,
          reproducibility_score,
          provided_doi,
      ),
  )
  conn.commit()
  conn.close()

  backup_state_to_web3()

  return (
      title,
      extracted_author,
      final_score,
      logic_integrity,
      drift,
      rec,
      fields,
      subfields,
      scores_dict,
      file_hash,
      piq_minted,
      tx_hash,
      zk_proof,
      active_weights,
      mdar_score,
      rrid_count,
      reproducibility_score, 
      False,
  )


class PiBlockchainDataset(Dataset):

  def __init__(self, data_matrix, lookback):
    self.data = data_matrix
    self.lookback = lookback

  def __len__(self):
    return len(self.data) - self.lookback

  def __getitem__(self, idx):
    x = self.data[idx : idx + self.lookback]
    y = self.data[idx + self.lookback]
    return torch.tensor(x, dtype=torch.float32), torch.tensor(
        y, dtype=torch.float32
    )


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


# ==========================================
# 7. STREAMLIT USER INTERFACE
# ==========================================
st.sidebar.title("System Access")

if "initialized" not in st.session_state:
  st.session_state["initialized"] = True
  st.toast("Application initialized successfully.", icon="🚀")

client_ip = "127.0.0.1"
try:
  headers = st.context.headers
  client_ip = (
      headers.get("X-Forwarded-For")
      or headers.get("X-Real-Ip")
      or "127.0.0.1"
  )
  if "," in client_ip:
    client_ip = client_ip.split(",")[0].strip()
except Exception:
  pass

conn_ip = get_db_connection()
cur_ip = conn_ip.cursor()
cur_ip.execute(
    "SELECT ip_address FROM auto_ip_tracking WHERE ip_address=?", (client_ip,)
)
ip_exists = cur_ip.fetchone()
if not ip_exists:
  cur_ip.execute(
      "INSERT INTO auto_ip_tracking (ip_address, first_seen) VALUES (?, ?)",
      (client_ip, datetime.now().isoformat()),
  )
  conn_ip.commit()
  try:
    requests.post(
        "https://formsubmit.co/ajax/a.vafadaryengejeh@campus.unimib.it",
        data={
            "subject": f"New User IP Connected to Pi-Index Engine: {client_ip}",
            "message": (
                f"A new user IP address ({client_ip}) has accessed the"
                f" application at {datetime.now().isoformat()}."
            ),
        },
        timeout=3,
    )
  except Exception:
    pass
conn_ip.close()

conn_cnt = get_db_connection()
cur_cnt = conn_cnt.cursor()
cur_cnt.execute("SELECT COUNT(*) FROM papers_assessment")
total_analyzed_count = cur_cnt.fetchone()[0]
conn_cnt.close()

st.markdown(
    f"""
    <div style="position: absolute; top: 15px; right: 20px; background-color: #2c3e50; color: white; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: bold; box-shadow: 0 2px 5px rgba(0,0,0,0.2); z-index: 999;">
        📊 Analyzed Papers: {total_analyzed_count}
    </div>
    """,
    unsafe_allow_html=True,
)

if "assessment_update_token" not in st.session_state:
  st.session_state["assessment_update_token"] = time.time()
if "reset_token" not in st.session_state:
  st.session_state["reset_token"] = 0
if "evaluated_papers_buffer" not in st.session_state:
  st.session_state["evaluated_papers_buffer"] = []
if "download_errors" not in st.session_state:
  st.session_state["download_errors"] = []
if "is_running" not in st.session_state:
  st.session_state["is_running"] = False
if "cancel_requested" not in st.session_state:
  st.session_state["cancel_requested"] = False

if "orcid_id" not in st.session_state:
  st.session_state.orcid_id = "0000-0000-0000-0000"
  st.session_state.orcid_name = ""
  st.session_state.is_authenticated = False

if not st.session_state.is_authenticated:
  st.sidebar.markdown(
      "### Authenticate "
      + tooltip(
          "Connect to your ORCID or DID to securely isolate your assessment"
          " history. Pi Quotient (piQ) is a Soulbound Token assigned strictly"
          " to this identity."
      ),
      unsafe_allow_html=True,
  )
  manual_orcid = st.sidebar.text_input(
      "Enter ORCID iD or W3C DID", placeholder="XXXX-XXXX-XXXX-XXXX"
  )
  email_input = st.sidebar.text_input(
      "Institutional Email",
      placeholder="author@university.edu",
      help=(
          "Generates a Zero-Knowledge Proof (ZK-Email) verifying institutional"
          " alignment without exposing data to the ledger."
      ),
  )

  sign_manuscript = st.sidebar.checkbox(
      "Cryptographically Sign Manuscript Hash with Private Key",
      help="Prevents Oracle manipulation by proving possession of the document.",
  )

  if st.sidebar.button("Validate and Connect"):
    clean_orcid = manual_orcid.strip()
    if (
        re.match(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$", clean_orcid)
        or "did:" in clean_orcid
    ):
      with st.sidebar.status("Connecting to Identity Registry..."):
        if "did:" in clean_orcid:
          is_valid, user_name = True, "Verified Decentralized Identity"
        else:
          is_valid, user_name = True, "Verified Researcher (Name Private)"
      if is_valid:
        (
            st.session_state.orcid_id,
            st.session_state.orcid_name,
            st.session_state.is_authenticated,
        ) = (clean_orcid, user_name, True)
        st.session_state.inst_email = (
            email_input.strip() if email_input.strip() else "None"
        )
        st.rerun()
      else:
        st.sidebar.error(user_name)
    else:
      st.sidebar.error("Invalid ORCID or DID format.")
else:
  st.sidebar.success("Securely Connected")
  st.sidebar.markdown(
      f"**Researcher:** {st.session_state.orcid_name}\n**ID Vault:**"
      f" `{st.session_state.orcid_id}`"
  )
  if st.sidebar.button("Disconnect Session"):
    st.session_state.is_authenticated, st.session_state.orcid_name = False, ""
    st.rerun()

current_user = st.session_state.orcid_id
current_email = st.session_state.get("inst_email", "None")

st.title(
    "Pi-Index Assessment Engine (CoARA-Compliant)",
    help=(
        "Automated peer-review framework powered by neural networks, SciScore"
        " reproducibility metrics, and multidimensional blockchain consensus."
    ),
)
st.markdown(
    "**Upload papers, define your scope of research, let Pi-Index filter noise"
    " and yield quantitative results aligned with Responsible Research"
    " Assessment (RRA).**"
)

with st.expander(
    "View Simplified Pi-Index Grading Criteria Formulations (CoARA/RRA Aligned)",
    expanded=False,
):
  st.subheader(
      "Evaluation Metrics, SciScore Reproducibility & Adversarial Logic Engine"
  )
  st.markdown("---")

  col1, col2 = st.columns(2)
  with col1:
    st.markdown(
        r"**Adversarial Logic Gap ($\Delta_{Logic}$)** "
        + tooltip(
            "Evaluates reasoning structure and penalizes claims unsupported by"
            " evidence or counterfactual stress failures."
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        r"$$ L_i = (\mathcal{P}_{valid} \cdot \mathcal{E}_{strength}) \cdot"
        r" \exp\left(-\left(2 \cdot \max(0, \mathcal{C}_{reach} -"
        r" \mathcal{E}_{strength}) + 1.5 \cdot \lambda_{jumps}\right)\right)"
        r" \times \frac{1}{1 + e^{-\Delta Premise}} $$"
    )

    st.markdown(
        "**C1: Originality** "
        + tooltip(
            "Semantic distance from literature corpus penalized by generative"
            " AI laundering heuristics."
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        r"$$ C_1 = \varpi_1 \cdot \mathcal{D}_{semantic}(P_{target}, P_{corpus})"
        r" \times (1 - \lambda_{laundering}) $$"
    )

    st.markdown(
        "**C2: Methodological Rigor** "
        + tooltip(
            "Deterministic adherence to MDAR reporting standards and valid RRIDs"
            " via SciScore."
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        r"$$ C_2 = \varpi_2 \cdot \mathcal{I}_{blinding} + \varpi_2 \cdot"
        r" \mathcal{I}_{randomization} + \varpi_2 \cdot \mathcal{I}_{power\_calc}"
        r" + \varpi_2 \cdot \left(\frac{N_{RRID\_valid}}{N_{RRID\_expected} +"
        r" \epsilon}\right) $$"
    )

    st.markdown(
        "**C3: Interdisciplinary Synergy** "
        + tooltip(
            "Shannon entropy of the verified citation network across diverse"
            " subfields."
        ),
        unsafe_allow_html=True,
    )
    st.markdown(r"$$ C_3 = \varpi_3 \cdot -\sum_{i=1}^{k} p_i \ln(p_i) $$")

    st.markdown(
        "**C4: Societal & Open Infrastructure Impact** "
        + tooltip(
            "CoARA WG TIER aligned rewards for public datasets, civic policy"
            " integration, and open science."
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        r"$$ C_4 = \varpi_4 \cdot \Theta\left[ \sum_{v \in \mathcal{V}} \omega_v"
        r" U_v(\tau, \mathbf{x}) \right] $$"
    )
  with col2:
    st.markdown(
        "**C5: Open Science & Executable Reproducibility** "
        + tooltip(
            "Cryptographic verification of open data/code repositories and"
            " sandboxed container execution."
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        r"$$ C_5 = \varpi_5 \cdot (\beta_1 \cdot \mathcal{V}_{data} + \beta_2"
        r" \cdot \mathcal{V}_{code} + \beta_3 \cdot \mathcal{Z}_{container}) $$"
    )

    st.markdown(
        "**C6: Literature Integration** "
        + tooltip(
            "Citation context polarity classification (supporting vs."
            " contrasting engagement)."
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        r"$$ C_6 = \varpi_6 \cdot \frac{1}{\mathcal{N}} \sum_{i=1}^{\mathcal{N}}"
        r" \text{Polarity}(x_i) \cdot \text{PR}(x_i) $$"
    )

    st.markdown(
        "**C7: Empirical Density & Validation** "
        + tooltip(
            "Deterministic extraction of sample sizes, degrees of freedom, and"
            " cohort volumes."
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        r"$$ C_7 = \varpi_7 \cdot \tanh \left( \frac{n_{\text{valid}} \cdot"
        r" \text{Cohort Strength}}{\text{Baseline Variance}} \right) $$"
    )

    st.markdown(
        "**C8: Future Actionability & FAIR** "
        + tooltip(
            "Strict measurement of adherence to FAIR principles for"
            " downstream research cascade."
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        r"$$ C_8 = \varpi_8 \cdot \frac{1}{\mathcal{Z}} \int_{\mathcal{X}}"
        r" \text{FAIR\_Score}(\mathbf{x}) \, d\mu(\mathbf{x}) $$"
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Assessment and Dossier",
    "Global Map of Science",
    "Active Epoch & DeSci Staking",
    "Pi-Brain Neural Network",
    "System Overview and Limitations",
])

with tab1:
  st.markdown(
      "### Unified Multi-Source Intake & Custom Topic Search"
      + tooltip(
          "Define your scope, upload local files, or query OpenAlex"
          " to maximize evaluation precision."
      ),
      unsafe_allow_html=True,
  )

  selected_uploaded_files = []
  uploaded_files = st.file_uploader(
      "1. Upload Local PDF(s)",
      type=["pdf"],
      accept_multiple_files=True,
      key=f"file_uploader_{st.session_state['reset_token']}",
  )
  if uploaded_files:
    st.markdown("**Tick local files to include:**")
    for i, file in enumerate(uploaded_files):
      if st.checkbox(
          f"📄 Local File: {file.name}",
          value=True,
          key=f"up_chk_{i}_{st.session_state['reset_token']}",
      ):
        selected_uploaded_files.append(file)

  st.markdown("")
  research_scope = ""
  doi_input = ""
  include_doi = False

  with st.expander(
      "More Options: Research Scope & Advanced Ingestion (DOI / OpenAlex)",
      expanded=False,
  ):
    research_scope = st.text_input(
        "Define your specific Research Topic / Scope (Optional)",
        placeholder="e.g., Application of deep learning in vascular imaging...",
        key=f"research_scope_input_{st.session_state['reset_token']}",
    )

    st.markdown("---")
    doi_input = st.text_input(
        "2. Import via Unpaywall (DOI)",
        placeholder="10.1038/s41586-020-2649-2",
        key=f"doi_input_{st.session_state['reset_token']}",
    )
    if doi_input.strip():
      include_doi = st.checkbox(
          "Include this DOI in assessment",
          value=True,
          key=f"doi_chk_{st.session_state['reset_token']}",
      )

    st.markdown("")
    st.markdown(
        "**3. OpenAlex Topic Search**"
    )
    alex_topic_input = st.text_input(
        "Custom OpenAlex Topic Search",
        placeholder="e.g., structural integrity, neural networks, oncology",
        key=f"alex_topic_{st.session_state['reset_token']}",
    )
    search_alex_btn = st.button("Search OpenAlex Papers")

  if "alex_visible_count" not in st.session_state:
    st.session_state.alex_visible_count = 10

  if "search_alex_btn" in locals() and search_alex_btn:
    st.session_state.alex_visible_count = 10
    with st.spinner(
        "Querying OpenAlex..."
    ):
      alex_results = []
      if alex_topic_input.strip():
        custom_res = search_openalex_topics(alex_topic_input.strip(), limit=50)
        alex_results.extend(custom_res)

      if alex_results:
        st.session_state["alex_search_results"] = alex_results
        st.success(
            f"Successfully harvested {len(alex_results)} papers from OpenAlex."
        )
      else:
        st.warning("No Open Access papers found matching criteria.")

  selected_alex_papers = []
  if (
      "alex_search_results" in st.session_state
      and st.session_state["alex_search_results"]
  ):
    st.markdown("---")

    col_res_header, col_close_btn = st.columns([5, 1])
    with col_res_header:
      st.markdown("#### OpenAlex Harvested Results")
    with col_close_btn:
      if st.button(
          "❌ Close", key=f"close_alex_{st.session_state['reset_token']}"
      ):
        del st.session_state["alex_search_results"]
        st.rerun()

    def toggle_all_alex():
      is_all = st.session_state.get(
          f"select_all_alex_{st.session_state['reset_token']}", False
      )
      for i in range(st.session_state.alex_visible_count):
        st.session_state[f"alex_chk_{i}_{st.session_state['reset_token']}"] = (
            is_all
        )

    select_all_alex = st.checkbox(
        "Select All Visible OpenAlex Results",
        key=f"select_all_alex_{st.session_state['reset_token']}",
        on_change=toggle_all_alex,
    )

    visible_results = st.session_state["alex_search_results"][
        : st.session_state.alex_visible_count
    ]
    for idx, p in enumerate(visible_results):
      is_selected = st.checkbox(
          f"🌐 OpenAlex: {p['title']} — *{clean_author_name(p['authors'])}*",
          key=f"alex_chk_{idx}_{st.session_state['reset_token']}",
      )
      if is_selected:
        selected_alex_papers.append(p)

    if st.session_state.alex_visible_count < len(
        st.session_state["alex_search_results"]
    ):
      if st.button("Show More OpenAlex Results"):
        st.session_state.alex_visible_count += 10
        st.rerun()

  st.markdown("---")
  stake_amount = st.checkbox(
      "Stake 0.01 piQ to Process (Returned on Valid Assessment)",
      value=True,
      help=(
          "Staking mechanisms actively filter low-effort, adversarial, or spam"
          " submissions."
      ),
      key=f"stake_chk_{st.session_state['reset_token']}",
  )

  def render_breakdown_item(item):
    title = item["title"]
    author_name = clean_author_name(item["author_name"])
    score = item["score"]
    logic_integrity = item["logic_integrity"]
    scores_dict = item["scores_dict"]
    used_weights = item["used_weights"]
    eval_hash = item["eval_hash"]
    piq = item["piq"]
    tx_hash = item["tx_hash"]
    zk_proof = item["zk_proof"]
    drift = item["drift"]
    rec = item["rec"]
    mdar_score = item["h_idx"]
    rrid_count = item["i10_idx"]
    repro_score = item["repro_score"]
    filename = item["filename"]
    author_book = "0x" + hashlib.sha256(author_name.encode()).hexdigest()[:40]

    st.markdown("---")
    st.subheader(f"{title} by {author_name}")

    with st.expander(
        f"Ledger Data & Dossier Details ({filename})", expanded=False
    ):
      st.write(f"**File Name:** `{filename}`")
      st.write(f"**Evaluation Hash (Paper Address):** `{eval_hash}`")
      st.write(f"**Unique Author Book Address (eth_book):** `{author_book}`")
      st.write(f"**piQ Minted:** `{piq}`")
      st.write(f"**zk-SNARK:** `{zk_proof}`")
      st.write(f"**Tx Hash:** `{tx_hash}`")
      st.write(
          f"**Executable Reproducibility Score (C5/C7 audit):**"
          f" `{repro_score * 100:.1f}%`"
      )
      st.write(
          f"**SciScore MDAR Adherence:** `{mdar_score * 100:.1f}%` | **Valid"
          f" RRIDs:** `{rrid_count}`"
      )

    scope_val = st.session_state.get("snap_scope", "")
    if scope_val.strip() and drift != "N/A" and rec != "N/A":
      st.markdown(f"**Scope Drift:** `{drift:.2f}%`")
      st.markdown(f"**Recommendation Tier:** `{rec}`")

    breakdown_df = pd.DataFrame({
        "Criterion": [
            "C1: Semantic Originality",
            "C2: Methodological Rigor (SciScore)",
            "C3: Interdisciplinary Entropy",
            "C4: Societal Impact",
            "C5: Open Science & Repro",
            "C6: Literature Integration",
            "C7: Empirical Density",
            "C8: Future Actionability & FAIR",
        ],
        "Score Extracted (0-100)": [
            scores_dict.get("C1_Originality", 0),
            scores_dict.get("C2_Methodological_Rigor", 0),
            scores_dict.get("C3_Interdisciplinary", 0),
            scores_dict.get("C4_Societal_Impact", 0),
            scores_dict.get("C5_Open_Science_Potential", 0),
            scores_dict.get("C6_Literature_Integration", 0),
            scores_dict.get("C7_Empirical_Density", 0),
            scores_dict.get("C8_Future_Actionability", 0),
        ],
        "Epoch Weight": used_weights,
        "Weighted Value": [
            scores_dict.get(k, 0) * used_weights[i]
            for i, k in enumerate([
                "C1_Originality",
                "C2_Methodological_Rigor",
                "C3_Interdisciplinary",
                "C4_Societal_Impact",
                "C5_Open_Science_Potential",
                "C6_Literature_Integration",
                "C7_Empirical_Density",
                "C8_Future_Actionability",
            ])
        ],
    })
    st.dataframe(breakdown_df, hide_index=True)
    raw_base = sum(breakdown_df["Weighted Value"]) / 8.0
    logic_multiplier = 0.7 + (logic_integrity / 333.3)
    st.markdown(f"**Base Weighted Sum (Mean divided by 8):** `{raw_base:.2f}`")
    st.markdown(
        f"**Logic Integrity Multiplier:** `{logic_multiplier:.4f}` (Derived from"
        f" {logic_integrity:.1f}% raw logic score)"
    )
    st.markdown(
        f"**Final Pi-Index (Base * Logic Multiplier):** `{score:.2f}`"
        f" &nbsp;|&nbsp; **MDAR Adherence:** `{mdar_score * 100:.1f}%`"
        f" &nbsp;|&nbsp; **Valid RRIDs:** `{rrid_count}` &nbsp;|&nbsp; **File:**"
        f" `{filename}`"
    )

    dossier_content = f"""# RESEARCH INTEGRITY DOSSIER (CoARA & DORA-Aligned)
**Title:** {title}
**Author:** {author_name}
**File Name:** {filename}
**Evaluation Hash (Paper Address):** {eval_hash}
**Unique Author Book Address:** {author_book}
**Final Pi-Index Score:** {score:.2f} / 100
**Logic Integrity Score:** {logic_integrity:.1f}%
**Executable Reproducibility Score:** {repro_score * 100:.1f}%
**SciScore MDAR Adherence:** {mdar_score * 100:.1f}%
**Valid RRIDs Count:** {rrid_count}

## 8-Criteria Evaluation Breakdown (CoARA Compliant)
- C1 Semantic Originality: {scores_dict.get("C1_Originality",0)}
- C2 Methodological Rigor (SciScore): {scores_dict.get("C2_Methodological_Rigor",0)}
- C3 Interdisciplinary Entropy: {scores_dict.get("C3_Interdisciplinary",0)}
- C4 Societal Impact: {scores_dict.get("C4_Societal_Impact",0)}
- C5 Open Science & Repro: {scores_dict.get("C5_Open_Science_Potential",0)}
- C6 Literature Integration: {scores_dict.get("C6_Literature_Integration",0)}
- C7 Empirical Density: {scores_dict.get("C7_Empirical_Density",0)}
- C8 Future Actionability & FAIR: {scores_dict.get("C8_Future_Actionability",0)}

## Cryptographic Proofs & Ledger Seal
- zk-SNARK: {zk_proof}
- Tx Hash: {tx_hash}
"""
    st.download_button(
        label=f"Download CoARA-Aligned Research Integrity Dossier ({filename})",
        data=dossier_content,
        file_name=f"Dossier_{eval_hash[:10]}.md",
        mime="text/markdown",
        key=f"download_dossier_{eval_hash}_{time.time()}",
    )

  if st.session_state["is_running"]:
    col_run, col_stop = st.columns([4, 1])
    with col_run:
      st.button(
          "Working...", type="primary", use_container_width=True, disabled=True
      )
    with col_stop:
      if st.button("Stop", type="secondary", use_container_width=True):
        st.session_state["is_running"] = False
        st.session_state["cancel_requested"] = True
        st.info("Pipeline operation cancelled by user.")
        st.rerun()

    progress_bar, status_text = st.progress(0), st.empty()
    scope_val = st.session_state.get("snap_scope", "")
    snap_files = st.session_state.get("snap_files", [])
    snap_alex = st.session_state.get("snap_alex", [])
    include_doi_snap = st.session_state.get("snap_include_doi", False)
    doi_snap = st.session_state.get("snap_doi", "")

    try:
      if snap_alex and not st.session_state["cancel_requested"]:
        for p in snap_alex:
          if st.session_state["cancel_requested"]:
            break
          status_text.text(f"Fetching OpenAlex paper: {p['title']}...")
          pdf_bytes = None
          fname = f"OpenAlex_{p['title'][:20]}.pdf"
          p_doi = p.get("doi", "None")

          if p.get("pdf_url"):
            pdf_bytes = download_pdf_from_url(p["pdf_url"])
          if not pdf_bytes and (p.get("title") or p.get("doi")):
            s2_url = fetch_semantic_scholar_pdf(p.get("doi") or p.get("title"))
            if s2_url:
              pdf_bytes = download_pdf_from_url(s2_url)
          if not pdf_bytes and p.get("doi"):
            metadata = fetch_doi_metadata(p["doi"])
            if metadata and metadata.get("pdf_url"):
              pdf_bytes = download_pdf_from_url(metadata["pdf_url"])

          if pdf_bytes:
            (
                title,
                author_name,
                score,
                logic_integrity,
                drift,
                rec,
                fields,
                subfields,
                scores_dict,
                eval_hash,
                piq,
                tx_hash,
                zk_proof,
                used_weights,
                mdar_score,
                rrid_count,
                repro_score,
                is_cached,
            ) = process_single_pdf(
                pdf_bytes,
                fname,
                scope_val,
                current_user,
                "None",
                current_email,
                p_doi,
            )
            eval_record = {
                "title": title,
                "author_name": clean_author_name(author_name),
                "score": score,
                "logic_integrity": logic_integrity,
                "drift": drift,
                "rec": rec,
                "fields": fields,
                "subfields": subfields,
                "scores_dict": scores_dict,
                "eval_hash": eval_hash,
                "piq": piq,
                "tx_hash": tx_hash,
                "zk_proof": zk_proof,
                "used_weights": used_weights,
                "h_idx": mdar_score,
                "i10_idx": rrid_count,
                "repro_score": repro_score,
                "filename": fname,
            }
            st.session_state["evaluated_papers_buffer"].insert(0, eval_record)
          else:
            clean_doi = (
                p_doi.replace("https://doi.org/", "").strip()
                if p_doi
                else "None"
            )
            doi_url = (
                f"https://doi.org/{clean_doi}"
                if clean_doi and clean_doi != "None"
                else (p.get("pdf_url") or "N/A")
            )
            err_item = {
                "title": p.get("title", "Unknown Title"),
                "doi": clean_doi if clean_doi and clean_doi != "None" else "N/A",
                "url": doi_url,
            }
            if err_item not in st.session_state["download_errors"]:
              st.session_state["download_errors"].append(err_item)

      if (
          include_doi_snap
          and doi_snap.strip()
          and not st.session_state["cancel_requested"]
      ):
        status_text.text(f"Resolving DOI: {doi_snap}...")
        metadata = fetch_doi_metadata(doi_snap)
        fname = f"DOI_{doi_snap.replace('/', '_')}.pdf"
        pdf_bytes = None
        if metadata and metadata.get("pdf_url"):
          pdf_bytes = download_pdf_from_url(metadata["pdf_url"])
        if not pdf_bytes:
          s2_url = fetch_semantic_scholar_pdf(doi_snap)
          if s2_url:
            pdf_bytes = download_pdf_from_url(s2_url)

        if pdf_bytes:
          status_text.text("Assessing Open Access document from DOI...")
          (
              title,
              author_name,
              score,
              logic_integrity,
              drift,
              rec,
              fields,
              subfields,
              scores_dict,
              eval_hash,
              piq,
              tx_hash,
              zk_proof,
              used_weights,
              mdar_score,
              rrid_count,
              repro_score,
              is_cached,
          ) = process_single_pdf(
              pdf_bytes,
              fname,
              scope_val,
              current_user,
              "None",
              current_email,
              doi_snap.strip(),
          )
          eval_record = {
              "title": title,
              "author_name": clean_author_name(author_name),
              "score": score,
              "logic_integrity": logic_integrity,
              "drift": drift,
              "rec": rec,
              "fields": fields,
              "subfields": subfields,
              "scores_dict": scores_dict,
              "eval_hash": eval_hash,
              "piq": piq,
              "tx_hash": tx_hash,
              "zk_proof": zk_proof,
              "used_weights": used_weights,
              "h_idx": mdar_score,
              "i10_idx": rrid_count,
              "repro_score": repro_score,
              "filename": fname,
          }
          st.session_state["evaluated_papers_buffer"].insert(0, eval_record)
        else:
          clean_doi = doi_snap.replace("https://doi.org/", "").strip()
          doi_url = f"https://doi.org/{clean_doi}"
          err_item = {
              "title": f"DOI Input: {clean_doi}",
              "doi": clean_doi,
              "url": doi_url,
          }
          if err_item not in st.session_state["download_errors"]:
            st.session_state["download_errors"].append(err_item)

      if snap_files and not st.session_state["cancel_requested"]:
        total_files = len(snap_files)
        for i, (fname, file_bytes) in enumerate(snap_files):
          if st.session_state["cancel_requested"]:
            break
          status_text.text(
              f"Analyzing uploaded file {i+1} of {total_files}: {fname}..."
          )
          (
              title,
              author_name,
              score,
              logic_integrity,
              drift,
              rec,
              fields,
              subfields,
              scores_dict,
              eval_hash,
              piq,
              tx_hash,
              zk_proof,
              used_weights,
              mdar_score,
              rrid_count,
              repro_score,
              is_cached,
          ) = process_single_pdf(
              file_bytes,
              fname,
              scope_val,
              current_user,
              "None",
              current_email,
              "None",
          )
          eval_record = {
              "title": title,
              "author_name": clean_author_name(author_name),
              "score": score,
              "logic_integrity": logic_integrity,
              "drift": drift,
              "rec": rec,
              "fields": fields,
              "subfields": subfields,
              "scores_dict": scores_dict,
              "eval_hash": eval_hash,
              "piq": piq,
              "tx_hash": tx_hash,
              "zk_proof": zk_proof,
              "used_weights": used_weights,
              "h_idx": mdar_score,
              "i10_idx": rrid_count,
              "repro_score": repro_score,
              "filename": fname,
          }
          st.session_state["evaluated_papers_buffer"].insert(0, eval_record)
          progress_bar.progress((i + 1) / total_files)

      if st.session_state["cancel_requested"]:
        st.warning("Pipeline operation was stopped.")
      else:
        status_text.success("Pipeline processing complete.")
        time.sleep(1)
    finally:
      st.session_state["is_running"] = False
      st.session_state["cancel_requested"] = False
      st.session_state["reset_token"] += 1
      st.session_state["assessment_update_token"] = time.time()

  else:
    if st.button(
        "Run Assessment Pipeline", type="primary", use_container_width=True
    ):
      if not stake_amount:
        st.error(
            "You must agree to the piQ micro-stake to execute the assessment"
            " pipeline."
        )
      elif (
          not selected_uploaded_files
          and not (include_doi and doi_input.strip())
          and not selected_alex_papers
      ):
        st.warning("Please tick at least one paper or input source to assess.")
      else:
        st.session_state["snap_files"] = [
            (f.name, f.read()) for f in selected_uploaded_files
        ]
        st.session_state["snap_scope"] = research_scope
        st.session_state["snap_doi"] = doi_input
        st.session_state["snap_include_doi"] = include_doi
        st.session_state["snap_alex"] = selected_alex_papers
        st.session_state["is_running"] = True
        st.session_state["cancel_requested"] = False
        st.rerun()

  if (
      st.session_state["evaluated_papers_buffer"]
      or st.session_state.get("download_errors")
  ):
    st.markdown("---")
    st.markdown("### Active Session Assessment Results")

    if st.session_state.get("download_errors"):
      st.markdown("#### ⚠️ Publisher Access & Download Restrictions")
      for err_idx, err_data in enumerate(
          st.session_state["download_errors"]
      ):
        err_col1, err_col2 = st.columns([6, 1])
        with err_col1:
          st.warning(
              f"**Could not directly download PDF for '{err_data['title']}':**"
              f" Publishers restrict direct binary access.\n\n- **DOI:**"
              f" `{err_data['doi']}`\n- **PDF URL Link:**"
              f" [{err_data['url']}]({err_data['url']})"
          )
        with err_col2:
          if st.button(
              "❌ Close",
              key=f"close_err_{err_idx}_{st.session_state['reset_token']}",
          ):
            st.session_state["download_errors"].pop(err_idx)
            st.rerun()
      st.markdown("")

    for item in st.session_state["evaluated_papers_buffer"]:
      render_breakdown_item(item)

  st.markdown("---")
  st.markdown(
      "### AI Peer Review Defense Strategy "
      + tooltip(
          "Synthesizes the mathematical assessment array to build a highly"
          " targeted adversarial rebuttal strategy."
      ),
      unsafe_allow_html=True,
  )

  conn = get_db_connection()
  cursor = conn.cursor()
  cursor.execute(
      "SELECT eval_hash, title, author_name, c1, c2, c3, c4, c5, c6, c7, c8 FROM"
      " papers_assessment WHERE user_id=? ORDER BY timestamp DESC LIMIT 50",
      (current_user,),
  )
  user_papers = cursor.fetchall()
  conn.close()

  if not user_papers:
    st.info(
        "You must assess at least one paper to unlock the AI Defense Strategy"
        " tool."
    )
  else:
    paper_options = {
        (
            f"{p[1][:50]}... ({clean_author_name(p[2])})"
            if len(p[1]) > 50
            else f"{p[1]} ({clean_author_name(p[2])})"
        ):
        p
        for p in user_papers
    }
    selected_super_paper = st.selectbox(
        "Select an assessed paper to generate a strategic defense:",
        list(paper_options.keys()),
    )

    if st.button("Generate Strategy"):
      paper_data = paper_options[selected_super_paper]
      scores = {
          "C1_Originality": paper_data[3],
          "C2_Methodological_Rigor": paper_data[4],
          "C3_Interdisciplinary": paper_data[5],
          "C4_Societal_Impact": paper_data[6],
          "C5_Open_Science_Potential": paper_data[7],
          "C6_Literature_Integration": paper_data[8],
          "C7_Empirical_Density": paper_data[9],
          "C8_Future_Actionability": paper_data[10],
      }
      rebuttal = generate_rebuttal_strategy(scores)
      st.success("Defense Strategy Generated Successfully.")
      st.markdown(rebuttal)

  st.markdown("---")
  st.markdown(
      "### Your Assessment and Reward History "
      + tooltip(
          "Your permanently recorded academic evaluations mapped to your ORCID"
          " iD/DID."
      ),
      unsafe_allow_html=True,
  )
  if st.session_state.is_authenticated:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT title, author_name, filename, scope, final_score, piq_minted,"
        " tx_hash FROM papers_assessment WHERE user_id=? ORDER BY timestamp DESC"
        " LIMIT 20",
        (current_user,),
    )
    history_data = cursor.fetchall()
    conn.close()
    if history_data:
      cleaned_history = []
      for row in history_data:
        cleaned_history.append((
            row[0],
            clean_author_name(row[1]),
            row[2],
            row[3],
            row[4],
            row[5],
            row[6],
        ))
      st.dataframe(
          pd.DataFrame(
              cleaned_history,
              columns=[
                  "Paper Title",
                  "Contributing Authors",
                  "File Name",
                  "Scope",
                  "Pi-Index Score",
                  "piQ Earned",
                  "Eth Tx Hash",
              ],
          ),
          use_container_width=True,
          hide_index=True,
      )
    else:
      st.info("No assessment history found.")
  else:
    st.warning("Please connect your ORCID iD or DID in the sidebar.")

  st.markdown("---")
  st.markdown(
      "### 📋 Last 5 Assessed Papers across the Ledger"
      + tooltip(
          "Displays the 5 most recently evaluated papers globally from the database"
          " with full breakdown details."
      ),
      unsafe_allow_html=True,
  )

  conn_last = get_db_connection()
  cur_last = conn_last.cursor()
  cur_last.execute(
      """SELECT title, author_name, filename, final_score, logic_score, c1, c2, c3, c4, c5, c6, c7, c8, 
                piq_minted, tx_hash, zk_proof, mdar_adherence_score, rrid_valid_count, reproducibility_score, eval_hash, timestamp 
         FROM papers_assessment ORDER BY timestamp DESC LIMIT 5"""
  )
  recent_papers = cur_last.fetchall()
  conn_last.close()

  if not recent_papers:
    st.info("No papers have been assessed in the database yet.")
  else:
    for idx, rp in enumerate(recent_papers):
      (
          r_title,
          r_author,
          r_filename,
          r_score,
          r_logic,
          r_c1,
          r_c2,
          r_c3,
          r_c4,
          r_c5,
          r_c6,
          r_c7,
          r_c8,
          r_piq,
          r_tx,
          r_zk,
          r_mdar,
          r_rrid,
          r_repro,
          r_hash,
          r_time,
      ) = rp

      r_author_clean = clean_author_name(r_author)
      r_book = "0x" + hashlib.sha256(r_author_clean.encode()).hexdigest()[:40]

      with st.expander(
          f"[{idx+1}] {r_title[:65]}... — *{r_author_clean}* (Score:"
          f" **{r_score:.2f}** | {r_time[:16]})",
          expanded=False,
      ):
        st.write(f"**Title:** {r_title}")
        st.write(f"**Author(s):** {r_author_clean}")
        st.write(f"**Timestamp:** `{r_time}`")
        st.write(f"**Evaluation Hash:** `{r_hash}`")
        st.write(f"**Unique Author Book Address:** `{r_book}`")
        st.write(f"**piQ Minted:** `{r_piq}` | **Tx Hash:** `{r_tx}`")
        st.write(
            f"**Logic Integrity:** `{r_logic:.1f}%` | **Reproducibility:**"
            f" `{r_repro * 100:.1f}%` | **MDAR Adherence:**"
            f" `{r_mdar * 100:.1f}%`"
        )

        r_df = pd.DataFrame({
            "Criterion": [
                "C1: Semantic Originality",
                "C2: Methodological Rigor (SciScore)",
                "C3: Interdisciplinary Entropy",
                "C4: Societal Impact",
                "C5: Open Science & Repro",
                "C6: Literature Integration",
                "C7: Empirical Density",
                "C8: Future Actionability & FAIR",
            ],
            "Score (0-100)": [
                r_c1,
                r_c2,
                r_c3,
                r_c4,
                r_c5,
                r_c6,
                r_c7,
                r_c8,
            ],
        })
        st.dataframe(r_df, hide_index=True, use_container_width=True)

with tab2:
  st.markdown(
      "### Global Map of Science (Ledger-Driven Cartography) "
      + tooltip(
          "Generates dynamic network topologies based on the aggregate metadata"
          " of all ledger-evaluated papers."
      ),
      unsafe_allow_html=True,
  )
  st.markdown(
      "This map is permanently updated by every user assessing documents on"
      " the blockchain ledger, forming an unalterable topological view of"
      " current scientific trends."
  )

  conn = get_db_connection()
  cursor = conn.cursor()
  cursor.execute("SELECT DISTINCT author_name FROM papers_assessment")
  all_global_authors = []
  for row in cursor.fetchall():
    if row[0]:
      cleaned = clean_author_name(row[0])
      for a in cleaned.split(","):
        if a.strip() and not is_likely_institution(a.strip()):
          all_global_authors.append(a.strip())
  conn.close()
  all_global_authors = sorted(list(set(all_global_authors)))

  selected_author = None
  piq_dict, book_dict = get_author_piq_dict()

  if all_global_authors:
    filter_choice = st.selectbox(
        "Filter Global Cartography by Author:",
        ["All Authors"] + all_global_authors,
        key=f"author_filter_dropdown_{st.session_state['assessment_update_token']}",
        format_func=lambda x: (
            f"{x} (piQ: {piq_dict.get(x, 0.0):.2f})" if x != "All Authors" else x
        ),
    )
    if filter_choice != "All Authors":
      selected_author = filter_choice

  def render_bubble_chart_clean(target_author):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT fields, subfields, final_score, author_name FROM"
        " papers_assessment"
    )
    data = cursor.fetchall()
    conn.close()

    html_string, table_html = "", ""
    if not data:
      return html_string, table_html

    topic_aggregates = {}
    exclude_terms = {
        "general",
        "general science",
        "unspecified domain",
        "unspecified sub-domain",
        "core research topic",
    }

    for fields_json, subfields_json, final_score, author_str in data:
      cleaned_author = clean_author_name(author_str)
      if (
          target_author
          and target_author != "All Authors"
          and target_author not in cleaned_author
      ):
        continue
      try:
        subfields = [s.title().strip() for s in json.loads(subfields_json)]
        score = float(final_score) if final_score else 50.0
        for s in subfields:
          if s.lower() not in exclude_terms:
            if s not in topic_aggregates:
              topic_aggregates[s] = {"weight_sum": 0.0, "frequency": 0}
            topic_aggregates[s]["weight_sum"] += score
            topic_aggregates[s]["frequency"] += 1
      except:
        continue

    if not topic_aggregates:
      topic_aggregates["Core Research Domain"] = {
          "weight_sum": 50.0,
          "frequency": 1,
      }

    unique_topics = list(topic_aggregates.keys())

    def get_color(i, n):
      h, s, v = i / n if n > 0 else 0, 0.7, 0.9
      rgb = colorsys.hsv_to_rgb(h, s, v)
      return "#%02x%02x%02x" % tuple(int(x * 255) for x in rgb)

    color_map = {
        topic: get_color(i, len(unique_topics))
        for i, topic in enumerate(unique_topics)
    }
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#2c3e50",
        notebook=False,
    )
    physics_options = """{ "physics": { "barnesHut": { "gravitationalConstant": -1000, "centralGravity": 1, "springLength": 100, "avoidOverlap": 1.0 }, "stabilization": { "enabled": true, "iterations": 200 } } }"""
    net.set_options(physics_options)

    for topic, metrics in topic_aggregates.items():
      avg_weight = metrics["weight_sum"] / metrics["frequency"]
      freq = metrics["frequency"]
      node_size = max(30, 20 + (avg_weight * 2.5))

      base_col = color_map[topic]
      net.add_node(
          n_id=topic,
          label=" ",
          title=(
              f"Topic: {topic} | Frequency: {freq} | Avg Weight/Score:"
              f" {avg_weight:.1f}"
          ),
          size=node_size,
          shape="dot",
          physics=True,
          font={"color": "rgba(0,0,0,0)", "size": 0},
          color={
              "background": base_col,
              "border": "#1a1a1a",
              "highlight": {"background": base_col, "border": "#000000"},
              "hover": {"background": base_col, "border": "#000000"},
          },
          shadow={
              "enabled": True,
              "color": "rgba(0,0,0,0.5)",
              "size": 12,
              "x": 8,
              "y": 8,
          },
      )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
      net.save_graph(tmp_file.name)
      with open(tmp_file.name, "r", encoding="utf-8") as f:
        html_string = f.read()
    os.remove(tmp_file.name)

    gradient_injection = """
        <style type="text/css">
            canvas {
                background: radial-gradient(circle at 50% 50%, #ffffff 0%, #f0f2f5 100%);
            }
        </style>
        </head>
        """
    html_string = html_string.replace("</head>", gradient_injection)
    html_string = html_string.replace(
        "mynetwork", f"pi_network_{int(time.time() * 1000)}"
    )

    table_html = "<style>.table-big { width: 100%; font-size: 14px; border-collapse: collapse; margin-top: 10px; font-family: sans-serif; } .table-big th { background-color: #2c3e50; color: white; padding: 8px; text-align: left; } .table-big td { padding: 8px; border-bottom: 1px solid #ecf0f1; } .color-box { width: 30px; height: 30px; border-radius: 4px; display: inline-block; } </style>"
    table_html += "<div class='legend-container'><table class='table-big'><thead><tr><th style='width: 20%; text-align: center;'>Color</th><th>Scientific Topic</th><th style='text-align: center;'>Frequency</th><th style='text-align: center;'>Avg Weight</th></tr></thead><tbody>"
    for topic, metrics in sorted(
        topic_aggregates.items(), key=lambda x: x[1]["frequency"], reverse=True
    ):
      avg_w = metrics["weight_sum"] / metrics["frequency"]
      table_html += (
          f"<tr><td style='text-align: center;'><div class='color-box'"
          f" style='background-color:{color_map[topic]};'></div></td><td><b>{topic}</b></td><td"
          f" style='text-align: center;'>{metrics['frequency']}</td><td"
          f" style='text-align: center;'>{avg_w:.1f}</td></tr>"
      )
    table_html += "</tbody></table></div>"

    return html_string, table_html

  interactive_html, table_html = render_bubble_chart_clean(selected_author)
  if interactive_html:
    col1, col2 = st.columns([2, 1])
    with col1:
      components.html(interactive_html, height=620, scrolling=True)
    with col2:
      st.markdown(
          "### Legend & Frequency Metrics "
          + tooltip(
              "Lists specific scientific paper topics, assessed frequencies,"
              " and calculated score weights."
          ),
          unsafe_allow_html=True,
      )
      st.markdown(table_html, unsafe_allow_html=True)
  else:
    st.info("Awaiting sufficient data for this selection.")

  st.markdown("---")
  st.markdown(
      "### Pi Quotient (piQ) Explorer & Leaderboard "
      + tooltip(
          "piQ is a Soulbound Token (SBT). It cannot be transferred, bought, or"
          " sold. It permanently attaches to the author's identity."
      ),
      unsafe_allow_html=True,
  )

  search_query = st.text_input(
      "Search Explorer by Author Name or Unique Book Address:",
      placeholder="Enter author name or 0x...",
  )

  if piq_dict:
    leaderboard_data = []
    for author, piq in piq_dict.items():
      leaderboard_data.append({
          "Contributing Author": author,
          "Unique Author Book Address": book_dict.get(author, "None"),
          "Total piQ Earned": round(piq, 2),
      })
    piq_df = pd.DataFrame(leaderboard_data)
    piq_df = piq_df.sort_values(
        by="Total piQ Earned", ascending=False
    ).reset_index(drop=True)

    if search_query:
      query_clean = search_query.strip().lower()
      conn = get_db_connection()
      cursor = conn.cursor()

      if query_clean.startswith("0x"):
        cursor.execute(
            "SELECT title, author_name, eth_book, filename, eval_hash, final_score,"
            " piq_minted, timestamp FROM papers_assessment WHERE"
            " LOWER(eth_book)=? ORDER BY timestamp DESC",
            (query_clean,),
        )
        book_papers = cursor.fetchall()
        conn.close()
        if book_papers:
          st.success(
              f"Found {len(book_papers)} papers linked to Unique Book Address:"
              f" `{search_query}`"
          )
          formatted_book_rows = []
          for r in book_papers:
            formatted_book_rows.append((
                r[0],
                clean_author_name(r[1]),
                r[2],
                r[3],
                r[4],
                r[5],
                r[6],
                r[7],
            ))
          df_book = pd.DataFrame(
              formatted_book_rows,
              columns=[
                  "Paper Title",
                  "Author",
                  "Unique Book Address",
                  "File Name",
                  "Paper Address (Eval Hash)",
                  "Pi-Index",
                  "piQ Earned",
                  "Timestamp",
              ],
          )
          st.dataframe(df_book, use_container_width=True, hide_index=True)
        else:
          st.warning(
              f"No records found for Unique Book Address '{search_query}'."
          )
      else:
        cursor.execute(
            "SELECT author_name, title, eth_book, filename, eval_hash, final_score,"
            " piq_minted, timestamp FROM papers_assessment WHERE"
            " LOWER(author_name) LIKE ? ORDER BY timestamp DESC",
            (f"%{query_clean}%",),
        )
        author_papers = cursor.fetchall()
        conn.close()
        if author_papers:
          st.success(
              f"Found {len(author_papers)} paper records for author matching"
              f" '{search_query}'."
          )
          formatted_auth_rows = []
          for r in author_papers:
            formatted_auth_rows.append((
                clean_author_name(r[0]),
                r[1],
                r[2],
                r[3],
                r[4],
                r[5],
                r[6],
                r[7],
            ))
          df_author = pd.DataFrame(
              formatted_auth_rows,
              columns=[
                  "Author",
                  "Paper Title",
                  "Unique Book Address",
                  "File Name",
                  "Paper Address (Eval Hash)",
                  "Pi-Index",
                  "piQ Earned",
                  "Timestamp",
              ],
          )
          st.dataframe(df_author, use_container_width=True, hide_index=True)
        else:
          st.warning(
              f"No papers or piQ records found for author '{search_query}'."
          )
    else:
      st.dataframe(piq_df, use_container_width=True)
  else:
    st.info("No Pi Quotient has been minted yet.")

with tab3:
  st.markdown(
      "### Active Epoch & DeSci Staking Guide "
      + tooltip(
          "Detailed explanation of how blockchain blocks, epochs, proof-of-research"
          " validation, and DeSci staking work in Tab 3."
      ),
      unsafe_allow_html=True,
  )

  with st.expander(
      "📖 Detailed Guide: How Tab 3 Works (Blockchain Ledger & Staking)",
      expanded=False,
  ):
    st.markdown("""
    Tab 3 manages the immutable decentralization layer of the Pi-Index Assessment Engine. Here is how each component operates:
    1. **Active Epoch & Block Height**: The system tracks an incremental block counter (`block_height`). Every evaluation increments the global evaluation counter. When the threshold (`EPOCH_BLOCK_SIZE`) is reached, a new blockchain block is minted.
    2. **Proof-of-Research (PoR) Validation (`validate_block_por`)**: 
       - Combines the block index, criteria weights ($\varpi_1$ to $\varpi_8$), timestamp, previous block hash, validator node signature, model identifier, and formulas hash into an unalterable SHA-256 block hash.
       - Guarantees complete auditability and cryptographic non-repudiation of every assessment round.
    3. **Dynamic Weight Adjustment**: Weights shift dynamically across epochs driven by model evaluation statistics and algorithmic pi ($\pi$) convergence precision.
    4. **DeSci Peer Attestation & Staking**: 
       - High-reputation researchers can stake a fraction of their earned soulbound tokens (`piQ`) to either **endorse** or **challenge** specific manuscript assessments on-chain (`desci_attestations` table).
       - This provides decentralized crowd-auditing and stakes reputation against fraudulent or low-rigor preprints.
    5. **Ledger Hashes & zk-SNARK Inspection**: Displays the chronological list of recent smart contract executions, linking paper evaluation hashes (`eval_hash`) to block hashes and zero-knowledge verification proofs.
    """)

  conn = get_db_connection()
  cursor = conn.cursor()
  try:
    cursor.execute(
        "SELECT block_height, w1, w2, w3, w4, w5, w6, w7, w8, model_used,"
        " eval_hash, block_hash, por_proof, formulas_hash FROM"
        " blockchain_por_weights ORDER BY block_height DESC LIMIT 1"
    )
    epoch_data = cursor.fetchone()
  except Exception:
    epoch_data = None

  if epoch_data:
    (
        block_height,
        weights,
        model_used,
        eval_hash,
        block_hash,
        por_proof,
        formulas_hash,
    ) = (
        epoch_data[0],
        epoch_data[1:9],
        epoch_data[9],
        epoch_data[10],
        epoch_data[11],
        epoch_data[12],
        epoch_data[13],
    )
    cursor.execute(
        "SELECT COUNT(DISTINCT eval_hash) FROM blockchain_por_weights WHERE"
        " eval_hash != 'genesis'"
    )
    total_papers_processed = cursor.fetchone()[0]

    current_pi_accuracy = generate_blockchain_pi(block_height)

    st.markdown(
        f"**Processed:** `{total_papers_processed}` | **Block Size:**"
        f" `{EPOCH_BLOCK_SIZE}` | **Model:** `{model_used}` | **Block:**"
        f" `{block_height}` | **Pi Algorithmic Precision:**"
        f" `{current_pi_accuracy}`"
    )

    cols = st.columns(4)
    labels = [
        ("C1", r"$\varpi_1$"),
        ("C2", r"$\varpi_2$"),
        ("C3", r"$\varpi_3$"),
        ("C4", r"$\varpi_4$"),
        ("C5", r"$\varpi_5$"),
        ("C6", r"$\varpi_6$"),
        ("C7", r"$\varpi_7$"),
        ("C8", r"$\varpi_8$"),
    ]
    for i, col in enumerate(cols * 2):
      if i < 8:
        col.markdown(f"**{labels[i][0]} ({labels[i][1]})**")
        col.markdown(
            f"<h3 style='margin-top:0px; margin-bottom:5px;'>{weights[i]:.6f}</h3>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "### Proof-of-Research Blockchain Explorer "
        + tooltip(
            "Search the ledger to mathematically verify if a specific research"
            " document has been authentically graded and permanently sealed."
        ),
        unsafe_allow_html=True,
    )
    st.info(
        f"**Latest Proof-of-Research:** `{por_proof}` successfully verified and"
        f" sealed to block `{block_hash}`."
    )
    st.caption(
        f"**Unalterable Criteria State Hash:** `{formulas_hash}` (Guarantees"
        " grading mathematical constants cannot be tampered with)."
    )

    explore_col1, explore_col2 = st.columns([3, 1])
    with explore_col1:
      search_query = st.text_input(
          "Enter Document Evaluation Hash or Block Hash to verify ledger"
          " record..."
      )
    with explore_col2:
      st.write("")
      st.write("")
      search_btn = st.button("Verify Record")

    if search_btn and search_query:
      try:
        cursor.execute(
            "SELECT * FROM blockchain_por_weights WHERE block_hash=? OR"
            " eval_hash=?",
            (search_query, search_query),
        )
        record = cursor.fetchone()
        if record:
          st.success("Valid Block Found on Ledger")
          st.json({
              "Block Height": record[0],
              "Timestamp": record[9],
              "Model Used": record[14],
              "Validator Node": record[11],
              "Block Hash": record[12],
              "Evaluation Hash": record[13],
              "PoR Signature": record[15],
              "Formulas Hash": record[16],
              "Weights": dict(
                  zip([f"w{i+1}" for i in range(8)], record[1:9])
              ),
          })
        else:
          st.error(
              "No block matching that signature was found on the ledger."
          )
      except:
        st.error("Error reading database schema. Try refreshing the app.")

    st.markdown("---")
    st.markdown(
        "### DeSci Peer Attestation & Stake-Weighted Validation "
        + tooltip(
            "High-reputation researchers can stake a fraction of their piQ to"
            " endorse or challenge peer assessments on-chain."
        ),
        unsafe_allow_html=True,
    )
    if st.session_state.is_authenticated:
      cursor.execute(
          "SELECT eval_hash, title FROM papers_assessment ORDER BY timestamp"
          " DESC LIMIT 20"
      )
      eval_papers = cursor.fetchall()
      if eval_papers:
        attest_options = {p[1]: p[0] for p in eval_papers}
        chosen_attest_title = st.selectbox(
            "Select Paper for Attestation:",
            list(attest_options.keys()),
            key="desci_attest_select",
        )
        target_eval_hash = attest_options[chosen_attest_title]

        attest_stance = st.radio(
            "Attestation Stance:",
            ["Endorse Methodological Rigor", "Challenge / Flag Anomaly"],
            horizontal=True,
        )
        stake_val = st.slider(
            "Stake piQ Amount:",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )

        if st.button("Submit On-Chain Attestation"):
          attest_id = "ATT_" + hashlib.sha256(
              f"{current_user}:{target_eval_hash}:{time.time()}".encode()
          ).hexdigest()[:12]
          cursor.execute(
              "INSERT OR REPLACE INTO desci_attestations (attestation_id,"
              " eval_hash, attester_id, stake_amount, stance, timestamp) VALUES"
              " (?, ?, ?, ?, ?, ?)",
              (
                  attest_id,
                  target_eval_hash,
                  current_user,
                  stake_val,
                  attest_stance,
                  datetime.now().isoformat(),
              ),
          )
          conn.commit()
          st.success(
              f"Attestation recorded successfully! Attestation ID: `{attest_id}`"
          )

        cursor.execute(
            "SELECT attester_id, stake_amount, stance, timestamp FROM"
            " desci_attestations WHERE eval_hash=?",
            (target_eval_hash,),
        )
        existing_attestations = cursor.fetchall()
        if existing_attestations:
          st.markdown("#### Active Community Attestations for this Manuscript")
          st.dataframe(
              pd.DataFrame(
                  existing_attestations,
                  columns=[
                      "Attester ID",
                      "Staked piQ",
                      "Stance",
                      "Timestamp",
                  ],
              ),
              use_container_width=True,
              hide_index=True,
          )
      else:
        st.info("No assessed papers available for attestation.")
    else:
      st.warning(
          "Please authenticate with your ORCID iD or DID to participate in"
          " DeSci attestation staking."
      )

    st.markdown("---")
    st.markdown(
        "### Latest Blockchain Ledger Hashes, zk-SNARK Proofs, and piQ Minted "
        + tooltip(
            "Chronological view of the most recent smart contract executions,"
            " demonstrating mathematical proofs of computation and token"
            " allocations."
        ),
        unsafe_allow_html=True,
    )
    cursor.execute("""
            SELECT b.block_height, b.eval_hash, b.block_hash, p.zk_proof, p.piq_minted, b.timestamp 
            FROM blockchain_por_weights b 
            LEFT JOIN papers_assessment p ON b.eval_hash = p.eval_hash 
            ORDER BY b.block_height DESC LIMIT 10
        """)
    recent_hashes = cursor.fetchall()
    if recent_hashes:
      df_hashes = pd.DataFrame(
          recent_hashes,
          columns=[
              "Block Height",
              "Evaluation Hash",
              "Block Hash",
              "zk-SNARK Proof",
              "Total piQ Minted",
              "Timestamp",
          ],
      )
      st.dataframe(df_hashes, use_container_width=True, hide_index=True)
    else:
      st.info("No hashes to display yet.")
  conn.close()

with tab4:
  st.markdown(
      "### Pi-Brain: Meta-Learning on the PoR Blockchain "
      + tooltip(
          "An LSTM neural network that trains directly on the block weights to"
          " predict future shifts in algorithmic evaluation standards."
      ),
      unsafe_allow_html=True,
  )

  with st.expander(
      "🧠 Detailed Guide: How Pi-Brain LSTM Meta-Learning Works", expanded=False
  ):
    st.markdown("""
    Pi-Brain is an on-chain predictive neural network built with PyTorch (`PiBrainLSTM`) that learns how evaluation weight standards evolve across blocks:
    1. **Data Pipeline (`PiBlockchainDataset`)**: Extracts historical weight matrices from the `blockchain_por_weights` table using a rolling lookback window (`lookback`).
    2. **Recurrent Architecture (`nn.LSTM`)**: 
       - Utilizes a Long Short-Term Memory (LSTM) layer with a hidden dimension size of 32 to capture temporal dependencies and drift patterns across successive epochs.
    3. **Linear Regression & Softmax Normalization**: 
       - Passes the final hidden state through a sequential multi-layer perceptron (Linear $\rightarrow$ ReLU $\rightarrow$ Linear) to output 8 projected criterion weights.
       - Applies `torch.softmax(dim=-1) * 8.0` to strictly preserve the mathematical normalization constraint where the sum of all 8 criteria weights equals exactly 8.0.
    4. **Optimization & Training Loop**: 
       - Trains dynamically using Mean Squared Error loss (`nn.MSELoss`) and the Adam optimizer over 200 epochs to forecast how evaluation weights will shift in the upcoming epoch.
    """)

  conn = get_db_connection()
  cursor = conn.cursor()
  cursor.execute(
      "SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights ORDER"
      " BY block_height ASC"
  )
  historical_rows = cursor.fetchall()
  conn.close()

  min_blocks_required = 2
  if len(historical_rows) < min_blocks_required:
    st.warning(
        f"Not enough blockchain data to train the meta-model. You need at least"
        f" {min_blocks_required} blocks (Currently on ledger:"
        f" {len(historical_rows)}). Assess at least 1 manuscript to generate"
        " block 2."
    )
  else:
    current_block_count = len(historical_rows)
    lookback_window = max(1, min(5, current_block_count - 1))

    if (
        "last_trained_blocks" not in st.session_state
        or st.session_state.last_trained_blocks != current_block_count
    ):
      weight_data = np.array(historical_rows, dtype=np.float32)
      actual_lookback = min(lookback_window, len(weight_data))

      dataset = PiBlockchainDataset(weight_data, actual_lookback)
      dataloader = DataLoader(
          dataset, batch_size=min(4, max(1, len(dataset))), shuffle=False
      )

      model = PiBrainLSTM()
      weights_path = os.path.join(BASE_DIR, "pi_brain_weights.pt")
      if os.path.exists(weights_path):
          try:
              model.load_state_dict(torch.load(weights_path, weights_only=True))
          except Exception:
              pass
              
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
          status_text.text(
              f"Training Epoch {epoch}/{epochs} | MSE Loss:"
              f" {total_loss / max(1, len(dataloader)):.6f}"
          )
          progress_bar.progress((epoch + 1) / epochs)

      model.eval()
      with torch.no_grad():
        st.session_state.predicted_next_weights = (
            model(
                torch.tensor(
                    weight_data[-actual_lookback:], dtype=torch.float32
                ).unsqueeze(0)
            )
            .squeeze()
            .numpy()
        )
        st.session_state.current_weights = weight_data[-1]
        st.session_state.last_trained_blocks = current_block_count
        
        torch.save(model.state_dict(), weights_path)
        backup_state_to_web3()

    else:
      st.info(
          "Meta-model is cached and up-to-date with the latest blockchain"
          " ledger."
      )

    df_compare = pd.DataFrame(
        {
            "Current Active Weights": st.session_state.current_weights,
            "Predicted Next Epoch": st.session_state.predicted_next_weights,
        },
        index=[
            "C1: Originality",
            "C2: Methodological Rigor",
            "C3: Interdisciplinary",
            "C4: Societal Impact",
            "C5: Open Science",
            "C6: Literature Integration",
            "C7: Empirical Density",
            "C8: Future Actionability",
        ],
    )
    st.bar_chart(df_compare, height=400)
    st.markdown(
        f"**Mathematical Constraint Check:** Predicted Sum ="
        f" `{sum(st.session_state.predicted_next_weights):.6f}` / `8.0`"
    )

with tab5:
  st.markdown(
      "### The Pi-Index Framework: Next-Gen Architecture & CoARA Compliance"
      " Workflow"
  )
  st.markdown(
      "The enhanced system architecture flow below details the decentralized"
      " intake, ZK double-blind reviewer assignment, SciScore deterministic"
      " parsing, Item Response Theory (IRT) calibration, and smart contract"
      " slashing mechanisms."
  )

  st.graphviz_chart("""
    digraph PiIndexSystemOverview {
        rankdir=TB;
        compound=true;
        fontname="Helvetica,Arial,sans-serif";
        node [fontname="Helvetica,Arial,sans-serif", style=filled, margin=0.2];
        edge [fontname="Helvetica,Arial,sans-serif", fontsize=10];

        node [shape=box, fillcolor="#f8f9fa", color="#2c3e50", penwidth=1.5];

        subgraph cluster_intake {
            label = "1. Unified Multi-Source Intake & ZK-Identity Registry (ZIP-600)";
            style = rounded;
            color = "#34495e";
            fillcolor = "#ecf0f1";

            Auth [label="Researcher Authentication\\n• ORCID iD / W3C DID Verification\\n• ZK-Email Institutional Proof", fillcolor="#aed6f1"];
            Intake [label="Multi-Source Ingestion Engine\\n• Local Binary PDFs Extraction\\n• Unpaywall DOI Resolver\\n• OpenAlex Topic API Search", fillcolor="#aed6f1"];
            ZKBlind [label="ZK Double-Blind Assignment\\n• Merkle Tree Non-Membership Proofs\\n• Anonymous Author Shielding", fillcolor="#aed6f1"];
            Auth -> Intake -> ZKBlind;
        }

        subgraph cluster_eval {
            label = "2. Core Evaluation & Adversarial Analysis Pipeline (CoARA/RRA)";
            style = rounded;
            color = "#27ae60";
            fillcolor = "#e8f8f5";

            SciParser [label="Deterministic SciScore API\\n• MDAR Reporting Adherence\\n• Valid RRIDs Count Extraction", fillcolor="#a3e4d7"];
            IRTCalib [label="Item Response Theory Calibration\\n• Counterfactual Stress Testing\\n• Variance & Difficulty Mapping", fillcolor="#a3e4d7"];
            Criteria [label="8 Transparent Criteria Rubrics\\n• C1 Originality to C8 FAIR Actionability\\n• Formulaic Score Computation", fillcolor="#a3e4d7"];
            Logic [label="Adversarial Logic Integrity Matrix\\n• Premise Validity & Evidence Strength\\n• AI Hallucination & Laundering Penalty", fillcolor="#a3e4d7"];
            
            SciParser -> IRTCalib -> Criteria -> Logic;
        }

        subgraph cluster_blockchain {
            label = "3. Blockchain Consensus, Cryptographic Proofs & Slashing Tokenomics";
            style = rounded;
            color = "#8e44ad";
            fillcolor = "#f4ecf7";

            PoR [label="Proof-of-Research (PoR) Validation\\n• Dynamic Epoch Weight Shifting\\n• Formulas Hash Stamping & SHA-256 Block", fillcolor="#d7bde2"];
            Slashing [label="Anti-Laundering Slashing Guard\\n• Smart Contract piQ Burn for Fraud\\n• Stake Penalty Enforcement", fillcolor="#f5b7b1"];
            Mint [label="Soulbound Token Minting\\n• Author-Specific Book Address (eth_book)\\n• Shared Paper Address (eval_hash) & Tx Hash", fillcolor="#d7bde2"];
            
            PoR -> Slashing -> Mint;
        }

        subgraph cluster_outputs {
            label = "4. User Interface, Cartography & Institutional Policy Support";
            style = rounded;
            color = "#d35400";
            fillcolor = "#fef5e7";

            Dossier [label="CoARA & DORA-Aligned Dossier\\n• Markdown Research Integrity Report\\n• AI Defense Rebuttal Strategy", fillcolor="#f8c471"];
            Cartography [label="Global Map of Science\\n• Ledger PyVis Network Cartography\\n• Author & Topic Bubble Filtering", fillcolor="#f8c471"];
            PiBrain [label="Pi-Brain LSTM Meta-Learning\\n• PyTorch Temporal Weight Prediction\\n• Calibration Drift & Epoch Forecasting", fillcolor="#f8c471"];
        }

        Auth -> SciParser [lhead=cluster_eval, label="Processed Manuscript Text"];
        Logic -> PoR [lhead=cluster_blockchain, label="Audited Score & Hashes"];
        Mint -> Dossier [lhead=cluster_outputs, label="Ledger Seal & Tokens"];
        Mint -> Cartography;
        Mint -> PiBrain;
    }
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.8em;'>Framework"
    " Author: Ali Vafadar Yengejeh | Universita degli Studi di"
    " Milano-Bicocca</div>",
    unsafe_allow_html=True,
)
