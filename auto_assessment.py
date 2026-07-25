
import os
import re
import json
import time
import math
import random
import sqlite3
import hashlib
import shutil
from datetime import datetime

import requests
import cloudscraper
import fitz
import numpy as np
from web3 import Web3
from groq import Groq

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 1. CONFIGURATION & ENVIRONMENT SETUP
# ==========================================
PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_TEXT_TOKENS = 12000
EPOCH_BLOCK_SIZE = 1

WEB3_PROVIDER_URI = os.getenv("WEB3_PROVIDER_URI", "https://ethereum-sepolia-rpc.publicnode.com")
ETH_ADMIN_PRIVATE_KEY = os.getenv("ETH_ADMIN_PRIVATE_KEY", "")
PIQ_CONTRACT_ADDRESS = os.getenv("PIQ_CONTRACT_ADDRESS", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
PINATA_API_KEY = os.getenv("PINATA_API_KEY", "")
PINATA_SECRET_API_KEY = os.getenv("PINATA_SECRET_API_KEY", "")
REGISTRY_CONTRACT_ADDRESS = os.getenv("REGISTRY_CONTRACT_ADDRESS", "")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing in environment variables.")

BASE_DIR = os.path.expanduser("~/Scientometric_Pi_Index")
os.makedirs(BASE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, "pi_index_main.db")

w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER_URI))
groq_client = Groq(api_key=GROQ_API_KEY)

HOT_TOPICS = [
    "Quantum Error Correction", "Generative AI in Oncology", "CRISPR-Cas12 Therapeutics",
    "Solid-State Battery Electrolytes", "Perovskite Solar Cell Efficiency",
    "Neuromorphic Computing Hardware", "Neural Radiance Fields 3D Reconstruction",
    "Carbon Capture Metal-Organic Frameworks", "Fusion Energy Plasma Confinement",
    "Exoplanet Atmospheric Spectroscopy"
]

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
# 2. STATE MANAGEMENT & DB SCHEMA
# ==========================================
def restore_state_from_web3():
    if not w3.is_connected() or not REGISTRY_CONTRACT_ADDRESS:
        return
    try:
        abi = '[{"inputs":[],"name":"getCID","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"}]'
        contract = w3.eth.contract(address=w3.to_checksum_address(REGISTRY_CONTRACT_ADDRESS), abi=json.loads(abi))
        cid = contract.functions.getCID().call()
        if cid:
            gateways = [
                f"https://ivory-worrying-boa-917.mypinata.cloud/ipfs/{cid}",
                f"https://gateway.pinata.cloud/ipfs/{cid}",
                f"https://ipfs.io/ipfs/{cid}"
            ]
            res = None
            for gw in gateways:
                try:
                    r = requests.get(gw, timeout=15)
                    if r.status_code == 200:
                        res = r
                        break
                except requests.RequestException:
                    continue
                    
            if res and res.status_code == 200:
                zip_path = BASE_DIR + "_restore.zip"
                with open(zip_path, 'wb') as fp:
                    fp.write(res.content)
                shutil.unpack_archive(zip_path, BASE_DIR)
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                print("State successfully restored from Web3/IPFS.")
    except Exception as e:
        print(f"Restore error: {e}")

def backup_state_to_web3():
    if not w3.is_connected() or not PINATA_API_KEY or not REGISTRY_CONTRACT_ADDRESS:
        return False
    try:
        shutil.make_archive(BASE_DIR, 'zip', BASE_DIR)
        zip_path = BASE_DIR + ".zip"
        headers = {"pinata_api_key": PINATA_API_KEY, "pinata_secret_api_key": PINATA_SECRET_API_KEY}
        with open(zip_path, 'rb') as fp:
            res = requests.post("https://api.pinata.cloud/pinning/pinFileToIPFS", files={"file": fp}, headers=headers)
        cid = res.json().get("IpfsHash")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if not cid:
            return False

        abi = '[{"inputs":[{"internalType":"string","name":"_cid","type":"string"}],"name":"updateCID","outputs":[],"stateMutability":"nonpayable","type":"function"}]'
        contract = w3.eth.contract(address=w3.to_checksum_address(REGISTRY_CONTRACT_ADDRESS), abi=json.loads(abi))
        account = w3.eth.account.from_key(ETH_ADMIN_PRIVATE_KEY)
        
        estimated_gas = contract.functions.updateCID(cid).estimate_gas({"from": account.address})
        tx = contract.functions.updateCID(cid).build_transaction({
            "from": account.address,
            "nonce": w3.eth.get_transaction_count(account.address),
            "gas": int(estimated_gas * 1.2),
            "gasPrice": w3.eth.gas_price,
        })
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=ETH_ADMIN_PRIVATE_KEY)
        try:
            tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            print(f"Automatic Backup Success! Tx Hash: {tx_hash.hex()}")
        except Exception as send_err:
            if "already known" in str(send_err):
                return True
            raise send_err
        return True
    except Exception as e:
        print(f"Failed backup to Web3: {e}")
        return False

def enforce_database_schema():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS papers_assessment 
                      (eval_hash TEXT PRIMARY KEY, user_id TEXT, title TEXT, filename TEXT, scope TEXT,
                       c1 REAL, c2 REAL, c3 REAL, c4 REAL, c5 REAL, c6 REAL, c7 REAL, c8 REAL, 
                       scope_alignment REAL, logic_score REAL, subfields TEXT, fields TEXT, 
                       author_name TEXT, final_score REAL, timestamp DATETIME)""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS blockchain_por_weights 
                      (block_height INTEGER PRIMARY KEY AUTOINCREMENT, 
                       w1 REAL, w2 REAL, w3 REAL, w4 REAL, w5 REAL, w6 REAL, w7 REAL, w8 REAL, 
                       timestamp DATETIME, previous_hash TEXT, validator_node TEXT, 
                       block_hash TEXT, eval_hash TEXT, model_used TEXT)""")
    cursor.execute("CREATE TABLE IF NOT EXISTS global_eval_counter (count INTEGER)")
    cursor.execute("SELECT COUNT(*) FROM global_eval_counter")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO global_eval_counter (count) VALUES (0)")
    
    target_columns = {
        "eth_book": "TEXT DEFAULT 'None'", "eth_wallet": "TEXT DEFAULT 'None'",
        "piq_minted": "REAL DEFAULT 0.0", "epc_minted": "REAL DEFAULT 0.0",
        "tx_hash": "TEXT DEFAULT 'Pending'", "zk_proof": "TEXT DEFAULT 'None'",
        "did": "TEXT DEFAULT 'None'", "zk_email_proof": "TEXT DEFAULT 'None'",
        "gaming_penalty": "REAL DEFAULT 0.0", "mdar_adherence_score": "REAL DEFAULT 0.0",
        "rrid_valid_count": "INTEGER DEFAULT 0", "credit_taxonomy_roles": "TEXT DEFAULT 'None'",
        "reproducibility_score": "REAL DEFAULT 0.0", "doi": "TEXT DEFAULT 'None'"
    }
    cursor.execute("PRAGMA table_info(papers_assessment)")
    existing = [row[1] for row in cursor.fetchall()]
    for col, dtype in target_columns.items():
        if col not in existing:
            try: cursor.execute(f"ALTER TABLE papers_assessment ADD COLUMN {col} {dtype}")
            except: pass
    conn.commit()
    conn.close()

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
            (g["block_height"], *g["weights"], g["timestamp"], g["previous_hash"], g["validator_node"], block_hash, g["eval_hash"], g["model_used"], g["por_proof"], g["formulas_hash"])
        )
        conn.commit()
    return conn

# ==========================================
# 3. OPENALEX HARVESTING & PROCESSING
# ==========================================
def fetch_random_hot_papers():
    topic = random.choice(HOT_TOPICS)
    print(f"Selected Hot Topic: '{topic}'")
    url = f"https://api.openalex.org/works?search={requests.utils.quote(topic)}&filter=is_oa:true&per_page=20"
    res = requests.get(url, timeout=10)
    if res.status_code == 200:
        results = res.json().get("results", [])
        extracted = []
        for item in results:
            pdf_url = (item.get("best_oa_location") or {}).get("pdf_url") or item.get("open_access", {}).get("oa_url", "")
            if pdf_url:
                extracted.append({
                    "title": item.get("title", "Untitled Paper"),
                    "doi": item.get("doi", ""),
                    "pdf_url": pdf_url,
                    "topic": topic
                })
        return extracted
    return []

def download_pdf(pdf_url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        res = requests.get(pdf_url, headers=headers, timeout=15)
        if res.status_code == 200 and b"%PDF" in res.content[:10]:
            return res.content
    except Exception:
        pass
    try:
        scraper = cloudscraper.create_scraper()
        res = scraper.get(pdf_url, timeout=20)
        if res.status_code == 200 and b"%PDF" in res.content[:10]:
            return res.content
    except Exception:
        pass
    return None

def process_paper_headless(pdf_bytes, filename, scope, doi):
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT eval_hash FROM papers_assessment WHERE eval_hash=?", (file_hash,))
    if cursor.fetchone():
        print(f"Paper {filename} already evaluated. Skipping.")
        conn.close()
        return

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = " ".join([page.get_text() for page in doc])[:MAX_TEXT_TOKENS]
    
    prompt = f"""Extract Metadata & 8 Criteria Variables (0.0 to 1.0) for this paper:
    `Extracted_Title`, `Extracted_Author`, `Extracted_Topics`.
    Audit Variables: `semantic_novelty`, `laundering_penalty`, `rigor_index`, `citation_entropy`, `societal_linkage`, `D_open`, `J_code`, `citation_polarity_score`, `empirical_density`, `fair_compliance`.
    Logic Mapping: `Evidence_Strength`, `Conclusion_Reach`, `Logical_Jumps`, `Premise_Validity`.
    Return ONLY valid JSON. Text: {full_text[:4000]}"""
    
    res = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=FALLBACK_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    raw_data = json.loads(res.choices[0].message.content)
    
    title = raw_data.get("Extracted_Title", filename)
    author = raw_data.get("Extracted_Author", "Unidentified")
    
    # Calculate scores
    scores = [
        round(raw_data.get("semantic_novelty", 0.7) * 100, 2),
        round(raw_data.get("rigor_index", 0.75) * 100, 2),
        round(raw_data.get("citation_entropy", 0.6) * 100, 2),
        round(raw_data.get("societal_linkage", 0.65) * 100, 2),
        round(raw_data.get("D_open", 0.7) * 100, 2),
        round(raw_data.get("citation_polarity_score", 0.7) * 100, 2),
        round(raw_data.get("empirical_density", 0.75) * 100, 2),
        round(raw_data.get("fair_compliance", 0.8) * 100, 2)
    ]
    
    final_score = float(np.mean(scores))
    author_book = "0x" + hashlib.sha256(author.encode()).hexdigest()[:40]

    cursor.execute("""INSERT OR REPLACE INTO papers_assessment 
                      (eval_hash, user_id, title, filename, scope, c1, c2, c3, c4, c5, c6, c7, c8, 
                       logic_score, scope_alignment, subfields, fields, author_name, final_score, 
                       timestamp, eth_book, piq_minted, tx_hash, zk_proof, doi) 
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                   (file_hash, "GitHub_Actions_Bot", title, filename, scope, *scores, 85.0, 90.0, 
                    json.dumps([scope]), json.dumps([scope]), author, final_score, 
                    datetime.now().isoformat(), author_book, round(final_score / 10.0, 2), 
                    "Auto_Executed", "0x_zkProof", doi))
    
    cursor.execute("UPDATE global_eval_counter SET count = count + 1")
    conn.commit()
    conn.close()
    print(f"Successfully assessed and logged: '{title}' by {author}")

# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    print("Starting Background Paper Assessment Cron...")
    restore_state_from_web3()
    
    papers = fetch_random_hot_papers()
    processed_count = 0
    
    for p in papers:
        print(f"Attempting download for: {p['title']}")
        pdf_bytes = download_pdf(p["pdf_url"])
        if pdf_bytes:
            process_paper_headless(pdf_bytes, f"Auto_{time.time()}.pdf", p["topic"], p["doi"])
            processed_count += 1
            if processed_count >= 10:  
                break
                
    if processed_count > 0:
        backup_state_to_web3()
        print("Background assessment cycle completed and backed up to Web3.")
    else:
        print("No new papers were processed in this run.")
