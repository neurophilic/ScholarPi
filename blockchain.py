import sqlite3
import hashlib
import time
from datetime import datetime
from config import DB_PATH

def init_system():
    """Set up the SQLite database tables if they don't exist yet."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=20.0)
    cursor = conn.cursor()
    
    # Main table for storing evaluated papers
    cursor.execute('''CREATE TABLE IF NOT EXISTS papers_assessment 
                      (eval_hash TEXT PRIMARY KEY, user_id TEXT, title TEXT, filename TEXT, scope TEXT,
                       c1 REAL, c2 REAL, c3 REAL, c4 REAL, 
                       c5 REAL, c6 REAL, c7 REAL, c8 REAL, 
                       scope_alignment REAL, logic_score REAL,
                       subfields TEXT, fields TEXT, author_name TEXT, final_score REAL, timestamp DATETIME)''')
                       
    try: cursor.execute("ALTER TABLE papers_assessment ADD COLUMN logic_score REAL DEFAULT 0.0")
    except: pass 
    try: cursor.execute("ALTER TABLE papers_assessment ADD COLUMN author_name TEXT DEFAULT 'Unknown Author'")
    except: pass 
        
    # Blockchain ledger table
    cursor.execute('''CREATE TABLE IF NOT EXISTS blockchain_por_weights 
                      (block_height INTEGER PRIMARY KEY AUTOINCREMENT, 
                       w1 REAL, w2 REAL, w3 REAL, w4 REAL, 
                       w5 REAL, w6 REAL, w7 REAL, w8 REAL, 
                       timestamp DATETIME, previous_hash TEXT, 
                       validator_node TEXT, block_hash TEXT, eval_hash TEXT, model_used TEXT, por_proof TEXT)''')
                       
    try: cursor.execute("ALTER TABLE blockchain_por_weights ADD COLUMN por_proof TEXT DEFAULT 'Genesis_Proof'")
    except: pass

    cursor.execute('''CREATE TABLE IF NOT EXISTS global_eval_counter (count INTEGER)''')
    
    # Create Genesis block if empty
    cursor.execute("SELECT COUNT(*) FROM blockchain_por_weights")
    if cursor.fetchone()[0] == 0:
        genesis_weights = [1.0] * 8
        prev_hash = "0" * 64
        timestamp = datetime.now().isoformat()
        val_node, block_hash, por_proof = validate_block_por(1, genesis_weights, timestamp, prev_hash, "genesis", "none", 100.0)
        
        cursor.execute('''INSERT INTO blockchain_por_weights 
                          (w1, w2, w3, w4, w5, w6, w7, w8, timestamp, previous_hash, validator_node, block_hash, eval_hash, model_used, por_proof) 
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                       (*genesis_weights, timestamp, prev_hash, val_node, block_hash, "genesis", "none", por_proof))
                       
    # Initialize the total paper counter
    cursor.execute("SELECT count FROM global_eval_counter")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO global_eval_counter (count) VALUES (0)")
        
    conn.commit()
    return conn

def validate_block_por(block_index, weights, timestamp, previous_hash, eval_hash, model_used, final_score):
    """
    Proof-of-Research (PoR) Consensus: 
    Validates a block by cryptographically sealing the peer-review evaluation metrics 
    (the eval_hash and the mathematical score) instead of wasting energy on PoW.
    """
    validator_node = "Validator_Pi_" + hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
    
    # The PoR proof mathematically binds the research effort to the blockchain
    por_proof = f"PoR_{eval_hash[:12]}_Score:{final_score:.2f}"
    
    data_string = f"{block_index}{weights}{timestamp}{previous_hash}{validator_node}{por_proof}{model_used}"
    block_hash = hashlib.sha256(data_string.encode('utf-8')).hexdigest()
    
    return validator_node, block_hash, por_proof
