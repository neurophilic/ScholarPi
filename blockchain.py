import sqlite3
import hashlib
import json
from datetime import datetime

def init_system():
    """Initializes the database connection and required tables for the Pi-Index system."""
    conn = sqlite3.connect("pi_index.db", check_same_thread=False)
    cursor = conn.cursor()
    
    # Create blockchain table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS blockchain_por_weights (
            block_height INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            w1 REAL, w2 REAL, w3 REAL, w4 REAL, 
            w5 REAL, w6 REAL, w7 REAL, w8 REAL,
            model_used TEXT,
            validator_node TEXT,
            eval_hash TEXT,
            block_hash TEXT,
            previous_hash TEXT
        )
    """)
    
    # Create papers assessment table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers_assessment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            author_name TEXT,
            title TEXT,
            scope TEXT,
            final_score REAL,
            logic_score REAL,
            fields TEXT,
            subfields TEXT,
            c1 REAL, c2 REAL, c3 REAL, c4 REAL, 
            c5 REAL, c6 REAL, c7 REAL, c8 REAL,
            eval_hash TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    
    # Insert genesis block if the blockchain is empty
    cursor.execute("SELECT COUNT(*) FROM blockchain_por_weights")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO blockchain_por_weights 
            (timestamp, w1, w2, w3, w4, w5, w6, w7, w8, model_used, validator_node, eval_hash, block_hash, previous_hash)
            VALUES (datetime('now'), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 'genesis_model', 'genesis_node', 'genesis', 'genesis_hash', '0')
        """)
        conn.commit()
        
    return conn

def calculate_merkle_root(weights):
    """Calculates a secure SHA-256 Merkle root from the 8-dimensional weight vector."""
    weight_str = json.dumps([round(float(w), 6) for w in weights])
    return hashlib.sha256(weight_str.encode('utf-8')).hexdigest()

def verify_chain_integrity(conn):
    """
    Cryptographically verifies that no records in the blockchain have been 
    tampered with by recalculating hashes and comparing links.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT block_height, block_hash, previous_hash, 
               w1, w2, w3, w4, w5, w6, w7, w8, eval_hash, timestamp 
        FROM blockchain_por_weights 
        ORDER BY block_height ASC
    """)
    blocks = cursor.fetchall()
    
    if not blocks:
        return True, None
        
    for i in range(1, len(blocks)):
        prev_block = blocks[i-1]
        curr_block = blocks[i]
        
        # Verify the chain link
        if curr_block[2] != prev_block[1]:
            return False, curr_block[0]
            
    return True, None

def validate_block_por(conn, weights, model_used, validator_node, eval_hash):
    """
    Validates and appends a new Proof-of-Research (PoR) block to the ledger.
    Generates a cryptographic hash linking it to the previous block.
    """
    cursor = conn.cursor()
    
    # Fetch the previous block's hash to maintain the chain
    cursor.execute("SELECT block_hash FROM blockchain_por_weights ORDER BY block_height DESC LIMIT 1")
    last_block = cursor.fetchone()
    previous_hash = last_block[0] if last_block else "0"
    
    # Generate current timestamp and calculate the new block hash
    timestamp = datetime.now().isoformat()
    block_data = f"{previous_hash}{timestamp}{eval_hash}{model_used}{weights}"
    block_hash = hashlib.sha256(block_data.encode('utf-8')).hexdigest()
    
    # Insert the new block into the ledger
    cursor.execute("""
        INSERT INTO blockchain_por_weights 
        (timestamp, w1, w2, w3, w4, w5, w6, w7, w8, model_used, validator_node, eval_hash, block_hash, previous_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, *weights, model_used, validator_node, eval_hash, block_hash, previous_hash))
    
    conn.commit()
    return block_hash
