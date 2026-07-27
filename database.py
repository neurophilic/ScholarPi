import sqlite3
import hashlib
import logging
from config import DB_PATH, GENESIS_BLOCK_CONFIG

_schema_initialized = False

def enforce_database_schema(conn: sqlite3.Connection):
    global _schema_initialized
    if _schema_initialized:
        return

    cursor = conn.cursor()

    cursor.execute("""CREATE TABLE IF NOT EXISTS papers_assessment 
                      (eval_hash TEXT PRIMARY KEY, user_id TEXT, title TEXT, filename TEXT, scope TEXT,
                       c1 REAL, c2 REAL, c3 REAL, c4 REAL, 
                       c5 REAL, c6 REAL, c7 REAL, c8 REAL, 
                       scope_alignment REAL, logic_score REAL,
                       subfields TEXT, fields TEXT, author_name TEXT, final_score REAL, timestamp DATETIME)""")

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='blockchain_por_weights'")
    table_exists = cursor.fetchone()
    
    if table_exists:
        cursor.execute("PRAGMA table_info(blockchain_por_weights)")
        columns = [row[1] for row in cursor.fetchall()]
        if "por_proof" not in columns or "formulas_hash" not in columns:
            cursor.execute("ALTER TABLE blockchain_por_weights RENAME TO old_blockchain_por_weights")
            cursor.execute("""CREATE TABLE blockchain_por_weights 
                              (block_height INTEGER PRIMARY KEY AUTOINCREMENT, 
                               w1 REAL, w2 REAL, w3 REAL, w4 REAL, w5 REAL, w6 REAL, w7 REAL, w8 REAL, 
                               timestamp DATETIME, previous_hash TEXT, validator_node TEXT, 
                               block_hash TEXT, eval_hash TEXT, model_used TEXT,
                               por_proof TEXT DEFAULT 'Genesis_Proof', formulas_hash TEXT DEFAULT 'Locked_State')""")
            try:
                cursor.execute("""INSERT INTO blockchain_por_weights 
                                  (block_height, w1, w2, w3, w4, w5, w6, w7, w8, timestamp, previous_hash, validator_node, block_hash, eval_hash, model_used)
                                  SELECT block_height, w1, w2, w3, w4, w5, w6, w7, w8, timestamp, previous_hash, validator_node, block_hash, eval_hash, model_used 
                                  FROM old_blockchain_por_weights""")
            except Exception:
                pass
            cursor.execute("DROP TABLE old_blockchain_por_weights")
    else:
        cursor.execute("""CREATE TABLE blockchain_por_weights 
                          (block_height INTEGER PRIMARY KEY AUTOINCREMENT, 
                           w1 REAL, w2 REAL, w3 REAL, w4 REAL, w5 REAL, w6 REAL, w7 REAL, w8 REAL, 
                           timestamp DATETIME, previous_hash TEXT, validator_node TEXT, 
                           block_hash TEXT, eval_hash TEXT, model_used TEXT,
                           por_proof TEXT DEFAULT 'Genesis_Proof', formulas_hash TEXT DEFAULT 'Locked_State')""")

    cursor.execute("CREATE TABLE IF NOT EXISTS global_eval_counter (count INTEGER)")
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
        "consensus_data": "TEXT DEFAULT '{}'",
        "evidence_report": "TEXT DEFAULT ''",
        "scilem_score": "REAL DEFAULT 50.0",
    }

    cursor.execute("PRAGMA table_info(papers_assessment)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    for col, dtype in target_columns_assessment.items():
        if col not in existing_cols:
            try:
                cursor.execute(f"ALTER TABLE papers_assessment ADD COLUMN {col} {dtype}")
            except Exception:
                pass

    # Performance Indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_eval_hash ON papers_assessment(eval_hash)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_eth_book ON papers_assessment(eth_book)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_author_name ON papers_assessment(author_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers_assessment(doi)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_por_eval_hash ON blockchain_por_weights(eval_hash)")

    conn.commit()
    _schema_initialized = True

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    enforce_database_schema(conn)
    
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

    return conn
