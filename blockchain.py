import sqlite3
import hashlib
import time
from datetime import datetime
from web3 import Web3
from config import DB_PATH, WEB3_PROVIDER_URI, ETH_ADMIN_PRIVATE_KEY, PICOIN_CONTRACT_ADDRESS

# Initialize Web3
w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER_URI))

def generate_blockchain_pi(block_height):
    """
    Generates Pi algorithmically using the Nilakantha series.
    The computational depth (iterations) is intrinsically linked to the blockchain's block height,
    making the generation dynamic rather than relying on a static string.
    """
    iterations = max(1, block_height * 50)  # Deeper blocks = more precision
    pi_approx = 3.0
    sign = 1.0
    for i in range(1, iterations + 1):
        n = i * 2
        pi_approx += sign * (4.0 / (n * (n + 1) * (n + 2)))
        sign *= -1.0
    return pi_approx

def init_system():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=20.0)
    cursor = conn.cursor()
    
    # 1. Main Assessment Table
    cursor.execute('''CREATE TABLE IF NOT EXISTS papers_assessment 
                      (eval_hash TEXT PRIMARY KEY, user_id TEXT, title TEXT, filename TEXT, scope TEXT,
                       c1 REAL, c2 REAL, c3 REAL, c4 REAL, 
                       c5 REAL, c6 REAL, c7 REAL, c8 REAL, 
                       scope_alignment REAL, logic_score REAL,
                       subfields TEXT, fields TEXT, author_name TEXT, final_score REAL, timestamp DATETIME,
                       eth_wallet TEXT, coins_minted REAL, tx_hash TEXT, zk_proof TEXT)''')
                       
    try: cursor.execute("ALTER TABLE papers_assessment ADD COLUMN logic_score REAL DEFAULT 0.0")
    except: pass 
    try: cursor.execute("ALTER TABLE papers_assessment ADD COLUMN author_name TEXT DEFAULT 'Unknown Author'")
    except: pass 
    try: cursor.execute("ALTER TABLE papers_assessment ADD COLUMN eth_wallet TEXT DEFAULT 'None'")
    except: pass 
    try: cursor.execute("ALTER TABLE papers_assessment ADD COLUMN coins_minted REAL DEFAULT 0.0")
    except: pass 
    try: cursor.execute("ALTER TABLE papers_assessment ADD COLUMN tx_hash TEXT DEFAULT 'Pending'")
    except: pass 
    try: cursor.execute("ALTER TABLE papers_assessment ADD COLUMN zk_proof TEXT DEFAULT 'None'")
    except: pass 
        
    # 2. Proof of Research Ledger Table
    cursor.execute('''CREATE TABLE IF NOT EXISTS blockchain_por_weights 
                      (block_height INTEGER PRIMARY KEY AUTOINCREMENT, 
                       w1 REAL, w2 REAL, w3 REAL, w4 REAL, 
                       w5 REAL, w6 REAL, w7 REAL, w8 REAL, 
                       timestamp DATETIME, previous_hash TEXT, 
                       validator_node TEXT, block_hash TEXT, eval_hash TEXT, model_used TEXT, 
                       por_proof TEXT, formulas_hash TEXT)''')
                       
    try: cursor.execute("ALTER TABLE blockchain_por_weights ADD COLUMN por_proof TEXT DEFAULT 'Genesis_Proof'")
    except: pass
    try: cursor.execute("ALTER TABLE blockchain_por_weights ADD COLUMN formulas_hash TEXT DEFAULT 'Locked_State'")
    except: pass

    cursor.execute('''CREATE TABLE IF NOT EXISTS global_eval_counter (count INTEGER)''')
    
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
                       
    cursor.execute("SELECT count FROM global_eval_counter")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO global_eval_counter (count) VALUES (0)")
        
    conn.commit()
    return conn

def validate_block_por(block_index, weights, timestamp, previous_hash, eval_hash, model_used, final_score, formulas_hash):
    """Proof-of-Research (PoR) Consensus sealing"""
    validator_node = "Validator_Pi_" + hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
    por_proof = f"PoR_{eval_hash[:12]}_Score:{final_score:.2f}"
    
    data_string = f"{block_index}{weights}{timestamp}{previous_hash}{validator_node}{por_proof}{model_used}{formulas_hash}"
    block_hash = hashlib.sha256(data_string.encode('utf-8')).hexdigest()
    
    return validator_node, block_hash, por_proof

def generate_zk_snark_proof(eval_hash, final_score, logic_score):
    """Simulates the generation of a zk-SNARK Groth16 proof for the smart contract."""
    circuit_input = f"{eval_hash}:{final_score}:{logic_score}:{time.time()}"
    return "0x0" + hashlib.sha3_256(circuit_input.encode('utf-8')).hexdigest()

def mint_pi_coin(wallet_address, amount, eval_hash, zk_proof):
    """Interacts with the Ethereum Smart Contract to mint PiCoin ($PIC)."""
    if not w3.is_connected() or wallet_address == "None" or not wallet_address:
        return "Not Connected / No Wallet"
        
    try:
        abi = '[{"inputs":[{"internalType":"address","name":"researcher","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"},{"internalType":"string","name":"evalHash","type":"string"},{"internalType":"bytes","name":"zkProof","type":"bytes"}],"name":"verifyProofAndMint","outputs":[],"stateMutability":"nonpayable","type":"function"}]'
        contract = w3.eth.contract(address=w3.to_checksum_address(PICOIN_CONTRACT_ADDRESS), abi=abi)
        account = w3.eth.account.from_key(ETH_ADMIN_PRIVATE_KEY)
        
        tx = contract.functions.verifyProofAndMint(
            w3.to_checksum_address(wallet_address),
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
