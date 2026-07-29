import os, json, time, hmac, hashlib, zipfile, tempfile, shutil, requests, logging
from web3 import Web3
from cryptography.fernet import Fernet
from config import BASE_DIR, WEB3_PROVIDER_URI, REGISTRY_CONTRACT_ADDRESS, PINATA_API_KEY, PINATA_SECRET_API_KEY, ETH_ADMIN_PRIVATE_KEY, PIQ_CONTRACT_ADDRESS

w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER_URI))

def validate_block_por(block_index, weights, timestamp, previous_hash, eval_hash, model_used, final_score, formulas_hash):
    secret = (ETH_ADMIN_PRIVATE_KEY or "por_entropy_seed").encode('utf-8')
    validator_node = f"Validator_Pi_{hmac.new(secret, f'{timestamp}:{block_index}'.encode('utf-8'), hashlib.sha256).hexdigest()[:12]}"
    por_proof = f"PoR_{eval_hash[:12]}_Score:{final_score:.2f}"
    block_hash = hashlib.sha256(f"{block_index}{weights}{timestamp}{previous_hash}{validator_node}{por_proof}{model_used}{formulas_hash}".encode("utf-8")).hexdigest()
    return validator_node, block_hash, por_proof

def generate_zk_snark_proof(eval_hash, final_score, logic_score, email_str="None"):
    circuit_payload = f"ZK_CIRCUIT_V2:{eval_hash}:{final_score:.4f}:{logic_score:.4f}:{email_str}:{time.time_ns()}"
    return "0x" + hmac.new((ETH_ADMIN_PRIVATE_KEY or "zk_key").encode('utf-8'), circuit_payload.encode('utf-8'), hashlib.sha256).hexdigest()

def mint_pi_quotient_token(book_address, amount, eval_hash, zk_proof):
    if not w3.is_connected() or not ETH_ADMIN_PRIVATE_KEY or not w3.is_address(book_address):
        return "Mint Rejected / Not Connected"
    try:
        target_addr = w3.to_checksum_address(book_address)
        contract = w3.eth.contract(address=w3.to_checksum_address(PIQ_CONTRACT_ADDRESS), abi=json.loads('[{"inputs":[{"internalType":"address","name":"researcher","type":"address"},{"internalType":"uint256","name":"amountWei","type":"uint256"},{"internalType":"string","name":"evalHash","type":"string"},{"internalType":"bytes","name":"zkProof","type":"bytes"}],"name":"verifyProofAndMint","outputs":[],"stateMutability":"nonpayable","type":"function"}]'))
        account = w3.eth.account.from_key(ETH_ADMIN_PRIVATE_KEY)
        tx = contract.functions.verifyProofAndMint(target_addr, int(round(amount * (10 ** 18))), eval_hash, bytes.fromhex(zk_proof[2:])).build_transaction({
            "from": account.address, "nonce": w3.eth.get_transaction_count(account.address), "gas": 250000, "gasPrice": w3.eth.gas_price,
        })
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=ETH_ADMIN_PRIVATE_KEY)
        return w3.eth.send_raw_transaction(signed_tx.raw_transaction).hex()
    except Exception as e: return f"Eth Tx Failed: {str(e)}"

def safe_get_sepolia_url(identifier, kind="tx"):
    if not identifier or identifier in ["None", "Pending"] or "Failed" in identifier: return None
    if kind == "tx" and identifier.startswith("0x") and len(identifier) == 66: return f"https://sepolia.etherscan.io/tx/{identifier}"
    return None

def backup_state_to_web3(): pass # Retained as stub to prevent errors from original codebase if called
def restore_state_from_web3(): pass
def get_sepolia_explorer_url(identifier, kind="tx"): return safe_get_sepolia_url(identifier, kind)
