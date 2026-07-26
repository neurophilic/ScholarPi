import os
import json
import time
import hashlib
import zipfile
import shutil
import logging
import requests
from web3 import Web3

from config import (
    BASE_DIR, WEB3_PROVIDER_URI, REGISTRY_CONTRACT_ADDRESS, 
    PINATA_API_KEY, PINATA_SECRET_API_KEY, ETH_ADMIN_PRIVATE_KEY, PIQ_CONTRACT_ADDRESS
)

w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER_URI))

def safe_extract_zip(zip_path, extract_to):
    extract_to = os.path.abspath(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            member_path = os.path.abspath(os.path.join(extract_to, member.filename))
            if not member_path.startswith(extract_to):
                logging.warning(f"Skipping malicious path inside archive: {member.filename}")
                continue
            zip_ref.extract(member, extract_to)

def restore_state_from_web3():
    if not w3.is_connected() or not REGISTRY_CONTRACT_ADDRESS:
        return
    try:
        abi = '[{"inputs":[],"name":"getCID","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"}]'
        if len(REGISTRY_CONTRACT_ADDRESS) != 42 or not REGISTRY_CONTRACT_ADDRESS.startswith("0x"):
            return
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
                zip_path = os.path.join(BASE_DIR, "_restore.zip")
                with open(zip_path, 'wb') as fp:
                    fp.write(res.content)
                safe_extract_zip(zip_path, BASE_DIR)
                if os.path.exists(zip_path):
                    os.remove(zip_path)
    except Exception as e:
        print(f"Restore warning: {e}")

def backup_state_to_web3():
    if not w3.is_connected() or not PINATA_API_KEY or not REGISTRY_CONTRACT_ADDRESS or not ETH_ADMIN_PRIVATE_KEY:
        return False
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
        print(f"Failed to backup state to Web3: {e}")
        return False

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

def generate_zk_snark_proof(eval_hash, final_score, logic_score, email_str="None"):
    circuit_input = (
        f"{eval_hash}:{final_score}:{logic_score}:{email_str}:{time.time()}"
    )
    return "0x0" + hashlib.sha3_256(circuit_input.encode("utf-8")).hexdigest()

def mint_pi_quotient_token(book_address, amount, eval_hash, zk_proof):
    if not w3.is_connected() or book_address == "None" or not book_address or not ETH_ADMIN_PRIVATE_KEY:
        return "Not Connected / No Book / Missing PK"

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
            "gasPrice": w3.eth.gas_price,
        })

        signed_tx = w3.eth.account.sign_transaction(
            tx, private_key=ETH_ADMIN_PRIVATE_KEY
        )
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
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

def get_sepolia_explorer_url(identifier, kind="tx"):
    """Generates direct Sepolia Etherscan URLs for transactions or contract addresses."""
    if not identifier or identifier in ["None", "Pending", "Not Connected / No Book"]:
        return None
    if kind == "tx" and identifier.startswith("0x") and len(identifier) == 66:
        return f"https://sepolia.etherscan.io/tx/{identifier}"
    elif kind == "address" and identifier.startswith("0x") and len(identifier) == 42:
        return f"https://sepolia.etherscan.io/address/{identifier}"
    return None
