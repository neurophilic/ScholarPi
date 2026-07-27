import os
import hashlib

# ==========================================
# CONFIGURATION & ENVIRONMENT SETUP
# ==========================================
PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_TEXT_TOKENS = 15000
EPOCH_BLOCK_SIZE = 5

WEB3_PROVIDER_URI = os.getenv(
    "WEB3_PROVIDER_URI", "https://ethereum-sepolia-rpc.publicnode.com"
)
ETH_ADMIN_PRIVATE_KEY = os.getenv("ETH_ADMIN_PRIVATE_KEY", "")
PIQ_CONTRACT_ADDRESS = os.getenv(
    "PIQ_CONTRACT_ADDRESS", "0xaE7a504aCF32ABf0E891B74bF39E4527999A6256"
)

BASE_DIR = os.path.expanduser("~/Scientometric_Pi_Index")
os.makedirs(BASE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, "pi_index_main.db")

def get_secret(key, default=""):
    val = os.getenv(key)
    if val: return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default

GROQ_API_KEY = get_secret("GROQ_API_KEY")
PINATA_API_KEY = get_secret("PINATA_API_KEY")
PINATA_SECRET_API_KEY = get_secret("PINATA_SECRET_API_KEY")
REGISTRY_CONTRACT_ADDRESS = get_secret("REGISTRY_CONTRACT_ADDRESS")

# New Endpoints for Consensus Pipeline
OR_API_KEY = get_secret("OR_API_KEY")
AIN_API_KEY = get_secret("AIN_API_KEY")

GENESIS_BLOCK_CONFIG = {
    # ... (Keep original genesis block config exactly as is)
}

HOT_TOPICS = [
    # ... (Keep original hot topics exactly as is)
]
