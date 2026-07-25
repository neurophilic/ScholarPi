import os
import streamlit as st

# --- CONFIGURATION & ENVIRONMENT SETUP ---
PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_TEXT_TOKENS = 12000
SEED_NUMBER = 42
EPOCH_BLOCK_SIZE = 5

# --- ETHEREUM & WEB3 SETTINGS ---
WEB3_PROVIDER_URI = os.getenv("WEB3_PROVIDER_URI", "https://sepolia.infura.io/v3/YOUR_INFURA_PROJECT_ID")
ETH_ADMIN_PRIVATE_KEY = os.getenv("ETH_ADMIN_PRIVATE_KEY", "0x0000000000000000000000000000000000000000000000000000000000000000")
EPC_CONTRACT_ADDRESS = os.getenv("EPC_CONTRACT_ADDRESS", "0xYourDeployedContractAddressHere")

BASE_DIR = os.path.abspath('./Scientometric_Pi_Index')
os.makedirs(BASE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, 'pi_index_main.db')

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
