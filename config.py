import os
import hashlib

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_TEXT_TOKENS = 15000
EPOCH_BLOCK_SIZE = 5

WEB3_PROVIDER_URI = os.getenv("WEB3_PROVIDER_URI", "https://ethereum-sepolia-rpc.publicnode.com")
ETH_ADMIN_PRIVATE_KEY = os.getenv("ETH_ADMIN_PRIVATE_KEY", "")
PIQ_CONTRACT_ADDRESS = os.getenv("PIQ_CONTRACT_ADDRESS", "0xaE7a504aCF32ABf0E891B74bF39E4527999A6256")

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
OR_API_KEY = get_secret("OR_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
ORCID_CLIENT_ID = get_secret("ORCID_CLIENT_ID")
ORCID_CLIENT_SECRET = get_secret("ORCID_CLIENT_SECRET")
ORCID_REDIRECT_URI = get_secret("ORCID_REDIRECT_URI")

GENESIS_BLOCK_CONFIG = {
    "block_height": 1,
    "weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "timestamp": "2026-01-01T00:00:00.000000",
    "previous_hash": "0" * 64,
    "validator_node": "Validator_Pi_Genesis",
    "eval_hash": "genesis",
    "model_used": "Genesis_Ensemble",
    "por_proof": "Genesis_Proof_Anchor",
    "formulas_hash": hashlib.sha256(b"C1:Semantic_Originality|C2:MDAR_Rigor|C3:Citation_Entropy|C4:Open_Infrastructure|C5:Containerized_Execution|C6:Citation_Polarity|C7:Empirical_Density|C8:Future_Actionability_FAIR|CoARA_Dossier_v2.0").hexdigest(),
}

HOT_TOPICS = [
    "Quantum Error Correction", "Generative AI in Oncology", "CRISPR-Cas12 Therapeutics",
    "Solid-State Battery Electrolytes", "Perovskite Solar Cell Efficiency",
    "Neuromorphic Computing Hardware", "Neural Radiance Fields 3D Reconstruction",
    "Carbon Capture Metal-Organic Frameworks", "Fusion Energy Plasma Confinement",
    "Exoplanet Atmospheric Spectroscopy"
]
