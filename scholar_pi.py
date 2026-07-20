import os
import sqlite3
import json
import hashlib
import time
import re
import requests
import colorsys
import math
import tempfile
import base64
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
from groq import Groq
from pyvis.network import Network

# --- MACHINE LEARNING IMPORTS ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- 1. CONFIGURATION & ENVIRONMENT ---
st.set_page_config(page_title="π-Index Assessment Engine", layout="wide")

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_TEXT_TOKENS = 12000 # ~2500 tokens to safely stay below the TPM limit
SEED_NUMBER = 42

# BLOCKCHAIN CONFIGURATION
EPOCH_BLOCK_SIZE = 5

BASE_DIR = os.path.abspath('./Scientometric_Pi_Index')
os.makedirs(BASE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, 'pi_index_main.db')

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("API Key not found! Please configure your environment variables or Streamlit Secrets.")
    st.stop()
client = Groq(api_key=GROQ_API_KEY)

# --- UTILITY FUNCTIONS ---
def extract_json_from_llm(text):
    """Robustly strip markdown formatting if the LLM wraps the JSON payload."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("
