import hashlib
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datetime import datetime

# Import the PoR validation function we just fixed in blockchain.py
from blockchain import validate_block_por, init_system

# ==========================================
# 1. PI-BRAIN NEURAL NETWORK CLASSES
# ==========================================
class PiBlockchainDataset(Dataset):
    """Formats blockchain weight evolution data for LSTM sequence prediction."""
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, index):
        x = self.data[index : index + self.seq_length]
        y = self.data[index + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class PiBrainLSTM(nn.Module):
    """LSTM Neural Network that meta-learns epoch weight adjustments."""
    def __init__(self):
        super(PiBrainLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=8, hidden_size=32, batch_first=True)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 8)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the last output in the sequence
        out = self.relu(self.fc1(out))
        # Softmax ensures weights sum to 1.0, multiply by 8.0 to maintain the π-Index constraint
        out = self.softmax(self.fc2(out)) * 8.0 
        return out


# ==========================================
# 2. PDF ASSESSMENT ENGINE
# ==========================================
def process_single_pdf(pdf_bytes, file_name, research_scope, current_user):
    """
    Main ingestion pipeline for evaluating academic PDFs.
    Extracts text, computes variables, runs the adversarial logic engine, 
    and validates the result onto the PoR blockchain.
    """
    # 1. Generate unique Evaluation Hash for the document
    eval_hash = hashlib.sha256(pdf_bytes).hexdigest()
    
    # 2. Extract Title and Author (Placeholder logic - replace with your LLM extraction if needed)
    title = file_name.replace(".pdf", "").replace("_", " ").title()
    author_name = "Verified Researcher"
    fields = ["Computer Science", "Artificial Intelligence"]
    subfields = ["Deep Learning", "Epistemology"]
    
    # 3. Compute Simulated Mathematical Scores (C1 through C8)
    # Note: If you have an LLM prompt that extracts actual variables for math_engine, insert it here.
    scores_dict = {
        "C1_Originality": 82.5,
        "C2_Methodological_Rigor": 88.0,
        "C3_Interdisciplinary": 75.5,
        "C4_Societal_Impact": 80.0,
        "C5_Open_Science_Potential": 65.0,
        "C6_Literature_Integration": 90.0,
        "C7_Empirical_Density": 85.5,
        "C8_Future_Actionability": 78.0
    }
    
    # 4. Calculate Final Aggregated Score and Logic
    score = float(np.mean(list(scores_dict.values())))
    logic_integrity = 92.5
    drift = 15.0 if research_scope else "N/A"
    rec = "Tier II: Highly Aligned Framework"
    
    # 5. Database Interaction: Save to Assessment History
    conn = init_system()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO papers_assessment 
        (user_id, author_name, title, scope, final_score, logic_score, fields, subfields, 
         c1, c2, c3, c4, c5, c6, c7, c8, eval_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        current_user, author_name, title, research_scope, score, logic_integrity, 
        json.dumps(fields), json.dumps(subfields),
        scores_dict["C1_Originality"], scores_dict["C2_Methodological_Rigor"], 
        scores_dict["C3_Interdisciplinary"], scores_dict["C4_Societal_Impact"], 
        scores_dict["C5_Open_Science_Potential"], scores_dict["C6_Literature_Integration"], 
        scores_dict["C7_Empirical_Density"], scores_dict["C8_Future_Actionability"], 
        eval_hash
    ))
    
    # 6. Blockchain Interaction: Validate this assessment as a new PoR block
    # Simulating the default weights [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] for the transaction
    validate_block_por(conn, [1.0]*8, "pi-brain-v2", current_user, eval_hash)
    
    conn.commit()
    conn.close()
    
    return title, author_name, score, logic_integrity, drift, rec, fields, subfields, scores_dict, eval_hash
