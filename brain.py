import os
import json
import time
import math
import random
import hashlib
import re
from datetime import datetime

import fitz
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from groq import Groq

from config import GROQ_API_KEY, PRIMARY_MODEL, FALLBACK_MODEL, MAX_TEXT_TOKENS, EPOCH_BLOCK_SIZE, BASE_DIR
from database import get_db_connection
from ledger import backup_state_to_web3, generate_zk_snark_proof, mint_pi_quotient_token, validate_block_por, generate_blockchain_pi
from integrations import clean_author_name, is_likely_institution, fetch_author_coara_metrics

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

class PidyneBlockchainDataset(Dataset):
    def __init__(self, data_matrix, lookback):
        self.data = data_matrix
        self.lookback = lookback

    def __len__(self):
        return len(self.data) - self.lookback

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.lookback]
        y = self.data[idx + self.lookback]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )

class PidyneLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_layer_size=32, output_size=8):
        super(PidyneLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_layer_size, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return torch.softmax(predictions, dim=-1) * 8.0

def get_evolving_system_context():
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT block_height, w1, w2, w3, w4, w5, w6, w7, w8 
            FROM blockchain_por_weights 
            ORDER BY block_height DESC LIMIT 1
        """)
        epoch_data = cursor.fetchone()
        
        cursor.execute("""
            SELECT stance, COUNT(*) 
            FROM desci_attestations 
            WHERE timestamp > datetime('now', '-7 days') 
            GROUP BY stance
        """)
        attestations = cursor.fetchall()
    finally:
        conn.close()

    context_str = "SYSTEM EVOLUTION CONTEXT:\n"
    if epoch_data:
        weights = epoch_data[1:9]
        max_idx = weights.index(max(weights))
        criteria_map = ["Originality", "Methodological Rigor", "Interdisciplinary", "Societal Impact", 
                        "Open Science", "Literature Integration", "Empirical Density", "FAIR Actionability"]
        context_str += f"- Current Blockchain Epoch {epoch_data[0]} heavily penalizes weak '{criteria_map[max_idx]}'. Apply maximum scrutiny to this dimension.\n"

    if attestations:
        context_str += "- Recent human peer-reviewers noted anomalies. Adjust strictness accordingly:\n"
        for stance, count in attestations:
            context_str += f"  * {count} recent human flags for: '{stance}'\n"

    return context_str

def harvest_fine_tuning_data(text_chunk, final_json_output, eval_hash):
    dataset_path = os.path.join(BASE_DIR, "scilem rlhf_dataset.jsonl")
    try:
        record = {
            "prompt": f"Extract Pi-Index Variables from this text:\n{text_chunk[:3000]}",
            "completion": json.dumps(final_json_output),
            "eval_hash": eval_hash,
            "timestamp": datetime.now().isoformat()
        }
        with open(dataset_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"Dataset harvest warning: {e}")

def adaptive_chunking(text, max_tokens):
    if len(text) <= max_tokens:
        return text
    front_matter = text[: int(max_tokens * 0.4)]
    back_matter = text[-int(max_tokens * 0.6) :]
    return front_matter + "\n...[TRUNCATED FOR TOKEN LIMITS]...\n" + back_matter

def evaluate_discriminator_and_divergence(text, model):
    if not groq_client: return 0.0, 0.5
    text_chunk = text[:5000]
    prompt = f"""Analyze this academic text for two adversarial threats:
1. Synthetic Hallucination / AI-Generated Preprint Flood (unnatural keyword stuffing, stylistic filler, or high-flown prose masking weak statistical substance).
2. Semantic-Empirical Divergence: Check if grandiose claims and equations drastically diverge from actual reported data variances.

Output a JSON object with two keys:
- "Gaming_Penalty": float from 0.0 (natural) to 1.0 (highly manipulated/synthetic).
- "Reproducibility_Score": float from 0.0 to 1.0 indicating whether code/data artifacts appear functional and verifiable.

Text: {text_chunk}"""
    
    for attempt in range(3):
        try:
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            res_json = json.loads(response.choices[0].message.content)
            return float(res_json.get("Gaming_Penalty", 0.0)), float(res_json.get("Reproducibility_Score", 0.5))
        except Exception as e:
            if any(k in str(e).lower() for k in ["413", "rate_limit_exceeded", "tokens", "429"]):
                time.sleep(2 ** attempt)
            else:
                break
    return 0.0, 0.5

def evaluate_scope_alignment(text, scope, model, text_limit):
    if not groq_client: return 0.0
    if not scope.strip():
        return 0.0
    text = adaptive_chunking(text, text_limit)
    prompt = f"""You are a research alignment tool. Read the following paper text and evaluate how well it aligns with this specific research scope/keyword: "{scope}"
Return ONLY a valid JSON object with a single key "Scope_Alignment" containing a float between 0.0 and 100.0.
Text: {text}"""

    for attempt in range(3):
        try:
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return float(json.loads(response.choices[0].message.content).get("Scope_Alignment", 0.0))
        except Exception as e:
            if any(k in str(e).lower() for k in ["413", "rate_limit_exceeded", "tokens", "429"]):
                time.sleep(2 ** attempt)
            else:
                break
    return 0.0

def extract_unpublished_authors_fallback(text):
    first_2k = text[:2500]
    lines = [line.strip() for line in first_2k.split("\n") if line.strip()]
    for line in lines[1:12]:
        clean_line = re.sub(r"[\d\*\†\‡\§\¶\(\)]", "", line).strip()
        if re.match(
            r"^[A-Z][a-z\.]+(\s+[A-Z]\.?)?\s+[A-Z][a-z]+(\s*,\s*[A-Z][a-z\.]+(\s+[A-Z]\.?)?\s+[A-Z][a-z]+)*$",
            clean_line,
        ):
            if len(clean_line) > 3 and not any(
                kw in clean_line.lower()
                for kw in [
                    "abstract", "introduction", "university", "department",
                    "contents", "journal", "bicocca", "milano", "institute",
                ]
            ):
                return clean_line
    return "Unidentified"

def evaluate_pdf_text_ensemble(text, model, text_limit, file_hash="unknown"):
    if not groq_client:
        return {
            "Extracted_Title": "Parsing Failed (No API Key)",
            "Extracted_Author": "Unidentified",
            "Extracted_Topics": "Core Research Domain",
            "Overall_Confidence": 0.0,
        }

    text = adaptive_chunking(text, text_limit)
    evolving_context = get_evolving_system_context()
    
    prompt = f"""You are the theoretical parser for the Pi-Index. Read the academic paper or draft manuscript and extract metadata and audit variables.
CRITICAL EQUITY & NORMALIZATION INSTRUCTION:
- Global research equity is paramount. Do NOT penalize non-native English writing styles.

{evolving_context}

CRITICAL INSTRUCTION FOR AUTHORS & TOPICS:
- Scan the first 2 pages carefully for human author names. Output as a clean comma-separated list of HUMAN author names (no brackets, no quotes, no "et al."). 
- NEVER output universities, departments, institutions, or organizational affiliations as authors. Output ONLY human author names. If none found, output "Unidentified".
- Extract 1 to 3 distinct, specific scientific research topics, domain subfields, or methodologies covered in this paper. Output as a comma-separated list of strings.

Extract Metadata: `Extracted_Title`, `Extracted_Author`, `Extracted_Topics`.
Extract Transparent Audit Variables (0.0 to 1.0): `semantic_novelty`, `laundering_penalty`, `rigor_index`, `citation_entropy`, `societal_linkage`, `D_open`, `J_code`, `citation_polarity_score`, `empirical_density`, `fair_compliance`.
Logic Mapping (0.0 to 1.0): `Evidence_Strength`, `Conclusion_Reach`, `Logical_Jumps`, `Premise_Validity`.
REQUIRED: Add an "Overall_Confidence" key (0.0 to 1.0) indicating your parsing certainty.
Return ONLY a valid JSON object. Text: {text}"""

    result_content = None
    for attempt in range(3):
        try:
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.1, 
                seed=random.randint(1, 1000),
                response_format={"type": "json_object"},
            )
            result_content = response.choices[0].message.content
            break
        except Exception as e:
            if any(k in str(e).lower() for k in ["413", "rate_limit_exceeded", "tokens", "429"]):
                time.sleep(2 ** attempt)
            else:
                break

    if result_content:
        try:
            parsed = json.loads(result_content)
            if isinstance(parsed, dict):
                harvest_fine_tuning_data(text, parsed, file_hash)
                return parsed
            elif isinstance(parsed, str):
                sub_parsed = json.loads(parsed)
                if isinstance(sub_parsed, dict):
                    harvest_fine_tuning_data(text, sub_parsed, file_hash)
                    return sub_parsed
        except Exception:
            pass
        
    return {
        "Extracted_Title": "Parsing Failed",
        "Extracted_Author": "Unidentified",
        "Extracted_Topics": "Core Research Domain",
        "Overall_Confidence": 0.0,
    }

def get_formulas_hash():
    criteria_state = (
        "C1:Semantic_Originality|C2:MDAR_Rigor|C3:Citation_Entropy|C4:Open_Infrastructure|C5:Containerized_Execution|C6:Citation_Polarity|C7:Empirical_Density|C8:FAIR_Actionability|CoARA_Dossier_v2.0"
    )
    return hashlib.sha256(criteria_state.encode("utf-8")).hexdigest()

def calculate_model_driven_weights(old_weights, scores, model_name, block_height):
    if "70b" in model_name:
        model_version, model_size = 3.3, 70.0
    else:
        model_version, model_size = 3.1, 8.0

    pi_accuracy = generate_blockchain_pi(block_height)
    delta_models = abs((3.3 * 70.0) - (3.1 * 8.0))
    mean_score = np.mean(scores)

    new_weights = []
    for i, old_w in enumerate(old_weights):
        stretched_score = max(
            1.0, min(100.0, mean_score + (scores[i] - mean_score) * 3.0)
        )
        weight_shift = (
            (model_version * model_size) / (delta_models * pi_accuracy)
        ) * ((stretched_score / 100.0) ** 2)
        w_new = old_w * 0.85 + (1.0 + weight_shift * 0.15) * 0.15
        new_weights.append(w_new)

    sum_of_weights = sum(new_weights)
    return [round((w / sum_of_weights) * 8.0, 6) for w in new_weights]

def compute_logical_integrity(extracted_logic_vars, gaming_penalty):
    evidence = extracted_logic_vars.get("Evidence_Strength", 0.5)
    conclusion_reach = extracted_logic_vars.get("Conclusion_Reach", 0.5)
    jumps = extracted_logic_vars.get("Logical_Jumps", 0.5)
    premise = extracted_logic_vars.get("Premise_Validity", 0.5)

    logic_gap = max(0.0, conclusion_reach - evidence)
    base_logic = (
        (premise * evidence)
        * np.exp(-(logic_gap * 2.0 + jumps * 1.5))
        * 100
    )
    logic_score = base_logic * (1.0 - (gaming_penalty * 0.9))
    return max(0.0, min(100.0, logic_score))

def compute_formulaic_criteria(
    vars_dict, reproducibility_score, sciscore_adherence=0.8
):
    scores = {}
    c1_raw = (
        vars_dict.get("semantic_novelty", 0.7)
        * 100
        * (1.0 - vars_dict.get("laundering_penalty", 0.1))
    )
    scores["C1_Originality"] = min(100.0, max(0.0, c1_raw))
    c2_raw = sciscore_adherence * vars_dict.get("rigor_index", 0.75) * 100
    scores["C2_Methodological_Rigor"] = min(100.0, max(0.0, c2_raw))
    c3_raw = vars_dict.get("citation_entropy", 0.6) * 100
    scores["C3_Interdisciplinary"] = min(100.0, max(0.0, c3_raw))
    c4_raw = vars_dict.get("societal_linkage", 0.65) * 100
    scores["C4_Societal_Impact"] = min(100.0, max(0.0, c4_raw))
    c5_raw = (
        (0.5 * vars_dict.get("D_open", 0.7))
        + (0.2 * vars_dict.get("J_code", 0.6))
        + (0.3 * reproducibility_score)
    ) * 100
    scores["C5_Open_Science_Potential"] = min(100.0, max(0.0, c5_raw))
    c6_raw = vars_dict.get("citation_polarity_score", 0.7) * 100
    scores["C6_Literature_Integration"] = min(100.0, max(0.0, c6_raw))
    c7_raw = vars_dict.get("empirical_density", 0.75) * 100
    scores["C7_Empirical_Density"] = min(100.0, max(0.0, c7_raw))
    c8_raw = vars_dict.get("fair_compliance", 0.8) * 100
    scores["C8_Future_Actionability"] = min(100.0, max(0.0, c8_raw))

    for key in scores:
        scores[key] = round(scores[key], 2)
    return scores

def calculate_complex_drift(alignment, scores):
    if not scores or alignment is None:
        return 0.0
    average_score = np.mean(scores)
    standard_deviation = np.std(scores)
    alignment_gap = (100.0 - alignment) / 100.0
    drift_metric = (
        100.0
        * (
            1.0
            - np.exp(
                -3.0
                * (alignment_gap ** 1.5)
                * (1.0 + (standard_deviation / 100.0))
                / (0.1 + (average_score / 100.0))
            )
        )
    )
    return float(max(0.0, min(100.0, drift_metric)))

def get_recommendation_spectrum(score, drift):
    if drift == "N/A":
        return "N/A"
    synergy = score * (1.0 - (drift / 100.0) ** 1.5)
    if synergy >= 85:
        return "Tier I: Core Paradigm (Optimal Synergy)"
    elif synergy >= 70:
        return "Tier II: Highly Aligned Framework"
    elif synergy >= 55:
        return "Tier III: Moderately Synergistic"
    elif synergy >= 40:
        return "Tier IV: Tangential Relevance"
    elif synergy >= 25:
        return "Tier V: Epistemic Divergence"
    else:
        return "Tier VI: Orthogonal / Unrelated Noise"

def generate_rebuttal_strategy(scores_dict):
    if not scores_dict:
        return "No scores available to generate a rebuttal strategy."

    weakest_criterion = min(scores_dict, key=scores_dict.get)
    strongest_criterion = max(scores_dict, key=scores_dict.get)

    strategy = (
        f"**Strategic Pivot:** Leverage your high score in"
        f" **{strongest_criterion.replace('_', ' ')}**"
        f" ({scores_dict[strongest_criterion]:.1f}/100) to distract from the"
        f" manuscript's primary vulnerability in"
        f" **{weakest_criterion.replace('_', ' ')}**"
        f" ({scores_dict[weakest_criterion]:.1f}/100).\n\n"
    )
    if "Originality" in weakest_criterion:
        strategy += (
            "**Defense Tactic:** Argue that the paper value lies in synthesis and"
            " rigorous validation rather than paradigm disruption."
        )
    elif "Rigor" in weakest_criterion:
        strategy += (
            "**Defense Tactic:** Pre-emptively acknowledge sample size limitations"
            " in the discussion section. Frame the methodology as an exploratory pilot."
        )
    else:
        strategy += (
            "**Defense Tactic:** Focus the reviewers attention on the empirical"
            " density of your dataset."
        )
    return strategy

def process_single_pdf(
    file_bytes,
    filename,
    scope,
    user_id,
    book_address="None",
    email="None",
    provided_doi="None",
    force_proceed=False,
):
    active_weights = [1.0] * 8
    works_count, cited_by_count, credit_role = 0.0, 0, "Data Curation"
    warnings_list = []

    if file_bytes is None or len(file_bytes) == 0:
        empty_scores = {
            k: 0.0
            for k in [
                "C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary",
                "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration",
                "C7_Empirical_Density", "C8_Future_Actionability",
            ]
        }
        warnings_list.append("Binary payload is empty or download/extraction failed.")
        return (
            "Download/Extraction Failed", "Unidentified", 0.0, 0.0, "N/A", "N/A",
            ["Unspecified Domain"], ["Unspecified Sub-domain"], empty_scores,
            "Failed", 0.0, "None", "None", active_weights, 0.85, 4, 0.0, False, warnings_list
        )

    file_hash = hashlib.sha256(file_bytes).hexdigest()
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT final_score, logic_score, title, fields, subfields, author_name,"
            " c1, c2, c3, c4, c5, c6, c7, c8, piq_minted, tx_hash, zk_proof,"
            " mdar_adherence_score, rrid_valid_count, reproducibility_score FROM"
            " papers_assessment WHERE eval_hash=?",
            (file_hash,),
        )
        cached_result = cursor.fetchone()

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            pdf_meta_author = doc.metadata.get("author", "").strip()
            
            text_blocks = []
            for page in doc:
                text_blocks.append(page.get_text("text", sort=True))
            full_text = "\n".join(text_blocks)
        except Exception as e:
            warnings_list.append(f"Invalid PDF structure or PyMuPDF parsing exception: {e}")
            full_text = ""

        if len(full_text.strip()) < 150:
            warnings_list.append("Sparse text layer detected (< 150 characters extracted; likely an image-only PDF scan).")

        scope_alignment = (
            evaluate_scope_alignment(full_text, scope, FALLBACK_MODEL, MAX_TEXT_TOKENS)
            if scope.strip()
            else 0.0
        )

        if cached_result and not force_proceed:
            score, logic_score, title, fields_str, subfields_str, author_name, *rest = (
                cached_result
            )
            c_scores = rest[:8]
            piq_minted, tx_hash, zk_proof, mdar_score, rrid_count, repro_score = (
                rest[8], rest[9], rest[10], rest[11], rest[12], rest[13],
            )
            fields = json.loads(fields_str) if fields_str else ["Unspecified Domain"]
            subfields = (
                json.loads(subfields_str) if subfields_str else ["Unspecified Sub-domain"]
            )

            drift = (
                calculate_complex_drift(scope_alignment, c_scores)
                if scope.strip()
                else "N/A"
            )
            rec = (
                get_recommendation_spectrum(score, drift) if scope.strip() else "N/A"
            )
            scores_dict = {
                "C1_Originality": c_scores[0],
                "C2_Methodological_Rigor": c_scores[1],
                "C3_Interdisciplinary": c_scores[2],
                "C4_Societal_Impact": c_scores[3],
                "C5_Open_Science_Potential": c_scores[4],
                "C6_Literature_Integration": c_scores[5],
                "C7_Empirical_Density": c_scores[6],
                "C8_Future_Actionability": c_scores[7],
            }

            cursor.execute(
                "SELECT w1, w2, w3, w4, w5, w6, w7, w8 FROM blockchain_por_weights"
                " WHERE eval_hash=?",
                (file_hash,),
            )
            weight_res = cursor.fetchone()
            used_weights = weight_res if weight_res else active_weights
            return (
                title, clean_author_name(author_name), score, logic_score, drift, rec,
                fields, subfields, scores_dict, file_hash, piq_minted, tx_hash, zk_proof,
                used_weights, mdar_score, rrid_count, repro_score, True, warnings_list,
            )

        gaming_penalty, reproducibility_score = evaluate_discriminator_and_divergence(
            full_text, FALLBACK_MODEL
        )
        if gaming_penalty > 0.40:
            warnings_list.append(f"Synthetic / AI-laundering gaming penalty ({gaming_penalty:.2f}) exceeds safety threshold (0.40).")

        try:
            raw_data = evaluate_pdf_text_ensemble(
                full_text, PRIMARY_MODEL, MAX_TEXT_TOKENS, file_hash
            )
            model_used = PRIMARY_MODEL
        except Exception:
            try:
                reduced_limit = int(MAX_TEXT_TOKENS * 0.6)
                raw_data = evaluate_pdf_text_ensemble(
                    full_text, FALLBACK_MODEL, reduced_limit, file_hash
                )
                model_used = FALLBACK_MODEL
            except Exception:
                warnings_list.append("LLM text ensemble extraction failed completely.")
                raw_data = {
                    "Extracted_Title": filename,
                    "Extracted_Author": "Unidentified",
                    "Extracted_Topics": "Core Research Domain",
                    "Overall_Confidence": 0.0,
                }

        if not isinstance(raw_data, dict):
            raw_data = {
                "Extracted_Title": filename,
                "Extracted_Author": "Unidentified",
                "Extracted_Topics": "Core Research Domain",
                "Overall_Confidence": 0.0,
            }

        confidence = float(raw_data.get("Overall_Confidence", 0.9))
        if confidence < 0.50:
            warnings_list.append(f"Low LLM parsing confidence score ({confidence * 100:.1f}% < 50%).")

        extracted_author_check = str(raw_data.get("Extracted_Author", "Unidentified"))
        extracted_title_check = str(raw_data.get("Extracted_Title", ""))

        if extracted_author_check.lower() in ["unidentified", "unknown", "none", "", "research scholar"]:
            warnings_list.append("Author metadata could not be reliably verified or identified in document header.")

        title = raw_data.get("Extracted_Title", filename)
        extracted_author = clean_author_name(extracted_author_check)
        extracted_topics = str(
            raw_data.get("Extracted_Topics", "Core Research Domain")
        ).strip()

        if (
            is_likely_institution(extracted_author)
            or not extracted_author
            or extracted_author.lower()
            in [
                "unknown", "unknown author", "none", "n/a",
                "research scholar", "unidentified",
            ]
            or extracted_author == os.path.splitext(filename)[0]
        ):
            if (
                pdf_meta_author.strip()
                and pdf_meta_author.lower() not in ["unknown", "none"]
                and not is_likely_institution(pdf_meta_author)
            ):
                extracted_author = clean_author_name(pdf_meta_author.strip())
            else:
                extracted_author = clean_author_name(
                    extract_unpublished_authors_fallback(full_text)
                )
                if is_likely_institution(extracted_author):
                    extracted_author = "Unidentified"

        if isinstance(extracted_topics, str):
            subfields = [
                s.strip().title() for s in extracted_topics.split(",") if s.strip()
            ]
        elif isinstance(extracted_topics, list):
            subfields = [
                str(s).strip().title() for s in extracted_topics if str(s).strip()
            ]
        else:
            subfields = ["Core Research Domain"]
        if not subfields:
            subfields = ["Core Research Domain"]
        fields = [subfields[0]]

        cursor.execute("UPDATE global_eval_counter SET count = count + 1")
        conn.commit()
        cursor.execute("SELECT count FROM global_eval_counter")
        total_evals = cursor.fetchone()[0]

        cursor.execute(
            "SELECT block_height, block_hash, w1, w2, w3, w4, w5, w6, w7, w8 FROM"
            " blockchain_por_weights ORDER BY block_height DESC LIMIT 1"
        )
        epoch_data = cursor.fetchone()
        block_height, previous_hash, old_weights = (
            epoch_data[0], epoch_data[1], epoch_data[2:],
        )

        variables = raw_data if isinstance(raw_data, dict) else {}
        scores_dict = compute_formulaic_criteria(
            variables, reproducibility_score, sciscore_adherence=0.82
        )
        scores = [
            scores_dict[k]
            for k in [
                "C1_Originality", "C2_Methodological_Rigor", "C3_Interdisciplinary",
                "C4_Societal_Impact", "C5_Open_Science_Potential", "C6_Literature_Integration",
                "C7_Empirical_Density", "C8_Future_Actionability",
            ]
        ]

        logic_integrity = compute_logical_integrity(raw_data, gaming_penalty)

        raw_final_score = float(np.dot(scores, old_weights)) / 8.0
        final_score = float(raw_final_score * (0.7 + (logic_integrity / 333.3)))
        formulas_hash = get_formulas_hash()

        if final_score < 60.0:
            warnings_list.append(f"Final score ({final_score:.2f}) is below quality floor (60.0).")

        if total_evals % EPOCH_BLOCK_SIZE == 0:
            active_weights = calculate_model_driven_weights(
                old_weights, scores, model_used, block_height
            )
            timestamp = datetime.now().isoformat()
            val_node, block_hash, por_proof = validate_block_por(
                block_height + 1,
                active_weights,
                timestamp,
                previous_hash,
                file_hash,
                model_used,
                final_score,
                formulas_hash,
            )
            cursor.execute(
                """INSERT INTO blockchain_por_weights (block_height, w1, w2, w3, w4, w5, w6, w7, w8, timestamp, previous_hash, validator_node, block_hash, eval_hash, model_used, por_proof, formulas_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    block_height + 1,
                    *active_weights,
                    timestamp,
                    previous_hash,
                    val_node,
                    block_hash,
                    file_hash,
                    model_used,
                    por_proof,
                    formulas_hash,
                ),
            )
        else:
            active_weights = old_weights

        piq_minted = 0.0
        if not warnings_list or force_proceed:
            co_authors = [a.strip() for a in extracted_author.split(",") if a.strip()]
            num_authors = max(1, len(co_authors))

            cursor.execute(
                "SELECT COUNT(*) FROM papers_assessment WHERE user_id = ?",
                (user_id,)
            )
            user_submission_count = cursor.fetchone()[0]
            decay_multiplier = 1.0 / math.sqrt(user_submission_count + 1)

            base_piq = (final_score / 10.0)
            piq_minted = round((base_piq / num_authors) * decay_multiplier, 2)

        zk_proof = generate_zk_snark_proof(
            file_hash, final_score, logic_integrity, "None"
        )
        unique_author_book = (
            "0x" + hashlib.sha256(extracted_author.encode()).hexdigest()[:40]
            if extracted_author != "Unidentified"
            else book_address
        )
        tx_hash = mint_pi_quotient_token(
            unique_author_book, piq_minted, file_hash, zk_proof
        )

        drift = (
            calculate_complex_drift(scope_alignment, scores)
            if scope.strip()
            else "N/A"
        )
        rec = (
            get_recommendation_spectrum(final_score, drift) if scope.strip() else "N/A"
        )

        mdar_score = 0.85
        rrid_count = 4
        credit_roles_str = json.dumps(
            [credit_role, "Methodology Validation", "Open Science Curation"]
        )

        cursor.execute(
            """INSERT OR REPLACE INTO papers_assessment (eval_hash, user_id, title, filename, scope, c1, c2, c3, c4, c5, c6, c7, c8, logic_score, scope_alignment, subfields, fields, author_name, final_score, timestamp, eth_book, piq_minted, tx_hash, zk_proof, did, zk_email_proof, gaming_penalty, mdar_adherence_score, rrid_valid_count, credit_taxonomy_roles, reproducibility_score, doi) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                file_hash, user_id, title, filename, scope, *scores,
                logic_integrity, scope_alignment, json.dumps(subfields),
                json.dumps(fields), extracted_author, final_score,
                datetime.now().isoformat(), unique_author_book, piq_minted,
                tx_hash, zk_proof, user_id, "None", gaming_penalty,
                mdar_score, rrid_count, credit_roles_str, reproducibility_score,
                provided_doi,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    backup_state_to_web3()

    return (
        title, extracted_author, final_score, logic_integrity, drift, rec,
        fields, subfields, scores_dict, file_hash, piq_minted, tx_hash, zk_proof,
        active_weights, mdar_score, rrid_count, reproducibility_score, False, warnings_list,
    )
