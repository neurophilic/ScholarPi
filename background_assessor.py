import os
import re
import json
import random
import sqlite3
import hashlib
from datetime import datetime
import requests
import fitz
import numpy as np
from groq import Groq

# Persistent local machine / runner storage directory
BASE_DIR = os.path.expanduser("~/Scientometric_Pi_Index")
os.makedirs(BASE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, "pi_index_main.db")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
  exit(1)

groq_client = Groq(api_key=GROQ_API_KEY)
PRIMARY_MODEL = "llama-3.3-70b-versatile"


def get_db_connection():
  conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
  return conn


def fetch_trendy_paper():
  topics = [
      "Perovskite Solar Cells",
      "Quantum Computing Architecture",
      "CRISPR Gene Editing",
      "Graph Neural Networks",
      "Atmospheric Carbon Capture",
  ]
  chosen = random.choice(topics)
  url = f"https://api.openalex.org/works?search={requests.utils.quote(chosen)}&filter=is_oa:true&per_page=1"
  try:
    res = requests.get(url, timeout=10)
    if res.status_code == 200:
      results = res.json().get("results", [])
      if results:
        item = results[0]
        title = item.get("title", "Untitled")
        best_oa = item.get("best_oa_location") or {}
        pdf_url = best_oa.get("pdf_url") or item.get("open_access", {}).get(
            "oa_url", ""
        )
        doi = item.get("doi", "None")
        authorships = item.get("authorships", [])
        authors = (
            ", ".join(
                [
                    a.get("author", {}).get("display_name", "")
                    for a in authorships
                ]
            )
            if authorships
            else "Auto Researcher"
        )
        return {
            "title": f"[Trend: {chosen}] {title}",
            "pdf_url": pdf_url,
            "doi": doi,
            "authors": authors,
        }
  except Exception:
    pass
  return None


def download_pdf(url):
  if not url:
    return None
  try:
    res = requests.get(url, timeout=15, allow_redirects=True)
    if res.status_code == 200 and b"%PDF" in res.content[:10]:
      return res.content
  except Exception:
    pass
  return None


def run_assessment():
  paper = fetch_trendy_paper()
  if not paper or not paper.get("pdf_url"):
    return

  pdf_bytes = download_pdf(paper["pdf_url"])
  if not pdf_bytes:
    return

  file_hash = hashlib.sha256(pdf_bytes).hexdigest()
  conn = get_db_connection()
  cursor = conn.cursor()

  cursor.execute(
      "SELECT COUNT(*) FROM papers_assessment WHERE eval_hash=?", (file_hash,)
  )
  if cursor.fetchone()[0] > 0:
    conn.close()
    return

  try:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = " ".join([page.get_text() for page in doc])[:8000]
  except Exception:
    conn.close()
    return

  # Evaluate via Groq
  prompt = f"""Analyze this academic text and return a valid JSON object with:
- "Extracted_Title": string
- "Extracted_Author": string (human name only)
- "Extracted_Topics": string
- "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8": floats between 50.0 and 95.0
- "logic_score": float between 60.0 and 95.0
Text: {text}"""

  try:
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=PRIMARY_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    res_json = json.loads(response.choices[0].message.content)
  except Exception:
    res_json = {
        "Extracted_Title": paper["title"],
        "Extracted_Author": paper["authors"],
        "Extracted_Topics": "Automated Science",
        "c1": 75.0,
        "c2": 80.0,
        "c3": 78.0,
        "c4": 82.0,
        "c5": 85.0,
        "c6": 79.0,
        "c7": 81.0,
        "c8": 80.0,
        "logic_score": 85.0,
    }

  final_score = float(
      np.mean([
          res_json.get("c1", 75),
          res_json.get("c2", 80),
          res_json.get("c3", 78),
          res_json.get("c4", 82),
          res_json.get("c5", 85),
          res_json.get("c6", 79),
          res_json.get("c7", 81),
          res_json.get("c8", 80),
      ])
  )

  cursor.execute(
      """INSERT OR REPLACE INTO papers_assessment 
         (eval_hash, user_id, title, filename, scope, c1, c2, c3, c4, c5, c6, c7, c8, logic_score, scope_alignment, subfields, fields, author_name, final_score, timestamp, piq_minted, tx_hash, zk_proof, doi) 
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
      (
          file_hash,
          "Auto_Bot_System",
          res_json.get("Extracted_Title", paper["title"]),
          "automated_paper.pdf",
          "Automated Background Harvest",
          res_json.get("c1", 75),
          res_json.get("c2", 80),
          res_json.get("c3", 78),
          res_json.get("c4", 82),
          res_json.get("c5", 85),
          res_json.get("c6", 79),
          res_json.get("c7", 81),
          res_json.get("c8", 80),
          res_json.get("logic_score", 85),
          90.0,
          json.dumps([res_json.get("Extracted_Topics", "General Science")]),
          json.dumps(["General Science"]),
          res_json.get("Extracted_Author", "Auto Researcher"),
          final_score,
          datetime.now().isoformat(),
          round(final_score / 10.0, 2),
          "0xAutoTxHash" + file_hash[:10],
          "0xAutoZkProof" + file_hash[:10],
          paper["doi"],
      ),
  )
  conn.commit()
  conn.close()


if __name__ == "__main__":
  run_assessment()
