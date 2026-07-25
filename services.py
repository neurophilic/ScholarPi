import re
import requests
import plotly.graph_objects as go
from io import BytesIO
import pandas as pd

# --- DOI INTEGRATION ---
def fetch_doi_metadata(doi):
    """Resolves a DOI to fetch metadata and open-access PDF links via Unpaywall."""
    clean_doi = doi.replace("https://doi.org/", "").replace("doi.org/", "").strip()
    unpaywall_url = f"https://api.unpaywall.org/v2/{clean_doi}?email=research@pi-index.org"
    
    try:
        response = requests.get(unpaywall_url, timeout=10)
        if response.status_code == 200:
            res = response.json()
            title = res.get("title", "Unknown Title")
            authors_list = res.get("z_authors", [])
            authors = ", ".join([a.get("family", "") for a in authors_list]) if authors_list else "Unknown Author"
            pdf_url = res.get("best_oa_location", {}).get("url_for_pdf", None) if res.get("best_oa_location") else None
            
            return {
                "title": title,
                "authors": authors,
                "pdf_url": pdf_url
            }
        return None
    except Exception:
        return None

def download_pdf_from_url(pdf_url):
    """Downloads a PDF from a raw URL."""
    try:
        res = requests.get(pdf_url, timeout=15)
        if res.status_code == 200:
            return res.content
        return None
    except Exception:
        return None

# --- SUPER FEATURE: AI REBUTTAL GENERATOR ---
def generate_rebuttal_strategy(scores_dict):
    """Analyzes the Pi-Index scores to build a strategic peer-review defense plan."""
    if not scores_dict:
        return "No scores available to generate a rebuttal strategy."
        
    weakest_criterion = min(scores_dict, key=scores_dict.get)
    strongest_criterion = max(scores_dict, key=scores_dict.get)
    
    strategy = f"### 🛡️ Automated Peer-Review Defense Strategy\n\n"
    strategy += f"**Strategic Pivot:** Leverage your high score in **{weakest_criterion.replace('_', ' ')}** ({scores_dict[strongest_criterion]:.1f}/100) to distract from the manuscript's primary vulnerability in **{weakest_criterion.replace('_', ' ')}** ({scores_dict[weakest_criterion]:.1f}/100).\n\n"
    
    if "Originality" in weakest_criterion:
        strategy += "**Defense Tactic:** Argue that the paper’s value lies in synthesis and rigorous validation rather than paradigm disruption. Emphasize that cumulative science requires foundational solidity over risky novelties."
    elif "Rigor" in weakest_criterion:
        strategy += "**Defense Tactic:** Pre-emptively acknowledge sample size limitations in the discussion section. Frame the methodology as an 'exploratory pilot' to lower the expectation of absolute statistical certainty."
    elif "Societal" in weakest_criterion:
        strategy += "**Defense Tactic:** Shift the narrative from immediate societal application to 'essential foundational groundwork'. Argue that downstream societal impact is impossible without this specific theoretical gap being closed."
    else:
        strategy += "**Defense Tactic:** Focus the reviewers' attention on the empirical density of your dataset. Acknowledge minor structural gaps but insist the volume of data speaks for itself."
        
    return strategy

# --- FEATURE 3: Comparison Radar Plot ---
def create_radar_comparison(title1, scores1, title2, scores2):
    categories = [
        'C1: Originality', 'C2: Method Rigor', 'C3: Interdisciplinary', 
        'C4: Societal Impact', 'C5: Open Science', 'C6: Lit Integration', 
        'C7: Empirical Density', 'C8: Actionability'
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[scores1.get(k, 0) for k in scores1], theta=categories, fill='toself', name=title1[:35] + ("..." if len(title1) > 35 else "")
    ))
    fig.add_trace(go.Scatterpolar(
        r=[scores2.get(k, 0) for k in scores2], theta=categories, fill='toself', name=title2[:35] + ("..." if len(title2) > 35 else "")
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, margin=dict(l=40, r=40, t=30, b=30))
    return fig

# --- FEATURE 4: Advanced Reporting & Exporting ---
def generate_latex_report(title, author, final_score, logic_score, scores_dict, eval_hash):
    return rf"""\documentclass{{article}}
\usepackage{{booktabs}}
\usepackage{{geometry}}
\geometry{{a4paper, margin=1in}}

\title{{\pi-Index Assessment Report:\\ \Large {title}}}
\author{{{author}}}
\date{{\today}}

\begin{document}
\maketitle

\section*{{Assessment Overview}}
\begin{{itemize}}
    \item \textbf{{Overall \pi-Index Score:}} {final_score:.2f} / 100.0
    \item \textbf{{Logical Integrity Index:}} {logic_score:.2f}\%
    \item \textbf{{Evaluation Hash:}} \texttt{{{eval_hash}}}
\end{itemize}

\section*{{Detailed Metric Breakdown}}
\begin{{table}}[h!]
\centering
\begin{{tabular}}{{llc}}
\toprule
\textbf{{Code}} & \textbf{{Metric Name}} & \textbf{{Score (0--100)}} \\
\midrule
C1 & Originality & {scores_dict.get('C1_Originality', 0.0):.2f} \\
C2 & Methodological Rigor & {scores_dict.get('C2_Methodological_Rigor', 0.0):.2f} \\
C3 & Interdisciplinary Capacity & {scores_dict.get('C3_Interdisciplinary', 0.0):.2f} \\
C4 & Societal Impact & {scores_dict.get('C4_Societal_Impact', 0.0):.2f} \\
C5 & Open Science Potential & {scores_dict.get('C5_Open_Science_Potential', 0.0):.2f} \\
C6 & Literature Integration & {scores_dict.get('C6_Literature_Integration', 0.0):.2f} \\
C7 & Empirical Density & {scores_dict.get('C7_Empirical_Density', 0.0):.2f} \\
C8 & Future Actionability & {scores_dict.get('C8_Future_Actionability', 0.0):.2f} \\
\bottomrule
\end{tabular}
\caption{{\pi-Index Multidimensional Assessment}}
\end{{table}}

\end{{document}}"""

def generate_bibtex(title, author, eval_hash):
    clean_title = re.sub(r'[^a-zA-Z0-9]', '', title)[:20]
    first_author = author.split()[0] if author else "Unknown"
    cite_key = f"{first_author}{clean_title}2026"
    return f"""@article{{{cite_key},
  title = {{{title}}},
  author = {{{author}}},
  note = {{Evaluated via \\pi-Index Engine, Evaluation Hash: {eval_hash}}},
  year = {{2026}}
}}"""
