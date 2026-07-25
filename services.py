import re
import requests
import plotly.graph_objects as go
from io import BytesIO
import pandas as pd

def fetch_doi_metadata(doi):
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
            return {"title": title, "authors": authors, "pdf_url": pdf_url}
        return None
    except Exception: return None

def download_pdf_from_url(pdf_url):
    try:
        res = requests.get(pdf_url, timeout=15)
        if res.status_code == 200: return res.content
        return None
    except Exception: return None

def generate_rebuttal_strategy(scores_dict):
    if not scores_dict: return "No scores available to generate a rebuttal strategy."
        
    weakest_criterion = min(scores_dict, key=scores_dict.get)
    strongest_criterion = max(scores_dict, key=scores_dict.get)
    
    strategy = f"**Strategic Pivot:** Leverage your high score in **{strongest_criterion.replace('_', ' ')}** ({scores_dict[strongest_criterion]:.1f}/100) to distract from the manuscript's primary vulnerability in **{weakest_criterion.replace('_', ' ')}** ({scores_dict[weakest_criterion]:.1f}/100).\n\n"
    
    if "Originality" in weakest_criterion:
        strategy += "**Defense Tactic:** Argue that the paper’s value lies in synthesis and rigorous validation rather than paradigm disruption. Emphasize that cumulative science requires foundational solidity over risky novelties."
    elif "Rigor" in weakest_criterion:
        strategy += "**Defense Tactic:** Pre-emptively acknowledge sample size limitations in the discussion section. Frame the methodology as an 'exploratory pilot' to lower the expectation of absolute statistical certainty."
    elif "Societal" in weakest_criterion:
        strategy += "**Defense Tactic:** Shift the narrative from immediate societal application to 'essential foundational groundwork'. Argue that downstream societal impact is impossible without this specific theoretical gap being closed."
    else:
        strategy += "**Defense Tactic:** Focus the reviewers' attention on the empirical density of your dataset. Acknowledge minor structural gaps but insist the volume of data speaks for itself."
        
    return strategy

def create_radar_comparison(title1, scores1, title2, scores2):
    categories = ['C1: Originality', 'C2: Method Rigor', 'C3: Interdisciplinary', 'C4: Societal Impact', 'C5: Open Science', 'C6: Lit Integration', 'C7: Empirical Density', 'C8: Actionability']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[scores1.get(k, 0) for k in scores1], theta=categories, fill='toself', name=title1[:35] + ("..." if len(title1) > 35 else "")))
    fig.add_trace(go.Scatterpolar(r=[scores2.get(k, 0) for k in scores2], theta=categories, fill='toself', name=title2[:35] + ("..." if len(title2) > 35 else "")))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, margin=dict(l=40, r=40, t=30, b=30))
    return fig
