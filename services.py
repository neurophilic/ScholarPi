import re
import urllib.parse
import xml.etree.ElementTree as ET
import requests
import plotly.graph_objects as go

# --- FEATURE 1: ArXiv Research Discovery ---
def search_arxiv(query, max_results=5):
    """Search ArXiv for papers and return metadata + direct PDF links."""
    encoded_query = urllib.parse.quote(query)
    url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}"
    
    try:
        response = requests.get(url, timeout=10)
        root = ET.fromstring(response.content)
        results = []
        
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip().replace('\n', ' ')
            authors = [a.find('{http://www.w3.org/2005/Atom}name').text for a in entry.findall('{http://www.w3.org/2005/Atom}author')]
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip().replace('\n', ' ')
            
            pdf_url = ""
            for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
                if link.attrib.get('title') == 'pdf':
                    pdf_url = link.attrib.get('href')
                    
            results.append({
                "title": title,
                "authors": ", ".join(authors),
                "summary": summary,
                "pdf_url": pdf_url
            })
        return results
    except Exception as e:
        return []

def download_arxiv_pdf(pdf_url):
    """Fetch PDF bytes directly from ArXiv URL for pipeline assessment."""
    try:
        res = requests.get(pdf_url, timeout=15)
        if res.status_code == 200:
            return res.content
        return None
    except Exception:
        return None


# --- FEATURE 3: Comparison Radar Plot ---
def create_radar_comparison(title1, scores1, title2, scores2):
    """Generate a Plotly Spider/Radar Chart comparing two evaluated papers across C1-C8."""
    categories = [
        'C1: Originality', 'C2: Method Rigor', 'C3: Interdisciplinary', 
        'C4: Societal Impact', 'C5: Open Science', 'C6: Lit Integration', 
        'C7: Empirical Density', 'C8: Actionability'
    ]
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[scores1.get(k, 0) for k in scores1],
        theta=categories,
        fill='toself',
        name=title1[:35] + ("..." if len(title1) > 35 else "")
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[scores2.get(k, 0) for k in scores2],
        theta=categories,
        fill='toself',
        name=title2[:35] + ("..." if len(title2) > 35 else "")
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        margin=dict(l=40, r=40, t=30, b=30)
    )
    return fig


# --- FEATURE 4: Advanced Reporting & Exporting ---
def generate_latex_report(title, author, final_score, logic_score, scores_dict, eval_hash):
    """Generate a complete downloadable LaTeX article detailing the assessment."""
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
C1 & Originality & {scores_dict.get('c1', 0.0):.2f} \\
C2 & Methodological Rigor & {scores_dict.get('c2', 0.0):.2f} \\
C3 & Interdisciplinary Capacity & {scores_dict.get('c3', 0.0):.2f} \\
C4 & Societal Impact & {scores_dict.get('c4', 0.0):.2f} \\
C5 & Open Science Potential & {scores_dict.get('c5', 0.0):.2f} \\
C6 & Literature Integration & {scores_dict.get('c6', 0.0):.2f} \\
C7 & Empirical Density & {scores_dict.get('c7', 0.0):.2f} \\
C8 & Future Actionability & {scores_dict.get('c8', 0.0):.2f} \\
\bottomrule
\end{tabular}
\caption{{\pi-Index Multidimensional Assessment}}
\end{{table}}

\end{{document}}"""

def generate_bibtex(title, author, eval_hash):
    """Generate BibTeX citation string for the evaluated manuscript."""
    clean_title = re.sub(r'[^a-zA-Z0-9]', '', title)[:20]
    first_author = author.split()[0] if author else "Unknown"
    cite_key = f"{first_author}{clean_title}2026"
    
    return f"""@article{{{cite_key},
  title = {{{title}}},
  author = {{{author}}},
  note = {{Evaluated via \\pi-Index Engine, Evaluation Hash: {eval_hash}}},
  year = {{2026}}
}}"""
