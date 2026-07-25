# services.py
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO

def search_arxiv(query, max_results):
    # Minimal placeholder returning a mock paper so the UI doesn't break
    return [{
        "title": f"Sample ArXiv Paper for '{query}'",
        "authors": "John Doe",
        "summary": "This is a placeholder abstract for the arxiv search.",
        "pdf_url": "https://arxiv.org/pdf/2101.00001.pdf"
    }]

def download_arxiv_pdf(url):
    return b"dummy_pdf_bytes"

def create_radar_comparison(name1, scores1, name2, scores2):
    categories = list(scores1.keys())
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=list(scores1.values()), theta=categories, fill='toself', name=name1))
    fig.add_trace(go.Scatterpolar(r=list(scores2.values()), theta=categories, fill='toself', name=name2))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True)
    return fig

def generate_latex_report(title, author, score, logic, scores_dict, eval_hash):
    return f"\\title{{{title}}}\n\\author{{{author}}}\n\\section{{Score: {score}}}"

def generate_bibtex(title, author, eval_hash):
    return f"@article{{{eval_hash[:8]},\n  title={{{title}}},\n  author={{{author}}},\n  year={{2026}}\n}}"

def export_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def export_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

def get_portfolio_stats(conn, user_id):
    cursor = conn.cursor()
    cursor.execute("SELECT final_score, logic_score, fields FROM papers_assessment WHERE user_id=?", (user_id,))
    rows = cursor.fetchall()
    if not rows:
        return {'total_papers': 0, 'avg_score': 0, 'max_score': 0, 'avg_logic': 0, 'unique_fields': 0}
    
    scores = [r[0] for r in rows if r[0]]
    logics = [r[1] for r in rows if r[1]]
    return {
        'total_papers': len(rows),
        'avg_score': sum(scores)/len(scores) if scores else 0,
        'max_score': max(scores) if scores else 0,
        'avg_logic': sum(logics)/len(logics) if logics else 0,
        'unique_fields': 5  # Placeholder for unique fields count
    }
