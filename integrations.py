import re, json, math, requests, cloudscraper, fitz, networkx as nx

def clean_author_name(author_str):
    if not author_str: return "Unidentified"
    try:
        if author_str.startswith("[") and author_str.endswith("]"):
            parsed = json.loads(author_str.replace("'", '"'))
            if isinstance(parsed, list): return ", ".join([str(a).strip() for a in parsed if str(a).strip()])
    except Exception: pass
    return author_str.replace("[", "").replace("]", "").replace("'", "").replace('"', "").strip()

def is_likely_institution(name):
    if not name: return True
    inst_keywords = ["university", "college", "institute", "department", "laboratory", "hospital", "center", "milano", "bicocca", "polytechnic", "academy", "corporation", "foundation"]
    for kw in inst_keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', name.lower()): return True
    return False

def fetch_core_text_by_doi(doi):
    clean_doi = doi.replace("https://doi.org/", "").strip()
    try:
        res = requests.get(f"https://api.core.ac.uk/v3/search/works?q=doi:{clean_doi}", headers={"User-Agent": "Pi-Index-Engine/1.0"}, timeout=10)
        if res.status_code == 200:
            results = res.json().get("results", [])
            if results:
                text = results[0].get("fullText") or results[0].get("abstract")
                if text and len(text.strip()) > 200: return text
    except: pass
    return None

def create_virtual_pdf_from_text(text, title="CORE Open Access Text"):
    try:
        doc = fitz.open()
        rect = fitz.Rect(50, 50, 550, 800)
        remaining = f"Title: {title}\n\n{text}"
        for _ in range(200):
            page = doc.new_page()
            leftover = page.insert_textbox(rect, remaining, fontsize=10, fontname="helv")
            if leftover >= 0: break
            chars_per_line = max(1, int(rect.width / (10 * 0.5)))
            lines_per_page = max(1, int(rect.height / (10 * 1.2)))
            fitted_chars = chars_per_line * lines_per_page
            if fitted_chars <= 0: break
            remaining = remaining[fitted_chars:]
        pdf_bytes = doc.write()
        doc.close()
        return pdf_bytes
    except: return None

def fetch_doi_metadata(doi):
    clean_doi = doi.replace("https://doi.org/", "").replace("doi.org/", "").strip()
    try:
        res = requests.get(f"https://api.unpaywall.org/v2/{clean_doi}?email=research@pi-index.org", timeout=10)
        if res.status_code == 200:
            data = res.json()
            authors = ", ".join([a.get("family", "") for a in data.get("z_authors", [])]) if data.get("z_authors") else "Unknown Author"
            pdf_url = data.get("best_oa_location", {}).get("url_for_pdf") if data.get("best_oa_location") else None
            return {"title": data.get("title", "Unknown Title"), "authors": authors, "pdf_url": pdf_url}
    except: pass
    return None

def fetch_semantic_scholar_pdf(doi):
    clean_query = doi.replace("https://doi.org/", "").strip()
    try:
        res = requests.get(f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(clean_query)}&limit=1&fields=openAccessPdf", timeout=10)
        if res.status_code == 200 and res.json().get("data"):
            oa_pdf = res.json()["data"][0].get("openAccessPdf")
            if oa_pdf and oa_pdf.get("url"): return oa_pdf["url"]
    except: pass
    return None

def download_pdf_from_url(pdf_url):
    if not pdf_url: return None
    if "arxiv.org/abs/" in pdf_url: pdf_url = pdf_url.replace("/abs/", "/pdf/") + ".pdf"
    try:
        res = requests.get(pdf_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15, allow_redirects=True)
        if res.status_code == 200 and (b"%PDF" in res.content[:10] or "application/pdf" in res.headers.get("Content-Type", "").lower()):
            return res.content
    except: pass
    try:
        scraper = cloudscraper.create_scraper()
        res = scraper.get(pdf_url, timeout=20, allow_redirects=True)
        if res.status_code == 200 and (b"%PDF" in res.content[:10] or "application/pdf" in res.headers.get("Content-Type", "").lower()):
            return res.content
    except: pass
    return None

def calculate_citation_topology(doi: str) -> float:
    clean_doi = doi.replace("https://doi.org/", "").strip() if doi and doi != "None" else None
    if not clean_doi: return 0.50
    try:
        res = requests.get(f"https://api.openalex.org/works/https://doi.org/{clean_doi}", timeout=10)
        if res.status_code != 200: return 0.50
        scores = [c.get("score", 0.0) for c in res.json().get("concepts", []) if c.get("score", 0.0) > 0]
        if len(scores) < 2: return 0.35
        probs = [s / sum(scores) for s in scores]
        max_entropy = math.log(len(probs))
        return max(0.1, min(1.0, (-sum(p * math.log(p) for p in probs) / max_entropy) if max_entropy > 0 else 0.0))
    except: return 0.50
