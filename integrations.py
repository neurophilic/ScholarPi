import re
import json
import requests
import cloudscraper
import fitz  
import networkx as nx

def clean_author_name(author_str):
    if not author_str:
        return "Unidentified"
    try:
        if author_str.startswith("[") and author_str.endswith("]"):
            parsed = json.loads(author_str.replace("'", '"'))
            if isinstance(parsed, list):
                return ", ".join([str(a).strip() for a in parsed if str(a).strip()])
    except:
        pass
    cleaned = (
        author_str.replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
    )
    return cleaned.strip()

def is_likely_institution(name):
    if not name:
        return True
    lower_name = name.lower()
    inst_keywords = [
        "university", "univ.", "college", "institute", "inst.",
        "department", "dept.", "laboratory", "lab", "hospital",
        "center", "centre", "faculty", "milano", "bicocca",
        "polytechnic", "academy", "school", "corporation", "inc",
        "llc", "ltd", "foundation", "fund", "council", "cnr",
        "inps", "iss", "università",
    ]
    for kw in inst_keywords:
        pattern = r'\b' + re.escape(kw.rstrip('.')) + r'\b'
        if re.search(pattern, lower_name):
            return True
    return False

def fetch_core_text_by_doi(doi):
    if not doi or doi == "None":
        return None
    
    clean_doi = doi.replace("https://doi.org/", "").strip()
    url = f"https://api.core.ac.uk/v3/search/works?q=doi:{clean_doi}"
    headers = {"User-Agent": "Pi-Index-Engine/1.0"}
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            results = data.get("results", [])
            if results:
                paper_data = results[0]
                text = paper_data.get("fullText") or paper_data.get("abstract")
                if text and len(text.strip()) > 200:
                    return text
    except Exception as e:
        print(f"CORE API lookup warning: {e}")
        
    return None

def create_virtual_pdf_from_text(text, title="CORE Open Access Text"):
    try:
        doc = fitz.open()  
        page = doc.new_page()
        rect = fitz.Rect(50, 50, 550, 800)
        page.insert_textbox(rect, f"Title: {title}\n\n{text}", fontsize=10, fontname="helv")
        
        pdf_bytes = doc.write()
        doc.close()
        return pdf_bytes
    except Exception as e:
        print(f"Virtual PDF generation error: {e}")
        
    return None

def fetch_author_coara_metrics(author_name):
    try:
        clean_name = clean_author_name(author_name)
        if (
            not clean_name
            or clean_name.lower() in ["unidentified", "unknown"]
            or is_likely_institution(clean_name)
        ):
            return 0.0, 0, "Data/Software Curation"
        first_author = clean_name.split(",")[0].strip()
        url = f"https://api.openalex.org/authors?search={first_author}"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get("results") and len(data["results"]) > 0:
                author_obj = data["results"][0]
                works_count = author_obj.get("works_count", 0)
                return (
                    float(works_count),
                    int(author_obj.get("cited_by_count", 0)),
                    "Open Access & Dataset Curation",
                )
    except Exception:
        pass
    return 0.0, 0, "Methodology & Validation"

def search_openalex_topics(topic_query, limit=100):
    try:
        url = f"https://api.openalex.org/works?search={requests.utils.quote(topic_query)}&filter=is_oa:true&per_page={limit}"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            results = res.json().get("results", [])
            extracted = []
            for item in results:
                title = item.get("title", "Untitled Paper")
                doi = item.get("doi", "")

                best_oa = item.get("best_oa_location") or {}
                pdf_url = best_oa.get("pdf_url") or item.get("open_access", {}).get("oa_url", "")

                authorships = item.get("authorships", [])
                authors_list = [
                    a.get("author", {}).get("display_name", "") for a in authorships
                ]
                authors_str = (
                    ", ".join([a for a in authors_list if a])
                    if authors_list
                    else "Unidentified"
                )

                if pdf_url or doi:
                    extracted.append({
                        "title": title,
                        "doi": doi,
                        "pdf_url": pdf_url,
                        "authors": authors_str,
                    })
            return extracted
    except Exception as e:
        print(f"OpenAlex Topic Fetch Error: {str(e)}")
    return []

def fetch_doi_metadata(doi):
    clean_doi = (
        doi.replace("https://doi.org/", "").replace("doi.org/", "").strip()
    )
    unpaywall_url = (
        f"https://api.unpaywall.org/v2/{clean_doi}?email=research@pi-index.org"
    )
    try:
        response = requests.get(unpaywall_url, timeout=10)
        if response.status_code == 200:
            res = response.json()
            title = res.get("title", "Unknown Title")
            authors_list = res.get("z_authors", [])
            authors = (
                ", ".join([a.get("family", "") for a in authors_list])
                if authors_list
                else "Unknown Author"
            )
            pdf_url = (
                res.get("best_oa_location", {}).get("url_for_pdf", None)
                if res.get("best_oa_location")
                else None
            )
            return {"title": title, "authors": authors, "pdf_url": pdf_url}
        return None
    except Exception:
        return None

def fetch_semantic_scholar_pdf(title_or_doi):
    if not title_or_doi:
        return None
    try:
        clean_query = title_or_doi.replace("https://doi.org/", "").strip()
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(clean_query)}&limit=1&fields=openAccessPdf,externalIds"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json().get("data", [])
            if data:
                oa_pdf = data[0].get("openAccessPdf")
                if oa_pdf and oa_pdf.get("url"):
                    return oa_pdf["url"]
    except Exception:
        pass
    return None

def download_pdf_from_url(pdf_url):
    if not pdf_url:
        return None

    if "arxiv.org/abs/" in pdf_url:
        pdf_url = pdf_url.replace("/abs/", "/pdf/") + ".pdf"
    elif (
        "ncbi.nlm.nih.gov/pmc/articles/PMC" in pdf_url
        and not pdf_url.endswith(".pdf")
    ):
        parts = pdf_url.split("PMC")
        if len(parts) > 1:
            pmc_id = parts[1].split("/")[0]
            pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf/"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
            " like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/pdf;q=0.9,image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://scholar.google.com/",
        "Connection": "keep-alive",
    }

    try:
        session = requests.Session()
        res = session.get(pdf_url, headers=headers, timeout=15, allow_redirects=True)
        content_type = res.headers.get("Content-Type", "").lower()
        if res.status_code == 200 and (
            b"%PDF" in res.content[:10] or "application/pdf" in content_type
        ):
            return res.content
    except Exception:
        pass

    try:
        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "desktop": True}
        )
        res = scraper.get(pdf_url, timeout=20, allow_redirects=True)
        content_type = res.headers.get("Content-Type", "").lower()
        if res.status_code == 200 and (
            b"%PDF" in res.content[:10] or "application/pdf" in content_type
        ):
            return res.content
    except Exception:
        pass

    return None

def calculate_citation_topology(doi: str) -> float:
    if not doi or doi == "None":
        return 0.50

    clean_doi = doi.replace("https://doi.org/", "").strip()
    url = f"https://api.openalex.org/works/https://doi.org/{clean_doi}"
    
    try:
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return 0.50
            
        data = res.json()
        referenced_works = data.get("referenced_works", [])
        
        if len(referenced_works) < 5:
            return 0.30
            
        G = nx.DiGraph()
        G.add_node(clean_doi)
        
        for ref in referenced_works:
            G.add_edge(clean_doi, ref)
            
        centrality = nx.degree_centrality(G)
        max_centrality = max(centrality.values()) if centrality else 1.0
        topological_entropy = max(0.1, 1.0 - (max_centrality / 2.0))
        
        return min(1.0, topological_entropy)
        
    except Exception as e:
        print(f"Topology mapping failed: {e}")
        return 0.50
