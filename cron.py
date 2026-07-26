import time
import random
import sys
import requests
from config import HOT_TOPICS
from ledger import restore_state_from_web3, backup_state_to_web3
from integrations import download_pdf_from_url
from brain import process_single_pdf

def run_headless_cron():
    print("Starting Background Paper Assessment Cron...")
    restore_state_from_web3()
    
    topic = random.choice(HOT_TOPICS)
    print(f"Selected Hot Topic: '{topic}'")
    
    url = f"https://api.openalex.org/works?search={requests.utils.quote(topic)}&filter=is_oa:true&per_page=15"
    papers = []
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            results = res.json().get("results", [])
            for item in results:
                pdf_url = (item.get("best_oa_location") or {}).get("pdf_url") or item.get("open_access", {}).get("oa_url", "")
                if pdf_url:
                    papers.append({
                        "title": item.get("title", "Untitled Paper"),
                        "doi": item.get("doi", ""),
                        "pdf_url": pdf_url,
                        "topic": topic
                    })
    except Exception as e:
        print(f"OpenAlex fetch warning: {e}")

    processed_count = 0
    for p in papers:
        print(f"Attempting download for: {p['title']}")
        pdf_bytes = download_pdf_from_url(p["pdf_url"])
        if pdf_bytes:
            process_single_pdf(
                pdf_bytes, 
                f"Auto_{time.time()}.pdf", 
                p["topic"], 
                "GitHub_Actions_Bot", 
                provided_doi=p["doi"]
            )
            processed_count += 1
            if processed_count >= 5:
                break
                
    if processed_count > 0:
        backup_state_to_web3()
        print("Background assessment cycle completed and backed up to Web3.")
    else:
        print("No new papers were processed in this run.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cron":
        run_headless_cron()
        sys.exit(0)
    else:
        print("Usage: python cron.py --cron")