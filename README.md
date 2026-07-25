
# Pi-Index Assessment Engine (CoARA-Compliant)

An automated, decentralized peer-review and research integrity framework powered by large language models (Groq Llama 3.3/3.1), SciScore reproducibility metrics, IPFS storage, and Ethereum Sepolia smart contract consensus. The engine evaluates academic preprints and published papers against an 8-criteria multidimensional rubric aligned with the Agreement on Reforming Research Assessment (CoARA) and DORA principles.

---

## Key Features

*   **Multi-Source Ingestion Engine:** Upload local binary PDFs, resolve publications dynamically via Unpaywall (DOI), or harvest open-access preprints using custom OpenAlex topic searches.
*   **CoARA & DORA-Aligned Rubric (C1-C8):** Evaluates manuscripts across 8 transparent criteria:
    *   *C1:* Semantic Originality (penalized by generative AI laundering heuristics)
    *   *C2:* Methodological Rigor & MDAR Adherence (SciScore standards)
    *   *C3:* Interdisciplinary Entropy (citation network Shannon entropy)
    *   *C4:* Societal & Open Infrastructure Impact
    *   *C5:* Open Science & Executable Reproducibility Potential
    *   *C6:* Literature Integration & Citation Polarity
    *   *C7:* Empirical Density & Statistical Cohort Strength
    *   *C8:* Future Actionability & FAIR Principles
*   **Adversarial Logic & Integrity Matrix:** Automatically detects synthetic hallucination, prompt-stuffing preprint floods, and semantic-empirical divergence.
*   **Decentralized State Persistence:** Automatically syncs application state and local SQLite databases via IPFS (Pinata) and records reference pointers on an Ethereum Sepolia smart contract registry.
*   **Proof-of-Research (PoR) Consensus:** Generates immutable block hashes, dynamic epoch weights, and Zero-Knowledge proofs for every evaluation round.
*   **Pi-Brain LSTM Meta-Learning:** A PyTorch-powered recurrent neural network (`PiBrainLSTM`) that trains on historical blockchain weights to forecast future shifts in evaluation criteria standards.
*   **Autonomous GitHub Actions Runner:** Executes headless background runs every 5 minutes to query hot scientific topics, assess batches of preprints, and seal results to the blockchain ledger.

---

## System Architecture

1.  **Intake & Identity:** Authenticate securely via ORCID iD / W3C DID. Ingest files from local storage, DOIs, or OpenAlex.
2.  **Evaluation Pipeline:** Parse text using chunked LLM ensembles, compute adversarial logic scores, and apply CoARA-compliant weights.
3.  **Blockchain Consensus:** Mint Soulbound Tokens (`piQ`), record proof-of-research signatures (`eval_hash`), and anchor storage CIDs to Ethereum Sepolia.
4.  **Outputs & Cartography:** Generate markdown integrity dossiers, AI defense rebuttal strategies, and interactive PyVis network maps of global science.

---

## Local Installation & Setup

### 1. Prerequisites
*   Python 3.11+
*   pip & git

### 2. Clone the Repository
bash
git clone [https://github.com/your-username/ScholarPi.git](https://github.com/your-username/ScholarPi.git)
cd ScholarPi
3. Install Dependencies
Bash
python -m pip install --upgrade pip
pip install requests cloudscraper PyMuPDF pandas numpy plotly pyvis torch web3 groq streamlit
4. Configure Environment Variables
Create a .env file in the root directory or configure your Streamlit Secrets (.streamlit/secrets.toml):

Code snippet
GROQ_API_KEY="your-groq-api-key"
PINATA_API_KEY="your-pinata-api-key"
PINATA_SECRET_API_KEY="your-pinata-secret-api-key"
ETH_ADMIN_PRIVATE_KEY="your-evm-private-key"
REGISTRY_CONTRACT_ADDRESS="0xYourRegistryContractAddress"
PIQ_CONTRACT_ADDRESS="0xYourPiQTokenContractAddress"
WEB3_PROVIDER_URI="[https://ethereum-sepolia-rpc.publicnode.com](https://ethereum-sepolia-rpc.publicnode.com)"
Running the Application
Launch the Streamlit user interface locally:

Bash
streamlit run app.py
Automated Background Runner (GitHub Actions)
The repository includes a headless background worker (auto_assessment.py) triggered via GitHub Actions on a 5-minute cron schedule or via manual workflow dispatch.

Configuring Repository Secrets
To allow GitHub Actions to pin state backups to IPFS and update the Ethereum registry, add the following secrets under your GitHub repository settings (Settings > Secrets and variables > Actions):

GROQ_API_KEY

PINATA_API_KEY

PINATA_SECRET_API_KEY

ETH_ADMIN_PRIVATE_KEY

REGISTRY_CONTRACT_ADDRESS

PIQ_CONTRACT_ADDRESS

License
Distributed under the MIT License. See LICENSE for more information.

Author
Ali Vafadar Yengejeh

Università degli Studi di Milano-Bicocca
