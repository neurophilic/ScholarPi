import streamlit as st
from database import get_db_connection

st.set_page_config(page_title="Pi-Index Engine", layout="wide", page_icon="🧠", initial_sidebar_state="expanded")

# Initialize DB & Queues on Boot
get_db_connection().close()

pg = st.navigation({
    "ScholarPi Modules": [
        st.Page("pages/1_intake.py", title="1. Intake & Ingestion", icon="📄"),
        st.Page("pages/2_brain_eval.py", title="2. Pidyne Brain & Jury", icon="🧠"),
        st.Page("pages/3_analytics.py", title="3. Analytics & Cartography", icon="🌐"),
        st.Page("pages/4_blockchain.py", title="4. Blockchain Explorer", icon="⛓️")
    ]
})
pg.run()
