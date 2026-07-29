import streamlit as st

# Must be the first Streamlit command
st.set_page_config(page_title="Pi-Index Assessment Engine", layout="wide")

from shared_ui import setup_global_state_and_sidebar

# Initialize session states, authentication, and the persistent sidebar
setup_global_state_and_sidebar()

# Define the multipage routing architecture
pg = st.navigation([
    st.Page("pages/1_Intake_Engine.py", title="Manuscript Intake", icon="📄"),
    st.Page("pages/2_Pidyne_Brain.py", title="Pidyne Brain & AI Jury", icon="🧠"),
    st.Page("pages/3_Analytics.py", title="Analytics & Map", icon="🌐"),
    st.Page("pages/4_Explorer.py", title="Blockchain Explorer", icon="⛓️"),
])

# Execute the active page
pg.run()
