import streamlit as st
from agents.multi_agent import StatsTool, RetrieverAgent, AnalystAgent
import os

# --- Page Config ---
st.set_page_config(
    page_title="IPL AI Analyst",
    page_icon="🏏",
    layout="wide"
)

# --- Header ---
st.title("🏏 IPL AI Analyst")
st.markdown("""
**Powered by TinyLlama + LoRA | RAG Architecture**
*Ask questions about IPL 2022 Batting, Match Commentary, and more.*
""")

# --- Sidebar: System Status ---
st.sidebar.header("⚙️ System Status")
status_text = st.sidebar.empty()

# --- Initialize System (Cached to run once) ---
@st.cache_resource
def load_system():
    status_text.text("Loading Data...")
    DATA_PATH = '/content/drive/MyDrive/AI_ML_Engineer_Tech_Test_Package/AI_ML_Engineer_Tech_Test_Package 2/data/Indian_Premier_League_2022-03-26/Indian_Premier_League_2022-03-26'
    
    # Load Tools
    tool = StatsTool(DATA_PATH)
    retriever = RetrieverAgent(tool)
    
    status_text.text("Loading AI Model...")
    analyst = AnalystAgent() # This loads the LoRA adapter
    
    status_text.text("✅ System Ready")
    return tool, retriever, analyst

# Load the agents
try:
    tool, retriever, analyst = load_system()
except Exception as e:
    st.error(f"Error loading system: {e}")
    st.stop()

# --- Main Interface ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat")
    query = st.text_input("Ask a question:", placeholder="e.g., How many runs did Kohli score?")
    
    if st.button("Analyze", type="primary"):
        if query:
            with st.spinner("🤖 Searching records & thinking..."):
                # 1. Retrieve
                context = retriever.retrieve(query)
                
                # 2. Generate
                answer = analyst.generate_answer(query, context)
                
                # 3. Display
                st.success("Analysis Complete")
                st.markdown(f"### 💡 Answer:")
                st.write(answer)
                
                # Save to session state for the sidebar to see
                st.session_state['last_context'] = context
        else:
            st.warning("Please enter a question.")

with col2:
    st.subheader("🔎 Retrieval Debugger")
    st.info("This section shows what the AI 'read' before answering.")
    
    if 'last_context' in st.session_state:
        with st.expander("View Retrieved Context", expanded=True):
            st.code(st.session_state['last_context'], language="text")
    else:
        st.write("*No query run yet.*")

# --- Footer ---
st.markdown("---")
st.caption("Built with Streamlit, Transformers, and Peft.")
