import streamlit as st
import asyncio
import pandas as pd

from orchestrator.courtroom_f import run_courtroom
from visualization.graph_viz import visualize_graph
from analysis.explainability import explain_risk

import streamlit.components.v1 as components


st.set_page_config(page_title="AI AML Courtroom", layout="wide")

st.title("⚖️ AI AML Courtroom with Explainability")

# Load risk scores
risk_df = pd.read_csv("risk_scores.csv")

account_id = st.selectbox(
    "Select Account",
    risk_df["node_id"].tolist()
)

risk_score = float(
    risk_df[risk_df["node_id"] == account_id]["risk_score"].values[0]
)

st.metric("Risk Score", risk_score)

# ---- Explainability ----
st.subheader("🧠 Why is this account risky?")

explanations = explain_risk(account_id)

for exp in explanations:
    st.write(f"• {exp}")

# ---- Graph ----
st.subheader("🔗 Transaction Network")

graph_file = visualize_graph(account_id)

with open(graph_file, 'r', encoding='utf-8') as f:
    components.html(f.read(), height=500)

# ---- Run Courtroom ----
if st.button("Run AML Courtroom"):

    with st.spinner("Running AI agents..."):

        rules = "regulatory_vectors.faiss"  # replace with FAISS later


        messages = asyncio.run(
            run_courtroom(account_id, risk_score, rules)
        )

        st.subheader("📜 Courtroom Transcript")

        for msg in messages:
            st.markdown(f"**{msg.source}**")
            st.write(msg.content)
            st.markdown("---")