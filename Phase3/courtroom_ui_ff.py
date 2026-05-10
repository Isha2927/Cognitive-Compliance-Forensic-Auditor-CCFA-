import streamlit as st
import asyncio
import pandas as pd

from orchestrator.courtroom_ff import run_courtroom
from visualization.graph_viz_ff import visualize_graph
from analysis.explainability_ff import explain_risk
from retrieval.retriever import retrieve_rules

import streamlit.components.v1 as components


st.set_page_config(page_title="AI AML Courtroom", layout="wide")

st.title("⚖️ AI AML Courtroom with Explainability + RAG")

# ---- Load risk scores ----
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

    with st.spinner("Retrieving AML rules + Running AI agents..."):

        # 🔥 Better query
        query = f"""
        Account risk score: {risk_score}
        Transaction connections: {len(explanations)}

        Find AML regulations related to:
        - money laundering
        - suspicious accounts
        - monitoring transactions
        """

        rules = retrieve_rules(query)

        # ---- Show retrieved rules ----
        st.subheader("📚 Retrieved AML Regulations")

        for r in rules:
            st.write(f"• {r}")

        # ---- Run courtroom ----
        messages = asyncio.run(
            run_courtroom(account_id, risk_score, rules)
        )

        st.subheader("📜 Courtroom Transcript")

        for msg in messages:
            st.markdown(f"**{msg.source}**")
            st.write(msg.content)
            st.markdown("---")