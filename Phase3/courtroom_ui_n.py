import streamlit as st
import asyncio
import pandas as pd

from orchestrator.courtroom_s import run_courtroom
from retrieval.regulatory_retriever import retrieve_rules

st.set_page_config(page_title="AI AML Courtroom", layout="wide")

st.title("🏛️ AI AML Compliance Courtroom")

st.write("Simulate regulatory debate between RBI and EU compliance officers.")

# Load risk score
df = pd.read_csv("risk_scores.csv")
sample = df.iloc[0]

account_id = int(sample["node_id"])
risk_score = float(sample["risk_score"])

st.subheader("Account Information")

st.write("Account ID:", account_id)
st.write("Risk Score:", risk_score)

if st.button("Start AI Courtroom Investigation"):

    st.subheader("Retrieving AML Regulations...")

    rules = retrieve_rules("money laundering regulations")

    st.success(f"{len(rules)} regulatory rules retrieved")

    st.subheader("Courtroom Debate")

    transcript = asyncio.run(run_courtroom(account_id, risk_score, rules))

    transcript = asyncio.run(run_courtroom(account_id, risk_score, rules))

    if transcript:
        for msg in transcript:

            if "RBI" in msg["speaker"]:
                st.markdown("🟠 **RBI Officer**")
            elif "EU" in msg["speaker"]:
                st.markdown("🔵 **EU Officer**")
            elif "Judge" in msg["speaker"]:
                st.markdown("⚖️ **Compliance Judge**")
            else:
                st.markdown("👤 **User Input**")

            st.write(msg["content"])
            st.divider()
    else:
        st.error("Courtroom failed to produce transcript")