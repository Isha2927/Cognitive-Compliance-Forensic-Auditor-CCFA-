import pandas as pd
import asyncio

from orchestrator.courtroomw import run_courtroom
from retrieval.regulatory_retriever import retrieve_rules

# Load risk scores
df = pd.read_csv("risk_scores.csv")

sample = df.iloc[0]

account_id = int(sample["node_id"])
risk_score = float(sample["risk_score"])

print("Account Under Investigation:", account_id)
print("Risk Score:", risk_score)

# Retrieve AML rules (Phase 1)
rules = retrieve_rules("anti money laundering suspicious account")

print("Retrieved Rules:", len(rules))

# Run AI courtroom
asyncio.run(run_courtroom(account_id, risk_score, rules))