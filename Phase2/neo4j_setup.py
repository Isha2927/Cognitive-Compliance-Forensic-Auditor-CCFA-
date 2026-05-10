from neo4j import GraphDatabase
import pandas as pd
from tqdm import tqdm

driver = GraphDatabase.driver(
    "neo4j://127.0.0.1:7687",
    auth=("neo4j", "password")
)

df = pd.read_csv("datasets/transactions.csv")

BATCH_SIZE = 1000

def insert_batch(tx, rows):

    tx.run("""
    UNWIND $rows AS row

    MERGE (a:Account {id: row.sender})
    MERGE (b:Account {id: row.receiver})

    CREATE (a)-[:TRANSACTION {
        tx_id: row.txid,
        type: row.txtype,
        amount: row.amount,
        timestamp: row.time,
        isFraud: row.fraud,
        alertID: row.alert
    }]->(b)
    """, rows=rows)


with driver.session() as session:

    batch = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Importing Transactions"):

        batch.append({
            "sender": row["SENDER_ACCOUNT_ID"],
            "receiver": row["RECEIVER_ACCOUNT_ID"],
            "txid": row["TX_ID"],
            "txtype": row["TX_TYPE"],
            "amount": float(row["TX_AMOUNT"]),
            "time": row["TIMESTAMP"],
            "fraud": int(row["IS_FRAUD"]),
            "alert": row["ALERT_ID"]
        })

        if len(batch) == BATCH_SIZE:
            session.execute_write(insert_batch, batch)
            batch = []

    if batch:
        session.execute_write(insert_batch, batch)

driver.close()

print("Data import completed")