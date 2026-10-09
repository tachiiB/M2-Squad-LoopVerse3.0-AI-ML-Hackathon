"""Smog Assistant Demo - interactive CLI.

Type a question, get an answer with sources. Type 'quit' to exit.
Covers: forecasts, health advice, policy, Roman Urdu.
Try: 'Should schools close in DHA tomorrow?' / 'DHA mein kal hawa kaisi hogi?'
"""

import glob
from vector_db import VectorDB
from ask import ask as ask_fn, set_retriever

print("Loading vector database...")
vdb = VectorDB(glob.glob("DOC-*.md"))
set_retriever(lambda q: vdb.search(q, top_k=3))
print(f"Ready [{vdb.vectors.shape[0]} chunks, {vdb.vectors.shape[1]}-dim vectors].")
print("Ask anything (or 'quit').\n")

i = 1
while True:
    try:
        q = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if q.lower() in ("quit", "exit", "q"):
        break
    if not q:
        continue
    r = ask_fn(q, f"demo-{i}")
    i += 1
    print(f"\nAssistant: {r['answer']}")
    print(f"Sources: {r['sources']} | Forecast used: {r['forecast_called']}\n")
