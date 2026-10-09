"""Lahore Smog Assistant - Streamlit UI.

Run:  pip install streamlit
      streamlit run app.py
"""

import streamlit as st
import glob
from vector_db import VectorDB
from ask import ask as ask_fn, set_retriever


@st.cache_resource
def load_system():
    vdb = VectorDB(glob.glob("DOC-*.md"))
    set_retriever(lambda q: vdb.search(q, top_k=3))
    return vdb


st.set_page_config(page_title="Lahore Smog Assistant", page_icon="🌫️")
st.title("🌫️ Lahore Smog Intelligence Assistant")
st.caption("Forecast + grounded advisory | 15 areas | Oct 29 – Nov 7")

vdb = load_system()
st.success(f"Vector DB ready: {vdb.vectors.shape[0]} chunks, "
           f"{vdb.vectors.shape[1]}-dim dense vectors")

q = st.text_input("Ask a question:",
                  placeholder="Should schools close in DHA tomorrow?")
if st.button("Ask") and q.strip():
    with st.spinner("Searching documents + forecast..."):
        r = ask_fn(q.strip(), "ui-1")
    st.subheader("Answer")
    st.write(r["answer"])
    st.caption(f"Sources: {r['sources']} | Forecast called: {r['forecast_called']}")

with st.expander("Try these"):
    st.write("- What is the PM2.5 forecast for Gulberg on 2026-11-03?")
    st.write("- Is it safe to exercise outdoors during smog?")
    st.write("- When do schools close due to smog?")
    st.write("- How is the air in Karachi tomorrow?")
    st.write("- DHA mein kal hawa kaisi hogi?")
