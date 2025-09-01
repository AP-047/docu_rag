import streamlit as st
from src._06_generator import generate_answer

# Page config
st.set_page_config(page_title="GenAI RAG Docs", layout="wide")

st.title("📖 GenAI RAG — Transformers Documentation Assistant")

# Sidebar: parameters
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of contexts (top_k)", 1, 10, 5)
    alpha = st.slider("Dense vs Sparse (α)", 0.0, 1.0, 0.5)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7)
    max_tokens = st.number_input("Max tokens", 16, 512, 256)

# Expose parameters to generator
from src._06_generator import TEMPERATURE, MAX_TOKENS
# Override defaults
import src._06_generator as gen_mod
gen_mod.TEMPERATURE = temperature
gen_mod.MAX_TOKENS = max_tokens

# User query
query = st.text_input("Ask a question about Hugging Face Transformers docs:")

if st.button("Generate Answer") and query:
    with st.spinner("Retrieving contexts and generating answer…"):
        answer = generate_answer(query, top_k=top_k, alpha=alpha)
    st.subheader("Answer")
    st.markdown(answer)

    st.subheader("Retrieved Contexts")
    # Optionally show raw contexts
    contexts = gen_mod.retrieve(query, top_k=top_k, alpha=alpha)
    for ctx in contexts:
        st.markdown(f"**[{ctx['id']}]** — {ctx['source_file']}")
        st.write(ctx['content'][:300] + "…")