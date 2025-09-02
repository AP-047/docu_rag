import streamlit as st
from src.generator import search_documents

st.set_page_config(
    page_title="GenAI RAG — Transformers Documentation Assistant",
    page_icon="📚",
    layout="wide"
)

# Sidebar settings
st.sidebar.header("Search Settings")
top_k = st.sidebar.slider("Number of contexts to retrieve", 1, 15, 10, key="top_k")
alpha = st.sidebar.slider("Dense vs Sparse balance (α)", 0.0, 1.0, 0.7, 
                         help="0.0 = pure BM25, 1.0 = pure FAISS", key="alpha")

st.title("📚 GenAI RAG — Transformers Documentation Assistant")
st.markdown("*Semantic search over Hugging Face Transformers documentation using BM25 + FAISS + Cross-Encoder reranking*")

# Main query input
query = st.text_input(
    "Ask a question about Hugging Face Transformers:",
    placeholder="e.g., How do I perform text classification using Transformers?",
    key="query_input"
)

if st.button("🔍 Search Documentation", key="search_button") and query:
    with st.spinner("Searching documentation..."):
        results = search_documents(query, top_k=top_k, alpha=alpha)
    
    if results:
        st.success(f"Found {len(results)} relevant documentation sections")
        
        # Display results
        for i, result in enumerate(results, 1):
            with st.expander(f"📄 Result {i}: {result['source']}", expanded=(i <= 3)):
                st.markdown("**Relevance Score:** {:.4f}".format(result.get('score', 0.0)))
                st.markdown("**Content:**")
                st.markdown(result['content'])
                
                # Add styling
                st.markdown("---")
    else:
        st.warning("No relevant documentation found. Try rephrasing your query.")

# Footer
st.markdown("---")
st.markdown(
    """
    ✅ Fully offline • ✅ CPU-friendly • ✅ Hybrid retrieval • ✅ Real-time search
    """
)