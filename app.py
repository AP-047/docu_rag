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
        
        for i, result in enumerate(results, 1):
            # Build GitHub URL
            raw = result['source']
            file_path = raw.rsplit("_", 1)[0].replace("\\", "/")
            gh_url = f"https://github.com/huggingface/transformers/blob/main/docs/{file_path}"

            # Use a simple title so link clicks inside the body don't toggle
            with st.expander(f"📄 Result {i}", expanded=(i <= 3)):
                # Render the clickable link inside the expander
                st.markdown(f"[{file_path}]({gh_url}){{:target=\"_blank\"}}", unsafe_allow_html=True)
                st.markdown("**Relevance Score:** {:.4f}".format(result.get('score', 0.0)))
                st.markdown("**Content:**")
                st.markdown(result['content'])
                st.markdown("---")
    else:
        st.warning("No relevant documentation found. Try rephrasing your query.")


# Sample questions
st.markdown(
    """
    <div style="color:gray; font-size:0.9rem; margin-top:0.5rem; font-weight:300;">
      You Can Ask:
      <ul style="margin-top:0.25rem; padding-left:1.2rem; line-height:1.4;">
        <li>How do I load a tokenizer?</li>
        <li>How do I perform text classification using Transformers?</li>
        <li>How do I use a model for question answering?</li>
        <li>How do I fine-tune a pre-trained model on my dataset?</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True
)