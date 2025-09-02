import streamlit as st
from pathlib import Path
from src.retriever import retrieve

# Configuration
ROOT = Path(__file__).parent.parent

def format_search_results(contexts: list) -> list:
    """
    Format retrieved contexts for display in Streamlit UI.
    
    Args:
        contexts: List of context dictionaries from retriever
        
    Returns:
        List of formatted result dictionaries
    """
    results = []
    
    for ctx in contexts:
        result = {
            'source': ctx.get('id', 'Unknown source'),
            'content': ctx.get('content', ''),
            'score': ctx.get('score', 0.0)
        }
        results.append(result)
    
    return results

def search_documents(query: str, top_k: int = 10, alpha: float = 0.7) -> list:
    """
    Search Transformers documentation using hybrid retrieval.
    
    Args:
        query: User's search query
        top_k: Number of results to return
        alpha: Balance between dense (1.0) and sparse (0.0) search
        
    Returns:
        List of formatted search results
    """
    try:
        # Retrieve relevant contexts using the hybrid approach
        contexts = retrieve(query, top_k=top_k, alpha=alpha)
        
        # Format results for display
        results = format_search_results(contexts)
        
        # Display search info in sidebar
        st.sidebar.success(f"Retrieved {len(results)} results")
        st.sidebar.info(f"Search mode: {'Dense-focused' if alpha > 0.7 else 'Balanced' if alpha > 0.3 else 'Sparse-focused'}")
        
        return results
        
    except Exception as e:
        st.sidebar.error(f"Search error: {str(e)}")
        return []

def get_search_summary(query: str, results: list) -> str:
    """
    Generate a brief summary of search results.
    
    Args:
        query: Original search query
        results: List of search results
        
    Returns:
        Summary string
    """
    if not results:
        return f"No documentation found for: '{query}'"
    
    # Extract key topics from top results
    top_content = " ".join([r['content'][:200] for r in results[:3]])
    
    return f"Found {len(results)} relevant sections about '{query}' in Transformers documentation."
