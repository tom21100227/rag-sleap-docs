import streamlit as st
import sys
from pathlib import Path

# Add the webapp directory to the Python path
sys.path.append(str(Path(__file__).parent))

from src.embeddings import EmbeddingManager
from src.memory import ConversationMemory
from src.rag_chain import RAGChain
from config.settings import RETRIEVAL_K


@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system with caching."""
    embedding_manager = EmbeddingManager()
    
    # Initialize vectorstore (will load existing if available)
    try:
        embedding_manager.initialize_vectorstore([])  # Empty list since we're loading existing
    except Exception as e:
        st.error(f"Failed to initialize vectorstore: {e}")
        st.error("Please run 'python scripts/initialize_db.py' first to set up the knowledge base.")
        st.stop()
    
    retriever = embedding_manager.get_retriever(k=RETRIEVAL_K)
    memory = ConversationMemory()
    rag_chain = RAGChain(retriever, memory)
    
    return rag_chain


def main():
    st.set_page_config(
        page_title="SLEAP Documentation Assistant",
        page_icon="🐍",
        layout="wide"
    )
    
    st.title("🐍 SLEAP Documentation Assistant")
    st.markdown("Ask questions about SLEAP, SLEAP-IO, and DREEM documentation!")
    
    # Initialize the RAG system
    try:
        rag_chain = initialize_rag_system()
    except Exception as e:
        st.error(f"Failed to initialize the system: {e}")
        st.stop()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("💬 Conversation Controls")
        
        if st.button("Clear Conversation"):
            rag_chain.memory.clear_memory()
            st.session_state.messages = []
            st.success("Conversation cleared!")
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        This assistant can help you with (hopefully):
        - SLEAP installation and setup
        - API documentation and usage
        - SLEAP-IO functionality
        - DREEM integration
        - Code examples and troubleshooting
        """)
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about SLEAP..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                with st.spinner("Thinking..."):
                    response = rag_chain.chat_with_memory(prompt)
                message_placeholder.markdown(response)
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                message_placeholder.markdown(error_msg)
                response = error_msg
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    if len(st.session_state.messages) == 0:
        # Example questions
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("How do I install SLEAP?", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "How do I install SLEAP?"})
                st.rerun()
        
        with col2:
            if st.button("What are the GPU requirements?", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "What are the GPU requirements?"})
                st.rerun()
        
        with col3:
            if st.button("How do I use DREEM with SLEAP?", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "How do I use DREEM with SLEAP predictions?"})
                st.rerun()


if __name__ == "__main__":
    main()
