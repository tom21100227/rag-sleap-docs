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
        
        # Add document display settings
        st.subheader("📄 Document Display")
        show_retrieved_docs = st.checkbox("Show retrieved documents", value=True)
        max_content_length = st.slider("Content preview length", 100, 1000, 300, 50)
        
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
        
        # Generate response and get retrieved documents
        try:
            with st.spinner("Thinking..."):
                result = rag_chain.chat_with_memory(prompt)
                response = result["response"]
                retrieved_docs = result["retrieved_docs"]
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            response = error_msg
            retrieved_docs = []
        
        # Show retrieved documents as a separate "Retriever" entity FIRST
        if show_retrieved_docs and retrieved_docs:
            with st.chat_message("assistant", avatar="🔍"):
                st.markdown("**Retriever**: I found the following relevant documents:")
                
                with st.expander(f"📚 Retrieved Documents ({len(retrieved_docs)} found)", expanded=False):
                    # Add tabs for different views
                    tab1, tab2 = st.tabs(["📋 Summary", "📖 Details"])
                    
                    with tab1:
                        # Summary view
                        sources = {}
                        for doc in retrieved_docs:
                            source = doc.metadata.get('source', 'Unknown')
                            sources[source] = sources.get(source, 0) + 1
                        
                        st.markdown("**Sources Retrieved:**")
                        for source, count in sources.items():
                            st.markdown(f"- {source}: {count} document(s)")
                    
                    with tab2:
                        # Detailed view
                        for i, doc in enumerate(retrieved_docs, 1):
                            with st.container():
                                st.markdown(f"### Document {i}")
                                
                                # Metadata in columns
                                col1, col2, col3 = st.columns([1, 1, 1])
                                with col1:
                                    st.markdown(f"**Source:** `{doc.metadata.get('source', 'Unknown')}`")
                                with col2:
                                    st.markdown(f"**File:** `{doc.metadata.get('file', 'Unknown')}`")
                                with col3:
                                    if 'object' in doc.metadata:
                                        st.markdown(f"**Object:** `{doc.metadata['object']}`")
                                
                                # Content
                                content_preview = doc.page_content[:max_content_length]
                                if len(doc.page_content) > max_content_length:
                                    content_preview += "..."
                                
                                st.markdown("**Content:**")
                                st.text_area(
                                    f"Document {i} Content",
                                    content_preview,
                                    height=150,
                                    key=f"doc_content_{i}_{len(st.session_state.messages)}",
                                    label_visibility="collapsed"
                                )
                                
                                if i < len(retrieved_docs):
                                    st.divider()
        
        # Show assistant response AFTER retriever
        with st.chat_message("assistant"):
            st.markdown(response)
        
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
