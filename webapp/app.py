import streamlit as st
import sys
from pathlib import Path
import time

# Add the webapp directory to the Python path
sys.path.append(str(Path(__file__).parent))

from src.embeddings import EmbeddingManager
from src.rag_chain import RAGChain
from config.settings import RETRIEVAL_K, QUERY_TRANSLATION_METHODS, DEFAULT_QUERY_METHOD, DEFAULT_USE_HYDE


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
    rag_chain = RAGChain(retriever)
    
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
        return
    
    # Initialize session state for tracking if we've had a conversation
    if "has_chatted" not in st.session_state:
        st.session_state.has_chatted = False
    
    # Sidebar for controls
    with st.sidebar:
        st.header("💬 Conversation Controls")
        
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.has_chatted = False  # Reset chat state
            st.success("Conversation cleared!")
            st.rerun()
        
        st.markdown("---")
        
        # RAG Controls
        st.subheader("🔧 Query Settings")

        # Filter out unimplemented methods
        available_methods = {k: v for k, v in QUERY_TRANSLATION_METHODS.items() 
                           if k not in ["Step Back", "Decomposition"]}

        query_method = st.selectbox(
            "Query Method:",
            options=list(available_methods.keys()),
            index=list(available_methods.keys()).index(DEFAULT_QUERY_METHOD),
            help="Choose how to enhance your query for better retrieval"
        )
        
        # Disable HyDE for now
        use_hyde = st.checkbox(
            "Use HyDE",
            value=False,
            disabled=True,
            help="Generate hypothetical documents to improve semantic matching (Currently disabled)"
        )
        
        # Auto-bypass RAG after first prompt (default behavior)
        bypass_rag_after_first = st.checkbox(
            "Bypass RAG after first prompt",
            value=True,  # Default to True for auto-bypass behavior
            help="Automatically switch to conversation mode after the first question"
        )
        
        # Determine if we should actually bypass RAG for this prompt
        should_bypass_rag = bypass_rag_after_first and st.session_state.has_chatted
        
        # Show current configuration
        if not should_bypass_rag and (query_method != "None" or use_hyde):
            config_text = f"🔍 Using: {query_method}"
            if use_hyde:
                config_text += " + HyDE"
            st.info(config_text)
        elif should_bypass_rag:
            st.info("💬 Conversation mode")
        elif not st.session_state.has_chatted:
            st.info("🔍 RAG mode (first question)")
        else:
            st.info("🔍 RAG mode")
            
        st.markdown("---")
        
        # Add document display settings
        st.subheader("📄 Document Display")
        show_retrieved_docs = st.checkbox("Show retrieved documents", value=True)
        max_content_length = st.slider("Content preview length", 100, 1000, 300, 50)
        
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "retriever":
            # Display retriever results
            with st.chat_message("assistant", avatar="🔍"):
                st.markdown("**Retriever**: I found the following relevant documents:")
                
                retrieved_docs = message.get("retrieved_docs", [])
                generated_queries = message.get("generated_queries")
                
                # Show generated queries if available
                if generated_queries and len(generated_queries) > 1:
                    with st.expander("🔄 Generated Queries", expanded=False):
                        for i, query in enumerate(generated_queries, 1):
                            st.markdown(f"{i}. {query}")
                
                if retrieved_docs:
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
                                        key=f"doc_content_{i}_{message['timestamp']}",
                                        label_visibility="collapsed"
                                    )
                                    
                                    if i < len(retrieved_docs):
                                        st.divider()
        else:
            # Display regular user/assistant messages
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
            # Prepare chat history (exclude the current message we just added)
            chat_history = st.session_state.messages[:-1] if st.session_state.messages else []
            
            with st.spinner("Retrieving documents..."):
                if should_bypass_rag:
                    result = rag_chain.chat_without_rag(prompt, chat_history=chat_history)
                else:
                    # Use the filtered available methods
                    available_methods = {k: v for k, v in QUERY_TRANSLATION_METHODS.items() 
                                       if k not in ["Step Back", "Decomposition"]}
                    
                    result = rag_chain.chat_with_memory(
                        prompt,
                        chat_history=chat_history,
                        query_method=available_methods[query_method],
                        use_hyde=use_hyde
                    )
            
            # Extract metadata and stream
            retrieved_docs = result["retrieved_docs"]
            generated_queries = result.get("generated_queries")
            response_stream = result["response_stream"]
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            # Create a simple generator for the error message
            def error_stream():
                yield error_msg
            response_stream = error_stream()
            retrieved_docs = []
            generated_queries = None
        
        # Mark that we've had a chat (this will auto-check bypass RAG on next rerun)
        st.session_state.has_chatted = True
        
        # Save retrieved documents to chat history for persistence
        if not should_bypass_rag and retrieved_docs:
            retriever_message = {
                "role": "retriever",
                "retrieved_docs": retrieved_docs,
                "generated_queries": generated_queries,
                "timestamp": int(time.time() * 1000)  # Unique timestamp for keys
            }
            st.session_state.messages.append(retriever_message)
            
            # Display the retriever results immediately (will also appear in history)
            with st.chat_message("assistant", avatar="🔍"):
                st.markdown("**Retriever**: I found the following relevant documents:")
                
                # Show generated queries if available
                if generated_queries and len(generated_queries) > 1:
                    with st.expander("🔄 Generated Queries", expanded=False):
                        for i, query in enumerate(generated_queries, 1):
                            st.markdown(f"{i}. {query}")
                
                if show_retrieved_docs:
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
                                        key=f"doc_content_{i}_{retriever_message['timestamp']}",
                                        label_visibility="collapsed"
                                    )
                                    
                                    if i < len(retrieved_docs):
                                        st.divider()
        
        # Show assistant response AFTER retriever with streaming
        with st.chat_message("assistant"):
            # Stream the response and accumulate it
            response = st.write_stream(response_stream)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
