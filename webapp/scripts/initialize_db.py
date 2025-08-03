#!/usr/bin/env python3
"""
Script to initialize the knowledge base by parsing documents and creating embeddings.
Run this script before using the webapp.
"""

import sys
from pathlib import Path

# Add the webapp directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.document_parser import DocumentParser
from src.embeddings import AdaptiveDocumentSplitter, EmbeddingManager
from config.settings import REPO_PATHS


def main():
    """Initialize the knowledge base."""
    print("🚀 Initializing SLEAP Documentation RAG System...")
    
    # Step 1: Parse all documents
    print("\n📚 Step 1: Parsing documents from repositories...")
    parser = DocumentParser(REPO_PATHS)
    raw_documents = parser.parse_all_repositories()
    
    if not raw_documents:
        print("❌ No documents found. Please check your repository paths in config/settings.py")
        return
    
    # Step 2: Split documents into chunks
    print("\n✂️  Step 2: Splitting documents into adaptive chunks...")
    splitter = AdaptiveDocumentSplitter()
    chunks = splitter.split_documents(raw_documents)
    
    # Step 3: Create embeddings and store in vector database  
    print("\n🔍 Step 3: Creating embeddings and storing in ChromaDB...")
    embedding_manager = EmbeddingManager()
    embedding_manager.initialize_vectorstore(chunks)
    
    print("\n✅ Knowledge base initialization complete!")
    print(f"📊 Total documents processed: {len(raw_documents)}")
    print(f"📋 Total chunks created: {len(chunks)}")
    print("\n🎉 You can now run the webapp with: streamlit run app.py")


if __name__ == "__main__":
    main()
