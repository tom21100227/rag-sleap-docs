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
from config.settings import REPO_PATHS, REMOTE_REPO_URL
import argparse
import subprocess

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Initialize the knowledge base.")
    parser.add_argument(
        '--clone-repos', action='store_true',
        help='Clone remote repositories before initializing the database'
    )
    return parser.parse_args()


def main():
    """Initialize the knowledge base."""
    args = parse_args()
    if args.clone_repos:
        print("\n🔄 Cloning remote repositories…")
        for name, url in REMOTE_REPO_URL.items():
            path = REPO_PATHS[name]
            if not path.exists():
                print(f"→ Cloning {name} into {path}…")
                subprocess.run(["git", "clone", url, str(path)], check=True)
            else:
                print(f"→ {name} already exists at {path}, skipping.")
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
