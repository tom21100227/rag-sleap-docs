#!/usr/bin/env python3
"""
Command-line interface for the SLEAP Documentation Assistant.
Alternative to the Streamlit webapp for quick testing.
"""

import sys
from pathlib import Path

# Add the webapp directory to the Python path
sys.path.append(str(Path(__file__).parent))

from src.embeddings import EmbeddingManager
from src.memory import ConversationMemory
from src.rag_chain import RAGChain
from config.settings import RETRIEVAL_K


def main():
    """Run the CLI version of the SLEAP assistant."""
    print("🐍 SLEAP Documentation Assistant - CLI Mode")
    print("=" * 50)
    
    # Initialize the RAG system
    print("Initializing system...")
    embedding_manager = EmbeddingManager()
    
    try:
        embedding_manager.initialize_vectorstore([])  # Load existing vectorstore
        retriever = embedding_manager.get_retriever(k=RETRIEVAL_K)
        memory = ConversationMemory()
        rag_chain = RAGChain(retriever, memory)
        print("✅ System initialized successfully!\n")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("Please run 'python scripts/initialize_db.py' first.")
        return
    
    print("You can now ask questions about SLEAP, SLEAP-IO, and DREEM.")
    print("Type 'quit' or 'exit' to stop, 'clear' to clear conversation history.\n")
    
    while True:
        try:
            question = input("🤔 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            if question.lower() == 'clear':
                memory.clear_memory()
                print("🧹 Conversation history cleared!\n")
                continue
            
            if not question:
                continue
            
            print("\n🤖 Assistant: ", end="")
            response = rag_chain.chat_with_memory(question)
            print(response)
            print("\n" + "─" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
