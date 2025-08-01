import os
from pathlib import Path
from typing import Dict, Any
import dotenv

# Load environment variables
dotenv.load_dotenv("../.local.env")

# Environment Configuration
LANGSMITH_TRACING_V2 = "true"
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
LANGSMITH_API_KEY = dotenv.get_key("../.local.env", "LANGSMITH_API_KEY")
LANGSMITH_PROJECT = "rag-sleap-docs"
GOOGLE_CLOUD_PROJECT_ID = dotenv.get_key("../.local.env", "GOOGLE_CLOUD_PROJECT_ID")

if not LANGSMITH_API_KEY:
    raise ValueError("LANGSMITH_API_KEY is not set in the environment variables.")
if not GOOGLE_CLOUD_PROJECT_ID:
    raise ValueError("GOOGLE_CLOUD_PROJECT_ID is not set in the environment variables.")

# Set environment variables
os.environ["LANGSMITH_TRACING_V2"] = LANGSMITH_TRACING_V2
os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT

# Repository Paths - Update these to match your local setup
REPO_PATHS = {
    "sleap": Path("/Users/chan/PersonalProjects/rag-sleap-docs/sleap"),
    "sleap-io": Path("/Users/chan/PersonalProjects/rag-sleap-docs/sleap-io"),
    "dreem": Path("/Users/chan/PersonalProjects/rag-sleap-docs/dreem")
}

# ChromaDB Configuration
CHROMA_DB_PATH = "../chroma_db"
COLLECTION_NAME = "sleap"

# Model Configuration
EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-2.0-flash-lite"
LLM_TEMPERATURE = 0.2

# Text Splitting Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
HEADERS_TO_SPLIT_ON = [("#", "H1"), ("##", "H2"), ("###", "H3")]

# Retrieval Configuration
RETRIEVAL_K = 6

# Prompt Template
RAG_PROMPT_TEMPLATE = """
You are a helpful AI assistant specialized in SLEAP (Social LEAP Estimates Animal Poses), SLEAP-IO, and DREEM documentation.

Chat History:
{chat_history}

Use the following context from the documentation to answer the user's question. If the answer cannot be found in the context, say "I don't have enough information in the provided documentation to answer that question."

Context:
{context}

Current Question: {question}

Instructions:
- Provide accurate, detailed answers based on the documentation
- Include code examples when relevant
- Mention specific function names, classes, or modules when applicable
- If discussing installation or setup, be specific about requirements
- For troubleshooting questions, provide step-by-step solutions
- Always cite which part of the documentation (SLEAP, SLEAP-IO, or DREEM) your answer comes from
- Reference previous conversation when relevant
- If user's question is not clear, ask for clarification
- If provided context is insufficient, inform the user politely. Never guess or provide information not in the documentation, unless user explicitly asks for general knowledge.
- Never answer question that is not related to the software SLEAP, it's documentation, or related experiments. When such request happens, you should only say "Sorry, I can only answer questions related to SLEAP, SLEAP-IO, or DREEM documentation."

Answer:
"""
