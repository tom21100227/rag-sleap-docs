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

REMOTE_REPO_URL = {
    "sleap": "https://github.com/talmolab/sleap.git",
    "sleap-io": "https://github.com/talmolab/sleap-io.git",
    "dreem": "https://github.com/talmolab/dreem.git",
    "sleap-nn": "https://github.com/talmolab/sleap-nn.git"
}

# Repository Paths - Update these to match your local setup
REPO_PATHS = {
    "sleap": Path("/home/chan/PersonalProjects/rag-sleap-docs/sleap"),
    "sleap-io": Path("/home/chan/PersonalProjects/rag-sleap-docs/sleap-io"),
    "dreem": Path("/home/chan/PersonalProjects/rag-sleap-docs/dreem"),
    "sleap-nn": Path("/home/chan/PersonalProjects/rag-sleap-docs/sleap-nn")
}

# ChromaDB Configuration
CHROMA_DB_PATH = "../chroma_db"
COLLECTION_NAME = "sleap"

# Model Configuration
EMBEDDING_MODEL = "text-embedding-004"
QUERY_LLM_MODEL = "gemini-2.5-flash-lite"
QUERY_LLM_TEMPERATURE = 0.1
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.4

# Text Splitting Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
HEADERS_TO_SPLIT_ON = [("#", "H1"), ("##", "H2"), ("###", "H3")]

# Adaptive Chunking Configuration
ADAPTIVE_CHUNKING = {
    "target_chunk_size": 1500,      # Sweet spot for context vs size
    "max_chunk_size": 2500,         # Hard limit before forced recursive splitting  
    "min_chunk_size": 600,          # Minimum viable chunk size
    "size_threshold_multiplier": 1.3,  # When to try next header level (more aggressive)
    "header_priorities": [
        ("#", "H1"),      # Try H1 first (largest sections)
        ("##", "H2"),     # Then H2 (medium sections) 
        ("###", "H3"),    # Finally H3 (smallest sections)
        ("####", "H4"),   # Even smaller if needed
        ("#####", "H5")   # Smallest header level
    ]
}

# Retrieval Configuration
RETRIEVAL_K = 6

# Query Translation Configuration
QUERY_TRANSLATION_METHODS = {
    "None": "none",
    "MultiQuery": "multi_query", 
    "RAG Fusion": "rag_fusion",
    "Decomposition": "decomposition",
    "Step Back": "step_back"
}

# Default query translation method
DEFAULT_QUERY_METHOD = "RAG Fusion"

# HyDE Configuration (can be applied with any method)
DEFAULT_USE_HYDE = False

# Query translation prompts
MULTI_QUERY_PROMPT = """
You are an AI language model assistant. Your task is to generate 3 different versions of the given user question about SLEAP, SLEAP-IO, SLEAP-NN, DREEM documentation to retrieve relevant documents from a vector database. 

By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.

Provide these alternative questions separated by newlines.
Original question: {question}
"""

RAG_FUSION_PROMPT = """
You are a helpful assistant that generates multiple search queries based on a single input query. \n

Generate multiple search queries related to: {question}, separated by newlines. Do not include any additional text, number, or explanations, just the queries.

Output (4 queries):
"""

DECOMPOSITION_PROMPT = """
You are a helpful assistant. Break down the following complex question about SLEAP, SLEAP-IO, or DREEM into 2-4 simpler sub-questions that together would help answer the original question.

Each sub-question should be specific and answerable from technical documentation.

Original question: {question}
"""

STEP_BACK_PROMPT = """
You are an expert at asking broader, more general questions. Given a specific question about SLEAP, SLEAP-IO, or DREEM, generate a more general "step back" question that would help provide broader context for answering the specific question.

For example:
- Specific: "How do I fix CUDA out of memory errors in SLEAP training?"
- Step back: "What are the memory requirements and optimization strategies for SLEAP training?"

Original question: {question}
"""

HYDE_PROMPT = """
You are an expert in SLEAP (Social LEAP Estimates Animal Poses), SLEAP-IO, and DREEM documentation. 

Write a detailed, technical passage that would appear in official documentation to answer this question: {question}

The passage should:
- Use technical terminology commonly found in SLEAP documentation
- Include specific function names, parameters, or configuration details when relevant
- Be written in the style of technical documentation
- Focus on practical implementation details

Write the hypothetical documentation passage:
"""

# Prompt Template
RAG_PROMPT_TEMPLATE = """
You are a helpful AI assistant specialized in SLEAP (Social LEAP Estimates Animal Poses), SLEAP-IO, SLEAP-NN, and DREEM documentation.

Chat History:
{chat_history}

Use the following context from the documentation to answer the user's question. If the answer cannot be found in the context, say "I don't have enough information in the provided documentation to answer that question." If you were to make any assumptions, you need to explicitly state that you are making an assumption based on the context provided.

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
