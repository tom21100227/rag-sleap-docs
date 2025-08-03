# SLEAP Documentation RAG Bot

A conversational AI assistant specialized in SLEAP (Social LEAP Estimates Animal Poses), SLEAP-IO, and DREEM documentation.

## Features

- **Intelligent document parsing** with AST-based code analysis
- **Adaptive chunking strategy** that creates larger, more contextual chunks
- **Smart header-level selection** based on content size (tries H1 first, then H2, H3, etc.)
- **Automatic chunk combining** for small related sections
- **Conversational interface** with memory using Google Vertex AI
- **ChromaDB vector storage** for efficient similarity search
- **Multiple interfaces**: Web UI (Streamlit) and CLI

## Adaptive Chunking Strategy

The system uses an intelligent chunking approach:

- **Target chunk size**: 2000 characters (vs. traditional 1000)
- **Maximum chunk size**: 4000 characters before forced splitting
- **Header-level adaptation**: Analyzes content to choose optimal split points
  - Large documents: Split by H1 headers first
  - Medium documents: Split by H2 headers
  - Small documents: Split by H3 headers or keep intact
- **Smart combining**: Merges small adjacent sections from the same file
- **Semantic preservation**: Keeps code docstrings and coherent sections intact

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.local.env` file in the root directory with:
```
LANGSMITH_API_KEY=your_langsmith_key
GOOGLE_CLOUD_PROJECT_ID=your_project_id
```

3. Ensure you have the SLEAP repositories cloned:
   - `/path/to/sleap`
   - `/path/to/sleap-io`
   - `/path/to/dreem`

4. Update the paths in `config/settings.py`

## Usage

### Initialize the knowledge base:
```bash
python scripts/initialize_db.py
```

### Analyze chunking strategy (optional):
```bash
make analyze-chunks  # Shows chunk size distribution and samples
```

### Run the webapp:
```bash
streamlit run app.py
```

## Project Structure

```
webapp/
├── app.py                 # Streamlit web interface
├── config/
│   └── settings.py        # Configuration settings
├── src/
│   ├── __init__.py
│   ├── document_parser.py # Document parsing logic
│   ├── embeddings.py      # Embedding and vector store management
│   ├── rag_chain.py       # RAG pipeline implementation
│   └── memory.py          # Conversation memory management
├── scripts/
│   └── initialize_db.py   # Database initialization script
└── requirements.txt
```
