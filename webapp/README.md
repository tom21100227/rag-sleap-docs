# SLEAP Documentation RAG Bot

A conversational AI assistant specialized in SLEAP (Social LEAP Estimates Animal Poses), SLEAP-IO, and DREEM documentation.

## Features

- Intelligent document parsing and indexing of SLEAP ecosystem documentation
- Smart embedding with AST-based code analysis
- Conversational interface with memory
- ChromaDB vector storage
- Google Vertex AI integration

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
