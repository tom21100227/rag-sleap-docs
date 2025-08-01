# SLEAP Documentation RAG Assistant - Usage Guide

## Quick Start

### 1. Initial Setup
```bash
cd webapp
make setup  # or ./setup.sh
```

### 2. Configure Environment
Copy the example environment file and fill in your credentials:
```bash
cp .env.example .local.env
```

Edit `.local.env` with your:
- `LANGSMITH_API_KEY`: Your LangSmith API key (optional, for monitoring)
- `GOOGLE_CLOUD_PROJECT_ID`: Your Google Cloud project ID

### 3. Update Repository Paths
Edit `config/settings.py` and update the `REPO_PATHS` dictionary with your local paths:
```python
REPO_PATHS = {
    "sleap": Path("/your/path/to/sleap"),
    "sleap-io": Path("/your/path/to/sleap-io"), 
    "dreem": Path("/your/path/to/dreem")
}
```

### 4. Initialize Knowledge Base
```bash
make init-db  # or python scripts/initialize_db.py
```

### 5. Run the Application

**Web Interface:**
```bash
make run-webapp  # or streamlit run app.py
```

**Command Line Interface:**
```bash
make run-cli  # or python cli.py
```

## Project Structure

```
webapp/
├── app.py                 # Streamlit web interface
├── cli.py                 # Command-line interface  
├── Makefile              # Easy commands
├── setup.sh              # Setup script
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
├── config/
│   └── settings.py        # Configuration settings
├── src/
│   ├── __init__.py
│   ├── document_parser.py # Document parsing logic
│   ├── embeddings.py      # Embedding and vector store management
│   ├── rag_chain.py       # RAG pipeline implementation
│   └── memory.py          # Conversation memory management
└── scripts/
    └── initialize_db.py   # Database initialization script
```

## How It Works

1. **Document Parsing**: The system uses AST (Abstract Syntax Tree) parsing to extract docstrings from Python code and parses Markdown/RST documentation files.

2. **Smart Chunking**: Documents are intelligently split based on their type:
   - Markdown files: Split by headers (H1, H2, H3)
   - RST files: Language-aware splitting
   - Code docstrings: Keep as single chunks or split if too large

3. **Vector Storage**: Uses ChromaDB for persistent vector storage with Google's text-embedding-004 model.

4. **Conversational RAG**: Implements a Retrieval-Augmented Generation pipeline with conversation memory using Google's Gemini model.

## Configuration Options

All configuration is centralized in `config/settings.py`:

- **Model Settings**: Change LLM model, temperature, embedding model
- **Chunking Settings**: Adjust chunk size, overlap, splitting headers
- **Retrieval Settings**: Configure number of documents to retrieve
- **Repository Paths**: Set local paths to your SLEAP repositories

## Troubleshooting

### "No documents found"
- Check repository paths in `config/settings.py`
- Ensure repositories are cloned locally
- Verify permissions to read the directories

### "Failed to initialize vectorstore"
- Run the initialization script first: `make init-db`
- Check Google Cloud credentials and project ID
- Ensure ChromaDB directory is writable

### Import errors
- Activate virtual environment: `source venv/bin/activate`
- Install dependencies: `make install-deps`

## Extending the System

### Adding New Repositories
1. Add repository path to `REPO_PATHS` in `config/settings.py`
2. Re-run initialization: `make init-db`

### Customizing the Prompt
Edit `RAG_PROMPT_TEMPLATE` in `config/settings.py`

### Adding New Document Types
Extend the `DocumentParser` class in `src/document_parser.py`

## Performance Tips

- The system caches the vector store, so subsequent runs are faster
- Increase `RETRIEVAL_K` for more comprehensive answers (but slower responses)
- Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` for different granularity of information
