.PHONY: help setup install-deps init-db analyze-chunks config-tuner run-webapp run-cli clean

help:  ## Show this help message
	@echo "SLEAP Documentation RAG Assistant"
	@echo "================================="
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make <target>\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

setup:  ## Set up the project (install deps, create venv)
	@echo "🚀 Setting up project..."
	@./setup.sh

install-deps:  ## Install Python dependencies
	@echo "📥 Installing dependencies..."
	@pip install -r requirements.txt

init-db:  ## Initialize the knowledge base
	@echo "🔍 Initializing knowledge base..."
	@python scripts/initialize_db.py $(if $(CLONE_REPOS),--clone-repos,)

analyze-chunks:  ## Analyze chunking strategy and show statistics
	@echo "🔍 Analyzing chunking strategy..."
	@python scripts/analyze_chunks.py

config-tuner:  ## Interactive tool to tune chunking configuration
	@echo "⚙️  Opening configuration tuner..."
	@python scripts/config_tuner.py

run-webapp:  ## Run the Streamlit webapp
	@echo "🌐 Starting webapp..."
	@streamlit run app.py

run-cli:  ## Run the CLI version
	@echo "💻 Starting CLI..."
	@python cli.py

clean:  ## Clean up generated files
	@echo "🧹 Cleaning up..."
	@rm -rf __pycache__ .pytest_cache .coverage
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
