#!/bin/bash

echo "🚀 Setting up SLEAP Documentation RAG Webapp..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if .local.env exists
if [ ! -f ".local.env" ]; then
    echo "⚠️  .local.env file not found."
    echo "📋 Please copy .env.example to .local.env and fill in your credentials:"
    echo "   cp .env.example .local.env"
    echo "   # Then edit .local.env with your API keys"
    echo ""
fi

echo "✅ Setup complete!"
echo ""
echo "🔧 Next steps:"
echo "1. Configure your .local.env file with API keys"
echo "2. Run: python scripts/initialize_db.py"
echo "3. Start the webapp: streamlit run app.py"
echo "   Or use CLI mode: python cli.py"
