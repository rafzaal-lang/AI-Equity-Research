# scripts/setup.sh
#!/bin/bash

# Setup script for AI Equity Research Platform

set -e

echo "🔧 Setting up AI Equity Research Platform..."

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2)
required_version="3.11"

if [[ "$python_version" < "$required_version" ]]; then
    echo "❌ Python 3.11+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip
pip install --upgrade pip
echo "✅ Pip upgraded"

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Create directories
mkdir -p templates static reports logs data
echo "✅ Created required directories"

# Copy template files if they don't exist
if [ ! -f ".env" ]; then
    if [ -f ".env.template" ]; then
        cp .env.template .env
        echo "✅ Created .env from template"
        echo "⚠️  Please edit .env and add your API keys"
    else
        echo "⚠️  Please create .env file with your API keys"
    fi
fi

# Download NLTK data (for sentiment analysis)
python3 -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    print('✅ NLTK data downloaded')
except:
    print('⚠️  Could not download NLTK data')
"

echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Copy HTML content to templates/index.html"
echo "3. Copy JavaScript content to static/app.js"
echo "4. Run: python main.py"
echo ""
echo "🚀 Ready to launch!"