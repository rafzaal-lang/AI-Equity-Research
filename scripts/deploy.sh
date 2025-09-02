# scripts/deploy.sh
#!/bin/bash

# Deployment script for AI Equity Research Platform

set -e

echo "🚀 Starting deployment process..."

# Check if required files exist
required_files=("main.py" "requirements.txt" "Dockerfile" "render.yaml")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file missing: $file"
        exit 1
    fi
done

echo "✅ All required files found"

# Check environment variables
if [ -z "$OPENAI_KEY" ]; then
    echo "⚠️  Warning: OPENAI_KEY not set"
fi

# Create necessary directories
mkdir -p templates static reports logs data
echo "✅ Created required directories"

# Build Docker image (if using Docker)
if [ "$1" == "docker" ]; then
    echo "🐳 Building Docker image..."
    docker build -t ai-equity-research:latest .
    echo "✅ Docker image built successfully"
fi

# Run tests (if they exist)
if [ -d "tests" ]; then
    echo "🧪 Running tests..."
    python -m pytest tests/ -v
    echo "✅ Tests passed"
fi

# Deploy to Render (if render CLI is available)
if command -v render &> /dev/null; then
    echo "☁️  Deploying to Render..."
    render deploy
    echo "✅ Deployed to Render successfully"
fi

echo "🎉 Deployment process completed!"
