#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$SCRIPT_DIR"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Setup .env
if [ ! -f ".env" ] && [ -f "$PROJECT_ROOT/.env.example" ]; then
    cp "$PROJECT_ROOT/.env.example" .env
    echo "Created .env from template. Edit it with your API key."
elif [ -f "$PROJECT_ROOT/.env" ] && [ ! -f ".env" ]; then
    cp "$PROJECT_ROOT/.env" .env
    echo "Copied .env from project root."
fi

echo ""
echo "Setup complete! Run with:"
echo "  cd $SCRIPT_DIR"
echo "  source .venv/bin/activate"
echo "  python main.py"
