#!/bin/bash

echo ""
echo "=========================================="
echo "Inflx Agent - Installation Script"
echo "=========================================="
echo ""

echo "[1/5] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.9+"
    exit 1
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $python_version"
echo ""

echo "[2/5] Installing dependencies..."
pip3 install -q -r requirements.txt
if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi
echo ""

echo "[3/5] Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env file from template"
    echo "⚠️  Please edit .env and add your OpenAI API key"
else
    echo "✓ .env file already exists"
fi
echo ""

echo "[4/5] Validating data files..."
python3 validate_data.py
if [ $? -eq 0 ]; then
    echo "✓ Data validation passed"
else
    echo "⚠️  Data validation found issues (see above)"
fi
echo ""

echo "[5/5] Running test..."
if grep -q "OPENAI_API_KEY=sk-" .env 2>/dev/null; then
    echo "Running quick test..."
    timeout 30 python3 test_conversation.py 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✓ Test passed"
    else
        echo "⚠️  Test skipped (add API key to run tests)"
    fi
else
    echo "⚠️  API key not configured - skipping test"
fi
echo ""

echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add: OPENAI_API_KEY=sk-your-key"
echo "2. Run: python3 agent.py"
echo ""
echo "Optional:"
echo "- Test: python3 test_conversation.py"
echo "- Validate: python3 validate_data.py"
echo ""