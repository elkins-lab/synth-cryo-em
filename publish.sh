#!/bin/bash
# publish.sh - Build and upload synth-cryo-em to PyPI

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting PyPI publication process...${NC}"

# 1. Clean previous builds
echo "Cleaning old build artifacts..."
rm -rf dist/ build/ *.egg-info

# 2. Ensure build tools are present
echo "Ensuring build and twine are installed..."
python3 -m pip install --upgrade build twine

# 3. Build the package
echo "Building the package..."
python3 -m build

# 4. Check the distribution
echo "Checking the distribution with twine..."
python3 -m twine check dist/*

# 5. Upload to PyPI
echo -e "${RED}Ready to upload to PyPI.${NC}"
read -p "Do you want to upload to PyPI now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 -m twine upload dist/*
    echo -e "${GREEN}Successfully uploaded to PyPI!${NC}"
else
    echo "Upload cancelled."
fi
