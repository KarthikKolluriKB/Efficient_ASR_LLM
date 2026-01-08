#!/bin/bash
# =============================================================================
# Download Common Voice Scripted Speech English from Mozilla Data Collective
# =============================================================================
# This downloads the smaller "Scripted Speech" subset (~smaller than full CV)
# 
# Prerequisites:
#   1. Create account at: https://datacollective.mozillafoundation.org
#   2. Get your API key from account settings
#
# Usage: 
#   bash scripts/download_cv_english.sh YOUR_API_KEY
# =============================================================================

set -e

# Configuration
OUTPUT_DIR="data"
DATASET_ID="cmj8u3p1w0075nxxbe8bedl00"
ARCHIVE_NAME="cv-scripted-en.tar.gz"

echo "============================================================"
echo "Download Common Voice Scripted Speech - English"
echo "============================================================"

# Check arguments
if [ -z "$1" ]; then
    echo ""
    echo "Usage: bash scripts/download_cv_english.sh API_KEY"
    echo ""
    echo "To get your API key:"
    echo "  1. Go to: https://datacollective.mozillafoundation.org"
    echo "  2. Create an account / Sign in"
    echo "  3. Go to Account Settings -> API Keys -> Create new key"
    echo "  4. Copy the API key and run this script"
    echo ""
    exit 1
fi

API_KEY="$1"

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# Step 1: Get download token
echo ""
echo "[Step 1/4] Getting download token..."
echo ""

RESPONSE=$(curl -s -X POST \
    "https://datacollective.mozillafoundation.org/api/datasets/${DATASET_ID}/download" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "Content-Type: application/json")

echo "Response: $RESPONSE"

# Extract download token from response (assuming JSON response with token field)
DOWNLOAD_TOKEN=$(echo "$RESPONSE" | grep -o '"downloadToken":"[^"]*"' | cut -d'"' -f4)

if [ -z "$DOWNLOAD_TOKEN" ]; then
    # Try alternative parsing
    DOWNLOAD_TOKEN=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('downloadToken', json.load(sys.stdin).get('token', '')))" 2>/dev/null || echo "")
fi

if [ -z "$DOWNLOAD_TOKEN" ]; then
    echo ""
    echo "ERROR: Could not extract download token from response"
    echo "Response was: $RESPONSE"
    echo ""
    echo "Please manually extract the token and run:"
    echo "  curl -X GET 'https://datacollective.mozillafoundation.org/api/datasets/${DATASET_ID}/download/YOUR_TOKEN' \\"
    echo "    -H 'Authorization: Bearer ${API_KEY}' -o '$ARCHIVE_NAME'"
    exit 1
fi

echo "Got download token: ${DOWNLOAD_TOKEN:0:20}..."

# Step 2: Download
echo ""
echo "[Step 2/4] Downloading Common Voice Scripted Speech English..."
echo ""

curl -X GET \
    "https://datacollective.mozillafoundation.org/api/datasets/${DATASET_ID}/download/${DOWNLOAD_TOKEN}" \
    -H "Authorization: Bearer ${API_KEY}" \
    -o "$ARCHIVE_NAME" \
    --progress-bar

# Check if download succeeded
if [ ! -f "$ARCHIVE_NAME" ] || [ ! -s "$ARCHIVE_NAME" ]; then
    echo "ERROR: Download failed. Check your API key."
    exit 1
fi

echo ""
echo "Download complete: $(ls -lh $ARCHIVE_NAME | awk '{print $5}')"

# Step 3: Extract
echo ""
echo "[Step 3/4] Extracting archive..."
echo ""
tar -xzf "$ARCHIVE_NAME"

# Step 4: Find extracted folder
echo ""
echo "[Step 4/4] Finding extracted folder..."

# List what was extracted
ls -la

EXTRACTED_DIR=$(find . -maxdepth 2 -type d -name "en" 2>/dev/null | head -1)
if [ -z "$EXTRACTED_DIR" ]; then
    EXTRACTED_DIR=$(find . -maxdepth 2 -type d -name "*english*" -o -name "*English*" 2>/dev/null | head -1)
fi

if [ -n "$EXTRACTED_DIR" ]; then
    echo "Found English data at: $EXTRACTED_DIR"
    
    # Check for TSV files
    if [ -f "$EXTRACTED_DIR/train.tsv" ]; then
        TRAIN_COUNT=$(wc -l < "$EXTRACTED_DIR/train.tsv")
        echo "Train samples: $TRAIN_COUNT"
    fi
else
    echo "Please check extracted contents and update path in:"
    echo "  scripts/download_cv_english.py"
fi

# Cleanup option
echo ""
read -p "Delete archive file to save space? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f "$ARCHIVE_NAME"
    echo "Archive deleted."
fi

echo ""
echo "============================================================"
echo "Download complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Check extracted folder structure"
echo "  2. Update CV_CORPUS_DIR in scripts/download_cv_english.py"
echo "  3. Run: python scripts/download_cv_english.py"
echo "  4. Train: python train.py --config configs/train_config_english_test.yaml"
echo ""

