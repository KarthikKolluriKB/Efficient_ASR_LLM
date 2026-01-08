#!/bin/bash
# =============================================================================
# Download Common Voice English Dataset from Mozilla
# =============================================================================
# Downloads directly from Common Voice data collective
# 
# Usage: 
#   bash scripts/download_cv_english.sh 'YOUR_DOWNLOAD_URL'
#
# Get URL from: https://commonvoice.mozilla.org/en/datasets
# =============================================================================

set -e

# Configuration
OUTPUT_DIR="data"
ARCHIVE_NAME="cv-corpus-en.tar.gz"

echo "============================================================"
echo "Download Common Voice English Dataset"
echo "============================================================"

# Check if URL provided
if [ -z "$1" ]; then
    echo ""
    echo "Usage: bash scripts/download_cv_english.sh 'DOWNLOAD_URL'"
    echo ""
    echo "To get the download URL:"
    echo "  1. Go to: https://commonvoice.mozilla.org/en/datasets"
    echo "  2. Select English language"
    echo "  3. Click 'Download Dataset Bundle'"
    echo "  4. Accept the terms"
    echo "  5. Right-click download button -> Copy link address"
    echo "  6. Run this script with the URL"
    echo ""
    exit 1
fi

DOWNLOAD_URL="$1"

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# Step 1: Download
echo ""
echo "[Step 1/3] Downloading Common Voice English..."
echo "This may take a while (~20-50GB)..."
echo ""
wget -O "$ARCHIVE_NAME" "$DOWNLOAD_URL" --show-progress

# Step 2: Extract
echo ""
echo "[Step 2/3] Extracting archive..."
echo "This may take a while..."
echo ""
tar -xzf "$ARCHIVE_NAME"

# Step 3: Find extracted folder
echo ""
echo "[Step 3/3] Finding extracted folder..."
EXTRACTED_DIR=$(ls -d cv-corpus-*/en 2>/dev/null | head -1)

if [ -z "$EXTRACTED_DIR" ]; then
    echo "WARNING: Could not find extracted English folder"
    echo "Please check the extracted contents and update the path in:"
    echo "  scripts/download_cv_english.py"
else
    echo "Found: $EXTRACTED_DIR"
    
    # Count files
    TRAIN_COUNT=$(wc -l < "$EXTRACTED_DIR/train.tsv" 2>/dev/null || echo "0")
    echo "Train samples: $TRAIN_COUNT"
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
echo "  1. Update CV_CORPUS_DIR in scripts/download_cv_english.py:"
echo "     CV_CORPUS_DIR = Path(\"$OUTPUT_DIR/$EXTRACTED_DIR\")"
echo ""
echo "  2. Process the data:"
echo "     python scripts/download_cv_english.py"
echo ""
echo "  3. Train:"
echo "     python train.py --config configs/train_config_english_test.yaml"
echo ""

