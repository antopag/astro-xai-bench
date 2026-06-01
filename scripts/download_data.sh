#!/bin/bash
# Download and preprocess PLAsTiCC dataset
# Requires: pip install kaggle && ~/.kaggle/kaggle.json configured
# Usage: bash scripts/download_data.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$PROJECT_ROOT/data/raw/plasticc"
PROCESSED_DIR="$PROJECT_ROOT/data/processed/plasticc"

mkdir -p "$RAW_DIR"

echo "=== Step 1: Downloading PLAsTiCC from Kaggle ==="
kaggle competitions download -c PLAsTiCC-2018 -p "$RAW_DIR"

echo "=== Step 2: Extracting ==="
cd "$RAW_DIR"
for f in *.zip; do
    [ -f "$f" ] && unzip -o "$f" && rm -f "$f"
done

echo "=== Step 3: Preprocessing ==="
cd "$PROJECT_ROOT"
python -c "
from src.data.plasticc import preprocess_plasticc
preprocess_plasticc('$RAW_DIR', '$PROCESSED_DIR')
"

echo "=== Done ==="
echo "Raw data: $RAW_DIR"
echo "Processed data: $PROCESSED_DIR"
