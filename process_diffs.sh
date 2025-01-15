#!/usr/bin/env bash

# This script processes each .diff file individually using files-to-prompt with CXML output.
# It maintains the exact directory structure and naming convention of the input files,
# adding .xml extension to the output files.

set -euo pipefail

# Base directories for input and output
DIFFS_DIR="diffs"
OUTPUT_BASE_DIR="ai_docs/gitdiffs"

# Create the output base directory if it doesn't exist
mkdir -p "$OUTPUT_BASE_DIR"

# Function to process a single diff file
process_diff_file() {
    local diff_file="$1"

    # Create the output path by:
    # 1. Taking the input path relative to DIFFS_DIR
    # 2. Prepending the output base directory
    # 3. Appending .xml extension
    local rel_path="${diff_file#$DIFFS_DIR/}"
    local output_file="$OUTPUT_BASE_DIR/$rel_path.xml"
    local output_dir="$(dirname "$output_file")"

    # Create the output directory structure if it doesn't exist
    mkdir -p "$output_dir"

    echo "Processing: $diff_file"
    echo "Output to: $output_file"

    # Run files-to-prompt with CXML output format
    if ! uv run files-to-prompt "$diff_file" --cxml -o "$output_file"; then
        echo "Error processing $diff_file" >&2
        return 1
    fi
}

# Find all .diff files and process them
# Using -print0 and read -d '' to handle filenames with spaces or special characters
find "$DIFFS_DIR" -type f -name "*.diff" -print0 | while IFS= read -r -d '' diff_file; do
    process_diff_file "$diff_file"
done

echo "Processing complete. All diff files have been converted to XML format."
echo "Check $OUTPUT_BASE_DIR for the output files."

# Make the script executable
chmod +x "$0"
