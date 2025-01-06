#!/bin/bash

# Export base requirements
uv export --no-hashes --prune langserve --prune notebook --format requirements-txt --extra dev  -o democracy_exe/requirements.txt

command -v yq >/dev/null 2>&1 || { echo >&2 "yq is required but it's not installed. run 'brew install yq' or 'pip install yq'"; exit 1; }

command -v langgraph >/dev/null 2>&1 || { echo >&2 "langgraph is required but it's not installed. run 'brew install langgraph-cli' or 'pip install langgraph-cli'"; exit 1; }

# Get all dependencies from pyproject.toml and process them
yq -r ".project.dependencies[]" pyproject.toml | while read -r dep; do
    pkg=$(echo "$dep" | gsed -E "s/([^>=<~!]+).*/\1/")
    ver=$(echo "$dep" | gsed -E "s/[^>=<~!]+(.+)/\1/")
    gsed -i "s|^$pkg==.*|$pkg$ver|g" democracy_exe/requirements.txt
done


# Get all dev-dependencies and process them
yq -r ".[\"tool.uv\"][\"dev-dependencies\"][]" pyproject.toml | while read -r dep; do
    pkg=$(echo "$dep" | gsed -E "s/([^>=<~!]+).*/\1/")
    ver=$(echo "$dep" | gsed -E "s/[^>=<~!]+(.+)/\1/")
    gsed -i "s|^$pkg==.*|$pkg$ver|g" democracy_exe/requirements.txt
done

# Remove exact version requirement for sse-starlette
gsed -i 's/^sse-starlette==.*/sse-starlette/g' democracy_exe/requirements.txt

gsed -i 's/^langserve\[all\]==.*/langserve\[all\]/g' democracy_exe/requirements.txt
gsed -i 's/^langserve==.*/langserve/g' democracy_exe/requirements.txt
gsed -i 's/^tenacity==.*/tenacity/g' democracy_exe/requirements.txt

git diff democracy_exe/requirements.txt
