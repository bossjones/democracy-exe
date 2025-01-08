# Prompt Generator

A tool for generating prompts by merging template variables from multiple sources.

## Installation

1. Make sure you have Python 3.7+ installed
2. Install UV: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Install dependencies:
```bash
uv pip install pyyaml loguru
```

## Usage

The script takes a template file, a YAML config file, and optionally a target file to process:

```bash
python generate_prompt.py template.txt config.yaml [target_file] [-o output.txt]
```

### Arguments

- `template.txt`: Template file with variables in Python format string style (e.g., `{variable_name}`)
- `config.yaml`: YAML file containing variable definitions
- `target_file`: Optional file to process (contents will be available as `{file_contents}`)
- `-o/--output`: Optional output file (defaults to stdout)

### Example

1. Create a template file (`template.txt`):
```markdown
# Task: Convert Python Codebase from {old_logger} to {new_logger}

## Current Codebase Summary
{codebase_summary}

## File Contents
```python
{file_contents}
```
```

2. Create a config file (`config.yaml`):
```yaml
old_logger: loguru
new_logger: structlog
codebase_summary: "A Python codebase using loguru for logging"
```

3. Run the script:
```bash
python generate_prompt.py template.txt config.yaml sample.py -o output.txt
```

## Features

- Merges variables from YAML config
- Loads file contents using UV
- Supports Python format string style templates
- Error handling and logging
- Optional output file

## Error Handling

The script will exit with a non-zero status code and log an error if:

- Template file is not found
- Config file is not found or invalid YAML
- Target file is not found
- Required variables are missing from config
- UV command fails
