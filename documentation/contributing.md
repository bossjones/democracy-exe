# Contributing

To contribute to this project, first checkout the code. Then follow these steps to set up your development environment:

## Development Setup

1. Install required system dependencies:
```bash
brew install taplo libmagic poppler tesseract pandoc qpdf tesseract-lang
brew install --cask libreoffice
```

2. Create a new virtual environment and install dependencies:
```bash
# Install dependencies using UV
uv sync
uv tool upgrade pyright
uv run pre-commit install
```

## Code Quality Tools

We use several tools to maintain code quality:

1. **Formatting and Linting**:
```bash
# Format all code
just fmt

# Run linters
just lint

# Run all code quality checks
just check
```

2. **Type Checking**:
```bash
# Run type checks
just typecheck
```

3. **Pre-commit Hooks**:
```bash
# Run all pre-commit hooks
just pre-commit-run-all
```

## Running Tests

We use pytest for testing. To run tests:

```bash
# Run all tests with coverage
just test

# Run tests in debug mode
just test-debug

# Run specific test file
uv run pytest -s --verbose --showlocals --tb=short path/to/test_file.py
```

### Test Coverage

After running tests, view the coverage report:
```bash
just open-coverage
```

## Documentation

Documentation uses [MkDocs](https://www.mkdocs.org/). To work with documentation:

```bash
# Serve documentation locally
just docs

# Build documentation
just docs_build

# Deploy documentation to GitHub Pages
just docs_deploy
```

## Development Workflow

1. Create a new branch for your feature/fix
2. Make your changes
3. Run code quality checks:
   ```bash
   just check
   just test
   ```
4. Format your code:
   ```bash
   just fmt
   ```
5. Commit your changes using commitizen:
   ```bash
   just commit
   ```
6. Push your changes and create a pull request

## Project Structure

```
democracy-exe/
├── democracy_exe/           # Main package directory
│   ├── agentic/            # Agentic system components
│   ├── ai/                 # AI/ML components
│   ├── bot_logger/         # Logging components
│   ├── chatbot/            # Discord chatbot components
│   └── utils/              # Utility functions
├── tests/                  # Test directory
├── docs/                   # Documentation
└── scripts/                # Utility scripts
```

## Additional Tools

- **Dependency Management**:
  ```bash
  # Upgrade dependencies
  just uv-upgrade-all

  # Sync dependencies
  just sync
  ```

- **Code Generation**:
  ```bash
  # Generate type stubs
  just createstubs
  ```

## Notes

- All Python files must include type annotations and docstrings following PEP 257
- Tests must use pytest (not unittest)
- Use `aiofiles` instead of built-in `open()` for async file operations
- Follow the project's import sorting standards using Ruff's isort rules
