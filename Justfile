set shell := ["zsh", "-cu"]

# just manual: https://github.com/casey/just/#readme

# Ignore the .env file that is only used by the web service
set dotenv-load := false

CURRENT_DIR := "$(pwd)"

base64_cmd := if "{{os()}}" == "macos" { "base64 -w 0 -i cert.pem -o ca.pem" } else { "base64 -w 0 -i cert.pem > ca.pem" }
grep_cmd := if "{{os()}}" =~ "macos" { "ggrep" } else { "grep" }

# Variables
PYTHON := "uv run python"

# Recipes
# Install the virtual environment and install the pre-commit hooks
install:
    @echo "🚀 Creating virtual environment using uv"
    uv sync
    uv run pre-commit install

# Run code quality tools.
check:
    @echo "🚀 Checking lock file consistency with 'pyproject.toml'"
    uv lock --locked
    @echo "🚀 Linting code: Running pre-commit"
    uv run pre-commit run -a
    @echo "🚀 Static type checking: Running mypy"
    uv run mypy
    @echo "🚀 Checking for obsolete dependencies: Running deptry"
    uv run deptry .

# Test the code with pytest
test:
    @echo "🚀 Testing code: Running pytest"
    {{PYTHON}} -m pytest --cov --cov-config=pyproject.toml --cov-report=xml

# Build wheel file
build: clean-build
    @echo "🚀 Creating wheel file"
    uvx --from build pyproject-build --installer uv

# Clean build artifacts
clean-build:
    @echo "🚀 Removing build artifacts"
    {{PYTHON}} -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

# Publish a release to PyPI.
publish:
    @echo "🚀 Publishing."
    uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

# Build and publish.
build-and-publish: build publish

# Test if documentation can be built without warnings or errors
docs-test:
    uv run mkdocs build -s

# Build and serve the documentation
docs:
    uv run mkdocs serve

help:
    @just --list

default: help


# Print the current operating system
info:
		print "OS: {{os()}}"

# Display system information
system-info:
	@echo "CPU architecture: {{ arch() }}"
	@echo "Operating system type: {{ os_family() }}"
	@echo "Operating system: {{ os() }}"

# verify python is running under pyenv
which-python:
		python -c "import sys;print(sys.executable)"

# when developing, you can use this to watch for changes and restart the server
autoreload-code:
	uv run watchmedo auto-restart --pattern "*.py" --recursive --signal SIGTERM uv run goobctl go

# Open the HTML coverage report in the default
local-open-coverage:
	./scripts/open-browser.py file://${PWD}/htmlcov/index.html

# Open the HTML coverage report in the default
open-coverage: local-open-coverage

# Run unit tests and open the coverage report
local-unittest:
	bash scripts/unittest-local
	./scripts/open-browser.py file://${PWD}/htmlcov/index.html

# Run all pre-commit hooks on all files
pre-commit-run-all:
	pre-commit run --all-files

# Install pre-commit hooks
pre-commit-install:
	pre-commit install

# Display the dependency tree of the project
pipdep-tree:
	pipdeptree --python .venv/bin/python3

# install uv tools globally
uv-tool-install:
	uv install invoke
	uv install pipdeptree
	uv install click

# Lint GitHub Actions workflow files
lint-github-actions:
	actionlint

# check that taplo is installed to lint/format TOML
check-taplo-installed:
	@command -v taplo >/dev/null 2>&1 || { echo >&2 "taplo is required but it's not installed. run 'brew install taplo'"; exit 1; }

# Format Python files using pre-commit
fmt-python:
	git ls-files '*.py' '*.ipynb' | xargs uv run pre-commit run --files

# Format Markdown files using pre-commit
fmt-markdown-pre-commit:
	git ls-files '*.md' | xargs uv run pre-commit run --files

# format pyproject.toml using taplo
fmt-toml:
	pre-commit run taplo-format --all-files

# SOURCE: https://github.com/PovertyAction/ipa-data-tech-handbook/blob/ed81492f3917ee8c87f5d8a60a92599a324f2ded/Justfile

# Format all markdown and config files
fmt-markdown:
	git ls-files '*.md' | xargs uv run mdformat

# Format a single markdown file, "f"
fmt-md f:
	uv run mdformat {{ f }}

# format all code using pre-commit config
fmt: fmt-python fmt-toml fmt-markdown fmt-markdown fmt-markdown-pre-commit

# lint python files using ruff
lint-python:
	pre-commit run ruff --all-files

# lint TOML files using taplo
lint-toml: check-taplo-installed
	pre-commit run taplo-lint --all-files

# lint yaml files using yamlfix
lint-yaml:
	pre-commit run yamlfix --all-files

# lint pyproject.toml and detect log_cli = true
lint-check-log-cli:
	pre-commit run detect-pytest-live-log --all-files

# Check format of all markdown files
lint-check-markdown:
	uv run mdformat --check .

# Lint all files in the current directory (and any subdirectories).
lint: lint-python lint-toml lint-check-log-cli lint-check-markdown

# generate type stubs for the project
createstubs:
	./scripts/createstubs.sh

# sweep init
sweep-init:
	uv run sweep init

# TODO: We should try out trunk
# By default, we use the following config that runs Trunk, an opinionated super-linter that installs all the common formatters and linters for your codebase. You can set up and configure Trunk for yourself by following https://docs.trunk.io/get-started.
# sandbox:
#   install:
#     - trunk init
#   check:
#     - trunk fmt {file_path}
#     - trunk check {file_path}

# Download AI models from Dropbox
download-models:
	curl -L 'https://www.dropbox.com/s/im6ytahqgbpyjvw/ScreenNetV1.pth?dl=1' > src/democracy_exe/data/ScreenNetV1.pth

# Perform a dry run of dependency upgrades
upgrade-dry-run:
	uv lock --update-all --all-features

# Upgrade all dependencies and sync the environment
sync-upgrade-all:
	uv sync --update-all --all-features

# Start a background HTTP server for test fixtures
http-server-background:
	#!/bin/bash
	# _PID=$(pgrep -f " -m http.server --bind localhost 19000 -d ./tests/fixtures")
	pkill -f " -m http.server --bind localhost 19000 -d ./tests/fixtures"
	python3 -m http.server --bind localhost 19000 -d ./tests/fixtures &
	echo $! > PATH.PID

# Start an HTTP server for test fixtures
http-server:
	#!/bin/bash
	# _PID=$(pgrep -f " -m http.server --bind localhost 19000 -d ./tests/fixtures")
	pkill -f " -m http.server --bind localhost 19000 -d ./tests/fixtures"
	python3 -m http.server --bind localhost 19000 -d ./tests/fixtures
	echo $! > PATH.PID

# Bump the version by major
major-version-bump:
	uv version
	uv version --bump major

# Bump the version by minor
minor-version-bump:
	uv version
	uv version --bump minor

# Bump the version by patch
patch-version-bump:
	uv version
	uv version --bump patch

# Bump the version by major
version-bump-major: major-version-bump

# Bump the version by minor
version-bump-minor: minor-version-bump

# Bump the version by patch
version-bump-patch: patch-version-bump

# Serve the documentation locally for preview
docs_preview:
    uv run mkdocs serve

# Build the documentation
docs_build:
    uv run mkdocs build

# Deploy the documentation to GitHub Pages
docs_deploy:
    uv run mkdocs gh-deploy --clean

# Generate a draft changelog
changelog:
    uv run towncrier build --version main --draft

# Checkout main branch and pull latest changes
gco:
    gco main
    git pull --rebase

# Show diff for LangChain migration
langchain-migrate-diff:
    langchain-cli migrate --include-ipynb --diff src

# Perform LangChain migration
langchain-migrate:
    langchain-cli migrate --include-ipynb src

# Get the ruff config
get-ruff-config:
	uv run ruff check --show-settings --config pyproject.toml -v -o ruff_config.toml >> ruff.log 2>&1

# Run lint and test
ci:
	uv run lint
	uv run test

# Open a manhole shell
manhole-shell:
	./scripts/manhole-shell

# Find the cassettes directories
find-cassettes-dirs:
	fd -td cassettes

# Delete the cassettes directories
delete-cassettes:
	fd -td cassettes -X rm -ri

# Install brew dependencies
brew-deps:
	brew install libmagic poppler tesseract pandoc qpdf tesseract-lang
	brew install --cask libreoffice

# install aicommits and configure it
init-aicommits:
	npm install -g aicommits
	aicommits config set OPENAI_KEY=$OCO_OPENAI_API_KEY type=conventional model=gpt-4o-mini
	aicommits hook install

# Run aider
aider:
	aider -c .aider.conf.yml --aiderignore .aiderignore

# Run aider with Claude
aider-claude:
	aider -c .aider.conf.yml --aiderignore .aiderignore --model 'anthropic/claude-3-5-sonnet-20241022'
