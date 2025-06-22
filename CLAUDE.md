# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

democracy-exe is an advanced, agentic Python application leveraging LangChain and LangGraph to orchestrate and manage a network of AI agents and subgraphs. The system emulates "managed democracy" principles from the Helldivers universe, automating decision-making and task delegation across multiple AI entities.

## Development Commands

### Environment Setup
- Install dependencies: `uv sync --all-extras`
- Install with pre-commit hooks: `make install`
- Run pre-commit checks: `uv run pre-commit run -a`

### Testing
- Run tests: `make test` (uses pytest with coverage)
- Run tests with verbose output: `make pytest`
- Run CI pipeline locally: `make ci`

### Code Quality
- Lint and type check: `make check`
- Type checking: `uv run mypy`
- Dependency check: `uv run deptry .`

### Building and Publishing
- Build wheel: `make build`
- Clean build artifacts: `make clean-build`
- Publish to PyPI: `make publish`

### Documentation
- Build docs: `make docs-test`
- Serve docs locally: `make docs`

### Bot Commands
- Run Discord bot: `uv run democracyctl run-bot` or `uv run democracyctl go`
- Run terminal bot: `uv run democracyctl run-terminal-bot`
- Show version info: `uv run democracyctl version --verbose`
- Show dependencies: `uv run democracyctl deps`

### LangGraph Studio
- The project includes LangGraph Studio configuration in `langgraph.json`
- Main graph endpoint: `./democracy_exe/agentic/workflows/react/graph.py:graph`

## Architecture Overview

### Core Structure
- **Entry Points**: `democracy_exe/__main__.py`, `democracy_exe/cli.py`
- **Main Application**: `democracy_exe/main.py` (sandbox agent entry point)
- **Settings**: `democracy_exe/aio_settings.py` (async settings management)

### Key Modules

#### Agentic System (`democracy_exe/agentic/`)
- **Main Graph**: `graph.py` - Core memory-enabled agent with LangGraph
- **Tools**: `tools/` - Various agent tools (file ops, browser, shell, etc.)
- **Workflows**: `workflows/react/` - ReAct agent implementation
- **Event Server**: `event_server.py` - Agentic event handling

#### AI Components (`democracy_exe/ai/`)
- **Agents**: Specialized agents (image analysis, research, social media, etc.)
- **Graphs**: Corresponding LangGraph implementations for each agent
- **State Management**: `state.py`, `base.py` - Core AI state handling

#### Chatbot System (`democracy_exe/chatbot/`)
- **Discord Bot**: `discord_bot.py` - Main Discord integration
- **Terminal Bot**: `terminal_bot.py` - Command-line interface bot
- **Handlers**: Message and attachment processing
- **Cogs**: Discord bot extensions (autocrop, image caption, Twitter)

#### Utilities (`democracy_exe/utils/`)
- **Image Processing**: `imgops.py`, `imitools.py` - Image operations
- **File Operations**: `file_functions.py`, `file_operations.py`
- **Async Utilities**: `async_.py`, `aiodbx.py`, `aiotweetpik.py`
- **Twitter Integration**: `twitter_utils/` - Twitter client and utilities

#### Clients (`democracy_exe/clients/`)
- **Gallery-dl**: `aio_gallery_dl.py` - Async gallery downloader
- **TweetPik**: `tweetpik.py` - Tweet image generation

### Configuration and Settings
- **Environment**: Uses `pydantic-settings` for configuration management
- **Async Settings**: All settings are async-compatible via `aio_settings.py`
- **LLM Configuration**: Supports multiple providers (OpenAI, Anthropic, etc.)

### Memory System
The agentic system implements sophisticated memory management:
- **Core Memories**: Fundamental user understanding (always available)
- **Recall Memories**: Contextual/semantic memories (retrieved as needed)
- **Vector Storage**: Uses SKLearnVectorStore for memory persistence
- **Memory Tools**: `save_recall_memory`, `search_memory`, `store_core_memory`

### Bot Architecture
- **Multi-Interface**: Supports both Discord and terminal interfaces
- **Extension System**: Discord cogs for modular functionality
- **Resource Management**: Memory limits and monitoring via `resource_manager.py`
- **Async First**: Built with asyncio throughout

## Testing Strategy

### Test Structure
- **Unit Tests**: `tests/unittests/` - Component testing
- **Integration Tests**: `tests/integration/` - End-to-end testing
- **Fixtures**: `tests/fixtures/` - Test data and media files
- **Test Utilities**: `tests/tests_utils/` - Testing helpers

### Key Test Areas
- Bot functionality (`test_discord_bot.py`)
- CLI commands (`test_cli.py`)
- Async components (`test_async_typer.py`)
- Settings management (`test_aio_settings.py`)

## Development Notes

### Dependency Management
- Uses `uv` for fast Python package management
- Multi-extras support for optional dependencies
- Lock file (`uv.lock`) for reproducible builds

### Code Style
- Uses `ruff` for linting and formatting
- `mypy` for static type checking
- `pre-commit` hooks for code quality

### Docker Support
- `Dockerfile` and `Dockerfile.debugging` for containerization
- LangGraph Studio Docker integration via `langgraph.json`

### Key Environment Variables
- `DISCORD_TOKEN`: Discord bot authentication
- `OPENAI_API_KEY`: OpenAI API access
- `ANTHROPIC_API_KEY`: Anthropic API access
- `TAVILY_API_KEY`: Web search functionality
- `PYTHONASYNCIODEBUG=1`: Async debugging
- `PYTHONFAULTHANDLER=1`: Enhanced error reporting

## Common Development Tasks

### Adding New Agent
1. Create agent in `democracy_exe/ai/agents/`
2. Create corresponding graph in `democracy_exe/ai/graphs/`
3. Register in main graph system
4. Add tests in `tests/`

### Adding New Tool
1. Create tool in `democracy_exe/agentic/tools/`
2. Import and register in `graph.py`
3. Add tests and documentation

### Adding CLI Command
1. Create command file in `democracy_exe/subcommands/` (ends with `_cmd.py`)
2. Export `APP` variable with Typer instance
3. Commands auto-load via `load_commands()` in `cli.py`

### Debugging
- Set `dev_mode=True` in settings for enhanced debugging
- Use `bpdb.pm()` for post-mortem debugging
- Enable LangChain debugging with `debug_langchain=True`
