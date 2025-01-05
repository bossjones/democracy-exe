# Democracy Exe - Quickstart Guide 🚀

This guide will help you get started with Democracy Exe, an advanced agentic system built using LangChain and LangGraph. The system provides a powerful framework for building and deploying AI agents that can work together to solve complex tasks.

## 📋 Prerequisites

- Python 3.12+ (earlier versions not tested)
- Docker (optional, for containerized execution)
- Just command runner (`brew install just` on macOS)
- Git
- UV package manager (`pip install uv`)

## 🚀 Quick Setup (5 minutes)

The fastest way to get started:

```bash
# Clone the repository
git clone https://github.com/bossjones/democracy-exe.git
cd democracy-exe

# Install dependencies using UV
uv sync

# Set up pre-commit hooks
just install

# Run the bot
just run
```

## 🛠 Project Structure

```
democracy_exe/
├── agentic/               # Agentic system components
│   ├── agents/           # Individual agent implementations
│   ├── workflows/        # Agent workflow definitions
│   └── tools/           # Agent tools and utilities
├── ai/                   # AI/ML components
│   ├── agents/          # AI agent implementations
│   ├── graphs/          # LangGraph workflow definitions
│   └── state.py         # State management
├── chatbot/              # Discord chatbot components
│   ├── ai/              # AI integration for chat
│   ├── cogs/            # Discord bot cogs
│   └── handlers/        # Message and event handlers
├── clients/              # API client implementations
├── data/                 # Data storage and models
├── models/               # Data models and schemas
├── services/             # Service implementations
├── shell/                # CLI components
├── utils/                # Utility functions
└── vendored/            # Vendored dependencies
```

## 🛠️ Development Environment Setup

### Python Setup

1. Install Python 3.12:
   ```bash
   # macOS (using brew)
   brew install python@3.12

   # Ubuntu
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt update
   sudo apt install python3.12 python3.12-venv python3.12-dev
   ```

2. Install UV package manager:
   ```bash
   pip install uv
   ```

### Dependencies

Install all dependencies using UV:

```bash
# Install dependencies
uv sync

# Install development dependencies
uv sync --dev
```

### API Keys & Configuration

Create a `.env` file in the project root:

```env
# Required API Keys
OPENAI_API_KEY=sk-...            # Your OpenAI API key
LANGSMITH_API_KEY=lsv2-...       # Your LangSmith API key (for tracing)

# Optional API Keys
ANTHROPIC_API_KEY=sk-ant-...     # Your Anthropic API key
TAVILY_API_KEY=tvly-...          # Your Tavily API key

# Discord Configuration (if using bot)
DISCORD_TOKEN=...                # Your Discord bot token
DISCORD_GUILD_ID=...            # Your Discord server ID
```

## 🚀 Running the Project

### Using Just Commands

The project uses Just for command automation. Common commands:

```bash
# Run the bot
just run

# Run tests
just test

# Run tests in debug mode
just test-debug

# Run linting
just lint

# Format code
just fmt

# Check types
just typecheck
```

### Docker Deployment

For containerized deployment:

```bash
# Build and start all services
just generate-langgraph-dockerfile
docker-compose up -d

# Stop services
docker-compose down
```

### Development Mode

For local development:

```bash
# Run in development mode with hot reload
just autoreload-code

# Run with debug logging
just test-debug
```

## 🧪 Testing & Development

```bash
# Run all tests
just test

# Run tests with coverage
just test-debug
just open-coverage

# Run specific test suites
just test-twitter-cog
just test-dropbox
just test-gallery-dl

# Regenerate test fixtures
just regenerate-cassettes
```

## 🔧 Common Operations

```bash
# Format code
just fmt

# Run linting
just lint

# Check types
just typecheck

# Generate documentation
just docs_preview

# Deploy documentation
just docs_deploy
```

## 🆘 Code Quality Tools

The project uses several tools to maintain code quality:

1. **Ruff** - For linting and formatting
   - Configuration in `pyproject.toml`
   - Run with `just lint`

2. **Pyright** - For static type checking
   - Configuration in `pyproject.toml`
   - Run with `just typecheck`

3. **Pre-commit** - For git hooks
   - Configuration in `.pre-commit-config.yaml`
   - Includes:
     - Code formatting
     - Import sorting
     - Type checking
     - Security checks

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Discord.py Documentation](https://discordpy.readthedocs.io/en/stable/)
- [UV Documentation](https://github.com/astral-sh/uv)

## 🆘 Troubleshooting

1. **Missing Dependencies**
   ```bash
   # Sync dependencies
   uv sync
   ```

2. **Test Failures**
   ```bash
   # Regenerate test fixtures
   just regenerate-cassettes
   ```

3. **Type Checking Errors**
   ```bash
   # Run type checking with detailed output
   just typecheck
   ```

4. **Discord Bot Issues**
   ```bash
   # Check Discord connection
   just test-discord-connection

   # View bot logs
   just tail-bot-logs
   ```

## 📖 Documentation

- [Full Documentation](https://bossjones.github.io/democracy-exe/)
- [API Reference](https://bossjones.github.io/democracy-exe/api/)
- [Contributing Guide](CONTRIBUTING.md)
- [Justfile](Justfile) for all available commands

## 🤝 Getting Help

- Open an issue on [GitHub](https://github.com/bossjones/democracy-exe/issues)
- Join our [Discord community](https://discord.gg/your-invite-link)
- Check the [FAQ](https://bossjones.github.io/democracy-exe/faq/)
