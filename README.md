**# democracy-exe

[![Release](https://img.shields.io/github/v/release/bossjones/democracy-exe)](https://img.shields.io/github/v/release/bossjones/democracy-exe)
[![Build status](https://img.shields.io/github/actions/workflow/status/bossjones/democracy-exe/main.yml?branch=main)](https://github.com/bossjones/democracy-exe/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/bossjones/democracy-exe/branch/main/graph/badge.svg)](https://codecov.io/gh/bossjones/democracy-exe)
[![Commit activity](https://img.shields.io/github/commit-activity/m/bossjones/democracy-exe)](https://img.shields.io/github/commit-activity/m/bossjones/democracy-exe)
[![License](https://img.shields.io/github/license/bossjones/democracy-exe)](https://img.shields.io/github/license/bossjones/democracy-exe)

democracy_exe is an advanced, agentic Python application leveraging LangChain and LangGraph to orchestrate and manage a network of AI agents and subgraphs. This system emulates the principles of "managed democracy" from the Helldivers universe, automating decision-making processes and task delegation across multiple AI entities. Based on Helldivers.

-   **Github repository**: <https://github.com/bossjones/democracy-exe/>
-   **Documentation** <https://bossjones.github.io/democracy-exe/>

## Getting started with your project

![LangGraph Architecture](@langgraph.png)

# Agentic Workflow

## Let The Code Write Itself
> Self Directed AI Coding

## Setup

- Install dependencies: `uv sync --all-extras`
- Run director
  - Version bump: `uv run python director.py --config specs/director_version_bump.yaml`
  - Create AI docs typer subcommand: `uv run python director.py --config specs/director_create_ai_docs_typer_subcommand.yaml`
  - Slider output format with two charts: `uv run python director.py --config specs/director_slider_output_format_two_charts.yaml`
  - Green output format: `uv run python director.py --config specs/director_green_output_format.yaml`



### 1. Create a New Repository

First, create a repository on GitHub with the same name as this project, and then run the following commands:

```bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:bossjones/democracy-exe.git
git push -u origin main
```

### 2. Set Up Your Development Environment

Then, install the environment and the pre-commit hooks with

```bash
make install
```

This will also generate your `uv.lock` file

### 3. Run the pre-commit hooks

Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

### 4. Commit the changes

Lastly, commit the changes made by the two steps above to your repository.

```bash
git add .
git commit -m 'Fix formatting issues'
git push origin main
```

You are now ready to start development on your project!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

To finalize the set-up for publishing to PyPI, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

## Releasing a new version

-   Create an API Token on [PyPI](https://pypi.org/).
-   Add the API Token to your projects secrets with the name `PYPI_TOKEN` by visiting [this page](https://github.com/bossjones/democracy-exe/settings/secrets/actions/new).
-   Create a [new release](https://github.com/bossjones/democracy-exe/releases/new) on Github.
-   Create a new tag in the form `*.*.*`.

For more details, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/cicd/#how-to-trigger-a-release).

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
**
