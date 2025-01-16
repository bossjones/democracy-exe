# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false

from __future__ import annotations

import os
import sys

from typing import Dict, List, Optional

import anthropic

from dotenv import find_dotenv, load_dotenv
from transformers import GPT2Tokenizer

from democracy_exe.aio_settings import aiosettings


# SOURCE: https://github.com/connor-john/ai-docs

load_dotenv(find_dotenv())

CLIENT = anthropic.Anthropic(api_key=aiosettings.anthropic_api_key.get_secret_value())

PULL_REQUEST_SYSTEM_PROMPT = """Your job is to act as a expert software engineer and provide outlined comprehensive feedback for merge requests in a appraochable and understanding way.
Key criteria outlined are:
Does this code change accomplish what it is supposed to do?
Can this solution be simplified?
Does this change add unwanted compile-time or run-time dependencies?
Is a framework, API, library, or service used that should not be used?
Could an additional framework, API, library, or service improve the solution?
Is the code at the right abstraction level?
Is the code modular enough?
Can a better solution be found in terms of maintainability, readability, performance, or security?
Does similar functionality already exist in the codebase? If yes, why isnt it reused?
Are there any best practices, design patterns or language-specific patterns that could substantially improve this code?
Does this code adhere to Object-Oriented Analysis and Design Principles, like the Single Responsibility Principle, Open-Close Principle, Liskov Substitution Principle, Interface Segregation, or Dependency Injection?
Can you think of any use case in which the code does not behave as intended?
Can you think of any inputs or external events that could break the code?
You will receive both a description of the changes, an outline of the changes and the existing codebase.
"""

BASIC_DOCS_SYSTEM_PROMPT = """Your job is to act as the expert software engineer and provide detailed technical documentation broken into readbile formats.
You will be able to read the majority of a code repository given as a converted single text with the Prefix "#File:" declaring the start of the new file e.g., # File: masters-and-sons-main/src/components/ArrowTable.tsx.
Typescipt projects are required to use pnpm for commands, python project if they have a pyproject.toml will be using poetry. MySQL databases are typically planetscale, and in that case we follow planetscales recommended use of prisma (if prisms ORM is used).
Documentation should be broken down into:
Introduction:
- Provide a brief overview of the project.
- Mention the purpose and core functionality of the code repository.
- Highlight the key features and potential use cases.
Codebase Overview:
- Provide an in-depth overview of the codebase architecture and design patterns used.
- Detail the modules, components, and their interactions within the application.
Development Environment Setup:
- Step-by-step instructions for setting up the development environment, including necessary tools and dependencies.
- Include guidelines for configuring IDEs, linters, and other development tools.
Code Repository Structure:
- Explain the repository structure, detailing the purpose of different directories and files.
- Document naming conventions, file organization, and any specific standards followed in the codebase.
Key Points of Complexity:
- Outline each keypoint of complexity
    - Breakdown each keypoint
Installation and Setup:
- Offer detailed instructions on installing and setting up the project.
- Include any prerequisites or dependencies required.
Getting Started:
- Guide users through a simple, initial setup or quick-start process.
- Include basic examples or a simple tutorial to help users begin using the project quickly.
The output should be detailed in standard markdown, using headers to improve readability.
"""

REFINED_DOCS_FOLLOW_UP_PROMPT = """You are instructed to further refine the documentation, there is no need to repeat basic content. Focus on delivering expert value insights for internal development handover outlined in the below criteria.
In-Depth Architecture Overview:
- Provide comprehensive diagrams and descriptions of the system architecture, including data flow, service interactions, and external dependencies.
- Detail the critical architectural decisions and their justifications, discussing the trade-offs and alternatives considered.
Advanced Codebase Insights:
- Delve into complex modules and components, explaining intricate details that are crucial for understanding system behavior.
- Document any non-obvious implementation strategies and optimizations, explaining why they were necessary and how they impact the system.
Environment and Toolchain Deep Dive:
- Offer a detailed guide on the development, testing, and production environments, including specific configurations, underlying infrastructure details, and environment parity practices.
- Describe the complete toolchain, including build systems, deployment pipelines, and any custom tooling.
Critical Dependency Analysis:
- Provide an exhaustive overview of external libraries, frameworks, and services the system depends on, including versioning, patching strategies, and known issues.
- Discuss how dependencies are managed, potential risks associated with them, and mitigation strategies.
Performance Considerations:
- Document performance benchmarks, profiling methods, and optimization strategies.
- Include case studies of past performance issues, their analysis, and the solutions implemented.
Security Protocols:
- Detail the security measures and protocols in place, including authentication, authorization, data encryption, and any proprietary security frameworks or libraries.
- Discuss security best practices, known vulnerabilities (and their mitigations), and any relevant security audits.
Testing and Quality Assurance:
- Elaborate on the testing strategy, including unit, integration, and end-to-end tests, highlighting any test-driven development (TDD) practices.
- Document the approach to continuous integration/continuous deployment (CI/CD), test automation, and quality benchmarks.
Troubleshooting and Debugging Guide:
- Provide a detailed guide for troubleshooting common issues, including debugging tips, logging best practices, and monitoring tools used.
- Share real-world incidents or outages, post-mortem analyses, and lessons learned.
Data Management and Migration:
- Explain the data architecture, schema design, and any data migration strategies or scripts.
- Discuss backup strategies, data integrity checks, and disaster recovery plans.
Developer Onboarding:
- Create a comprehensive onboarding guide for new developers, including step-by-step setup instructions, key contacts, and resources for getting up to speed.
- Include a glossary of terms, acronyms, and jargon used within the project to aid in understanding the documentation and codebase.
"""


def check_prompt_token_size(prompt: str) -> int:
    """Use GPT-2 to check the approximate number of tokens in a prompt.

    Args:
        prompt: The text prompt to tokenize

    Returns:
        int: Number of tokens in the prompt
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return len(tokens)


async def request_message(
    system_prompt: str, messages: list[dict[str, str]]
) -> anthropic.types.Message:
    """Send message to Anthropic.

    Args:
        system_prompt: The system prompt to use for the message
        messages: List of message dictionaries containing role and content

    Returns:
        anthropic.types.Message: Response from the Anthropic API
    """
    response = await CLIENT.messages.create(
        model="claude-3-opus-20240229",
        system=system_prompt,
        max_tokens=4096,
        messages=messages,
    )

    return response


def read_file(file_path: str) -> str | None:
    """Read the text file containing the repo content.

    Args:
        file_path: Path to the file to read

    Returns:
        Optional[str]: Contents of the file if it exists, None otherwise

    Raises:
        FileNotFoundError: If the file does not exist
    """
    if os.path.exists(file_path):
        with open(file_path, encoding="utf-8") as file:
            repo_content = file.read()
            return repo_content
    else:
        print("Error: The file does not exist.")
        raise FileNotFoundError(f"File not found: {file_path}")


def _generate_docs(file_path: str) -> None:
    """Generate docs for the provided repo using Claude Opus.

    Args:
        file_path: Path to the file containing repository content

    Raises:
        SystemExit: If user chooses not to proceed with doc generation
        FileNotFoundError: If the input file does not exist
    """
    repo_name = str(os.path.splitext(os.path.basename(file_path))[0]).replace(
        "_code", ""
    )
    file_content = read_file(file_path)
    input_prompt = f"Given this repo. \n{file_content}\ncomplete your instruction"
    token_size = check_prompt_token_size(input_prompt)

    proceed_check = input(
        f"Input token size is: {token_size}. Do you wish to proceed? (Y/N)"
    )
    if str(proceed_check).upper() != "Y":
        print("Exiting")
        sys.exit(1)

    messages = [
        {"role": "user", "content": input_prompt},
    ]
    response = request_message(BASIC_DOCS_SYSTEM_PROMPT, messages)

    message = response.content[0].text
    with open(f"{repo_name}-docs.md", "w", encoding="utf-8") as file:
        file.write(message)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python api.py <txt file path>")
        sys.exit(1)

    _generate_docs(sys.argv[1])
