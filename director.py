#!/usr/bin/env python3
# pyright: reportInvalidTypeForm=false

# Import required libraries for data validation and typing
from __future__ import annotations

import argparse
import subprocess
import sys

from pathlib import Path
from typing import List, Literal, Optional

import yaml

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from openai import OpenAI
from pydantic import BaseModel


# Define a model for evaluation results that includes success status and optional feedback
class EvaluationResult(BaseModel):
    success: bool
    feedback: str | None


# Define configuration model for the Director class with all required settings
class DirectorConfig(BaseModel):
    prompt: str  # The instruction prompt for code generation
    coder_model: str  # The model to use for code generation
    evaluator_model: Literal["gpt-4o", "gpt-4o-mini", "o1-mini", "o1-preview"]  # Model for evaluation
    max_iterations: int  # Maximum number of attempts to generate correct code
    execution_command: str  # Command to execute for testing the generated code
    context_editable: list[str]  # List of files that can be modified
    context_read_only: list[str]  # List of files that should not be modified
    evaluator: Literal["default"]  # Type of evaluator to use


# Main class that orchestrates the AI-driven code generation and evaluation process
class Director:
    """
    Self Directed AI Coding Assistant
    """

    def __init__(self, config_path: str):
        # Initialize Director with configuration from the specified path
        self.config = self.validate_config(Path(config_path))
        self.llm_client = OpenAI()

    @staticmethod
    def validate_config(config_path: Path) -> DirectorConfig:
        """Validate the yaml config file and return DirectorConfig object."""
        # Check if the config file exists
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load and parse the YAML configuration file
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        # Handle markdown prompt files - load content if prompt is a .md file
        if config_dict["prompt"].endswith(".md"):
            prompt_path = Path(config_dict["prompt"])
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
            with open(prompt_path) as f:
                config_dict["prompt"] = f.read()

        # Create DirectorConfig object from the loaded dictionary
        config = DirectorConfig(**config_dict)

        # Validate that the evaluator model is one of the allowed values
        allowed_evaluator_models = {"gpt-4o", "gpt-4o-mini", "o1-mini", "o1-preview"}
        if config.evaluator_model not in allowed_evaluator_models:
            raise ValueError(f"evaluator_model must be one of {allowed_evaluator_models}, got {config.evaluator_model}")

        # Ensure there is at least one editable file specified
        if not config.context_editable:
            raise ValueError("At least one editable context file must be specified")

        # Verify all specified files (both editable and read-only) exist
        for path in config.context_editable:
            if not Path(path).exists():
                raise FileNotFoundError(f"Editable context file not found: {path}")

        for path in config.context_read_only:
            if not Path(path).exists():
                raise FileNotFoundError(f"Read-only context file not found: {path}")

        return config

    def parse_llm_json_response(self, str) -> str:
        """
        Parse and fix the response from an LLM that is expected to return JSON.
        """
        # Handle case where response doesn't contain code blocks
        if "```" not in str:
            str = str.strip()
            self.file_log(f"raw pre-json-parse: {str}", print_message=False)
            return str

        # Extract content from markdown code blocks by removing backticks and language identifiers
        str = str.split("```", 1)[-1].split("\n", 1)[-1]
        str = str.rsplit("```", 1)[0]
        str = str.strip()

        self.file_log(f"post-json-parse: {str}", print_message=False)
        return str

    def file_log(self, message: str, print_message: bool = True):
        # Print message to console if requested
        if print_message:
            print(message)
        # Append message to log file for record keeping
        with open("director_log.txt", "a+") as f:
            f.write(message + "\n")

    # ------------- Key Director Methods -------------

    def create_new_ai_coding_prompt(
        self,
        iteration: int,
        base_input_prompt: str,
        execution_output: str,
        evaluation: EvaluationResult,
    ) -> str:
        # For first iteration, use the original prompt
        if iteration == 0:
            return base_input_prompt
        else:
            # For subsequent iterations, create a prompt that includes feedback and previous attempt results
            return f"""
# Generate the next iteration of code to achieve the user's desired result based on their original instructions and the feedback from the previous attempt.
> Generate a new prompt in the same style as the original instructions for the next iteration of code.

## This is your {iteration}th attempt to generate the code.
> You have {self.config.max_iterations - iteration} attempts remaining.

## Here's the user's original instructions for generating the code:
{base_input_prompt}

## Here's the output of your previous attempt:
{execution_output}

## Here's feedback on your previous attempt:
{evaluation.feedback}"""

    def ai_code(self, prompt: str):
        # Initialize the AI model for code generation
        model = Model(self.config.coder_model)
        # Create a coder instance with specified configuration
        coder = Coder.create(
            main_model=model,
            io=InputOutput(yes=True),
            fnames=self.config.context_editable,
            read_only_fnames=self.config.context_read_only,
            auto_commits=False,
            suggest_shell_commands=False,
        )
        # Run the code generation with the provided prompt
        coder.run(prompt)

    def execute(self) -> str:
        """Execute the tests and return the output as a string."""
        # Run the configured execution command and capture its output
        result = subprocess.run(
            self.config.execution_command.split(),
            capture_output=True,
            text=True,
        )
        # Log the execution output for debugging
        self.file_log(
            f"Execution output: \n{result.stdout + result.stderr}",
            print_message=False,
        )
        return result.stdout + result.stderr

    def evaluate(self, execution_output: str) -> EvaluationResult:
        # Verify we're using the default evaluator
        if self.config.evaluator != "default":
            raise ValueError(f"Custom evaluator {self.config.evaluator} not implemented")

        # Create maps of file contents for both editable and read-only files
        map_editable_fname_to_files = {
            Path(fname).name: Path(fname).read_text() for fname in self.config.context_editable
        }

        map_read_only_fname_to_files = {
            Path(fname).name: Path(fname).read_text() for fname in self.config.context_read_only
        }

        # Construct the evaluation prompt with all necessary context and instructions
        evaluation_prompt = f"""Evaluate this execution output and determine if it was successful based on the execution command, the user's desired result, the editable files, checklist, and the read-only files.

## Checklist:
- Is the execution output reporting success or failure?
- Did we miss any tasks? Review the User's Desired Result to see if we have satisfied all tasks.
- Did we satisfy the user's desired result?
- Ignore warnings

## User's Desired Result:
{self.config.prompt}

## Editable Files:
{map_editable_fname_to_files}

## Read-Only Files:
{map_read_only_fname_to_files}

## Execution Command:
{self.config.execution_command}

## Execution Output:
{execution_output}

## Response Format:
> Be 100% sure to output JSON.parse compatible JSON.
> That means no new lines.

Return a structured JSON response with the following structure: {{
    success: bool - true if the execution output generated by the execution command matches the Users Desired Result
    feedback: str | None - if unsuccessful, provide detailed feedback explaining what failed and how to fix it, or None if successful
}}"""

        # Log the evaluation prompt for debugging
        self.file_log(
            f"Evaluation prompt: ({self.config.evaluator_model}):\n{evaluation_prompt}",
            print_message=False,
        )

        try:
            # Attempt to get evaluation from the primary model
            completion = self.llm_client.chat.completions.create(
                model=self.config.evaluator_model,
                messages=[
                    {
                        "role": "user",
                        "content": evaluation_prompt,
                    },
                ],
            )

            # Log the model's response
            self.file_log(
                f"Evaluation response: ({self.config.evaluator_model}):\n{completion.choices[0].message.content}",
                print_message=False,
            )

            # Parse and validate the response as an EvaluationResult
            evaluation = EvaluationResult.model_validate_json(
                self.parse_llm_json_response(completion.choices[0].message.content)
            )

            return evaluation
        except Exception as e:
            # Log the error and attempt to use fallback model
            self.file_log(
                f"Error evaluating execution output for '{self.config.evaluator_model}'. Error: {e}. Falling back to gpt-4o & structured output."
            )

            # Try evaluation with fallback model (gpt-4o)
            completion = self.llm_client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": evaluation_prompt,
                    },
                ],
                response_format=EvaluationResult,
            )

            # Return parsed response if available, otherwise raise error
            message = completion.choices[0].message
            if message.parsed:
                return message.parsed
            else:
                raise ValueError("Failed to parse the response")

    def direct(self):
        # Initialize variables for tracking progress
        evaluation = EvaluationResult(success=False, feedback=None)
        execution_output = ""
        success = False

        # Main iteration loop for attempting code generation
        for i in range(self.config.max_iterations):
            self.file_log(f"\nIteration {i + 1}/{self.config.max_iterations}")

            # Step 1: Create a new prompt based on previous results
            self.file_log("🧠 Creating new prompt...")
            new_prompt = self.create_new_ai_coding_prompt(i, self.config.prompt, execution_output, evaluation)

            # Step 2: Generate code using AI
            self.file_log("🤖 Generating AI code...")
            self.ai_code(new_prompt)

            # Step 3: Execute the generated code
            self.file_log(f"💻 Executing code... '{self.config.execution_command}'")
            execution_output = self.execute()

            # Step 4: Evaluate the results
            self.file_log(f"🔍 Evaluating results... '{self.config.evaluator_model}' + '{self.config.evaluator}'")
            evaluation = self.evaluate(execution_output)

            # Log evaluation results
            self.file_log(f"🔍 Evaluation result: {'✅ Success' if evaluation.success else '❌ Failed'}")
            if evaluation.feedback:
                self.file_log(f"💬 Feedback: \n{evaluation.feedback}")

            # Break loop if successful
            if evaluation.success:
                success = True
                self.file_log(f"\n🎉 Success achieved after {i + 1} iterations! Breaking out of iteration loop.")
                break
            else:
                self.file_log(
                    f"\n🔄 Continuing with next iteration... Have {self.config.max_iterations - i - 1} attempts remaining."
                )

        # Log final status if not successful
        if not success:
            self.file_log("\n🚫 Failed to achieve success within the maximum number of iterations.")

        self.file_log("\nDone.")


# Entry point for command line execution
if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Run the AI Coding Director with a config file")
    parser.add_argument(
        "--config",
        type=str,
        default="specs/basic.yaml",
        help="Path to the YAML config file",
    )

    # Parse arguments and run the director
    args = parser.parse_args()
    director = Director(args.config)
    director.direct()
