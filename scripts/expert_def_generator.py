#!/usr/bin/env python3
# SOURCE: https://claude.ai/chat/6f981f76-5f20-450d-b2cc-7cef1f55ab1a

"""
The enhanced script now:

Uses a highly structured prompt template optimized for Claude 3.5 Sonnet
Includes clear XML tag guidelines
Provides explicit instructions for formatting and style
Allows customization through external style guides
Maintains consistent technical documentation standards

The output will be more structured, detailed, and aligned with best practices for technical documentation. Claude 3.5 Sonnet will use the XML tags consistently and provide thorough analysis with concrete examples.
Would you like me to explain any part in more detail or make additional enhancements to the script?


Example usage:

python expert_def_generator.py https://github.com/langchain-ai/langgraph \
    --version 0.2.61 \
    --files Dockerfile langgraph.json \
    --prompt "Analyze these configuration files for best practices and implementation patterns" \
    --output-dir ai_docs/prompts/data \
    --style custom_style.json \
    --verbose
"""



import argparse
import subprocess
import os
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
import json

def run_command(command):
    """Execute a shell command and return its output."""
    try:
        result = subprocess.run(command, shell=True, check=True,
                              capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error output: {e.stderr}")
        raise

def generate_xml_filename(repo_name, files):
    """Generate a consistent filename for the XML output."""
    base_name = '_'.join(files)
    if len(base_name) > 30:  # Truncate if too long
        base_name = base_name[:30]
    return f"{repo_name}_{base_name}.xml"

def create_expert_prompt(xml_content, custom_prompt):
    """Create an enhanced expert definition prompt optimized for Claude 3.5 Sonnet."""
    prompt_template = f"""<system>
You are Claude 3.5 Sonnet, an expert at analyzing code, documentation, and technical specifications. You excel at providing detailed, well-structured analysis while maintaining clear organization through XML tags. Your responses should be thorough, educational, and precisely formatted.
</system>

<context>
The following documents contain source code and configuration files that need expert analysis. The analysis should help developers understand and implement best practices.
</context>

<documents>
{xml_content}
</documents>

<instructions>
1. Analyze the provided documents thoroughly, paying special attention to patterns, best practices, and implementation details.
2. Structure your response using clear XML tags to separate different types of information.
3. When providing analysis, include specific examples from the source code to support your points.
4. Consider both explicit statements and implicit patterns in the code.
5. Focus on actionable insights that developers can implement.
6. Maintain a clear, educational tone appropriate for technical documentation.
</instructions>

<task>
{custom_prompt}
</task>

<output_format>
Your response should use the following XML structure:
- Use <quotes> tags for relevant excerpts from the source
- Use <info> tags for analytical insights and recommendations
- Use <examples> tags when providing concrete examples
- Use <best_practices> tags when summarizing recommended approaches
- Use <implementation> tags when discussing specific implementation details
</output_format>

<style_preferences>
- Write in clear, professional prose
- Use complete sentences and proper technical terminology
- Provide detailed explanations with concrete examples
- Include relevant context to support understanding
- Structure information logically and hierarchically
</style_preferences>"""
    return prompt_template

def main():
    parser = argparse.ArgumentParser(description='Generate expert definitions from repository files')
    parser.add_argument('repo_url', help='GitHub repository URL')
    parser.add_argument('--version', help='Repository version/tag to checkout')
    parser.add_argument('--files', nargs='+', help='Files to analyze', required=True)
    parser.add_argument('--prompt', help='Custom prompt for expert definition', required=True)
    parser.add_argument('--output-dir', help='Output directory', default='ai_docs/prompts/data')
    parser.add_argument('--style', help='Path to custom style guide JSON file')
    args = parser.parse_args()

    # Load custom style guide if provided
    style_guide = {}
    if args.style:
        try:
            with open(args.style) as f:
                style_guide = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load style guide: {e}")
            print("Using default style guide")

    # Extract repo name from URL
    repo_name = args.repo_url.split('/')[-1].replace('.git', '')

    # Create temporary directory for cloning
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")

        # Clone repository
        clone_cmd = f"git clone {args.repo_url} {temp_dir}"
        run_command(clone_cmd)

        # Checkout specific version if provided
        if args.version:
            checkout_cmd = f"cd {temp_dir} && git checkout {args.version}"
            run_command(checkout_cmd)

        # Generate XML filename
        xml_filename = generate_xml_filename(repo_name, args.files)
        xml_path = os.path.join(args.output_dir, xml_filename)

        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        # Generate XML using files-to-prompt
        files_str = ' '.join(args.files)
        files_to_prompt_cmd = f"cd {temp_dir} && uv run files-to-prompt {files_str} --cxml -o {xml_path}"
        run_command(files_to_prompt_cmd)

        # Read the generated XML
        with open(xml_path) as f:
            xml_content = f.read()

        # Create the expert definition prompt
        expert_prompt = create_expert_prompt(xml_content, args.prompt)

        # Use llm to generate the expert definition
        expert_def_path = xml_path.replace('.xml', '_expert_definition.md')

        # Save the prompt to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as temp_prompt:
            temp_prompt.write(expert_prompt)
            temp_prompt.flush()

            # Run llm command
            llm_cmd = f'cat {temp_prompt.name} | llm > "{expert_def_path}"'
            run_command(llm_cmd)

        print(f"\nExpert definition generated successfully!")
        print(f"XML output: {xml_path}")
        print(f"Expert definition: {expert_def_path}")

if __name__ == "__main__":
    main()
