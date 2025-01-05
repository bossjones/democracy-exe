# @timeit_decorator
# async def run_python(prompt: str) -> dict:
#     """
#     Executes a Python script from the scratch_pad_dir based on the user's prompt.
#     Returns the output and a success or failure status.
#     """
#     scratch_pad_dir = os.getenv("SCRATCH_PAD_DIR", "./scratchpad")
#     memory_content = memory_manager.get_xml_for_prompt(["*"])

#     # Step 1: Select the file based on the prompt
#     select_file_prompt = f"""
# <purpose>
#     Select a Python file to execute based on the user's prompt.
# </purpose>

# <instructions>
#     <instruction>Based on the user's prompt and the list of available Python files, infer which file the user wants to execute.</instruction>
#     <instruction>If no file matches, return an empty string for 'file'.</instruction>
# </instructions>

# <available-files>
#     {", ".join([f for f in os.listdir(scratch_pad_dir) if f.endswith('.py')])}
# </available-files>

# <memory-content>
#     {memory_content}
# </memory-content>

# <user-prompt>
#     {prompt}
# </user-prompt>
#     """

#     file_selection_response = structured_output_prompt(
#         select_file_prompt,
#         FileReadResponse,
#         llm_model=model_name_to_id[ModelName.fast_model],
#     )

#     if not file_selection_response.file:
#         return {
#             "status": "error",
#             "message": "No matching Python file found for the given prompt.",
#         }

#     file_path = os.path.join(scratch_pad_dir, file_selection_response.file)

#     if not os.path.exists(file_path):
#         return {
#             "status": "error",
#             "message": f"File '{file_selection_response.file}' does not exist in '{scratch_pad_dir}'.",
#         }

#     # Read the Python code from the selected file
#     try:
#         with open(file_path, "r") as f:
#             python_code = f.read()
#     except Exception as e:
#         return {"status": "error", "message": f"Failed to read the file: {str(e)}"}

#     # Execute the Python code using run_uv_script
#     output = run_uv_script(python_code)

#     # Save the output to a file with '_output' suffix
#     output_file_name = os.path.splitext(file_selection_response.file)[0] + "_output.txt"
#     output_file_path = os.path.join(scratch_pad_dir, output_file_name)
#     with open(output_file_path, "w") as f:
#         f.write(output)

#     # Determine success based on presence of errors
#     if "Traceback" in output or "Error" in output:
#         success = False
#         error_message = output
#     else:
#         success = True
#         error_message = None

#     return {
#         "status": "success" if success else "failure",
#         "error": error_message,
#         "file_name": file_selection_response.file,
#         "output_file": output_file_name,
#     }
