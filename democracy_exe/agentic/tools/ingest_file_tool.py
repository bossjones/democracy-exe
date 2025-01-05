# @timeit_decorator
# async def ingest_file(prompt: str) -> dict:
#     """
#     Selects a file based on the user's prompt, reads its content, and returns the file data.
#     """
#     scratch_pad_dir = os.getenv("SCRATCH_PAD_DIR", "./scratchpad")

#     # Step 1: Select the file based on the prompt
#     select_file_prompt = f"""
# <purpose>
#     Select a file from the available files based on the user's prompt.
# </purpose>

# <instructions>
#     <instruction>Based on the user's prompt and the list of available files, infer which file the user wants to ingest.</instruction>
#     <instruction>If no file matches, return an empty string for 'file'.</instruction>
# </instructions>

# <available-files>
#     {", ".join(os.listdir(scratch_pad_dir))}
# </available-files>

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
#             "ingested_content": None,
#             "message": "No matching file found for the given prompt.",
#             "success": False,
#         }

#     file_path = os.path.join(scratch_pad_dir, file_selection_response.file)

#     if not os.path.exists(file_path):
#         return {
#             "ingested_content": None,
#             "message": f"File '{file_selection_response.file}' does not exist in '{scratch_pad_dir}'.",
#             "success": False,
#         }

#     # Read the file content
#     try:
#         with open(file_path, "r") as f:
#             file_content = f.read()
#     except Exception as e:
#         return {
#             "ingested_content": None,
#             "message": f"Failed to read the file: {str(e)}",
#             "success": False,
#         }

#     return {
#         "ingested_content": file_content,
#         "message": "Successfully ingested content",
#         "success": True,
#     }
