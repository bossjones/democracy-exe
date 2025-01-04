# @timeit_decorator
# async def discuss_file(prompt: str, model: ModelName = ModelName.base_model) -> dict:
#     """
#     Discuss a file's content based on the user's prompt, considering the current memory content.
#     """
#     scratch_pad_dir = os.getenv("SCRATCH_PAD_DIR", "./scratchpad")
#     focus_file = personalization.get("focus_file")

#     if focus_file:
#         file_path = os.path.join(scratch_pad_dir, focus_file)
#         if not os.path.exists(file_path):
#             return {"status": "Focus file not found", "file_name": focus_file}
#     else:
#         # List available files in SCRATCH_PAD_DIR
#         available_files = os.listdir(scratch_pad_dir)
#         available_files_str = ", ".join(available_files)

#         # Build the structured prompt to select the file
#         select_file_prompt = f"""
# <purpose>
#     Select a file from the available files based on the user's prompt.
# </purpose>

# <instructions>
#     <instruction>Based on the user's prompt and the list of available files, infer which file the user wants to discuss.</instruction>
#     <instruction>If no file matches, return an empty string for 'file'.</instruction>
# </instructions>

# <available-files>
#     {available_files_str}
# </available-files>

# <user-prompt>
#     {prompt}
# </user-prompt>
#         """

#         # Call the LLM to select the file
#         file_selection_response = structured_output_prompt(
#             select_file_prompt,
#             FileReadResponse,
#             llm_model=model_name_to_id[ModelName.fast_model],
#         )

#         if not file_selection_response.file:
#             return {"status": "No matching file found"}

#         file_path = os.path.join(scratch_pad_dir, file_selection_response.file)

#     # Read the content of the file
#     with open(file_path, "r") as f:
#         file_content = f.read()

#     # Get all memory content
#     memory_content = memory_manager.get_xml_for_prompt(["*"])

#     # Build the structured prompt to discuss the file content
#     discuss_file_prompt = f"""
# <purpose>
#     Discuss the content of the file based on the user's prompt and the current memory content.
# </purpose>

# <instructions>
#     <instruction>Based on the user's prompt, the file content, and the current memory content, provide a relevant discussion or analysis.</instruction>
#     <instruction>Be concise and focus on the aspects mentioned in the user's prompt.</instruction>
#     <instruction>Consider the current memory content when discussing the file, if relevant.</instruction>
#     <instruction>Keep responses short and concise. Keep response under 3 sentences for concise conversations.</instruction>
# </instructions>

# <file-content>
# {file_content}
# </file-content>

# {memory_content}

# <user-prompt>
# {prompt}
# </user-prompt>
#     """

#     # Call the LLM to discuss the file content
#     discussion = chat_prompt(discuss_file_prompt, model_name_to_id[model])

#     return {
#         "status": "File discussed",
#         "file_name": os.path.basename(file_path),
#         "discussion": discussion,
#     }
