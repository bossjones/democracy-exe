# @timeit_decorator
# async def delete_file(prompt: str, force_delete: bool = False) -> dict:
#     """
#     Delete a file based on the user's prompt.
#     """
#     scratch_pad_dir = os.getenv("SCRATCH_PAD_DIR", "./scratchpad")

#     # Ensure the scratch pad directory exists
#     os.makedirs(scratch_pad_dir, exist_ok=True)

#     # List available files in SCRATCH_PAD_DIR
#     available_files = os.listdir(scratch_pad_dir)
#     available_files_str = ", ".join(available_files)

#     # Build the structured prompt to select the file and determine 'force_delete' status
#     select_file_prompt = f"""
#     <purpose>
#         Select a file from the available files to delete.
#     </purpose>

#     <instructions>
#         <instruction>Based on the user's prompt and the list of available files, infer which file the user wants to delete.</instruction>
#         <instruction>If no file matches, return an empty string for 'file'.</instruction>
#     </instructions>

#     <available-files>
#         {available_files_str}
#     </available-files>

#     <user-prompt>
#         {prompt}
#     </user-prompt>
#     """

#     # Call the LLM to select the file and determine 'force_delete'
#     file_delete_response = structured_output_prompt(
#         select_file_prompt, FileDeleteResponse
#     )

#     # Check if a file was selected
#     if not file_delete_response.file:
#         result = {"status": "No matching file found"}
#     else:
#         selected_file = file_delete_response.file
#         file_path = os.path.join(scratch_pad_dir, selected_file)

#         # Check if the file exists
#         if not os.path.exists(file_path):
#             result = {"status": "File does not exist", "file_name": selected_file}
#         # If 'force_delete' is False, prompt for confirmation
#         elif not force_delete:
#             result = {
#                 "status": "Confirmation required",
#                 "file_name": selected_file,
#                 "message": f"Are you sure you want to delete '{selected_file}'? Say force delete if you want to delete.",
#             }
#         else:
#             # Proceed to delete the file
#             os.remove(file_path)
#             result = {"status": "File deleted", "file_name": selected_file}

#     return result
