"""
This type stub file was generated by pyright.
"""

from typing import List

class UnknownEditFormat(ValueError):
    def __init__(self, edit_format, valid_formats) -> None:
        ...
    


class MissingAPIKeyError(ValueError):
    ...


class FinishReasonLength(Exception):
    ...


def wrap_fence(name): # -> tuple[str, str]:
    ...

all_fences = ...
class Coder:
    abs_fnames = ...
    abs_read_only_fnames = ...
    repo = ...
    last_aider_commit_hash = ...
    aider_edited_files = ...
    last_asked_for_commit_time = ...
    repo_map = ...
    functions = ...
    num_exhausted_context_windows = ...
    num_malformed_responses = ...
    last_keyboard_interrupt = ...
    num_reflections = ...
    max_reflections = ...
    edit_format = ...
    yield_stream = ...
    temperature = ...
    auto_lint = ...
    auto_test = ...
    test_cmd = ...
    lint_outcome = ...
    test_outcome = ...
    multi_response_content = ...
    partial_response_content = ...
    commit_before_message = ...
    message_cost = ...
    message_tokens_sent = ...
    message_tokens_received = ...
    add_cache_headers = ...
    cache_warming_thread = ...
    num_cache_warming_pings = ...
    suggest_shell_commands = ...
    detect_urls = ...
    ignore_mentions = ...
    chat_language = ...
    file_watcher = ...
    @classmethod
    def create(self, main_model=..., edit_format=..., io=..., from_coder=..., summarize_from_coder=..., **kwargs):
        ...
    
    def clone(self, **kwargs):
        ...
    
    def get_announcements(self): # -> list[Any]:
        ...
    
    def __init__(self, main_model, io, repo=..., fnames=..., read_only_fnames=..., show_diffs=..., auto_commits=..., dirty_commits=..., dry_run=..., map_tokens=..., verbose=..., stream=..., use_git=..., cur_messages=..., done_messages=..., restore_chat_history=..., auto_lint=..., auto_test=..., lint_cmds=..., test_cmd=..., aider_commit_hashes=..., map_mul_no_files=..., commands=..., summarizer=..., total_cost=..., analytics=..., map_refresh=..., cache_prompts=..., num_cache_warming_pings=..., suggest_shell_commands=..., chat_language=..., detect_urls=..., ignore_mentions=..., file_watcher=..., auto_copy_context=...) -> None:
        ...
    
    def setup_lint_cmds(self, lint_cmds): # -> None:
        ...
    
    def show_announcements(self): # -> None:
        ...
    
    def add_rel_fname(self, rel_fname): # -> None:
        ...
    
    def drop_rel_fname(self, fname): # -> Literal[True] | None:
        ...
    
    def abs_root_path(self, path): # -> str:
        ...
    
    fences = ...
    fence = ...
    def show_pretty(self): # -> bool:
        ...
    
    def get_abs_fnames_content(self): # -> Generator[tuple[Any, str | Any], Any, None]:
        ...
    
    def choose_fence(self): # -> None:
        ...
    
    def get_files_content(self, fnames=...): # -> str:
        ...
    
    def get_read_only_files_content(self): # -> str:
        ...
    
    def get_cur_message_text(self): # -> Literal['']:
        ...
    
    def get_ident_mentions(self, text): # -> set[str | Any]:
        ...
    
    def get_ident_filename_matches(self, idents): # -> set[Any]:
        ...
    
    def get_repo_map(self, force_refresh=...): # -> str | LiteralString | None:
        ...
    
    def get_repo_messages(self): # -> list[Any]:
        ...
    
    def get_readonly_files_messages(self): # -> list[Any]:
        ...
    
    def get_chat_files_messages(self): # -> list[Any]:
        ...
    
    def get_images_message(self, fnames): # -> dict[str, Any] | None:
        ...
    
    def run_stream(self, user_message): # -> Generator[Any, Any, None]:
        ...
    
    def init_before_message(self): # -> None:
        ...
    
    def run(self, with_message=..., preproc=...): # -> str | None:
        ...
    
    def copy_context(self): # -> None:
        ...
    
    def get_input(self):
        ...
    
    def preproc_user_input(self, inp): # -> Any | List[str] | None:
        ...
    
    def run_one(self, user_message, preproc): # -> None:
        ...
    
    def check_and_open_urls(self, exc, friendly_msg=...): # -> list[Any]:
        """Check exception for URLs, offer to open in a browser, with user-friendly error msgs."""
        ...
    
    def check_for_urls(self, inp: str) -> List[str]:
        """Check input for URLs and offer to add them to the chat."""
        ...
    
    def keyboard_interrupt(self): # -> None:
        ...
    
    def summarize_start(self): # -> None:
        ...
    
    def summarize_worker(self): # -> None:
        ...
    
    def summarize_end(self): # -> None:
        ...
    
    def move_back_cur_messages(self, message): # -> None:
        ...
    
    def get_user_language(self): # -> str | None:
        ...
    
    def get_platform_info(self): # -> str:
        ...
    
    def fmt_system_prompt(self, prompt):
        ...
    
    def format_chat_chunks(self): # -> ChatChunks:
        ...
    
    def format_messages(self): # -> ChatChunks:
        ...
    
    def warm_cache(self, chunks): # -> None:
        ...
    
    def send_message(self, inp):
        ...
    
    def reply_completed(self): # -> None:
        ...
    
    def show_exhausted_error(self): # -> None:
        ...
    
    def lint_edited(self, fnames): # -> Literal['']:
        ...
    
    def update_cur_messages(self): # -> None:
        ...
    
    def get_file_mentions(self, content): # -> set[Any]:
        ...
    
    def check_for_file_mentions(self, content): # -> str | None:
        ...
    
    def send(self, messages, model=..., functions=...): # -> Generator[Any, Any, None]:
        ...
    
    def show_send_output(self, completion): # -> None:
        ...
    
    def show_send_output_stream(self, completion): # -> Generator[Any, Any, None]:
        ...
    
    def live_incremental_response(self, final): # -> None:
        ...
    
    def render_incremental_response(self, final): # -> str:
        ...
    
    def calculate_and_show_tokens_and_cost(self, messages, completion=...): # -> None:
        ...
    
    def show_usage_report(self): # -> None:
        ...
    
    def get_multi_response_content(self, final=...): # -> str:
        ...
    
    def get_rel_fname(self, fname): # -> str:
        ...
    
    def get_inchat_relative_files(self): # -> list[str | Any]:
        ...
    
    def is_file_safe(self, fname): # -> bool | None:
        ...
    
    def get_all_relative_files(self): # -> list[Any | str]:
        ...
    
    def get_all_abs_files(self): # -> list[Any | str]:
        ...
    
    def get_addable_relative_files(self): # -> set[Any | str]:
        ...
    
    def check_for_dirty_commit(self, path): # -> None:
        ...
    
    def allowed_to_edit(self, path): # -> Literal[True] | None:
        ...
    
    warning_given = ...
    def check_added_files(self): # -> None:
        ...
    
    def prepare_to_edit(self, edits): # -> list[Any]:
        ...
    
    def apply_updates(self): # -> set[Any]:
        ...
    
    def parse_partial_args(self): # -> Any | None:
        ...
    
    def get_context_from_history(self, history): # -> Literal['']:
        ...
    
    def auto_commit(self, edited, context=...): # -> None:
        ...
    
    def show_auto_commit_outcome(self, res): # -> None:
        ...
    
    def show_undo_hint(self): # -> None:
        ...
    
    def dirty_commit(self): # -> Literal[True] | None:
        ...
    
    def get_edits(self, mode=...): # -> list[Any]:
        ...
    
    def apply_edits(self, edits): # -> None:
        ...
    
    def apply_edits_dry_run(self, edits):
        ...
    
    def run_shell_commands(self): # -> str:
        ...
    
    def handle_shell_commands(self, commands_str, group): # -> str | None:
        ...
    

