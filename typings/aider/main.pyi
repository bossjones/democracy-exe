"""
This type stub file was generated by pyright.
"""

def check_config_files_for_yes(config_files): # -> bool:
    ...

def get_git_root(): # -> git.types.PathLike | None:
    """Try and guess the git repo, since the conf.yml can be at the repo root"""
    ...

def guessed_wrong_repo(io, git_root, fnames, git_dname): # -> str | None:
    """After we parse the args, we can determine the real repo. Did we guess wrong?"""
    ...

def make_new_repo(git_root, io): # -> Repo | None:
    ...

def setup_git(git_root, io): # -> git.types.PathLike | None:
    ...

def check_gitignore(git_root, io, ask=...):
    ...

def check_streamlit_install(io): # -> Literal[True] | None:
    ...

def write_streamlit_credentials(): # -> None:
    ...

def launch_gui(args): # -> None:
    ...

def parse_lint_cmds(lint_cmds, io): # -> dict[Any, Any] | None:
    ...

def generate_search_path_list(default_file, git_root, command_line_file): # -> list[str]:
    ...

def register_models(git_root, model_settings_fname, io, verbose=...): # -> Literal[1] | None:
    ...

def load_dotenv_files(git_root, dotenv_fname, encoding=...): # -> list[Any]:
    ...

def register_litellm_models(git_root, model_metadata_fname, io, verbose=...): # -> Literal[1] | None:
    ...

def sanity_check_repo(repo, io): # -> bool:
    ...

def main(argv=..., input=..., output=..., force_git_root=..., return_coder=...):
    ...

def is_first_run_of_new_version(io, verbose=...): # -> bool:
    """Check if this is the first run of a new version/executable combination"""
    ...

def check_and_load_imports(io, is_first_run, verbose=...): # -> None:
    ...

def load_slow_imports(swallow=...): # -> None:
    ...

if __name__ == "__main__":
    status = ...
