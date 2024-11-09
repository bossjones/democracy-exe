# """files_import module tests"""

# from __future__ import annotations

# from pathlib import Path

# import pytest

# from democracy_exe.ai.collections import create_collection
# from democracy_exe.utils.files_import import index_file_folder


# @pytest.mark.skip(reason="This test is not yet implemented")
# @pytest.mark.flaky()
# def test_index_file_folder():
#     """Test index_file_folder function"""
#     create_collection("empty", {})
#     folder_path = f"{Path(__file__).parent.parent}/files/docs"
#     result = index_file_folder(file_path=folder_path, collection="empty")
#     assert result == 3


# @pytest.mark.skip(reason="This test is not yet implemented")
# @pytest.mark.flaky()
# def test_index_file_folder_fail():
#     """Test index_file_folder failure"""
#     with pytest.raises(FileNotFoundError) as exc:
#         index_file_folder(file_path="/not/found", collection="test")
#     assert str(exc.value) == "File /not/found does not exist."
