from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os.path
import sys
import textwrap
import typing

from _asyncio import Future
from asyncio.tasks import _GatheringFuture
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime
from enum import Enum
from functools import partial
from pathlib import Path, PosixPath
from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Literal, Optional, Tuple, Type, Union

import pytest

from pytest_mock.plugin import MockerFixture

from democracy_exe.utils import vidops


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture()
def sample_video() -> PosixPath:
    return Path("tests/fixtures/song.mp4")


def test_calculate_bitrate():
    """Test the calculate_bitrate function."""
    assert vidops.calculate_bitrate(60, 10) == 1333
    assert vidops.calculate_bitrate(120, 5) == 333
    assert vidops.calculate_bitrate(30, 20) == 5333


@pytest.mark.asyncio()
async def test_duration_video(
    sample_video: PosixPath, tmp_path: PosixPath, mocker: MockerFixture
) -> typing.Iterator[Future | _GatheringFuture]:
    """Test the process_video function."""
    duration: float = await vidops.get_duration(sample_video)
    assert duration == 36.133333


@pytest.mark.asyncio()
async def test_process_video(
    sample_video: PosixPath, tmp_path: PosixPath, mocker: MockerFixture
) -> typing.Iterator[Future | _GatheringFuture]:
    """Test the process_video function."""
    await vidops.process_video(sample_video)


@pytest.mark.asyncio()
async def test_process_audio(
    sample_video: PosixPath, tmp_path: PosixPath, mocker: MockerFixture
) -> typing.Iterator[Future | _GatheringFuture]:
    """Test the process_audio function."""
    await vidops.process_audio(sample_video)


@pytest.mark.asyncio()
async def test_process_video_low_bitrate(
    sample_video: PosixPath, tmp_path: PosixPath, mocker: MockerFixture
) -> typing.Iterator[Future | _GatheringFuture]:
    """Test the process_video function with a low bitrate scenario."""
    await vidops.process_video(sample_video)


@pytest.mark.asyncio()
async def test_process_audio_low_bitrate(
    sample_video: PosixPath, tmp_path: PosixPath, mocker: MockerFixture
) -> typing.Iterator[Future | _GatheringFuture]:
    """Test the process_audio function with a low bitrate scenario."""
    await vidops.process_audio(sample_video)
