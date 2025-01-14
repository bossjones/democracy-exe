# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
# pylint: disable=no-name-in-module
# pyright: reportInvalidTypeForm=false
# pyright: reportUndefinedVariable=false

# disable nsupported-membership-test
# pylint: disable=unsupported-membership-test
# pylint: disable=unsubscriptable-object
# pylint: disable=no-member
"""test_settings"""

from __future__ import annotations

import asyncio
import datetime
import os

from collections.abc import Iterable, Iterator
from pathlib import Path, PosixPath
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import pytest_asyncio

from _pytest.monkeypatch import MonkeyPatch
from pydantic import (
    AliasChoices,
    AmqpDsn,
    BaseModel,
    Field,
    ImportString,
    PostgresDsn,
    RedisDsn,
    SecretBytes,
    SecretStr,
    ValidationError,
    field_serializer,
    model_validator,
)

import pytest

from democracy_exe import aio_settings
from democracy_exe.aio_settings import AioSettings, ModelConfigError, SecurityError
from democracy_exe.utils.file_functions import tilda


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch

IS_RUNNING_ON_GITHUB_ACTIONS = bool(os.environ.get("GITHUB_ACTOR"))


class TestMoreSettings:
    """Test settings configuration."""

    def test_api_key_defaults(self) -> None:
        """Test API key default values."""
        settings = AioSettings()

        # Check SecretStr fields
        assert isinstance(settings.tavily_api_key, SecretStr)
        assert isinstance(settings.brave_search_api_key, SecretStr)
        assert isinstance(settings.unstructured_api_key, SecretStr)

        # Check URL field
        assert settings.unstructured_api_url == "https://api.unstructured.io/general/v0/general"

    def test_langchain_integration_settings(self) -> None:
        """Test LangChain integration settings."""
        settings = AioSettings()

        # Check default values
        assert settings.langchain_endpoint == "https://api.smith.langchain.com"
        assert settings.langchain_hub_api_url == "https://api.hub.langchain.com"
        assert settings.langchain_project == "democracy_exe"

    def test_langchain_integration_env_override(self, monkeypatch: MonkeyPatch) -> None:
        """Test environment variable override for LangChain integration settings.

        Args:
            monkeypatch: Pytest fixture for modifying environment
        """
        # Set environment variables
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_LANGCHAIN_ENDPOINT", "https://custom.langchain.com")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_LANGCHAIN_HUB_API_URL", "https://custom.hub.langchain.com")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_LANGCHAIN_PROJECT", "custom_project")

        settings = AioSettings()

        # Verify environment overrides
        assert settings.langchain_endpoint == "https://custom.langchain.com"
        assert settings.langchain_hub_api_url == "https://custom.hub.langchain.com"
        assert settings.langchain_project == "custom_project"


# TODO: Make sure os,environ unsets values while running tests
@pytest.mark.unittest()
class TestSettings:
    def test_defaults(
        self,
    ) -> None:  # sourcery skip: extract-method
        """Test default settings."""
        test_settings = aio_settings.AioSettings()
        assert isinstance(test_settings.openai_api_key, SecretStr)
        assert test_settings.monitor_host == "localhost"
        assert test_settings.monitor_port == 50102
        assert test_settings.debug_langchain is True
        assert test_settings.audit_log_send_channel == ""
        assert isinstance(test_settings.enable_ai, bool)

    @pytest_asyncio.fixture
    async def test_integration_with_deleted_envs(self, monkeypatch: MonkeyPatch) -> None:
        # import bpdb
        # bpdb.set_trace()
        # paranoid about weird libraries trying to read env vars during testing
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_DISCORD_TOKEN", "fake_discord_token")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_DISCORD_TOKEN", "fake_discord_token")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_DISCORD_ADMIN_USER_ID", 1337)
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_DISCORD_SERVER_ID", 1337)
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_DISCORD_CLIENT_ID", 8008)
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_OPENAI_API_KEY", "fake_openai_key")
        monkeypatch.setenv("OPENAI_API_KEY", "fake_openai_key")
        monkeypatch.setenv("PINECONE_API_KEY", "fake_pinecone_key")
        monkeypatch.setenv("PINECONE_INDEX", "fake_test_index")
        await asyncio.sleep(0.05)

        test_settings: aio_settings.AioSettings = aio_settings.AioSettings()
        assert test_settings.discord_admin_user_id == 1337
        assert test_settings.discord_client_id == 8008
        assert test_settings.discord_server_id == 1337
        assert test_settings.discord_token == "fake_discord_token"
        assert test_settings.openai_api_key == "fake_openai_key"
        assert test_settings.pinecone_api_key == "fake_pinecone_key"
        assert test_settings.pinecone_index == "fake_test_index"

    def test_postgres_defaults(self):
        test_settings = aio_settings.AioSettings()
        assert test_settings.postgres_host == "localhost"
        assert test_settings.postgres_port == 8432
        assert test_settings.postgres_password == "langchain"
        assert test_settings.postgres_driver == "psycopg"
        assert test_settings.postgres_database == "langchain"
        assert test_settings.postgres_collection_name == "langchain"
        assert test_settings.postgres_user == "langchain"
        assert test_settings.enable_postgres == True

    def test_postgres_url(self):
        test_settings = aio_settings.AioSettings()
        expected_url = "postgresql+psycopg://langchain:langchain@localhost:8432/langchain"
        assert test_settings.postgres_url == expected_url

    @pytest.mark.parametrize(
        "host,port,user,password,driver,database,expected",
        [
            (
                "testhost",
                5432,
                "testuser",
                "testpass",
                "psycopg",
                "testdb",
                "postgresql+psycopg://testuser:testpass@testhost:5432/testdb",
            ),
            (
                "127.0.0.1",
                5433,
                "admin",
                "securepass",
                "psycopg",
                "production",
                "postgresql+psycopg://admin:securepass@127.0.0.1:5433/production",
            ),
        ],
    )
    def test_custom_postgres_url(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        driver: str,
        database: str,
        expected: str,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """Test custom PostgreSQL URL construction.

        Args:
            host: Database host
            port: Database port
            user: Database user
            password: Database password
            driver: Database driver
            database: Database name
            expected: Expected URL string
        """
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_POSTGRES_HOST", host)
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_POSTGRES_PORT", str(port))
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_POSTGRES_USER", user)
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_POSTGRES_PASSWORD", password)
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_POSTGRES_DRIVER", driver)
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_POSTGRES_DATABASE", database)
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_ENABLE_POSTGRES", "false")

        custom_settings = aio_settings.AioSettings(
            postgres_host=host,
            postgres_port=port,
            postgres_user=user,
            postgres_pass=SecretStr(password),
            postgres_driver=driver,
            postgres_db=database,
        )
        # import bpdb

        # bpdb.set_trace()
        # reload settings
        custom_settings.__init__()
        assert str(custom_settings.postgres_url) == expected

    @pytest.mark.asyncio()
    async def test_postgres_env_variables(self, monkeypatch: MonkeyPatch):
        test_settings = aio_settings.AioSettings()

        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_POSTGRES_HOST", "envhost")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_POSTGRES_PORT", "5555")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_POSTGRES_USER", "envuser")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_POSTGRES_PASSWORD", "envpass")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_POSTGRES_DRIVER", "envdriver")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_POSTGRES_DATABASE", "envdb")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_ENABLE_POSTGRES", "false")

        # reload settings
        test_settings.__init__()

        assert test_settings.postgres_host == "envhost"
        assert test_settings.postgres_port == 5555
        assert test_settings.postgres_user == "envuser"
        assert test_settings.postgres_password == "envpass"
        assert test_settings.postgres_driver == "envdriver"
        assert test_settings.postgres_database == "envdb"
        assert test_settings.enable_postgres == False

        expected_url = "postgresql+envdriver://envuser:envpass@envhost:5555/envdb"
        assert test_settings.postgres_url == expected_url

    # -------

    def test_redis_defaults(self):
        test_settings = aio_settings.AioSettings()
        assert test_settings.redis_host == "localhost"
        assert test_settings.redis_port == 8600
        assert test_settings.redis_user is None
        assert test_settings.redis_pass is None
        assert test_settings.redis_base is None
        assert isinstance(test_settings.enable_redis, bool)

    def test_redis_url(self) -> None:
        """Test Redis URL construction."""
        test_settings = aio_settings.AioSettings()
        expected_url = "redis://localhost:8600"  # Updated to include database number
        assert str(test_settings.redis_url) == expected_url

    @pytest.mark.parametrize(
        "host,port,user,password,base,expected",
        [
            (
                "testhost",
                6379,
                "testuser",
                "testpass",
                0,
                "redis://testuser:testpass@testhost:6379/0",
            ),
            (
                "127.0.0.1",
                6380,
                None,
                None,
                1,
                "redis://127.0.0.1:6380/1",
            ),
        ],
    )
    def test_custom_redis_url(
        self,
        host: str,
        port: int,
        user: str | None,
        password: str | None,
        base: int,
        expected: str,
        monkeypatch: MonkeyPatch,
    ):
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_ENABLE_REDIS", "false")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_REDIS_HOST", host)
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_REDIS_PORT", str(port))
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_REDIS_USER", user)
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_REDIS_PASS", password)
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_REDIS_BASE", str(base))

        custom_settings = aio_settings.AioSettings(
            redis_host=host,
            redis_port=port,
            redis_user=user,
            redis_pass=password,
            redis_base=base,
        )
        assert str(custom_settings.redis_url) == expected

    @pytest.mark.asyncio()
    async def test_redis_env_variables(self, monkeypatch: MonkeyPatch) -> None:
        """Test Redis settings from environment variables."""
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_REDIS_HOST", "envhost")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_REDIS_PORT", "7777")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_REDIS_USER", "envuser")
        monkeypatch.setenv(
            "DEMOCRACY_EXE_CONFIG_REDIS_PASS", "envpassword123"
        )  # Longer password that meets requirements
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_REDIS_BASE", "2")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_ENABLE_REDIS", "true")

        test_settings = aio_settings.AioSettings()
        assert test_settings.redis_host == "envhost"
        assert test_settings.redis_port == 7777
        assert test_settings.redis_user == "envuser"
        assert test_settings.redis_pass.get_secret_value() == "envpassword123"
        assert test_settings.redis_base == 2
        assert test_settings.enable_redis is True

        expected_url = "redis://envuser:envpassword123@envhost:7777/2"
        assert str(test_settings.redis_url) == expected_url

    def test_model_settings(self):
        test_settings = aio_settings.AioSettings()
        assert isinstance(test_settings.discord_admin_user_id, int)
        assert isinstance(test_settings.discord_admin_user_invited, bool)
        assert isinstance(test_settings.discord_client_id, str)
        assert isinstance(test_settings.discord_general_channel, int)
        assert isinstance(test_settings.discord_server_id, int)
        assert isinstance(test_settings.discord_token, SecretStr)
        assert isinstance(test_settings.enable_ai, bool)
        assert isinstance(test_settings.enable_chroma, bool)
        assert isinstance(test_settings.enable_postgres, bool)
        assert isinstance(test_settings.enable_redis, bool)
        assert isinstance(test_settings.enable_sentry, bool)
        assert isinstance(test_settings.experimental_redis_memory, bool)
        assert isinstance(test_settings.oco_openai_api_key, SecretStr)
        assert isinstance(test_settings.openai_api_key, SecretStr)
        assert isinstance(test_settings.pinecone_api_key, SecretStr)
        assert isinstance(test_settings.rag_answer_accuracy_feature_flag, bool)
        assert isinstance(test_settings.rag_answer_hallucination_feature_flag, bool)
        assert isinstance(test_settings.rag_answer_v_reference_feature_flag, bool)
        assert isinstance(test_settings.rag_doc_relevance_and_hallucination_feature_flag, bool)
        assert isinstance(test_settings.rag_doc_relevance_feature_flag, bool)
        assert isinstance(test_settings.rag_string_embedding_distance_metrics_feature_flag, bool)
        assert test_settings.chat_history_buffer == 10
        assert test_settings.chat_model == "gpt-4o-mini"
        assert test_settings.editor in ["lvim", "vim", "nvim"]
        assert test_settings.eval_max_concurrency == 4
        assert test_settings.git_editor in ["lvim", "vim", "nvim"]
        assert test_settings.globals_always_use_cpu == False
        assert test_settings.globals_ckpt_convert == False
        assert test_settings.globals_full_precision == False
        assert test_settings.globals_internet_available == True
        assert test_settings.globals_log_tokenization == False
        assert test_settings.globals_try_patchmatch == True
        assert str(test_settings.groq_api_key) in [
            "**********",
            "",
            None,
            SecretStr(""),
            SecretStr("**********"),
            "None",
        ]
        assert test_settings.helpfulness_feature_flag == False
        assert test_settings.helpfulness_testing_feature_flag == False
        assert test_settings.http_client_debug_enabled == False
        assert str(test_settings.langchain_api_key) in [
            "**********",
            "",
            None,
            SecretStr(""),
            SecretStr("**********"),
            "None",
        ]
        assert test_settings.langchain_debug_logs == False
        assert test_settings.langchain_endpoint == "https://api.smith.langchain.com"
        assert str(test_settings.langchain_hub_api_key) in [
            "**********",
            "",
            None,
            SecretStr(""),
            SecretStr("**********"),
        ]
        assert test_settings.langchain_hub_api_url == "https://api.hub.langchain.com"
        assert test_settings.langchain_project == "democracy_exe"
        assert isinstance(test_settings.langchain_tracing_v2, bool)
        assert test_settings.llm_embedding_model_name == "text-embedding-3-large"
        assert test_settings.llm_model_name == "gpt-4o-mini"
        assert test_settings.llm_temperature == 0.0
        assert isinstance(test_settings.local_test_debug, bool)
        assert isinstance(test_settings.local_test_enable_evals, bool)
        assert isinstance(test_settings.log_pii, bool)
        assert test_settings.max_retries == 9
        assert test_settings.max_tokens == 900
        assert test_settings.monitor_host == "localhost"
        assert test_settings.monitor_port == 50102
        assert test_settings.oco_ai_provider == "openai"
        assert test_settings.oco_language == "en"
        assert test_settings.oco_model == "gpt-4o"
        assert test_settings.oco_prompt_module == "conventional-commit"
        assert test_settings.oco_tokens_max_input == 4096
        assert test_settings.oco_tokens_max_output == 500
        assert test_settings.openai_embeddings_model == "text-embedding-3-large"
        assert test_settings.pinecone_env in ["us-east-1", "local"]
        assert test_settings.postgres_collection_name == "langchain"
        assert test_settings.postgres_database == "langchain"
        assert test_settings.postgres_driver == "psycopg"
        assert test_settings.postgres_host == "localhost"
        assert test_settings.postgres_password == "langchain"
        assert test_settings.postgres_port == 8432
        assert test_settings.postgres_url == "postgresql+psycopg://langchain:langchain@localhost:8432/langchain"
        assert test_settings.postgres_user == "langchain"
        assert test_settings.prefix == "?"
        assert test_settings.provider == "openai"
        assert isinstance(test_settings.python_debug, bool)
        assert isinstance(test_settings.pythonasynciodebug, bool)
        assert isinstance(test_settings.pythondevmode, bool)
        assert test_settings.question_to_ask == "What is the main cause of climate change?"
        assert test_settings.redis_host == "localhost"
        assert str(test_settings.redis_pass) in ["**********", "", None, SecretStr(""), SecretStr("**********"), "None"]
        assert test_settings.redis_port == 8600
        assert str(test_settings.redis_url) == "redis://localhost:8600"
        assert test_settings.redis_user is None
        assert test_settings.retry_stop_after_attempt == 3
        assert test_settings.retry_wait_exponential_max == 5
        assert test_settings.retry_wait_exponential_min == 1
        assert test_settings.retry_wait_exponential_multiplier == 2
        assert test_settings.retry_wait_fixed == 15
        assert str(test_settings.sentry_dsn) in ["**********", "", None, SecretStr(""), SecretStr("**********"), "None"]
        assert str(test_settings.tavily_api_key) in [
            "**********",
            "",
            None,
            SecretStr(""),
            SecretStr("**********"),
            "None",
        ]
        assert str(test_settings.unstructured_api_key) in [
            "**********",
            "",
            None,
            SecretStr(""),
            SecretStr("**********"),
        ]
        assert test_settings.unstructured_api_url == "https://api.unstructured.io/general/v0/general"
        assert test_settings.vision_model == "gpt-4o"

    def test_model_zoo(self):
        test_settings = aio_settings.AioSettings()
        assert isinstance(test_settings.MODEL_ZOO, set)
        assert "gpt-4o" in test_settings.MODEL_ZOO
        assert "text-embedding-3-large" in test_settings.MODEL_ZOO
        assert "claude-3-opus" in test_settings.MODEL_ZOO

    def test_model_config(self):
        test_settings = aio_settings.AioSettings()
        assert isinstance(test_settings.MODEL_CONFIG, dict)

        # Test older models
        assert "gpt-4-0613" in test_settings.MODEL_CONFIG
        assert test_settings.MODEL_CONFIG["gpt-4-0613"]["max_tokens"] == 8192

        # Test newer models
        assert "gpt-4o-mini-2024-07-18" in test_settings.MODEL_CONFIG
        assert test_settings.MODEL_CONFIG["gpt-4o-mini-2024-07-18"]["max_tokens"] == 900

        # Test Claude models
        assert "claude-3-opus-20240229" in test_settings.MODEL_CONFIG
        assert test_settings.MODEL_CONFIG["claude-3-opus-20240229"]["max_tokens"] == 2048

    def test_model_point(self):
        test_settings = aio_settings.AioSettings()
        assert isinstance(test_settings.MODEL_POINT, dict)
        assert "gpt-4o" in test_settings.MODEL_POINT
        assert test_settings.MODEL_POINT["gpt-4o"] == "gpt-4o-2024-08-06"
        assert test_settings.MODEL_POINT["claude-3-opus"] == "claude-3-opus-20240229"

    def test_embedding_config(self):
        test_settings = aio_settings.AioSettings()
        assert isinstance(test_settings.EMBEDDING_CONFIG, dict)

        # Test older embeddings
        assert "text-embedding-ada-002" in test_settings.EMBEDDING_CONFIG
        assert test_settings.EMBEDDING_CONFIG["text-embedding-ada-002"]["max_tokens"] == 8191

        # Test newer embeddings
        assert "text-embedding-3-large" in test_settings.EMBEDDING_CONFIG
        assert test_settings.EMBEDDING_CONFIG["text-embedding-3-large"]["max_tokens"] == 8191

    def test_embedding_model_dimensions(self):
        test_settings = aio_settings.AioSettings()
        assert isinstance(test_settings.EMBEDDING_MODEL_DIMENSIONS_DATA, dict)
        assert "text-embedding-3-large" in test_settings.EMBEDDING_MODEL_DIMENSIONS_DATA
        assert test_settings.EMBEDDING_MODEL_DIMENSIONS_DATA["text-embedding-3-large"] == 1024
        assert test_settings.EMBEDDING_MODEL_DIMENSIONS_DATA["text-embedding-ada-002"] == 1536

    def test_feature_flags(self) -> None:
        """Test feature flags."""
        test_settings = aio_settings.AioSettings()
        assert isinstance(test_settings.helpfulness_feature_flag, bool)
        assert isinstance(test_settings.rag_answer_accuracy_feature_flag, bool)
        assert isinstance(test_settings.rag_answer_v_reference_feature_flag, bool)
        assert isinstance(test_settings.compare_models_feature_flag, bool)
        assert isinstance(test_settings.document_relevance_feature_flag, bool)

    @pytest.mark.parametrize(
        "model_name,expected_tokens,expected_output_tokens",
        [
            ("gpt-4o-mini-2024-07-18", 900, 16384),
            ("gpt-4o-2024-08-06", 128000, 16384),
            ("claude-3-opus-20240229", 2048, 16384),
            ("gpt-4", 8192, 4096),
        ],
    )
    def test_model_token_limits(
        self,
        model_name: str,
        expected_tokens: int,
        expected_output_tokens: int,
    ) -> None:
        """Test model token limits.

        Args:
            model_name: Name of the model to test
            expected_tokens: Expected token limit
            expected_output_tokens: Expected output token limit
        """
        test_settings = aio_settings.AioSettings(llm_model_name=model_name)
        assert test_settings.llm_max_tokens == expected_tokens
        assert test_settings.llm_max_output_tokens == expected_output_tokens

    def test_redis_password_validation(self) -> None:
        """Test Redis password validation."""
        # Test with short password (should fail)
        with pytest.raises(SecurityError) as exc_info:
            test_settings = aio_settings.AioSettings(redis_pass=SecretStr("short"))
        assert "Redis password must be at least 8 characters" in str(exc_info.value)

        # Test with valid password (should pass)
        test_settings = aio_settings.AioSettings(redis_pass=SecretStr("validpassword123"))
        assert test_settings.redis_pass.get_secret_value() == "validpassword123"

    @pytest.mark.skip_until(
        deadline=datetime.datetime(2025, 1, 25),
        strict=True,
        msg="Need to find a good url to test this with, will do later",
    )
    @pytest.mark.parametrize(
        "env_vars,expected_values",
        [
            (
                {
                    "LLM_STREAMING": "True",
                    "LLM_PROVIDER": "anthropic",
                    "LLM_MAX_RETRIES": "5",
                    "LLM_DOCUMENT_LOADER_TYPE": "unstructured",
                    "LLM_VECTORSTORE_TYPE": "pinecone",
                    "LLM_EMBEDDING_MODEL_TYPE": "text-embedding-ada-002",
                },
                {
                    "llm_streaming": True,
                    "llm_provider": "anthropic",
                    "llm_max_retries": 5,
                    "llm_document_loader_type": "unstructured",
                    "llm_vectorstore_type": "pinecone",
                    "llm_embedding_model_type": "text-embedding-ada-002",
                },
            ),
            (
                {},
                {
                    "llm_streaming": False,
                    "llm_provider": "openai",
                    "llm_max_retries": 3,
                    "llm_document_loader_type": "pymupdf",
                    "llm_vectorstore_type": "pgvector",
                    "llm_embedding_model_type": "text-embedding-3-large",
                },
            ),
        ],
    )
    def test_llm_settings_from_env(
        self, monkeypatch: MonkeyPatch, env_vars: dict[str, str], expected_values: dict[str, object]
    ) -> None:
        """Test LLM settings loaded from environment variables."""
        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)

        settings = aio_settings.AioSettings()
        for key, expected_value in expected_values.items():
            assert getattr(settings, key) == expected_value

    @pytest.mark.skip_until(
        deadline=datetime.datetime(2025, 1, 25),
        strict=True,
        msg="Need to find a good url to test this with, will do later",
    )
    def test_model_validation(self):
        """Test model validation for invalid models."""
        with pytest.raises(ModelConfigError) as exc_info:
            aio_settings.AioSettings(llm_model_name="invalid-model")
        assert "Invalid model name" in str(exc_info.value)
        assert "available_models" in exc_info.value.context

        with pytest.raises(ModelConfigError) as exc_info:
            aio_settings.AioSettings(llm_embedding_model_name="invalid-embedding")
        assert "Invalid embedding model name" in str(exc_info.value)
        assert "available_models" in exc_info.value.context

    @pytest.mark.parametrize(
        "temperature,valid",
        [
            # (-0.1, False),
            (0.0, True),
            (0.7, True),
            (1.0, True),
            (2.0, True),
            # (2.1, False),
        ],
    )
    def test_temperature_validation(self, temperature: float, valid: bool):
        """Test temperature validation with various values."""
        if valid:
            settings = aio_settings.AioSettings(llm_temperature=temperature)
            assert settings.llm_temperature == temperature
        else:
            with pytest.raises(ValidationError) as exc_info:
                aio_settings.AioSettings(llm_temperature=temperature)
            assert "Temperature must be between 0.0 and 2.0" in str(exc_info.value)

    @pytest.mark.skip_until(
        deadline=datetime.datetime(2025, 1, 25),
        strict=True,
        msg="Need to find a good url to test this with, will do later",
    )
    def test_retry_delay_validation(self):
        """Test retry delay validation."""
        # Test negative delay
        with pytest.raises(ValidationError) as exc_info:
            aio_settings.AioSettings(llm_retry_delay=-1)
        assert "Retry delay must be non-negative" in str(exc_info.value)

        # Test max delay less than initial delay
        with pytest.raises(ValidationError) as exc_info:
            aio_settings.AioSettings(llm_retry_delay=5, llm_retry_max_delay=3)
        assert "Maximum retry delay must be greater than initial delay" in str(exc_info.value)

        # Test valid delays
        settings = aio_settings.AioSettings(llm_retry_delay=1, llm_retry_max_delay=5)
        assert settings.llm_retry_delay == 1
        assert settings.llm_retry_max_delay == 5

    @pytest.mark.skip_until(
        deadline=datetime.datetime(2025, 1, 25),
        strict=True,
        msg="Need to find a good url to test this with, will do later",
    )
    def test_model_token_updates(self) -> None:
        """Test model token updates."""
        settings = aio_settings.AioSettings(llm_model_name="gpt-4o-mini-2024-07-18")
        assert settings.llm_max_tokens == 900
        assert settings.llm_max_output_tokens == 16384

        settings.llm_model_name = "gpt-4o-2024-08-06"
        assert settings.llm_max_tokens == 128000
        assert settings.llm_max_output_tokens == 16384

        settings.llm_model_name = "claude-3-opus-20240229"
        assert settings.llm_max_tokens == 2048
        assert settings.llm_max_output_tokens == 16384

    @pytest.mark.skip_until(
        deadline=datetime.datetime(2025, 1, 25),
        strict=True,
        msg="Need to find a good url to test this with, will do later",
    )
    def test_embedding_dimension_updates(self):
        """Test automatic embedding dimension updates."""
        settings = aio_settings.AioSettings(llm_embedding_model_name="text-embedding-3-large")
        assert settings.embedding_model_dimensions == 1024

        settings = aio_settings.AioSettings(llm_embedding_model_name="text-embedding-ada-002")
        assert settings.embedding_model_dimensions == 1536

    @pytest.mark.skip_until(
        deadline=datetime.datetime(2025, 1, 25),
        strict=True,
        msg="Need to find a good url to test this with, will do later",
    )
    def test_debug_settings(self, monkeypatch: MonkeyPatch) -> None:
        """Test debug and development settings."""
        monkeypatch.setenv("DEBUG_AIDER", "false")
        monkeypatch.setenv("DEBUG_LANGGRAPH_STUDIO", "false")
        monkeypatch.setenv("PYTHONFAULTHANDLER", "false")
        monkeypatch.setenv("PYTHONASYNCIODEBUG", "false")
        monkeypatch.setenv("PYTHONDEVMODE", "false")
        monkeypatch.setenv("LANGCHAIN_DEBUG_LOGS", "false")

        settings = aio_settings.AioSettings()

        # Check debug settings
        assert isinstance(settings.debug_aider, bool)
        assert settings.debug_aider is False

        assert isinstance(settings.debug_langgraph_studio, bool)
        assert settings.debug_langgraph_studio is False

        assert isinstance(settings.python_fault_handler, bool)
        assert settings.python_fault_handler is False

        assert isinstance(settings.pythonasynciodebug, bool)
        assert settings.pythonasynciodebug is False

        assert isinstance(settings.pythondevmode, bool)
        assert settings.pythondevmode is False

        assert isinstance(settings.langchain_debug_logs, bool)
        assert settings.langchain_debug_logs is False

    @pytest.mark.skip_until(
        deadline=datetime.datetime(2025, 1, 25),
        strict=True,
        msg="Need to find a good url to test this with, will do later",
    )
    def test_debug_settings_env_override(self, monkeypatch: MonkeyPatch) -> None:
        """Test environment variable override for debug settings.

        Args:
            monkeypatch: Pytest fixture for modifying environment
        """
        monkeypatch.setenv("DEBUG_AIDER", "true")
        monkeypatch.setenv("DEBUG_LANGGRAPH_STUDIO", "true")
        monkeypatch.setenv("PYTHONFAULTHANDLER", "true")

        settings = aio_settings.AioSettings()
        assert settings.debug_aider is True
        assert settings.debug_langgraph_studio is True
        assert settings.python_fault_handler is True

    @pytest.mark.skip_until(
        deadline=datetime.datetime(2025, 1, 25),
        strict=True,
        msg="Need to find a good url to test this with, will do later",
    )
    def test_editor_settings(self, monkeypatch: MonkeyPatch) -> None:
        """Test editor settings."""
        monkeypatch.setenv("EDITOR", "vim")
        monkeypatch.setenv("VISUAL", "vim")
        monkeypatch.setenv("GIT_EDITOR", "vim")

        settings = aio_settings.AioSettings()

        # Check editor settings
        assert settings.editor == "vim"
        assert settings.visual == "vim"
        assert settings.git_editor == "vim"

    @pytest.mark.skip_until(
        deadline=datetime.datetime(2025, 1, 25),
        strict=True,
        msg="Need to find a good url to test this with, will do later",
    )
    def test_editor_settings_env_override(self, monkeypatch: MonkeyPatch) -> None:
        """Test environment variable override for editor settings.

        Args:
            monkeypatch: Pytest fixture for modifying environment
        """
        monkeypatch.setenv("EDITOR", "nvim")
        monkeypatch.setenv("VISUAL", "nvim")
        monkeypatch.setenv("GIT_EDITOR", "nvim")

        settings = aio_settings.AioSettings()
        assert settings.editor == "nvim"
        assert settings.visual == "nvim"
        assert settings.git_editor == "nvim"

    @pytest.mark.skip_until(
        deadline=datetime.datetime(2025, 1, 25),
        strict=True,
        msg="Need to find a good url to test this with, will do later",
    )
    def test_tweetpik_settings(self) -> None:
        """Test TweetPik settings."""
        settings = aio_settings.AioSettings()

        # Check TweetPik settings
        assert isinstance(settings.tweetpik_api_key, SecretStr)
        assert isinstance(settings.tweetpik_authorization, SecretStr)

        assert settings.tweetpik_bucket_id == "323251495115948625"
        assert settings.tweetpik_theme == "dim"
        assert settings.tweetpik_dimension == "instagramFeed"

    @pytest.mark.skip_until(
        deadline=datetime.datetime(2025, 1, 25),
        strict=True,
        msg="Need to find a good url to test this with, will do later",
    )
    def test_tweetpik_settings_env_override(self, monkeypatch: MonkeyPatch) -> None:
        """Test environment variable override for TweetPik settings.

        Args:
            monkeypatch: Pytest fixture for modifying environment
        """
        monkeypatch.setenv("TWEETPIK_API_KEY", "test-api-key")
        monkeypatch.setenv("TWEETPIK_AUTHORIZATION", "test-auth")
        monkeypatch.setenv("TWEETPIK_BUCKET_ID", "test-bucket")
        monkeypatch.setenv("TWEETPIK_THEME", "light")
        monkeypatch.setenv("TWEETPIK_DIMENSION", "custom")

        settings = aio_settings.AioSettings()
        assert settings.tweetpik_api_key.get_secret_value() == "test-api-key"
        assert settings.tweetpik_authorization.get_secret_value() == "test-auth"
        assert settings.tweetpik_bucket_id == "test-bucket"
        assert settings.tweetpik_theme == "light"
        assert settings.tweetpik_dimension == "custom"

    def test_tool_settings(self) -> None:
        """Test tool settings."""
        settings = aio_settings.AioSettings()

        # Check tool settings
        assert isinstance(settings.tool_allowlist, list)
        assert "tavily_search" in settings.tool_allowlist
        assert "magic_function" in settings.tool_allowlist

        assert isinstance(settings.extension_allowlist, list)
        assert "democracy_exe.chatbot.cogs.twitter" in settings.extension_allowlist

        assert settings.tavily_search_max_results == 3

    def test_agent_settings(self) -> None:
        """Test agent settings."""
        settings = aio_settings.AioSettings()

        # Check agent settings
        assert settings.agent_type == "adaptive_rag"
        assert settings.agent_type in ["plan_and_execute", "basic", "advanced", "adaptive_rag"]

    # def test_agent_settings_env_override(self, monkeypatch: MonkeyPatch) -> None:
    #     """Test environment variable override for agent settings.

    #     Args:
    #         monkeypatch: Pytest fixture for modifying environment
    #     """
    #     monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_AGENT_TYPE", "basic")
    #     settings = aio_settings.AioSettings()
    #     assert settings.agent_type == "basic"

    # def test_memory_settings(self) -> None:
    #     """Test memory settings."""
    #     settings = aio_settings.AioSettings()

    #     # Check memory settings
    #     assert settings.llm_memory_type == "memorysaver"
    #     assert settings.llm_memory_enabled is True
    #     assert settings.llm_human_loop_enabled is False

    @pytest.mark.skip_until(
        deadline=datetime.datetime(2025, 1, 25),
        strict=True,
        msg="Need to find a good url to test this with, will do later",
    )
    def test_memory_settings_env_override(self, monkeypatch: MonkeyPatch) -> None:
        """Test environment variable override for memory settings.

        Args:
            monkeypatch: Pytest fixture for modifying environment
        """
        monkeypatch.setenv("LLM_MEMORY_TYPE", "custom")
        monkeypatch.setenv("LLM_MEMORY_ENABLED", "false")
        monkeypatch.setenv("LLM_HUMAN_LOOP_ENABLED", "true")

        settings = aio_settings.AioSettings()
        assert settings.llm_memory_type == "custom"
        assert settings.llm_memory_enabled is False
        assert settings.llm_human_loop_enabled is True

    def test_text_processing_settings(self) -> None:
        """Test text processing settings."""
        settings = aio_settings.AioSettings()

        # Check text processing settings
        assert settings.text_chunk_size == 2000
        assert settings.text_chunk_overlap == 200
        assert settings.text_splitter == "{}"

    @pytest.mark.skip_until(
        deadline=datetime.datetime(2025, 1, 25),
        strict=True,
        msg="Need to find a good url to test this with, will do later",
    )
    def test_qa_settings(self, monkeypatch: MonkeyPatch) -> None:
        """Test QA and summarization settings."""
        # monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_LLM_MEMORY_TYPE", "custom")
        monkeypatch.setenv("LLM_MEMORY_ENABLED", "false")
        # monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_LLM_HUMAN_LOOP_ENABLED", "true")
        monkeypatch.setenv("LLM_HUMAN_LOOP_ENABLED", "true")

        settings = aio_settings.AioSettings()

        # Check QA settings
        assert isinstance(settings.qa_completion_llm, dict)
        assert settings.qa_completion_llm.get("_type") == "openai-chat"
        assert settings.qa_completion_llm.get("model_name") == "gpt-4o-mini"
        assert settings.qa_completion_llm.get("temperature") == 0
        assert settings.qa_completion_llm.get("max_tokens") == 1000
        assert settings.qa_completion_llm.get("verbose") is True

        assert isinstance(settings.qa_followup_llm, dict)
        assert settings.qa_followup_llm.get("_type") == "openai-chat"
        assert settings.qa_followup_llm.get("model_name") == "gpt-4o-mini"
        assert settings.qa_followup_llm.get("max_tokens") == 200

        assert isinstance(settings.summarize_llm, dict)
        assert settings.summarize_llm.get("_type") == "openai-chat"
        assert settings.summarize_llm.get("model_name") == "gpt-4o"
        assert settings.summarize_llm.get("max_tokens") == 2000

    # ... rest of existing tests ...
