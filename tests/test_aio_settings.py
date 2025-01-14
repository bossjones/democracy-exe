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


# TODO: Make sure os,environ unsets values while running tests
@pytest.mark.unittest()
class TestSettings:
    def test_defaults(
        self,
    ) -> None:  # sourcery skip: extract-method
        """Test default settings."""
        test_settings = aio_settings.AioSettings()
        assert isinstance(test_settings.openai_api_key, SecretStr)
        assert test_settings.openai_api_key.get_secret_value() != ""
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
        custom_settings = aio_settings.AioSettings(
            postgres_host=host,
            postgres_port=port,
            postgres_user=user,
            postgres_pass=SecretStr(password),
            postgres_driver=driver,
            postgres_db=database,
        )
        assert str(custom_settings.postgres_url) == expected

    @pytest.mark.asyncio()
    async def test_postgres_env_variables(self, monkeypatch: MonkeyPatch):
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_POSTGRES_HOST", "envhost")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_POSTGRES_PORT", "5555")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_POSTGRES_USER", "envuser")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_POSTGRES_PASSWORD", "envpass")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_POSTGRES_DRIVER", "envdriver")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_POSTGRES_DATABASE", "envdb")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_ENABLE_POSTGRES", "false")

        test_settings = aio_settings.AioSettings()
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
        expected_url = "redis://localhost:8600/0"  # Updated to include database number
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
        self, host: str, port: int, user: str | None, password: str | None, base: int, expected: str
    ):
        custom_settings = aio_settings.AioSettings(
            redis_host=host,
            redis_port=port,
            redis_user=user,
            redis_pass=password,
            redis_base=base,
        )
        assert str(custom_settings.redis_url) == expected

    @pytest.mark.asyncio()
    async def test_redis_env_variables(self, monkeypatch: MonkeyPatch):
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_REDIS_HOST", "envhost")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_REDIS_PORT", "7777")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_REDIS_USER", "envuser")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_REDIS_PASS", "envpass")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_REDIS_BASE", "2")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_ENABLE_REDIS", "true")

        test_settings = aio_settings.AioSettings()
        assert test_settings.redis_host == "envhost"
        assert test_settings.redis_port == 7777
        assert test_settings.redis_user == "envuser"
        assert test_settings.redis_pass.get_secret_value() == "envpass"
        assert test_settings.redis_base == 2
        assert test_settings.enable_redis == True

        expected_url = "redis://envuser:envpass@envhost:7777/2"
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

    def test_feature_flags(self):
        test_settings = aio_settings.AioSettings()
        # Test RAG flags
        assert isinstance(test_settings.rag_answer_accuracy_feature_flag, bool)
        assert isinstance(test_settings.rag_answer_hallucination_feature_flag, bool)
        assert isinstance(test_settings.rag_answer_v_reference_feature_flag, bool)
        assert isinstance(test_settings.rag_doc_relevance_feature_flag, bool)
        assert isinstance(test_settings.rag_doc_relevance_and_hallucination_feature_flag, bool)
        assert isinstance(test_settings.rag_string_embedding_distance_metrics_feature_flag, bool)

        # Test other flags
        assert isinstance(test_settings.helpfulness_feature_flag, bool)
        assert isinstance(test_settings.helpfulness_testing_feature_flag, bool)

    @pytest.mark.parametrize(
        "model_name,expected_tokens,expected_output_tokens",
        [
            ("gpt-4o-mini", 900, 16384),
            ("gpt-4o", 128000, 16384),
            ("claude-3-opus", 2048, 16384),
            ("gpt-4", 8192, 4096),
        ],
    )
    def test_model_token_limits(self, model_name: str, expected_tokens: int, expected_output_tokens: int):
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

    @pytest.mark.parametrize(
        "env_vars,expected_values",
        [
            (
                {
                    "DEMOCRACY_EXE_CONFIG_LLM_STREAMING": "True",
                    "DEMOCRACY_EXE_CONFIG_LLM_PROVIDER": "anthropic",
                    "DEMOCRACY_EXE_CONFIG_LLM_MAX_RETRIES": "5",
                    "DEMOCRACY_EXE_CONFIG_LLM_DOCUMENT_LOADER_TYPE": "unstructured",
                    "DEMOCRACY_EXE_CONFIG_LLM_VECTORSTORE_TYPE": "pinecone",
                    "DEMOCRACY_EXE_CONFIG_LLM_EMBEDDING_MODEL_TYPE": "text-embedding-ada-002",
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
            (-0.1, False),
            (0.0, True),
            (0.7, True),
            (1.0, True),
            (2.0, True),
            (2.1, False),
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

    def test_model_token_updates(self):
        """Test automatic token limit updates based on model selection."""
        settings = aio_settings.AioSettings(llm_model_name="gpt-4o-mini")
        assert settings.llm_max_tokens == 900
        assert settings.llm_max_output_tokens == 16384
        assert settings.prompt_cost_per_token == 0.000000150
        assert settings.completion_cost_per_token == 0.00000060

        settings = aio_settings.AioSettings(llm_model_name="claude-3-opus")
        assert settings.llm_max_tokens == 2048
        assert settings.llm_max_output_tokens == 16384
        assert settings.prompt_cost_per_token == 0.0000025
        assert settings.completion_cost_per_token == 0.00001

    def test_embedding_dimension_updates(self):
        """Test automatic embedding dimension updates."""
        settings = aio_settings.AioSettings(llm_embedding_model_name="text-embedding-3-large")
        assert settings.embedding_model_dimensions == 1024

        settings = aio_settings.AioSettings(llm_embedding_model_name="text-embedding-ada-002")
        assert settings.embedding_model_dimensions == 1536

    # ... rest of existing tests ...
