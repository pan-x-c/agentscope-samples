# -*- coding: utf-8 -*-
import os
import shutil
import tempfile
from unittest.mock import Mock, patch

import pytest
from agentscope.formatter import DashScopeChatFormatter
from agentscope.mcp import StdIOStatefulClient
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel

from deep_research.agent_deep_research.deep_research_agent import (
    DeepResearchAgent,
)

# Import the main function to be tested
from deep_research.agent_deep_research.main import main


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture to set required environment variables"""
    monkeypatch.setenv("TAVILY_API_KEY", "test_tavily_key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test_dashscope_key")
    return {
        "TAVILY_API_KEY": "test_tavily_key",
        "DASHSCOPE_API_KEY": "test_dashscope_key",
    }


@pytest.fixture
def temp_working_dir():
    """Create a temporary working directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_tavily_client():
    """Create a mocked Tavily client"""
    client = Mock(spec=StdIOStatefulClient)
    client.name = "tavily_mcp"
    client.connect = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_formatter():
    """Create a mocked formatter"""
    return Mock(spec=DashScopeChatFormatter)


@pytest.fixture
def mock_memory():
    """Create a mocked memory instance"""
    return Mock(spec=InMemoryMemory)


@pytest.fixture
def mock_model():
    """Create a mocked model instance"""
    model = Mock(spec=DashScopeChatModel)
    model.call = AsyncMock(return_value=Mock(content="test response"))
    return model


@pytest.fixture
def mock_agent(mock_model, mock_formatter, mock_memory, mock_tavily_client):
    """Create a mocked DeepResearchAgent instance"""
    agent = Mock(spec=DeepResearchAgent)
    agent.return_value = agent  # Make the mock instance return itself
    agent.model = mock_model
    agent.formatter = mock_formatter
    agent.memory = mock_memory
    agent.search_mcp_client = mock_tavily_client
    return agent


class AsyncMock(Mock):
    """Helper class for async mocks"""

    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class TestDeepResearchAgent:
    """Test suite for Deep Research Agent functionality"""

    def test_agent_initialization(
        self,
        mock_model,
        mock_tavily_client,
        temp_working_dir,
    ):
        """Test agent initialization with valid parameters"""
        agent = DeepResearchAgent(
            name="Friday",
            sys_prompt="You are a helpful assistant named Friday.",
            model=mock_model,
            formatter=DashScopeChatFormatter(),
            memory=InMemoryMemory(),
            search_mcp_client=mock_tavily_client,
            tmp_file_storage_dir=temp_working_dir,
        )

        assert agent.name == "Friday"
        assert agent.sys_prompt == "You are a helpful assistant named Friday."
        assert agent.tmp_file_storage_dir == temp_working_dir
        assert os.path.exists(temp_working_dir)

    @pytest.mark.asyncio
    async def test_main_function_success(
        self,
        mock_env_vars,
        mock_tavily_client,
        mock_model,
        temp_working_dir,
    ):
        """Test main function with successful execution"""
        # Mock the StdIOStatefulClient constructor
        with patch(
            "deep_research.agent_deep_research.main.StdIOStatefulClient",
            return_value=mock_tavily_client,
        ):
            # Mock the DeepResearchAgent constructor
            with patch(
                "deep_research.agent_deep_research.main.DeepResearchAgent",
                autospec=True,
            ) as mock_agent_class:
                mock_agent_instance = Mock()
                mock_agent_instance.return_value = mock_agent_instance
                mock_agent_instance.__call__ = AsyncMock(
                    return_value=Msg("Friday", "Test response", "assistant"),
                )
                mock_agent_class.return_value = mock_agent_instance

                # Mock os.makedirs
                with patch("os.makedirs") as mock_makedirs:
                    # Run the main function with a test query
                    test_query = "Test research question"
                    msg = Msg("Bob", test_query, "user")

                    await main(test_query)

                    # Verify initialization calls
                    mock_makedirs.assert_called_once_with(
                        temp_working_dir,
                        exist_ok=True,
                    )
                    mock_agent_class.assert_called_once()

                    # Verify agent was called with the correct message
                    mock_agent_instance.__call__.assert_called_once_with(msg)

    @pytest.mark.asyncio
    async def test_main_function_with_missing_env_vars(self):
        """Test main function handles missing environment variables"""
        # Test missing Tavily API key
        with patch.dict(os.environ, clear=True):
            with pytest.raises(Exception):
                await main("Test query")

    @pytest.mark.asyncio
    async def test_main_function_connection_failure(
        self,
        mock_env_vars,
        temp_working_dir,
    ):
        """Test main function handles connection failures"""
        # Mock the StdIOStatefulClient to raise an exception
        with patch(
            "deep_research.agent_deep_research.main.StdIOStatefulClient",
        ) as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.connect = AsyncMock(
                side_effect=Exception("Connection failed"),
            )
            mock_client.return_value = mock_client_instance

            # Run the main function and expect exception
            with pytest.raises(Exception) as exc_info:
                await main("Test query")

            assert "Connection failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_agent_cleanup(
        self,
        mock_env_vars,
        mock_tavily_client,
    ):
        """Test proper cleanup of resources"""
        with patch(
            "deep_research.agent_deep_research.main.StdIOStatefulClient",
            return_value=mock_tavily_client,
        ):
            # Run main function
            await main("Test query")

            # Verify client close was called
            mock_tavily_client.close.assert_called_once()

    def test_working_directory_creation(self, temp_working_dir):
        """Test working directory is created correctly"""
        test_dir = os.path.join(temp_working_dir, "test_subdir")

        # Test directory creation
        os.makedirs(test_dir, exist_ok=True)
        assert os.path.exists(test_dir)

        # Test exist_ok=True behavior
        os.makedirs(test_dir, exist_ok=True)  # Should not raise error


class TestErrorHandling:
    """Test suite for error handling scenarios"""

    @pytest.mark.asyncio
    async def test_model_failure(self, mock_env_vars, mock_tavily_client):
        """Test handling of model failures"""
        with patch(
            "deep_research.agent_deep_research.main.StdIOStatefulClient",
            return_value=mock_tavily_client,
        ):
            with patch(
                "deep_research.agent_deep_research.main.DeepResearchAgent",
            ) as mock_agent_class:
                mock_agent = Mock()
                mock_agent.__call__ = AsyncMock(
                    side_effect=Exception("Model error"),
                )
                mock_agent_class.return_value = mock_agent

                with pytest.raises(Exception) as exc_info:
                    await main("Test query")

                assert "Model error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_filesystem_errors(self, mock_env_vars, mock_tavily_client):
        """Test handling of filesystem errors"""
        # Test with invalid directory path
        invalid_dir = "/invalid/path/that/does/not/exist"

        with patch.dict(os.environ, {"AGENT_OPERATION_DIR": invalid_dir}):
            with patch(
                "os.makedirs",
                side_effect=PermissionError("Permission denied"),
            ):
                with pytest.raises(PermissionError):
                    await main("Test query")

    @pytest.mark.asyncio
    async def test_logging_output(
        self,
        mock_env_vars,
        mock_tavily_client,
        caplog,
    ):
        """Test logging output is generated correctly"""
        with patch(
            "deep_research.agent_deep_research.main.StdIOStatefulClient",
            return_value=mock_tavily_client,
        ):
            with patch(
                "deep_research.agent_deep_research.main.DeepResearchAgent",
            ) as mock_agent_class:
                mock_agent = Mock()
                mock_agent.__call__ = AsyncMock(
                    return_value=Msg("Friday", "Test response", "assistant"),
                )
                mock_agent_class.return_value = mock_agent

                await main("Test query")

                # Verify debug logs are present
                assert any(
                    "DEBUG" in record.levelname for record in caplog.records
                )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
