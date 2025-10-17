# -*- coding: utf-8 -*-
import os
from unittest.mock import patch

import pytest
from agentscope.formatter import DashScopeChatFormatter
from agentscope.mcp import StdIOStatefulClient
from agentscope.memory import InMemoryMemory
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit

from browser_use.agent_browser.browser_agent import BrowserAgent


class TestBrowserAgentSingleton:
    _instance = None

    @classmethod
    def get_instance(cls) -> BrowserAgent:
        """Singleton access method"""
        if cls._instance is None:
            cls._instance = BrowserAgent(
                name="BrowserBot",
                model=DashScopeChatModel(
                    api_key=os.environ.get("DASHSCOPE_API_KEY"),
                    model_name="qwen-max",
                    stream=True,
                ),
                formatter=DashScopeChatFormatter(),
                memory=InMemoryMemory(),
                toolkit=Toolkit(),
                max_iters=50,
                start_url="https://www.google.com",
            )
        return cls._instance

    def test_singleton_pattern(self) -> None:
        """Test that only one instance of BrowserAgent is created"""
        instance1 = TestBrowserAgentSingleton.get_instance()
        instance2 = TestBrowserAgentSingleton.get_instance()

        assert (
            instance1 is instance2
        ), "BrowserAgent instances are not the same"

    def test_instance_properties(self) -> None:
        """Test browser agent instance properties"""
        instance = TestBrowserAgentSingleton.get_instance()

        assert instance.name == "BrowserBot"
        assert isinstance(instance.model, DashScopeChatModel)
        assert isinstance(instance.formatter, DashScopeChatFormatter)
        assert isinstance(instance.memory, InMemoryMemory)
        assert isinstance(instance.toolkit, Toolkit)
        assert instance.max_iters == 50
        assert instance.start_url == "https://www.google.com"

    @pytest.mark.asyncio
    async def test_browser_connection(self, monkeypatch) -> None:
        """Test browser connection functionality"""

        # Mock async methods
        async def mock_connect():
            return True

        async def mock_close():
            return True

        # Patch the StdIOStatefulClient
        with patch("agentscope.mcp.StdIOStatefulClient.connect", mock_connect):
            with patch("agentscope.mcp.StdIOStatefulClient.close", mock_close):
                instance = TestBrowserAgentSingleton.get_instance()

                # Test connection
                connected = await instance.toolkit._mcp_clients[0].connect()
                assert connected is True

                # Test cleanup
                closed = await instance.toolkit._mcp_clients[0].close()
                assert closed is True


if __name__ == "__main__":
    pytest.main(["-v", __file__])
