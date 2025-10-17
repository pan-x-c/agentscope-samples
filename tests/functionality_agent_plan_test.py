# -*- coding: utf-8 -*-
# test_main.py
import os
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from agentscope.agent import ReActAgent, UserAgent
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit
from agentscope.message import Msg
from agentscope.formatter import DashScopeChatFormatter
from agentscope.plan import PlanNotebook
from agentscope.tool import (
    execute_shell_command,
    execute_python_code,
    write_text_file,
    insert_text_file,
    view_text_file,
)

from browser_use.functionality.plan.main_agent_managed_plan import main


class TestMainFunctionality:
    """Test suite for the main.py functionality"""

    @pytest.fixture
    def mock_toolkit(self):
        """Create a mocked Toolkit instance"""
        return Mock(spec=Toolkit)

    @pytest.fixture
    def mock_model(self):
        """Create a mocked DashScopeChatModel"""
        model = Mock(spec=DashScopeChatModel)
        model.call = AsyncMock(return_value=Mock(content="test response"))
        return model

    @pytest.fixture
    def mock_formatter(self):
        """Create a mocked DashScopeChatFormatter"""
        return Mock(spec=DashScopeChatFormatter)

    @pytest.fixture
    def mock_plan_notebook(self):
        """Create a mocked PlanNotebook"""
        return Mock(spec=PlanNotebook)

    @pytest.fixture
    def mock_agent(
        self,
        mock_model,
        mock_formatter,
        mock_toolkit,
        mock_plan_notebook,
    ):
        """Create a mocked ReActAgent instance"""
        agent = Mock(spec=ReActAgent)
        agent.model = mock_model
        agent.formatter = mock_formatter
        agent.toolkit = mock_toolkit
        agent.plan_notebook = mock_plan_notebook
        agent.__call__ = AsyncMock(
            return_value=Msg("assistant", "test response", role="assistant"),
        )
        return agent

    @pytest.fixture
    def mock_user(self):
        """Create a mocked UserAgent instance"""
        user = Mock(spec=UserAgent)
        user.__call__ = AsyncMock(
            return_value=Msg("user", "exit", role="user"),
        )
        return user

    def test_toolkit_initialization(self):
        """Test toolkit initialization and tool registration"""
        toolkit = Toolkit()
        # Register all required tools
        toolkit.register_tool_function(execute_shell_command)
        toolkit.register_tool_function(execute_python_code)
        toolkit.register_tool_function(write_text_file)
        toolkit.register_tool_function(insert_text_file)
        toolkit.register_tool_function(view_text_file)

        # ✅ 通过 hasattr 和 callable 验证工具是否注册成功
        assert hasattr(toolkit, "execute_shell_command")
        assert hasattr(toolkit, "execute_python_code")
        assert hasattr(toolkit, "write_text_file")
        assert hasattr(toolkit, "insert_text_file")
        assert hasattr(toolkit, "view_text_file")

        assert callable(toolkit.execute_shell_command)
        assert callable(toolkit.execute_python_code)
        assert callable(toolkit.write_text_file)
        assert callable(toolkit.insert_text_file)
        assert callable(toolkit.view_text_file)

    @pytest.mark.asyncio
    async def test_agent_initialization(
        self,
        mock_model,
        mock_formatter,
        mock_toolkit,
        mock_plan_notebook,
    ):
        """Test ReActAgent initialization"""
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test_key"}):
            agent = ReActAgent(
                name="Friday",
                sys_prompt="You're a helpful assistant named Friday.",
                model=mock_model,
                formatter=mock_formatter,
                toolkit=mock_toolkit,
                enable_meta_tool=True,
                plan_notebook=mock_plan_notebook,
            )

            assert agent.name == "Friday"
            assert (
                agent.sys_prompt == "You're a helpful assistant named Friday."
            )
            assert agent.model == mock_model
            assert agent.formatter == mock_formatter
            assert agent.toolkit == mock_toolkit
            assert agent.enable_meta_tool is True
            assert agent.plan_notebook == mock_plan_notebook

    @pytest.mark.asyncio
    async def test_message_loop_exits_on_exit(self, mock_agent, mock_user):
        """Test the message loop exits when user sends 'exit'"""
        with patch("main.asyncio.sleep") as mock_sleep, patch.dict(
            os.environ,
            {"DASHSCOPE_API_KEY": "test_key"},
        ):
            # 避免无限循环
            mock_sleep.side_effect = asyncio.TimeoutError()

            # 替换 main.py 中的 agent 和 user
            with patch("main.ReActAgent", return_value=mock_agent), patch(
                "main.UserAgent",
                return_value=mock_user,
            ):
                try:
                    await main()
                except asyncio.TimeoutError:
                    pass  # 期望的退出方式

                # ✅ 验证 agent 和 user 被正确调用
                mock_agent.__call__.assert_awaited_once()
                mock_user.__call__.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_full_message_flow(self, mock_agent, mock_user):
        """Test the complete message flow between agent and user"""
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test_key"}):
            # 模拟 agent 返回的响应
            mock_agent.__call__ = AsyncMock(
                side_effect=[
                    Msg("assistant", "response 1", role="assistant"),
                    Msg("assistant", "response 2", role="assistant"),
                ],
            )

            # 模拟 user 返回的响应
            mock_user.__call__ = AsyncMock(
                side_effect=[
                    Msg("user", "first message", role="user"),
                    Msg("user", "exit", role="user"),
                ],
            )

            # 替换 main.py 中的 agent 和 user
            with patch("main.ReActAgent", return_value=mock_agent), patch(
                "main.UserAgent",
                return_value=mock_user,
            ):
                try:
                    await main()
                except asyncio.TimeoutError:
                    pass  # 期望的退出方式

                # ✅ 验证消息流程
                assert mock_agent.__call__.await_count == 2
                assert mock_user.__call__.await_count == 2

                # ✅ 验证最终消息是 "exit"
                final_msg = mock_user.__call__.call_args_list[-1][0][0]
                assert final_msg.get_text_content() == "exit"

    @pytest.mark.asyncio
    async def test_main_runs_without_error(self, mock_agent, mock_user):
        """Test the main function runs without raising exceptions"""
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test_key"}), patch(
            "main.ReActAgent",
            return_value=mock_agent,
        ), patch("main.UserAgent", return_value=mock_user), patch(
            "main.asyncio.sleep",
            AsyncMock(),
        ):
            # 使用 asyncio.run(main()) 来启动测试
            try:
                await main()
            except Exception as e:
                pytest.fail(f"main() raised an unexpected exception: {e}")
