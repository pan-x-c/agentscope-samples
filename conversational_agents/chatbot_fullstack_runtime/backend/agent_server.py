# -*- coding: utf-8 -*-
import asyncio
import os

from agentscope_runtime.engine import LocalDeployManager, Runner
from agentscope_runtime.engine.agents.llm_agent import LLMAgent
from agentscope_runtime.engine.llms import QwenLLM
from agentscope_runtime.engine.services.context_manager import ContextManager
from agentscope_runtime.engine.services.session_history_service import (
    InMemorySessionHistoryService,
)


def local_deploy():
    asyncio.run(_local_deploy())


async def _local_deploy():
    from dotenv import load_dotenv

    load_dotenv()

    server_port = int(os.environ.get("SERVER_PORT", "8090"))
    server_endpoint = os.environ.get("SERVER_ENDPOINT", "agent")

    llm_agent = LLMAgent(
        model=QwenLLM(),
        name="llm_agent",
        description="A simple LLM agent to generate a short ",
    )

    session_history_service = InMemorySessionHistoryService()
    context_manager = ContextManager(
        session_history_service=session_history_service,
    )
    runner = Runner(
        agent=llm_agent,
        context_manager=context_manager,
    )

    deploy_manager = LocalDeployManager(host="localhost", port=server_port)
    try:
        deployment_info = await runner.deploy(
            deploy_manager,
            endpoint_path=f"/{server_endpoint}",
        )

        print("✅ Service deployed successfully!")
        print(f"   URL: {deployment_info['url']}")
        print(f"   Endpoint: {deployment_info['url']}/{server_endpoint}")
        print("\nAgent Service is running in the background.")

        while True:
            await asyncio.sleep(1)

    except (KeyboardInterrupt, asyncio.CancelledError):
        # This block will be executed when you press Ctrl+C.
        print("\nShutdown signal received. Stopping the service...")
        if deploy_manager.is_running:
            await deploy_manager.stop()
        print("✅ Service stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")
        if deploy_manager.is_running:
            await deploy_manager.stop()


if __name__ == "__main__":
    local_deploy()
