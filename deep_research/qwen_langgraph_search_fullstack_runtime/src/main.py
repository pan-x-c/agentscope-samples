# -*- coding: utf-8 -*-
import asyncio
import json

from qwen_langgraph_search.src.configuration import Configuration
from qwen_langgraph_search.src.custom_search_tool import CustomSearchTool
from qwen_langgraph_search.src.graph_openai_compatible import WebSearchGraph
from qwen_langgraph_search.src.llm_utils import call_dashscope

if __name__ == "__main__":
    custom_search_tool = CustomSearchTool(search_engine="quark")
    graph = WebSearchGraph(
        json.loads(Configuration().model_dump_json()),
        call_dashscope,
        custom_search_tool,
    )

    user_input = input("Type in your question or press q to quit\n")
    while user_input != "q":
        question = user_input
        use_agentengine = True

        try:
            res = asyncio.run(graph.run(question))
            print(res)
        except Exception as e:
            print(f"An error occurred: {e}")

        user_input = input("Type in your question or press q to quit\n")
