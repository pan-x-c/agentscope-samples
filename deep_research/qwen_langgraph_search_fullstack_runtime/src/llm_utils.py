# -*- coding: utf-8 -*-
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

from openai import OpenAI
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)


def extract_json_from_qwen(qwen_result: str) -> str:
    sql = ""
    pattern = r"```json(.*?)```"

    sql_code_snippets = re.findall(pattern, qwen_result, re.DOTALL)

    if len(sql_code_snippets) > 0:
        sql = sql_code_snippets[-1].strip()

    return sql


def call_dashscope(**args: Any) -> ChatCompletion:
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        **args,
    )
    stream = args.get("stream", False)
    if stream:
        try:
            completion = postprocess_completion(completion)
            return completion
        except Exception as e:
            print(
                f"Error occurred when postprocess_completion on "
                f"'stream=True'. {e}",
            )
            default_message = ChatCompletionMessage(
                role="assistant",
                content="Error in calling LLM",  # 默认内容
            )
            default_choice = Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=default_message,
            )
            default_chat_completion = ChatCompletion(
                id="chatcmpl-1234567890",
                choices=[default_choice],
                created=int(datetime.now().timestamp()),
                model=args["model"],
                object="chat.completion",
                service_tier="default",
                system_fingerprint=None,
                usage=None,
            )
            return default_chat_completion
    return completion


def merge_fields(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, str):
            target[key] = target.get(key, "") + value
        elif value is not None and isinstance(value, dict):
            merge_fields(target[key], value)


def merge_chunk(final_response: Dict[str, Any], delta: Dict[str, Any]) -> None:
    delta.pop("role", None)
    merge_fields(final_response, delta)

    tool_calls = delta.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        index = int(tool_calls[0].pop("index"))  # Convert index to integer
        if "tool_calls" not in final_response:
            final_response["tool_calls"] = {}
        final_response["tool_calls"][index] = final_response["tool_calls"].get(
            index,
            {},
        )
        final_response["tool_calls"][index].pop("type", None)
        merge_fields(final_response["tool_calls"][index], tool_calls[0])


def postprocess_completion(completion: Iterator) -> ChatCompletion:
    message: Dict[str, Any] = {
        "content": "",
        "role": "assistant",
        "function_call": None,
        "tool_calls": defaultdict(
            lambda: {
                "function": {"arguments": "", "name": ""},
                "id": "",
                "type": "",
            },
        ),
        "reasoning_content": "",
        "refusal": "",
    }
    last_chunk: Optional[Any] = None

    for chunk in completion:
        try:
            delta = json.loads(chunk.choices[0].delta.json())
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from chunk: {e}")
            continue
        delta.pop("role", None)
        merge_chunk(message, delta)
        finish_reason = chunk.choices[0].finish_reason
        logprobs = chunk.choices[0].logprobs
        last_chunk = chunk

    # 显式声明类型
    tool_calls_list: List[Dict[str, Any]] = list(
        message.get("tool_calls", {}).values(),
    )
    message["tool_calls"] = tool_calls_list

    tool_calls = None
    if message["tool_calls"]:
        tool_calls = []
        for tool_call in message["tool_calls"]:  # 类型已明确为 Dict
            function = Function(
                arguments=tool_call["function"]["arguments"],
                name=tool_call["function"]["name"],
            )
            tool_call_object = ChatCompletionMessageToolCall(
                id=tool_call["id"],
                function=function,
                type=tool_call["type"],
            )
            tool_calls.append(tool_call_object)

    chat_message = ChatCompletionMessage(
        content=message["content"],
        role=message["role"],
        function_call=message["function_call"],
        tool_calls=tool_calls,
        reasoning_content=message["reasoning_content"],
        refusal=message["refusal"],
    )
    choices = [
        Choice(
            finish_reason=finish_reason,
            index=0,
            message=chat_message,
            logprobs=logprobs,
        ),
    ]

    completion = ChatCompletion(
        id=last_chunk.id,
        choices=choices,
        created=last_chunk.created,
        model=last_chunk.model,
        object="chat.completion",
        service_tier=last_chunk.service_tier,
        system_fingerprint=last_chunk.system_fingerprint,
        usage=last_chunk.usage,
    )
    return completion
