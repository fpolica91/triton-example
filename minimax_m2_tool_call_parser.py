import json
import re
import uuid
from typing import Union

from engine.utils.tokenizer import AnyTokenizer
from engine.utils.tool_call_parsers.tool_call_parser import (
    ToolCallParser,
    ToolParserManager,
)
from schemas.openai import (
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallChunk,
    ChatCompletionMessageToolCalls,
    ChatCompletionResponseMessage,
    ChatCompletionStreamResponseDelta,
    Function1,
    Function2,
)


@ToolParserManager.register_module("minimax_m2")
class MinimaxM2ToolParser(ToolCallParser):

    TOOL_CALL_RE = re.compile(
        r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL
    )
    INVOKE_RE = re.compile(
        r'<invoke\s+name=["\']([^"\']+)["\']>(.*?)</invoke>', re.DOTALL
    )
    PARAM_RE = re.compile(
        r'<parameter\s+name=["\']([^"\']+)["\']>(.*?)</parameter>', re.DOTALL
    )

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self.tool_call_start_token = "<minimax:tool_call>"
        self.tool_call_end_token = "</minimax:tool_call>"
        # streaming state
        self.accumulated_text = ""
        self.current_tool_index = 0
        self.header_sent = False
        self.current_function_name = None
        self.accumulated_params = {}
        self.in_tool_call = False
        self.streamed_args_for_tool = []

    def _extract_tool_calls(self, text: str):
        """Extract structured tool calls from MiniMax XML format."""
        results = []
        for tc_match in self.TOOL_CALL_RE.finditer(text):
            tc_body = tc_match.group(1)
            for inv_match in self.INVOKE_RE.finditer(tc_body):
                fn_name = inv_match.group(1)
                inv_body = inv_match.group(2)
                params = {}
                for p_match in self.PARAM_RE.finditer(inv_body):
                    p_name = p_match.group(1)
                    p_value = p_match.group(2)
                    # try to cast to appropriate types
                    params[p_name] = self._cast_value(p_value)
                results.append({"name": fn_name, "arguments": params})
        return results

    @staticmethod
    def _cast_value(value: str):
        """Try to cast string value to int, float, bool, or keep as string."""
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    def parse_tool_calls(
        self, full_text: str, role: str, backend: str
    ) -> ChatCompletionResponseMessage:
        if self.tool_call_start_token not in full_text:
            return ChatCompletionResponseMessage(
                tool_calls=None, content=full_text, role=role
            )

        try:
            fn_calls = self._extract_tool_calls(full_text)
            if not fn_calls:
                return ChatCompletionResponseMessage(
                    tool_calls=None, content=full_text, role=role
                )

            tool_calls = ChatCompletionMessageToolCalls(
                root=[
                    ChatCompletionMessageToolCall(
                        id=f"call_{uuid.uuid4().hex[:24]}",
                        type="function",
                        function=Function1(
                            name=fc["name"],
                            arguments=json.dumps(fc["arguments"]),
                        ),
                    )
                    for fc in fn_calls
                ]
            )

            # extract any content before the first tool call
            idx = full_text.find(self.tool_call_start_token)
            content_before = full_text[:idx].strip() if idx > 0 else ""

            return ChatCompletionResponseMessage(
                tool_calls=tool_calls, content=content_before or None, role=role
            )
        except Exception:
            return ChatCompletionResponseMessage(
                tool_calls=None, content=full_text, role=role
            )

    def parse_tool_calls_streaming(
        self, current_text: str, delta_text: str, backend: str
    ) -> Union[ChatCompletionStreamResponseDelta, None]:
        # if no tool call token seen yet, just stream content
        if self.tool_call_start_token not in current_text:
            return ChatCompletionStreamResponseDelta(content=delta_text)

        # once we see tool call start, stop streaming content and accumulate
        if not self.in_tool_call:
            self.in_tool_call = True
            # send any remaining content before the tool call tag
            idx = current_text.find(self.tool_call_start_token)
            prefix = current_text[:idx]
            if prefix and not self.header_sent:
                self.header_sent = True
                already_sent = current_text[: -len(delta_text)] if delta_text else ""
                unsent = prefix[len(already_sent):]
                if unsent.strip():
                    return ChatCompletionStreamResponseDelta(content=unsent)
            return None

        # check if we have a complete tool call block
        if self.tool_call_end_token in current_text:
            fn_calls = self._extract_tool_calls(current_text)
            if fn_calls and len(fn_calls) > self.current_tool_index:
                fc = fn_calls[self.current_tool_index]
                self.current_tool_index += 1
                args_json = json.dumps(fc["arguments"])
                delta = ChatCompletionStreamResponseDelta(
                    tool_calls=[
                        ChatCompletionMessageToolCallChunk(
                            index=self.current_tool_index - 1,
                            type="function",
                            id=f"call_{uuid.uuid4().hex[:24]}",
                            function=Function2(
                                name=fc["name"],
                                arguments=args_json,
                            ).model_dump(exclude_none=True),
                        )
                    ]
                )
                return delta

        return None
