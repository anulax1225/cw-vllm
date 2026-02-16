"""
Output Parser — Thin wrapper around vLLM's native reasoning and tool parsers.

Handles reasoning extraction and tool call parsing on raw engine output
without using OpenAIServingChat. Instantiates vLLM's own parser classes
and calls their extract methods in the right order.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

logger = logging.getLogger("vllm-manager")

# Import compatibility — vLLM moved these modules between versions
try:
    # vLLM >= 0.9.x
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager
except ImportError:
    # vLLM 0.7.x - 0.8.x
    from vllm.entrypoints.openai.reasoning_parsers import ReasoningParserManager
    from vllm.entrypoints.openai.tool_parsers import ToolParserManager


@dataclass
class ParsedOutput:
    """Result of parsing raw model output."""

    reasoning_content: Optional[str] = None
    content: Optional[str] = None
    tool_calls: list = field(default_factory=list)  # vLLM ToolCall objects
    tools_called: bool = False
    finish_reason: str = "stop"


class OutputParser:
    """
    Wraps vLLM's ReasoningParser and ToolParser instances.
    Created once per model load (shared for non-streaming).
    For streaming, create a new instance per request (parsers have internal state).
    """

    def __init__(
        self,
        tokenizer,
        reasoning_parser_name: Optional[str] = None,
        tool_parser_name: Optional[str] = None,
    ):
        self.reasoning_parser = None
        self.tool_parser = None

        if reasoning_parser_name:
            try:
                cls = ReasoningParserManager.get_reasoning_parser(
                    reasoning_parser_name
                )
                self.reasoning_parser = cls(tokenizer)
                logger.debug(f"Reasoning parser initialized: {reasoning_parser_name}")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize reasoning parser '{reasoning_parser_name}': {e}"
                )

        if tool_parser_name:
            try:
                cls = ToolParserManager.get_tool_parser(tool_parser_name)
                self.tool_parser = cls(tokenizer)
                logger.debug(f"Tool parser initialized: {tool_parser_name}")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize tool parser '{tool_parser_name}': {e}"
                )

    @property
    def has_reasoning(self) -> bool:
        return self.reasoning_parser is not None

    @property
    def has_tools(self) -> bool:
        return self.tool_parser is not None

    def parse(
        self,
        model_output: str,
        request=None,
        finish_reason: str = "stop",
    ) -> ParsedOutput:
        """
        Non-streaming: parse complete model output.

        Pipeline:
        1. Extract reasoning (thinking) -> splits into reasoning + content
        2. Extract tool calls from content -> splits into tool_calls + remaining
        """
        result = ParsedOutput(finish_reason=finish_reason)

        # Step 1: Reasoning extraction
        if self.reasoning_parser and request:
            try:
                reasoning, content = self.reasoning_parser.extract_reasoning(
                    model_output, request
                )
                result.reasoning_content = reasoning
                working_text = content if content is not None else ""
            except Exception as e:
                logger.warning(f"Reasoning extraction failed, using raw text: {e}")
                working_text = model_output
        else:
            working_text = model_output

        # Step 2: Tool call extraction (from content only, not reasoning)
        if self.tool_parser and request:
            try:
                tool_result = self.tool_parser.extract_tool_calls(
                    working_text, request
                )
                result.tools_called = tool_result.tools_called
                result.tool_calls = tool_result.tool_calls
                result.content = tool_result.content
            except Exception as e:
                logger.warning(f"Tool call extraction failed, using raw text: {e}")
                result.content = working_text
        else:
            result.content = working_text

        # Step 3: Adjust finish_reason if tools were called
        if result.tools_called:
            result.finish_reason = "tool_calls"

        return result

    def parse_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request=None,
    ):
        """
        Streaming: parse a single delta.
        Returns vLLM's DeltaMessage (already OpenAI-compatible), or None.

        Order: reasoning first, then tool calls on content.
        """
        # Step 1: Reasoning extraction on the delta
        if self.reasoning_parser:
            try:
                delta_msg = self.reasoning_parser.extract_reasoning_streaming(
                    previous_text,
                    current_text,
                    delta_text,
                    previous_token_ids,
                    current_token_ids,
                    delta_token_ids,
                )
            except Exception as e:
                logger.warning(f"Streaming reasoning extraction failed: {e}")
                return None

            if delta_msg is not None:
                # If reasoning parser produced content and we have a tool parser,
                # check if the content portion contains tool calls
                if (
                    self.tool_parser
                    and delta_msg.content is not None
                    and request is not None
                ):
                    try:
                        tool_delta = (
                            self.tool_parser.extract_tool_calls_streaming(
                                previous_text,
                                current_text,
                                delta_text,
                                previous_token_ids,
                                current_token_ids,
                                delta_token_ids,
                                request,
                            )
                        )
                        if tool_delta is not None and tool_delta.tool_calls:
                            delta_msg.tool_calls = tool_delta.tool_calls
                            delta_msg.content = None
                    except Exception as e:
                        logger.warning(
                            f"Streaming tool extraction failed: {e}"
                        )
                return delta_msg

        # Step 2: No reasoning parser — go straight to tool parsing
        if self.tool_parser and request:
            try:
                return self.tool_parser.extract_tool_calls_streaming(
                    previous_text,
                    current_text,
                    delta_text,
                    previous_token_ids,
                    current_token_ids,
                    delta_token_ids,
                    request,
                )
            except Exception as e:
                logger.warning(f"Streaming tool extraction failed: {e}")
                return None

        # Step 3: No parsers — caller uses delta_text as-is
        return None
