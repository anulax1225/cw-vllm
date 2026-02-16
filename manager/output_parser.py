"""
Output Parser — Thin wrapper around vLLM's native reasoning and tool parsers.

Mirrors the approach in vLLM's serving_chat.py:
  - Reasoning parser is STATEFUL and called on every delta
  - After reasoning ends, tool parser is called on the content portion
  - current_token_ids (cumulative) is passed as delta_token_ids
    (this matches vLLM's own serving_chat.py behavior)
  - Tool parser also requires a ChatCompletionRequest object
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

logger = logging.getLogger("vllm-manager")

# Import compatibility — vLLM moved these modules between versions.
#
# ReasoningParserManager:
#   vLLM >= 0.9:  vllm.reasoning
#   vLLM 0.7-0.8: vllm.entrypoints.openai.reasoning_parsers
#
# ToolParserManager:
#   vLLM >= 0.12: vllm.tool_parsers  (top-level re-export)
#   vLLM 0.6-0.11: vllm.entrypoints.openai.tool_parsers

# --- Reasoning parser ---
ReasoningParserManager = None
try:
    from vllm.reasoning import ReasoningParserManager
except ImportError:
    try:
        from vllm.entrypoints.openai.reasoning_parsers import ReasoningParserManager
    except ImportError:
        logger.info("ReasoningParserManager not available in this vLLM version")

# --- Tool parser ---
ToolParserManager = None
try:
    from vllm.entrypoints.openai.tool_parsers import ToolParserManager
except ImportError:
    try:
        from vllm.tool_parsers import ToolParserManager
    except ImportError:
        logger.info("ToolParserManager not available in this vLLM version")


def _resolve(obj, *names):
    """Return the first attribute found on obj from a list of names."""
    for name in names:
        fn = getattr(obj, name, None)
        if fn is not None:
            return fn
    return None


@dataclass
class ParsedOutput:
    """Result of parsing raw model output."""

    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: list = field(default_factory=list)
    finish_reason: str = "stop"

    @property
    def tools_called(self) -> bool:
        return len(self.tool_calls) > 0


class OutputParser:
    """
    Parse model output for reasoning, content, and tool calls.

    Uses vLLM's native parser classes, following the same approach as
    vLLM's serving_chat.py — the reasoning parser is stateful and tracks
    the <think>...</think> boundary internally across streaming calls.
    """

    def __init__(
        self,
        tokenizer,
        reasoning_parser_name: Optional[str] = None,
        tool_parser_name: Optional[str] = None,
    ):
        self.reasoning_parser = None
        self.tool_parser = None

        # Streaming state — tracks whether reasoning has ended so we know
        # when to start forwarding content to the tool parser.
        self._reasoning_ended = False

        logger.info(
            f"[OutputParser] init: reasoning={reasoning_parser_name}, "
            f"tools={tool_parser_name}, "
            f"ReasoningParserManager={'available' if ReasoningParserManager else 'MISSING'}, "
            f"ToolParserManager={'available' if ToolParserManager else 'MISSING'}"
        )

        # Initialize reasoning parser
        if reasoning_parser_name and ReasoningParserManager is not None:
            try:
                cls = ReasoningParserManager.get_reasoning_parser(
                    reasoning_parser_name
                )
                self.reasoning_parser = cls(tokenizer)
                methods = [
                    m for m in dir(self.reasoning_parser)
                    if m.startswith("extract") and callable(getattr(self.reasoning_parser, m))
                ]
                logger.info(
                    f"[OutputParser] Reasoning parser: "
                    f"{type(self.reasoning_parser).__name__}, "
                    f"methods={methods}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to init reasoning parser '{reasoning_parser_name}': {e}"
                )

        # Initialize tool parser
        if tool_parser_name and ToolParserManager is not None:
            try:
                cls = ToolParserManager.get_tool_parser(tool_parser_name)
                self.tool_parser = cls(tokenizer)
                methods = [
                    m for m in dir(self.tool_parser)
                    if m.startswith("extract") and callable(getattr(self.tool_parser, m))
                ]
                logger.info(
                    f"[OutputParser] Tool parser: "
                    f"{type(self.tool_parser).__name__}, "
                    f"methods={methods}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to init tool parser '{tool_parser_name}': {e}"
                )

    @property
    def has_reasoning(self) -> bool:
        return self.reasoning_parser is not None

    @property
    def has_tools(self) -> bool:
        return self.tool_parser is not None

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    def parse(
        self,
        model_output: str,
        request=None,
        finish_reason: str = "stop",
    ) -> ParsedOutput:
        """
        Non-streaming: parse complete model output.

        Pipeline:
        1. Extract reasoning -> splits into reasoning + content
        2. Extract tool calls from content
        """
        result = ParsedOutput(finish_reason=finish_reason)

        # Step 1: Reasoning
        #   v0.12+:  extract_reasoning(model_output, request)
        #   v0.9-11: extract_reasoning_content(model_output, request)
        if self.reasoning_parser and request:
            try:
                fn = _resolve(
                    self.reasoning_parser,
                    "extract_reasoning",
                    "extract_reasoning_content",
                )
                if fn is None:
                    raise AttributeError("No extract_reasoning[_content] found")
                reasoning, content = fn(model_output, request)
                result.reasoning_content = reasoning
                working_text = content if content is not None else ""
            except Exception as e:
                logger.warning(f"Reasoning extraction failed: {e}")
                working_text = model_output
        else:
            working_text = model_output

        # Step 2: Tool calls (from content, not reasoning)
        if self.tool_parser and request:
            try:
                tool_info = self.tool_parser.extract_tool_calls(
                    working_text, request
                )
                if tool_info.tools_called:
                    result.tool_calls = tool_info.tool_calls
                    result.content = tool_info.content or ""
                else:
                    result.content = working_text
            except Exception as e:
                logger.warning(f"Tool extraction failed: {e}")
                result.content = working_text
        else:
            result.content = working_text

        if result.tools_called:
            result.finish_reason = "tool_calls"
        return result

    # ------------------------------------------------------------------
    # Streaming — mirrors serving_chat.chat_completion_stream_generator
    # ------------------------------------------------------------------

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

        Returns a DeltaMessage-like object (from vLLM) with attributes:
          .reasoning_content / .reasoning  — thinking text
          .content                         — response text
          .tool_calls                      — tool call deltas
        Or None if the parser is buffering / suppressing a special token.

        Follows serving_chat.py logic:
        1. While reasoning is active → call reasoning parser
        2. Once reasoning ends → call tool parser on content deltas
        3. current_token_ids is passed as delta_token_ids (cumulative)
           because vLLM's parsers use `token_id in delta_token_ids`
        """
        if not delta_text:
            return None

        # ---- Reasoning phase ----
        if self.reasoning_parser and not self._reasoning_ended:
            fn = _resolve(
                self.reasoning_parser,
                "extract_reasoning_streaming",
                "extract_reasoning_content_streaming",
            )
            if fn is None:
                logger.warning("No streaming reasoning extract method found")
                return None

            try:
                delta_msg = fn(
                    previous_text,
                    current_text,
                    delta_text,
                    previous_token_ids,
                    current_token_ids,
                    # CRITICAL: Pass actual delta_token_ids, NOT cumulative.
                    #
                    # The parser logic (DeepSeekR1/Qwen3/BaseThinking) is:
                    #   if think_end_token_id in delta_token_ids:
                    #       → split this chunk into reasoning + content
                    #   elif think_end_token_id in previous_token_ids:
                    #       → pure content mode
                    #   else:
                    #       → pure reasoning mode
                    #
                    # If we pass cumulative IDs as delta, think_end appears
                    # in delta for EVERY chunk after </think>, causing the
                    # parser to run find(think_end_token) on text that doesn't
                    # contain it → find() returns -1 → garbled splits.
                    delta_token_ids,
                )
            except Exception as e:
                logger.warning(f"Streaming reasoning failed: {e}")
                return None

            if delta_msg is None:
                return None

            # Check if reasoning has ended — the parser returns content
            # when it sees </think>. After that, switch to tool parsing.
            content = getattr(delta_msg, "content", None)
            reasoning = (
                getattr(delta_msg, "reasoning_content", None)
                or getattr(delta_msg, "reasoning", None)
            )

            if content is not None:
                # Reasoning just ended — content is the first post-think text
                self._reasoning_ended = True
                logger.info("[stream] Reasoning ended, switching to content/tool mode")

            return delta_msg

        # ---- Content / Tool phase (after reasoning) ----
        if self.tool_parser and request:
            fn = _resolve(
                self.tool_parser,
                "extract_tool_calls_streaming",
            )
            if fn is not None:
                try:
                    delta_msg = fn(
                        previous_text,
                        current_text,
                        delta_text,
                        previous_token_ids,
                        current_token_ids,
                        delta_token_ids,  # actual delta
                        request,
                    )
                    return delta_msg
                except Exception as e:
                    logger.warning(f"Streaming tool extraction failed: {e}")

        # ---- Fallback: no parser or no request — pass through as content ----
        # Create a simple object with .content attribute
        return _SimpleDelta(content=delta_text)


class _SimpleDelta:
    """Minimal DeltaMessage stand-in when no parser is active."""

    __slots__ = ("content", "reasoning_content", "reasoning", "tool_calls")

    def __init__(self, content=None, reasoning_content=None, tool_calls=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.reasoning = None
        self.tool_calls = tool_calls
