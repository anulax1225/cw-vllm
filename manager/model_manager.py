"""
Model Manager — Embedded vLLM Engine with Ollama-style semantics.

Manages AsyncLLMEngine lifecycle:
- Auto-load on inference request
- Auto-switch when model changes
- Auto-unload after keep_alive inactivity timeout
"""

import asyncio
import gc
import inspect
import json
import logging
import time
import uuid
from typing import AsyncIterator, Optional

import torch
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from cache_manager import HFCacheManager
from config import ManagerConfig
from model_capabilities import ModelCapabilities, detect_capabilities
from output_parser import OutputParser, ParsedOutput
from preflight import PreflightError, classify_engine_error, preflight_check
from tokenizer_resolver import ensure_tokenizer_available, resolve_tokenizer

logger = logging.getLogger("vllm-manager")


async def maybe_await(obj):
    """Handle vLLM methods that may or may not be async."""
    if inspect.isawaitable(obj):
        return await obj
    return obj


class ModelManager:
    """
    Manages vLLM engine lifecycle with Ollama-style semantics:
    - Auto-load on inference request
    - Auto-switch when request model ≠ loaded model
    - Auto-unload after keep_alive inactivity timeout
    """

    def __init__(self, config: ManagerConfig):
        self.config = config

        # Engine state
        self.engine: Optional[AsyncLLMEngine] = None
        self.tokenizer = None
        self.current_model: Optional[str] = None
        self.state: str = "idle"  # idle | loading | ready | unloading | error
        self.error_message: Optional[str] = None
        self._last_preflight = None  # Store last preflight result for debugging
        self.caps: Optional[ModelCapabilities] = None  # Detected model capabilities
        self.parser: Optional[OutputParser] = None  # Output parser for reasoning/tools

        # Keep-alive
        self.last_used: float = 0
        self.keep_alive: int = config.keep_alive_seconds  # seconds, -1 = forever, 0 = immediate
        self._keep_alive_task: Optional[asyncio.Task] = None

        # Concurrency
        self._lock = asyncio.Lock()

    # ── Ollama-style auto-load ──────────────────────────────────────

    async def ensure_loaded(self, model: str, keep_alive: Optional[int] = None) -> None:
        """
        Ensure the requested model is loaded and ready.
        If a different model is loaded, switch (unload + load).
        If no model is loaded, load it.

        This is called before every inference request — the Ollama pattern.
        The keep_alive parameter can override the default per-request.
        """
        # Fast path: model already loaded and ready
        if self.current_model == model and self.state == "ready":
            self.last_used = time.time()
            if keep_alive is not None:
                self.keep_alive = keep_alive
                self._restart_keep_alive_timer()
            return

        async with self._lock:
            # Double-check after acquiring lock
            if self.current_model == model and self.state == "ready":
                self.last_used = time.time()
                return

            # Need to load (possibly after unloading current)
            await self._load(model)

            if keep_alive is not None:
                self.keep_alive = keep_alive

    # ── Load / Unload ───────────────────────────────────────────────

    async def _load(self, model: str) -> None:
        """Create vLLM engine for the given model."""

        # Unload current model if any
        if self.engine is not None:
            await self._unload()

        self.state = "loading"
        self.current_model = model
        self.error_message = None
        self._last_preflight = None
        logger.info(f"Loading model: {model}")

        try:
            # Resolve tokenizer for community-quantized models
            tokenizer = self.config.vllm_tokenizer or None
            if not tokenizer:
                tokenizer = resolve_tokenizer(model, hf_token=self.config.hf_token)

            if tokenizer:
                logger.info(f"Using external tokenizer: {tokenizer}")
                ensure_tokenizer_available(tokenizer, hf_token=self.config.hf_token)

            # Preflight validation (runs before engine creation)
            engine_overrides = {}
            if self.config.vllm_preflight_enabled:
                preflight_result = await preflight_check(
                    model_id=model,
                    config=self.config,
                    resolved_tokenizer=tokenizer,
                )
                self._last_preflight = preflight_result

                if preflight_result.blocked:
                    raise PreflightError(preflight_result)

                # Log warnings
                for warning in preflight_result.warnings:
                    logger.warning(f"[PREFLIGHT] {warning}")

                engine_overrides = preflight_result.engine_overrides

            # Detect model capabilities from config
            model_config = (
                preflight_result.model_config
                if self.config.vllm_preflight_enabled and preflight_result
                else {}
            )
            self.caps = detect_capabilities(
                model, model_config, self._get_user_overrides()
            )

            # Build engine args with preflight overrides
            # Note: tool_call_parser, reasoning_parser are NOT engine args —
            # they are used by OutputParser (instantiated after engine creation)
            engine_args = AsyncEngineArgs(
                model=model,
                tokenizer=tokenizer or model,
                gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                tensor_parallel_size=self.config.vllm_tensor_parallel_size,
                max_model_len=engine_overrides.get(
                    "max_model_len", self.config.vllm_max_model_len
                ),
                quantization=engine_overrides.get(
                    "quantization", self.config.vllm_quantization or None
                ),
                load_format=engine_overrides.get(
                    "load_format", "auto"
                ),
                trust_remote_code=engine_overrides.get(
                    "trust_remote_code", self.config.vllm_trust_remote_code
                ),
                dtype=engine_overrides.get("dtype", self.config.vllm_dtype),
                enforce_eager=engine_overrides.get(
                    "enforce_eager", self.config.vllm_enforce_eager
                ),
                cpu_offload_gb=engine_overrides.get("cpu_offload_gb", 0),
            )

            # CPU-specific settings
            if self.config.is_cpu:
                engine_args.device = "cpu"

            # Create engine (this downloads + loads the model)
            logger.info(f"Creating AsyncLLMEngine for {model}...")
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)

            # Get tokenizer for chat template
            self.tokenizer = await maybe_await(self.engine.get_tokenizer())

            # Create output parser using vLLM's native parser classes
            self.parser = OutputParser(
                tokenizer=self.tokenizer,
                reasoning_parser_name=self.caps.reasoning_parser,
                tool_parser_name=self.caps.tool_call_parser,
            )
            logger.info(
                f"Output parser initialized: "
                f"reasoning={self.caps.reasoning_parser}, "
                f"tools={self.caps.tool_call_parser}"
            )

            self.state = "ready"
            self.last_used = time.time()
            self._start_keep_alive_timer()

            logger.info(f"Model ready: {model}")

        except PreflightError as e:
            self.state = "error"
            self.error_message = e.result.block_reason
            self._last_preflight = e.result
            self.engine = None
            self.tokenizer = None
            self.parser = None
            logger.error(f"Preflight blocked {model}: {e.result.block_reason}")
            raise

        except torch.cuda.OutOfMemoryError as e:
            self.state = "error"
            error_result = classify_engine_error(e)
            self.error_message = error_result.message
            self._cleanup_gpu()
            logger.error(f"OOM loading {model}: {e}")
            raise RuntimeError(self.error_message) from e

        except Exception as e:
            self.state = "error"
            error_result = classify_engine_error(e)
            self.error_message = error_result.message
            self._cleanup_gpu()
            logger.error(f"Failed to load {model}: {e}")
            raise

    def _cleanup_gpu(self) -> None:
        """Nuclear GPU cleanup after failed load attempts."""
        self.engine = None
        self.tokenizer = None
        self.parser = None

        # Aggressive garbage collection
        gc.collect()
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        self.state = "idle"
        logger.info("GPU memory cleanup complete")

    async def _unload(self) -> None:
        """Destroy engine and free all GPU memory."""
        if self.engine is None:
            self.state = "idle"
            return

        model = self.current_model
        self.state = "unloading"
        logger.info(f"Unloading model: {model}")

        self._stop_keep_alive_timer()

        # Destroy engine
        # vLLM's AsyncLLMEngine holds GPU tensors — deleting it triggers cleanup
        engine = self.engine
        self.engine = None
        self.tokenizer = None
        self.current_model = None
        self.caps = None
        self.parser = None
        del engine

        # Force GPU memory release
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self.state = "idle"
        logger.info(f"Unloaded {model}, GPU memory freed")

    async def unload(self) -> None:
        """Public unload (acquires lock)."""
        async with self._lock:
            await self._unload()

    # ── Chat Completion ────────────────────────────────────────────

    async def chat_completion(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
        tools: list = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        Generate chat completion using the loaded model.
        Returns OpenAI-compatible response format.

        Uses vLLM's native parsers (via OutputParser) for reasoning
        and tool call extraction from raw engine output.
        """
        if self.engine is None or self.tokenizer is None:
            raise RuntimeError("No model loaded")

        self.last_used = time.time()

        # Build a minimal request object for parser consumption
        parser_request = self._build_parser_request(
            messages=messages, tools=tools, tool_choice=tool_choice
        )

        # Apply chat template (include tools and custom template if applicable)
        template_kwargs = {}
        if tools and self.caps and self.caps.supports_tools:
            template_kwargs["tools"] = tools
        if self.caps and self.caps.chat_template:
            template_kwargs["chat_template"] = self.caps.chat_template

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
        except Exception as e:
            logger.warning(f"Chat template failed, using fallback: {e}")
            # Fallback: simple concatenation
            prompt = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in messages
            ) + "\nassistant:"

        # Build sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=kwargs.get("top_p", 1.0),
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            presence_penalty=kwargs.get("presence_penalty", 0.0),
            stop=kwargs.get("stop"),
        )

        request_id = str(uuid.uuid4())

        if stream:
            return self._stream_chat_completion(
                prompt, sampling_params, request_id, parser_request
            )
        else:
            return await self._complete_chat(
                prompt, sampling_params, request_id, parser_request
            )

    async def _complete_chat(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str,
        parser_request=None,
    ) -> dict:
        """Non-streaming chat completion with reasoning/tool parsing."""
        results_generator = self.engine.generate(prompt, sampling_params, request_id)

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        if final_output is None or not final_output.outputs:
            raise RuntimeError("No output generated")

        output = final_output.outputs[0]
        raw_text = output.text
        raw_finish_reason = self._map_finish_reason(output.finish_reason)

        # Parse output for reasoning and tool calls
        if self.parser and (self.parser.has_reasoning or self.parser.has_tools):
            parsed = self.parser.parse(
                model_output=raw_text,
                request=parser_request,
                finish_reason=raw_finish_reason,
            )
        else:
            parsed = ParsedOutput(content=raw_text, finish_reason=raw_finish_reason)

        # Build message
        message = {"role": "assistant", "content": parsed.content}

        if parsed.reasoning_content is not None:
            message["reasoning_content"] = parsed.reasoning_content

        if parsed.tools_called:
            message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in parsed.tool_calls
            ]

        return {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.current_model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": parsed.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(final_output.prompt_token_ids),
                "completion_tokens": len(output.token_ids),
                "total_tokens": len(final_output.prompt_token_ids)
                + len(output.token_ids),
            },
        }

    async def _stream_chat_completion(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str,
        parser_request=None,
    ) -> AsyncIterator[str]:
        """Streaming chat completion (SSE) with reasoning/tool parsing."""
        results_generator = self.engine.generate(prompt, sampling_params, request_id)

        # Per-request parser instance (streaming parsers have internal state)
        has_parsers = self.parser and (
            self.parser.has_reasoning or self.parser.has_tools
        )
        if has_parsers:
            request_parser = OutputParser(
                tokenizer=self.tokenizer,
                reasoning_parser_name=self.caps.reasoning_parser if self.caps else None,
                tool_parser_name=self.caps.tool_call_parser if self.caps else None,
            )
        else:
            request_parser = None

        previous_text = ""
        previous_token_ids: list[int] = []

        async for request_output in results_generator:
            if not request_output.outputs:
                continue

            output = request_output.outputs[0]
            current_text = output.text
            current_token_ids = list(output.token_ids)
            delta_text = current_text[len(previous_text):]
            delta_token_ids = current_token_ids[len(previous_token_ids):]
            is_finished = output.finish_reason is not None

            if delta_text or is_finished:
                # Build SSE chunk delta
                chunk_delta = {}

                if request_parser and delta_text:
                    delta_msg = request_parser.parse_streaming(
                        previous_text=previous_text,
                        current_text=current_text,
                        delta_text=delta_text,
                        previous_token_ids=previous_token_ids,
                        current_token_ids=current_token_ids,
                        delta_token_ids=delta_token_ids,
                        request=parser_request,
                    )

                    if delta_msg is not None:
                        if getattr(delta_msg, "reasoning_content", None) is not None:
                            chunk_delta["reasoning_content"] = (
                                delta_msg.reasoning_content
                            )
                        if getattr(delta_msg, "content", None) is not None:
                            chunk_delta["content"] = delta_msg.content
                        if getattr(delta_msg, "tool_calls", None):
                            chunk_delta["tool_calls"] = [
                                tc.model_dump()
                                if hasattr(tc, "model_dump")
                                else tc
                                for tc in delta_msg.tool_calls
                            ]
                    else:
                        # Parser returned None — pass raw delta as content
                        chunk_delta["content"] = delta_text
                elif delta_text:
                    # No parser — raw delta is content
                    chunk_delta["content"] = delta_text

                # Emit content/reasoning chunk if we have delta
                if chunk_delta:
                    chunk = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": self.current_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": chunk_delta,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                # Send finish reason on last chunk
                if is_finished:
                    finish_reason = self._map_finish_reason(output.finish_reason)

                    # Check if tools were called across the full output
                    if self.parser and self.parser.has_tools and parser_request:
                        full_parsed = self.parser.parse(
                            current_text, parser_request
                        )
                        if full_parsed.tools_called:
                            finish_reason = "tool_calls"

                    chunk = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": self.current_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

            previous_text = current_text
            previous_token_ids = current_token_ids

        yield "data: [DONE]\n\n"

    @staticmethod
    def _map_finish_reason(reason) -> str:
        """Map vLLM finish reason to OpenAI format."""
        if reason is None:
            return None
        reason_str = str(reason).lower()
        if "stop" in reason_str:
            return "stop"
        elif "length" in reason_str:
            return "length"
        elif "tool" in reason_str:
            return "tool_calls"
        return "stop"

    def _build_parser_request(
        self,
        messages: list,
        tools: list = None,
        tool_choice: str = "auto",
    ):
        """Build a minimal request object for parser consumption.

        vLLM's parsers expect a ChatCompletionRequest. We try to construct one;
        if it fails (Pydantic validation too strict), we return None and parsers
        will fall back gracefully.
        """
        if not self.parser or (
            not self.parser.has_reasoning and not self.parser.has_tools
        ):
            return None

        try:
            from vllm.entrypoints.openai.protocol import ChatCompletionRequest

            return ChatCompletionRequest(
                model=self.current_model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice if tools else "none",
            )
        except Exception as e:
            logger.debug(
                f"Could not build ChatCompletionRequest, parsers will use None: {e}"
            )
            return None

    def _get_user_overrides(self) -> dict:
        """Collect explicit user overrides that should not be auto-detected."""
        overrides = {}
        if self.config.vllm_reasoning_parser:
            overrides["VLLM_REASONING_PARSER"] = self.config.vllm_reasoning_parser
        if self.config.vllm_tool_parser:
            overrides["VLLM_TOOL_PARSER"] = self.config.vllm_tool_parser
        if self.config.vllm_chat_template:
            overrides["VLLM_CHAT_TEMPLATE"] = self.config.vllm_chat_template
        return overrides

    # ── Keep-Alive Timer ────────────────────────────────────────────

    def _start_keep_alive_timer(self) -> None:
        """Start background task that unloads model after inactivity."""
        self._stop_keep_alive_timer()

        if self.keep_alive == 0:
            # keep_alive=0 means unload immediately after request
            # (handled separately after inference completes)
            return

        if self.keep_alive == -1:
            # -1 = keep loaded forever
            return

        self._keep_alive_task = asyncio.create_task(self._keep_alive_loop())

    def _stop_keep_alive_timer(self) -> None:
        """Stop the keep-alive timer."""
        if self._keep_alive_task and not self._keep_alive_task.done():
            self._keep_alive_task.cancel()
            self._keep_alive_task = None

    def _restart_keep_alive_timer(self) -> None:
        """Restart the timer with potentially new keep_alive value."""
        self._start_keep_alive_timer()

    async def _keep_alive_loop(self) -> None:
        """Periodically check if model should be unloaded due to inactivity."""
        try:
            while True:
                await asyncio.sleep(30)  # Check every 30 seconds

                if self.state != "ready" or self.keep_alive == -1:
                    continue

                elapsed = time.time() - self.last_used
                if elapsed >= self.keep_alive:
                    logger.info(
                        f"Keep-alive expired: {self.current_model} "
                        f"idle for {elapsed:.0f}s (limit: {self.keep_alive}s)"
                    )
                    async with self._lock:
                        # Re-check after acquiring lock (could have been used)
                        if time.time() - self.last_used >= self.keep_alive:
                            await self._unload()
                    break  # Timer done after unload
        except asyncio.CancelledError:
            pass

    # ── Status ──────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Get engine status for /api/status endpoint."""
        idle_for = time.time() - self.last_used if self.last_used > 0 else None

        # Calculate will_unload_in
        will_unload_in = None
        if idle_for and self.keep_alive > 0 and self.state == "ready":
            will_unload_in = max(0, round(self.keep_alive - idle_for, 1))

        # Format keep_alive for display
        if self.keep_alive == -1:
            keep_alive_str = "forever"
        elif self.keep_alive == 0:
            keep_alive_str = "0 (unload after each request)"
        else:
            keep_alive_str = f"{self.keep_alive}s"

        return {
            "state": self.state,
            "model": self.current_model,
            "keep_alive": keep_alive_str,
            "idle_seconds": round(idle_for, 1) if idle_for else None,
            "will_unload_in": will_unload_in,
            "error": self.error_message,
            "gpu_memory": self._get_gpu_info(),
            "capabilities": {
                "family": self.caps.family,
                "supports_thinking": self.caps.supports_thinking,
                "supports_tools": self.caps.supports_tools,
                "reasoning_parser": self.caps.reasoning_parser,
                "tool_call_parser": self.caps.tool_call_parser,
            }
            if self.caps
            else None,
        }

    @staticmethod
    def _get_gpu_info() -> Optional[dict]:
        """Get GPU memory info."""
        if not torch.cuda.is_available():
            return None
        try:
            return {
                "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
                "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            }
        except Exception:
            return None

    @staticmethod
    def _get_total_vram_gb() -> float:
        """Get total GPU VRAM in GB."""
        if not torch.cuda.is_available():
            return 0.0
        try:
            return torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            return 0.0
