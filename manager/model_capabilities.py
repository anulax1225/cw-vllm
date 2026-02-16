"""
Model Capabilities Detection.

Auto-detects reasoning parser, tool call parser, and chat template
for a model based on its architecture and name.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("vllm-manager")

# Bundled templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass
class ModelCapabilities:
    """Detected model capabilities for reasoning and tool calling."""

    # Reasoning
    reasoning_parser: Optional[str] = None  # "qwen3", "deepseek_r1", etc.
    default_enable_thinking: bool = True  # server-wide default

    # Tool calling
    enable_tool_choice: bool = False  # --enable-auto-tool-choice
    tool_call_parser: Optional[str] = None  # "hermes", "mistral", etc.
    chat_template: Optional[str] = None  # path to .jinja override or None

    # Metadata (for /api/status)
    family: str = "unknown"  # "qwen3", "llama3", etc.
    supports_thinking: bool = False
    supports_tools: bool = False


# Type alias for match functions
MatchFn = Callable[[str, str], bool]


# Model family definitions
# Order matters — more specific entries before generic ones. First match wins.
MODEL_FAMILIES: list[dict] = [
    # ── Qwen3 ──────────────────────────────────────────────────
    {
        "family": "qwen3",
        "match": lambda arch, name: ("Qwen3" in arch or "qwen3" in name.lower()),
        "reasoning_parser": "qwen3",
        "default_enable_thinking": True,
        "tool_call_parser": "hermes",
        "supports_thinking": True,
        "supports_tools": True,
    },
    # ── QwQ (Qwen reasoning-only) ──────────────────────────────
    {
        "family": "qwq",
        "match": lambda arch, name: "qwq" in name.lower(),
        "reasoning_parser": "deepseek_r1",
        "tool_call_parser": "hermes",
        "supports_thinking": True,
        "supports_tools": True,
    },
    # ── Qwen2.5 ────────────────────────────────────────────────
    {
        "family": "qwen2.5",
        "match": lambda arch, name: (
            "Qwen2" in arch
            and ("qwen2.5" in name.lower() or "qwen2-5" in name.lower())
        ),
        "reasoning_parser": None,
        "tool_call_parser": "hermes",
        "supports_thinking": False,
        "supports_tools": True,
    },
    # ── DeepSeek R1 / V3 ───────────────────────────────────────
    {
        "family": "deepseek_r1",
        "match": lambda arch, name: (
            "DeepSeek" in arch
            and ("r1" in name.lower() or "v3" in name.lower())
        ),
        "reasoning_parser": "deepseek_r1",
        "tool_call_parser": "hermes",
        "supports_thinking": True,
        "supports_tools": True,
    },
    # ── DeepSeek V2 / Coder ────────────────────────────────────
    {
        "family": "deepseek_v2",
        "match": lambda arch, name: "DeepSeek" in arch,
        "reasoning_parser": None,
        "tool_call_parser": "hermes",
        "supports_thinking": False,
        "supports_tools": True,
    },
    # ── Llama 3.x / 4.x ───────────────────────────────────────
    {
        "family": "llama3",
        "match": lambda arch, name: (
            "Llama" in arch
            and any(
                v in name.lower()
                for v in ["llama-3", "llama3", "llama-4", "llama4"]
            )
        ),
        "reasoning_parser": None,
        "tool_call_parser": "llama3_json",
        "chat_template_file": "tool_chat_template_llama3_json.jinja",
        "supports_thinking": False,
        "supports_tools": True,
    },
    # ── Mistral / Mixtral ──────────────────────────────────────
    {
        "family": "mistral",
        "match": lambda arch, name: any(a in arch for a in ["Mistral", "Mixtral"]),
        "reasoning_parser": None,
        "tool_call_parser": "mistral",
        "chat_template_file": "tool_chat_template_mistral_parallel.jinja",
        "supports_thinking": False,
        "supports_tools": True,
    },
    # ── Gemma 3 ────────────────────────────────────────────────
    {
        "family": "gemma3",
        "match": lambda arch, name: (
            "Gemma" in arch
            and ("gemma-3" in name.lower() or "gemma3" in name.lower())
        ),
        "reasoning_parser": None,
        "tool_call_parser": "pythonic",
        "chat_template_file": "tool_chat_template_gemma3_pythonic.jinja",
        "supports_thinking": False,
        "supports_tools": True,
    },
    # ── IBM Granite 3.2+ ───────────────────────────────────────
    {
        "family": "granite",
        "match": lambda arch, name: "Granite" in arch,
        "reasoning_parser": "granite",
        "default_enable_thinking": False,  # opt-in via chat_template_kwargs
        "tool_call_parser": "granite",
        "supports_thinking": True,
        "supports_tools": True,
    },
    # ── Hermes (NousResearch) ──────────────────────────────────
    {
        "family": "hermes",
        "match": lambda arch, name: "hermes" in name.lower(),
        "reasoning_parser": None,
        "tool_call_parser": "hermes",
        "supports_thinking": False,
        "supports_tools": True,
    },
    # ── InternLM ───────────────────────────────────────────────
    {
        "family": "internlm",
        "match": lambda arch, name: "InternLM" in arch,
        "reasoning_parser": None,
        "tool_call_parser": "internlm",
        "supports_thinking": False,
        "supports_tools": True,
    },
    # ── Hunyuan A13B ───────────────────────────────────────────
    {
        "family": "hunyuan",
        "match": lambda arch, name: "hunyuan" in name.lower(),
        "reasoning_parser": "hunyuan_a13b",
        "tool_call_parser": "hermes",
        "supports_thinking": True,
        "supports_tools": True,
    },
    # ── Phi-3 / Phi-4 ─────────────────────────────────────────
    {
        "family": "phi",
        "match": lambda arch, name: "Phi" in arch,
        "reasoning_parser": None,
        "tool_call_parser": "hermes",
        "supports_thinking": False,
        "supports_tools": True,
    },
    # ── Fallback: unknown model ────────────────────────────────
    # No explicit match — _detect_fallback_parsers() handles defaults.
]


def _resolve_template_path(filename: str) -> Optional[str]:
    """
    Find a chat template file. Search order:
    1. Bundled in manager's templates/ directory
    2. vLLM's examples/ directory (if installed from source)
    """
    # 1. Our bundled templates
    bundled = TEMPLATES_DIR / filename
    if bundled.exists():
        return str(bundled)

    # 2. vLLM's examples directory
    try:
        import vllm

        vllm_root = Path(vllm.__file__).parent.parent
        vllm_template = vllm_root / "examples" / filename
        if vllm_template.exists():
            return str(vllm_template)
    except Exception:
        pass

    logger.warning(f"Chat template {filename} not found, using model default")
    return None


def _detect_fallback_parsers(
    model_id: str,
    config: dict,
    default_tool_parser: str = "hermes",
    default_reasoning_parser: str = "deepseek_r1",
) -> ModelCapabilities:
    """
    Detect sensible default parsers for an unknown model by inspecting
    its chat template and tokenizer vocabulary for known patterns.

    Strategy:
      1. If the model's chat template contains <think> tags → enable
         deepseek_r1 reasoning parser (generic <think>...</think>)
      2. If the chat template contains <tool_call> or tool_call patterns
         → enable hermes tool parser
      3. If no template clues, still default to the configured defaults
         as they are safe no-ops when the model doesn't emit those tokens.
    """
    caps = ModelCapabilities()
    caps.family = "unknown"

    # Try to read the chat template from the model's tokenizer_config.json
    chat_template_str = _get_chat_template_string(model_id, config)

    has_think = False
    has_tool_call = False

    if chat_template_str:
        tpl_lower = chat_template_str.lower()
        # Check for <think> reasoning support
        has_think = "<think>" in tpl_lower or "think" in tpl_lower
        # Check for Hermes-style tool_call tags
        has_tool_call = (
            "<tool_call>" in tpl_lower
            or "tool_call" in tpl_lower
            or "tool_calls" in tpl_lower
        )

    # Apply configured defaults — these parsers are safe no-ops if the model
    # doesn't emit the relevant tokens. Better to have them ready than
    # to miss tool calls or reasoning from an unknown model.
    if default_reasoning_parser:
        caps.reasoning_parser = default_reasoning_parser
        caps.default_enable_thinking = True
        caps.supports_thinking = True

    if default_tool_parser:
        caps.tool_call_parser = default_tool_parser
        caps.enable_tool_choice = True
        caps.supports_tools = True

    # Log what we detected
    evidence = []
    if has_think:
        evidence.append("<think> in template")
    if has_tool_call:
        evidence.append("<tool_call> in template")
    if not evidence:
        evidence.append("no template clues, using configured defaults")

    logger.info(
        f"[caps] {model_id} -> unknown family, fallback parsers: "
        f"reasoning={caps.reasoning_parser or 'none'}, "
        f"tools={caps.tool_call_parser or 'none'} "
        f"({', '.join(evidence)})"
    )

    return caps


def _get_chat_template_string(model_id: str, config: dict) -> Optional[str]:
    """
    Try to extract the chat template string from the model.
    Checks tokenizer_config.json in the HF cache.
    """
    try:
        from huggingface_hub import scan_cache_dir
        from huggingface_hub.constants import HF_HUB_CACHE
        import json, os

        cache_dir = Path(os.environ.get("HF_HOME", HF_HUB_CACHE))
        if not cache_dir.exists():
            return None

        cache_info = scan_cache_dir(cache_dir)
        for repo in cache_info.repos:
            if repo.repo_id == model_id and repo.repo_type == "model":
                for rev in repo.revisions:
                    tokenizer_config = rev.snapshot_path / "tokenizer_config.json"
                    if tokenizer_config.exists():
                        with open(tokenizer_config) as f:
                            tok_config = json.load(f)
                        template = tok_config.get("chat_template")
                        if isinstance(template, str):
                            return template
                        # Some models have a list of templates
                        if isinstance(template, list) and template:
                            # Return the first/default one
                            for t in template:
                                if isinstance(t, dict) and "template" in t:
                                    return t["template"]
                                if isinstance(t, str):
                                    return t
    except Exception as e:
        logger.debug(f"Could not read chat template for {model_id}: {e}")

    return None


def detect_capabilities(
    model_id: str,
    config: dict,
    user_overrides: Optional[dict] = None,
    default_tool_parser: str = "hermes",
    default_reasoning_parser: str = "deepseek_r1",
) -> ModelCapabilities:
    """
    Auto-detect reasoning parser, tool parser, and chat template
    for a model based on its architecture and name.

    Args:
        model_id: HuggingFace model ID (e.g. "Qwen/Qwen3-8B")
        config: Model's config.json as a dict
        user_overrides: Explicit settings from env vars that should
                       not be overridden (VLLM_REASONING_PARSER, etc.)
        default_tool_parser: Fallback tool parser for unknown models
                            (from VLLM_DEFAULT_TOOL_PARSER env var)
        default_reasoning_parser: Fallback reasoning parser for unknown models
                                 (from VLLM_DEFAULT_REASONING_PARSER env var)

    Returns:
        ModelCapabilities with detected parsers and template
    """
    user_overrides = user_overrides or {}
    caps = ModelCapabilities()

    # Extract architecture and name for matching
    architectures = config.get("architectures", [])
    arch_str = " ".join(architectures)
    name_str = config.get("_name_or_path", model_id)

    # Find matching family
    for family in MODEL_FAMILIES:
        match_fn = family["match"]
        if match_fn(arch_str, name_str):
            caps.family = family["family"]
            caps.supports_thinking = family.get("supports_thinking", False)
            caps.supports_tools = family.get("supports_tools", False)

            # Reasoning parser
            if "VLLM_REASONING_PARSER" not in user_overrides:
                caps.reasoning_parser = family.get("reasoning_parser")
                caps.default_enable_thinking = family.get(
                    "default_enable_thinking", True
                )
            else:
                caps.reasoning_parser = user_overrides["VLLM_REASONING_PARSER"]

            # Tool call parser
            if "VLLM_TOOL_PARSER" not in user_overrides:
                caps.tool_call_parser = family.get("tool_call_parser")
                caps.enable_tool_choice = caps.tool_call_parser is not None
            else:
                caps.tool_call_parser = user_overrides["VLLM_TOOL_PARSER"]
                caps.enable_tool_choice = bool(caps.tool_call_parser)

            # Chat template override
            if "VLLM_CHAT_TEMPLATE" not in user_overrides:
                template_file = family.get("chat_template_file")
                if template_file:
                    caps.chat_template = _resolve_template_path(template_file)
            else:
                caps.chat_template = user_overrides["VLLM_CHAT_TEMPLATE"]

            logger.info(
                f"[caps] {model_id} -> family={caps.family} "
                f"reasoning={caps.reasoning_parser} "
                f"tools={caps.tool_call_parser} "
                f"template={'custom' if caps.chat_template else 'default'}"
            )
            return caps

    # ── No match — apply smart fallback defaults ──
    # Use configured defaults (hermes + deepseek_r1 by default).
    # These parsers are no-ops if the model doesn't emit their tokens.
    caps = _detect_fallback_parsers(
        model_id, config,
        default_tool_parser=default_tool_parser,
        default_reasoning_parser=default_reasoning_parser,
    )

    # Still respect user overrides
    if "VLLM_REASONING_PARSER" in user_overrides:
        caps.reasoning_parser = user_overrides["VLLM_REASONING_PARSER"]
    if "VLLM_TOOL_PARSER" in user_overrides:
        caps.tool_call_parser = user_overrides["VLLM_TOOL_PARSER"]
        caps.enable_tool_choice = bool(caps.tool_call_parser)
    if "VLLM_CHAT_TEMPLATE" in user_overrides:
        caps.chat_template = user_overrides["VLLM_CHAT_TEMPLATE"]

    return caps
