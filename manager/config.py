"""
vLLM Manager Configuration.

Uses pydantic-settings for env var parsing with sensible defaults.
All settings are configurable via environment variables.
"""

import re
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings


def parse_duration(duration_str: str) -> Optional[int]:
    """
    Parse Ollama-style duration string to seconds.

    Examples:
        "5m" -> 300
        "1h" -> 3600
        "30s" -> 30
        "0" -> 0 (unload immediately after request)
        "-1" -> -1 (never unload, keep forever)
        "300" -> 300 (raw seconds)
    """
    if not duration_str:
        return 300  # Default 5 minutes

    duration_str = duration_str.strip().lower()

    # Special cases
    if duration_str == "-1":
        return -1  # Never unload
    if duration_str == "0":
        return 0  # Unload immediately

    # Try parsing with unit suffix
    match = re.match(r'^(-?\d+\.?\d*)\s*(s|m|h|d)?$', duration_str)
    if not match:
        return 300  # Default on invalid format

    value = float(match.group(1))
    unit = match.group(2) or 's'

    multipliers = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
    return int(value * multipliers.get(unit, 1))


class ManagerConfig(BaseSettings):
    """Configuration from environment variables."""

    # Manager settings
    manager_port: int = 8000

    # Keep-alive: seconds before unloading idle model
    # -1 = never unload, 0 = unload immediately after each request
    vllm_keep_alive: str = "5m"

    # HuggingFace token (either name works)
    hugging_face_hub_token: str = ""
    huggingface_api_key: str = ""

    # vLLM engine settings
    vllm_gpu_memory_utilization: float = 0.7
    vllm_tensor_parallel_size: int = 1
    vllm_max_model_len: Optional[int] = None
    vllm_quantization: str = ""
    vllm_target_device: str = ""  # "cpu" for CPU mode, empty for GPU
    vllm_tokenizer: str = ""  # Explicit tokenizer override for community-quantized models

    # Preflight validation settings
    vllm_trust_remote_code: bool = True  # Default enabled for user-chosen models
    vllm_preflight_enabled: bool = True  # Can disable for debugging
    vllm_auto_cap_context: bool = True  # Auto-reduce max_model_len if OOM likely
    vllm_enforce_eager: bool = False  # Force eager mode (no CUDA graphs)
    vllm_dtype: str = "auto"  # Explicit dtype override

    # Capability overrides (None = auto-detect from model family)
    vllm_reasoning_parser: Optional[str] = None  # VLLM_REASONING_PARSER
    vllm_tool_parser: Optional[str] = None  # VLLM_TOOL_PARSER
    vllm_chat_template: Optional[str] = None  # VLLM_CHAT_TEMPLATE

    # Fallback defaults for unknown models (used when auto-detect finds no match).
    # "hermes" is the most common tool call format (<tool_call>...</tool_call>)
    # "deepseek_r1" uses generic <think>...</think> tags adopted by many models.
    # These parsers are safe no-ops when the model doesn't emit those tokens.
    # Set to "" to disable fallback (unknown models get no parsers).
    vllm_default_tool_parser: str = "hermes"
    vllm_default_reasoning_parser: str = "deepseek_r1"

    # KV cache offloading — fraction of system RAM allowed for KV offload buffer.
    # KV offloading moves KV cache blocks to CPU via async DMA (no pinned memory
    # needed, works on WSL). Set to 0 to disable KV offloading entirely.
    vllm_kv_offload_max_ram_fraction: float = 0.5

    @field_validator("vllm_max_model_len", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        """Treat empty strings from env vars as None."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator(
        "vllm_reasoning_parser",
        "vllm_tool_parser",
        "vllm_chat_template",
        mode="before",
    )
    @classmethod
    def empty_str_to_none_caps(cls, v):
        """Treat empty strings from env vars as None for capability overrides."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator(
        "vllm_default_tool_parser",
        "vllm_default_reasoning_parser",
        mode="before",
    )
    @classmethod
    def normalize_default_parsers(cls, v):
        """Treat empty strings as disabled, strip whitespace."""
        if isinstance(v, str):
            return v.strip()
        return v

    model_config = {"env_prefix": "", "extra": "ignore"}

    @property
    def hf_token(self) -> str:
        """Get HuggingFace token from either env var."""
        return self.hugging_face_hub_token or self.huggingface_api_key

    @property
    def keep_alive_seconds(self) -> int:
        """Parse keep_alive string to seconds."""
        return parse_duration(self.vllm_keep_alive)

    @property
    def is_cpu(self) -> bool:
        """Check if running in CPU mode."""
        return self.vllm_target_device == "cpu"
