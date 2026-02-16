"""
vLLM Manager Preflight Validation.

Runs validation checks before engine creation to catch configuration issues early
and provide actionable error messages.
"""

from __future__ import annotations

import gc, json, logging, os, struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from config import ManagerConfig

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Data Structures
# ════════════════════════════════════════════════════════════════════════════


class CheckSeverity(str, Enum):
    """Severity level for preflight checks."""

    BLOCK = "block"  # Cannot proceed, must fix
    WARN = "warn"  # Can proceed with potential issues
    AUTOFIX = "autofix"  # Automatically corrected


class CheckId(str, Enum):
    """Identifiers for preflight checks."""

    GPU_REQUIRED_QUANTIZATION = "GPU_REQUIRED_QUANTIZATION"
    UNSUPPORTED_ARCHITECTURE = "UNSUPPORTED_ARCHITECTURE"
    TRUST_REMOTE_CODE = "TRUST_REMOTE_CODE"
    OOM_ESTIMATE = "OOM_ESTIMATE"
    CONTEXT_LENGTH_MISMATCH = "CONTEXT_LENGTH_MISMATCH"
    CPU_DTYPE_INCOMPATIBLE = "CPU_DTYPE_INCOMPATIBLE"
    MISSING_TOKENIZER = "MISSING_TOKENIZER"
    VLLM_V1_CPU_INCOMPATIBLE = "VLLM_V1_CPU_INCOMPATIBLE"
    GATED_MODEL_NO_TOKEN = "GATED_MODEL_NO_TOKEN"
    GPU_COMPUTE_CAP_TOO_LOW = "GPU_COMPUTE_CAP_TOO_LOW"
    CPU_ALL_QUANT_UNSUPPORTED = "CPU_ALL_QUANT_UNSUPPORTED"
    MOE_ON_CPU = "MOE_ON_CPU"
    ENFORCE_EAGER_LOW_VRAM = "ENFORCE_EAGER_LOW_VRAM"
    NO_CONFIG = "NO_CONFIG"
    OOM_RUNTIME = "OOM_RUNTIME"
    OOM_KV_CACHE = "OOM_KV_CACHE"
    CUDA_TOOLKIT_MISMATCH = "CUDA_TOOLKIT_MISMATCH"
    CUDA_GRAPH_ERROR = "CUDA_GRAPH_ERROR"
    INCOMPATIBLE_REMOTE_CODE = "INCOMPATIBLE_REMOTE_CODE"
    UNKNOWN_ENGINE_ERROR = "UNKNOWN_ENGINE_ERROR"
    KV_OFFLOAD_ENABLED = "KV_OFFLOAD_ENABLED"


@dataclass
class CheckResult:
    """Result of a single preflight check."""

    check_id: CheckId
    severity: CheckSeverity
    message: str
    suggestion: Optional[str] = None
    auto_fixed: bool = False
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.check_id.value,
            "severity": self.severity.value,
            "message": self.message,
            "suggestion": self.suggestion,
            "auto_fixed": self.auto_fixed,
        }


@dataclass
class PlatformInfo:
    """Detected platform capabilities."""

    has_gpu: bool
    gpu_name: Optional[str] = None
    gpu_memory_bytes: int = 0
    gpu_compute_capability: Optional[tuple[int, int]] = None
    cpu_ram_bytes: int = 0
    is_wsl: bool = False
    vllm_version: Optional[str] = None

    @classmethod
    def detect(cls) -> PlatformInfo:
        """Detect platform capabilities."""
        import psutil

        info = cls(
            has_gpu=False,
            cpu_ram_bytes=psutil.virtual_memory().total,
        )

        # Detect GPU
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                info.has_gpu = True
                info.gpu_name = props.name
                info.gpu_memory_bytes = props.total_memory
                info.gpu_compute_capability = (props.major, props.minor)
        except ImportError:
            pass

        # Detect WSL (Windows Subsystem for Linux)
        # WSL does not support CUDA pinned memory (UVA), which breaks cpu_offload
        try:
            with open("/proc/version", "r") as f:
                version_str = f.read().lower()
                if "microsoft" in version_str or "wsl" in version_str:
                    info.is_wsl = True
        except Exception:
            pass

        # Detect vLLM version
        try:
            import vllm

            info.vllm_version = getattr(vllm, "__version__", None)
        except ImportError:
            pass

        return info


# Module-level cached platform info
_cached_platform: Optional[PlatformInfo] = None


def get_platform() -> PlatformInfo:
    """Get cached platform info (detected once per process)."""
    global _cached_platform
    if _cached_platform is None:
        _cached_platform = PlatformInfo.detect()
    return _cached_platform


@dataclass
class PreflightResult:
    """Result of all preflight checks."""

    model: str
    passed: bool = True
    blocked: bool = False
    block_reason: Optional[str] = None
    checks: list[CheckResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    engine_overrides: dict = field(default_factory=dict)
    platform: Optional[PlatformInfo] = None
    model_config: dict = field(default_factory=dict)  # Preserve for capabilities detection

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "model": self.model,
            "passed": self.passed,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "checks": [c.to_dict() for c in self.checks],
            "warnings": self.warnings,
            "engine_overrides": self.engine_overrides,
            "platform": {
                "has_gpu": self.platform.has_gpu,
                "gpu_name": self.platform.gpu_name,
                "gpu_memory_gb": (
                    round(self.platform.gpu_memory_bytes / 1e9, 1)
                    if self.platform.gpu_memory_bytes
                    else None
                ),
                "gpu_compute_capability": self.platform.gpu_compute_capability,
                "cpu_ram_gb": round(self.platform.cpu_ram_bytes / 1e9, 1),
                "vllm_version": self.platform.vllm_version,
            }
            if self.platform
            else None,
        }


class PreflightError(Exception):
    """Raised when preflight checks block model loading."""

    def __init__(self, result: PreflightResult):
        self.result = result
        super().__init__(result.block_reason)


# ════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ════════════════════════════════════════════════════════════════════════════

def _try_kv_offload(
    model_config: dict,
    platform,  # PlatformInfo
    kv_cache_bytes: int,
    available_vram: int,
    weight_bytes: int,
    max_model_len: int,
    kv_offload_max_ram_fraction: float,
    details: dict,
) -> "Optional[CheckResult]":
    """
    Determine if KV cache offloading to CPU can avoid capping context length.

    KV offloading (--kv-offloading-size) moves evicted KV cache blocks to CPU
    RAM asynchronously via DMA. Unlike cpu_offload_gb (model weight offloading),
    this has minimal impact on per-token latency because transfers don't block
    the forward pass.

    Returns a CheckResult with AUTOFIX severity if KV offloading is viable,
    or None if it's not (caller should fall back to capping context).

    The offloading buffer is capped at kv_offload_max_ram_fraction of system RAM
    to prevent OOM on the host side.
    """
    if not platform.has_gpu:
        return None

    # KV offloading requires vLLM >= 0.11 (offloading connector)
    # and works best on >= 0.12 (contiguous block layout)
    try:
        import vllm
        version = getattr(vllm, "__version__", "0.0.0")
        parts = version.split(".")
        major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        if major == 0 and minor < 11:
            return None  # Too old for native KV offloading
    except (ImportError, ValueError, IndexError):
        return None

    # Check if KV offloading is disabled (fraction == 0)
    if kv_offload_max_ram_fraction <= 0:
        return None

    # Calculate how much KV cache exceeds available VRAM (after weights)
    vram_for_kv = available_vram - weight_bytes
    if vram_for_kv <= 0:
        # Weights alone fill VRAM — KV offloading alone won't help
        return None

    kv_overflow_bytes = kv_cache_bytes - vram_for_kv
    if kv_overflow_bytes <= 0:
        # KV cache fits in VRAM, no offloading needed
        return None

    # Calculate the CPU RAM budget for KV offloading
    ram_budget_bytes = int(platform.cpu_ram_bytes * kv_offload_max_ram_fraction)

    # The offloading buffer doesn't need to hold the ENTIRE KV cache overflow.
    # It's a circular/LRU buffer: vLLM evicts blocks to CPU when GPU is full,
    # and loads them back when needed. A buffer ~= the overflow is a good target,
    # with some headroom for double-buffering.
    offload_buffer_bytes = int(kv_overflow_bytes * 1.2)  # 20% headroom

    if offload_buffer_bytes > ram_budget_bytes:
        # KV offloading would exceed the RAM budget — not viable at full context.
        # Caller should fall back to capping context length.
        details["kv_offload_rejected"] = True
        details["kv_offload_needed_gb"] = round(offload_buffer_bytes / 1e9, 1)
        details["kv_offload_ram_budget_gb"] = round(ram_budget_bytes / 1e9, 1)
        details["kv_offload_max_ram_fraction"] = kv_offload_max_ram_fraction
        return None

    offload_size_gb = max(1, int(offload_buffer_bytes / 1e9 + 0.5))  # Round up, min 1GB

    details["kv_offload_size_gb"] = offload_size_gb
    details["kv_overflow_bytes"] = kv_overflow_bytes
    details["kv_offload_ram_budget_gb"] = round(ram_budget_bytes / 1e9, 1)
    details["kv_offload_max_ram_fraction"] = kv_offload_max_ram_fraction

    return CheckResult(
        check_id=CheckId.KV_OFFLOAD_ENABLED,
        severity=CheckSeverity.AUTOFIX,
        message=f"KV cache ({kv_cache_bytes/1e9:.1f}GB) exceeds available VRAM for KV "
        f"({vram_for_kv/1e9:.1f}GB) by {kv_overflow_bytes/1e9:.1f}GB. "
        f"Auto-enabling native KV cache offloading to CPU "
        f"(buffer: {offload_size_gb}GB, "
        f"RAM budget: {round(ram_budget_bytes/1e9, 1)}GB = "
        f"{kv_offload_max_ram_fraction:.0%} of system RAM).",
        auto_fixed=True,
        details=details,
    )



def _get_snapshot_path(model_id: str) -> Optional[Path]:
    """Get the HuggingFace cache snapshot path for a model."""
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.constants import HF_HUB_CACHE

    cache_dir = Path(os.environ.get("HF_HOME", HF_HUB_CACHE))
    if not cache_dir.exists():
        return None

    try:
        cache_info = scan_cache_dir(cache_dir)
        for repo in cache_info.repos:
            if repo.repo_id == model_id and repo.repo_type == "model":
                for rev in repo.revisions:
                    if rev.snapshot_path.exists():
                        return rev.snapshot_path
    except Exception:
        pass

    return None


def _load_model_config(snapshot_path: Path) -> Optional[dict]:
    """Load config.json from model snapshot."""
    config_path = snapshot_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return None


def kv_cache_per_token(config: dict) -> int:
    """
    Calculate bytes per token for KV cache.

    Formula: 2 (K+V) × num_layers × num_kv_heads × head_dim × dtype_bytes

    Example for Qwen2.5-7B (28 layers, 4 KV heads, 128 head_dim):
        2 × 28 × 4 × 128 × 2 = 57,344 bytes ≈ 56 KB per token
        For 32K context: 56 KB × 32,768 = 1.79 GB
    """
    num_layers = config.get("num_hidden_layers", 32)
    num_attention_heads = config.get("num_attention_heads", 32)
    hidden_size = config.get("hidden_size", 4096)
    head_dim = config.get("head_dim", hidden_size // num_attention_heads)
    num_kv_heads = config.get("num_key_value_heads", num_attention_heads)  # GQA fallback
    dtype_bytes = 2  # float16/bfloat16 for KV cache

    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes


def estimate_model_memory(
    snapshot_path: Path,
    model_config: dict,
    max_model_len: Optional[int] = None,
) -> tuple[int, dict]:
    """
    Estimate model memory requirements from actual file sizes on disk.

    Uses the actual safetensor file sizes which correctly reflects quantization.

    Returns:
        (estimated_bytes, details_dict)
    """
    details: dict = {"files": [], "param_count": 0}

    # Sum actual file sizes on disk (most accurate for quantized models)
    weight_bytes = 0
    for st_file in snapshot_path.glob("*.safetensors"):
        try:
            file_size = st_file.stat().st_size
            weight_bytes += file_size
            details["files"].append({"name": st_file.name, "size": file_size})
        except Exception:
            pass

    # Also check for .bin files (older format)
    for bin_file in snapshot_path.glob("*.bin"):
        if "pytorch_model" in bin_file.name or "model" in bin_file.name:
            try:
                file_size = bin_file.stat().st_size
                weight_bytes += file_size
                details["files"].append({"name": bin_file.name, "size": file_size})
            except Exception:
                pass

    details["weight_bytes"] = weight_bytes

    # KV cache estimation using helper
    context_len = max_model_len or model_config.get(
        "max_position_embeddings",
        model_config.get("max_sequence_length", 4096),
    )
    kv_per_token = kv_cache_per_token(model_config)
    kv_cache_bytes = kv_per_token * context_len
    details["kv_per_token"] = kv_per_token
    details["kv_cache_bytes"] = kv_cache_bytes
    details["context_len"] = context_len

    # Total with overhead:
    # - CUDA context: ~1GB fixed overhead
    # - Activations/intermediates: ~20% of model size
    cuda_context_bytes = 1 * 1024 * 1024 * 1024  # 1GB
    activation_overhead = int((weight_bytes + kv_cache_bytes) * 0.20)
    total_estimate = weight_bytes + kv_cache_bytes + cuda_context_bytes + activation_overhead
    details["cuda_context_bytes"] = cuda_context_bytes
    details["activation_overhead"] = activation_overhead
    details["total_estimate"] = total_estimate

    return total_estimate, details


# ════════════════════════════════════════════════════════════════════════════
# Individual Check Functions
# ════════════════════════════════════════════════════════════════════════════


def check_gpu_required_quantization(
    model_id: str,
    model_config: dict,
    quantization: Optional[str],
    is_cpu: bool,
) -> Optional[CheckResult]:
    """
    Check #1: MXFP4/NVFP4/EXL2 quantization requires GPU (CUDA kernels).
    """
    if not is_cpu:
        return None

    # GPU-only quantization methods
    gpu_only_quants = {"mxfp4", "nvfp4", "fp4", "exl2"}
    gpu_only_patterns = {"mxfp4", "nvfp4", "fp4", "exl2", "6bit-g128"}

    quant = (quantization or "").lower()
    config_quant = (
        model_config.get("quantization_config", {}).get("quant_method", "").lower()
    )
    model_lower = model_id.lower()

    detected_quant = quant or config_quant

    if detected_quant in gpu_only_quants:
        return CheckResult(
            check_id=CheckId.GPU_REQUIRED_QUANTIZATION,
            severity=CheckSeverity.BLOCK,
            message=f"Cannot load on CPU: quantization method '{detected_quant.upper()}' "
            f"requires GPU (CUDA kernels: FlashInfer/Marlin/Triton).",
            suggestion="Use a non-quantized model, GPTQ, or AWQ for CPU. "
            "Or switch to GPU mode.",
            details={"quant_method": detected_quant},
        )

    # Check model name patterns
    for pattern in gpu_only_patterns:
        if pattern in model_lower:
            return CheckResult(
                check_id=CheckId.GPU_REQUIRED_QUANTIZATION,
                severity=CheckSeverity.BLOCK,
                message=f"Cannot load on CPU: model appears to use GPU-only "
                f"quantization (detected '{pattern}' in name).",
                suggestion="Use a non-quantized model or switch to GPU mode.",
                details={"detected_pattern": pattern},
            )

    return None


def check_cpu_all_quant_unsupported(
    model_id: str,
    model_config: dict,
    quantization: Optional[str],
    is_cpu: bool,
) -> Optional[CheckResult]:
    """
    Check #11: AWQ/GPTQ and most quantization methods require GPU.
    CPU backend only supports unquantized models (float32/bfloat16).
    """
    if not is_cpu:
        return None

    # ALL quantization that needs GPU kernels
    gpu_quants = {
        "awq",
        "gptq",
        "gptq_marlin",
        "awq_marlin",
        "marlin",
        "exl2",
        "mxfp4",
        "nvfp4",
        "fp8",
        "aqlm",
        "qqq",
        "hqq",
        "compressed-tensors",
        "bitsandbytes",
    }
    # Only GGUF sort-of works on CPU (but vLLM support is limited)
    cpu_ok_quants = {"gguf"}

    quant = (quantization or "").lower()
    config_quant = (
        model_config.get("quantization_config", {}).get("quant_method", "").lower()
    )
    model_lower = model_id.lower()

    detected = quant or config_quant

    if detected and detected not in cpu_ok_quants:
        if detected in gpu_quants:
            return CheckResult(
                check_id=CheckId.CPU_ALL_QUANT_UNSUPPORTED,
                severity=CheckSeverity.BLOCK,
                message=f"Cannot load {detected.upper()}-quantized model on CPU. "
                f"Quantization methods require GPU CUDA kernels.",
                suggestion="Use an unquantized model (bfloat16) for CPU inference, "
                "or switch to Ollama/llama.cpp for GGUF support.",
                details={"quant_method": detected},
            )

    # Check by filename patterns
    quant_patterns = {"-awq": "AWQ", "-gptq": "GPTQ", "-exl2": "EXL2", "-fp8": "FP8"}
    for pattern, name in quant_patterns.items():
        if pattern in model_lower:
            return CheckResult(
                check_id=CheckId.CPU_ALL_QUANT_UNSUPPORTED,
                severity=CheckSeverity.BLOCK,
                message=f"Cannot load {name}-quantized model on CPU. "
                f"Detected '{pattern}' in model name.",
                suggestion="Use an unquantized model (bfloat16) for CPU inference.",
                details={"detected_pattern": pattern, "quant_name": name},
            )

    return None


def check_unsupported_architecture(model_config: dict) -> Optional[CheckResult]:
    """
    Check #2: Model architecture must be supported by vLLM.
    """
    architectures = model_config.get("architectures", [])

    if not architectures:
        return None  # Can't check without architectures field

    # Known supported architectures (fallback if vLLM import fails)
    known_archs = {
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen2MoeForCausalLM",
        "Phi3ForCausalLM",
        "PhiForCausalLM",
        "GemmaForCausalLM",
        "Gemma2ForCausalLM",
        "GPT2LMHeadModel",
        "GPTNeoXForCausalLM",
        "GPTJForCausalLM",
        "MixtralForCausalLM",
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "Starcoder2ForCausalLM",
        "StarCoderForCausalLM",
        "FalconForCausalLM",
        "OPTForCausalLM",
        "BloomForCausalLM",
        "MPTForCausalLM",
        "BaichuanForCausalLM",
        "InternLMForCausalLM",
        "InternLM2ForCausalLM",
        "ChatGLMForConditionalGeneration",
        "CohereForCausalLM",
        "DbrxForCausalLM",
        "OlmoForCausalLM",
        "Olmo2ForCausalLM",
        "ArcticForCausalLM",
        "JambaForCausalLM",
        "MambaForCausalLM",
        "Mamba2ForCausalLM",
        "ExaoneForCausalLM",
        "GraniteForCausalLM",
        "GraniteMoeForCausalLM",
    }

    # Try to get supported archs from vLLM
    supported = known_archs
    try:
        from vllm.model_executor.models import ModelRegistry

        if hasattr(ModelRegistry, "get_supported_archs"):
            supported = set(ModelRegistry.get_supported_archs())
    except (ImportError, AttributeError):
        pass

    unsupported = [arch for arch in architectures if arch not in supported]

    if unsupported:
        return CheckResult(
            check_id=CheckId.UNSUPPORTED_ARCHITECTURE,
            severity=CheckSeverity.BLOCK,
            message=f"Model architecture '{unsupported[0]}' is not supported by vLLM.",
            suggestion="Check vLLM documentation for supported models, "
            "or use a compatible model variant.",
            details={"architectures": architectures, "unsupported": unsupported},
        )

    return None


def check_trust_remote_code(
    model_config: dict,
    trust_remote_code: bool,
) -> Optional[CheckResult]:
    """
    Check #3: Detect if model requires trust_remote_code and auto-fix if needed.
    """
    # auto_map indicates custom model code
    auto_map = model_config.get("auto_map", {})

    if auto_map and any(k.startswith("AutoModel") for k in auto_map):
        if not trust_remote_code:
            return CheckResult(
                check_id=CheckId.TRUST_REMOTE_CODE,
                severity=CheckSeverity.AUTOFIX,
                message="Model contains custom code (auto_map). "
                "Enabling trust_remote_code=True.",
                auto_fixed=True,
                details={"auto_map_keys": list(auto_map.keys())},
            )

    return None


def _try_bnb_quantization(
    model_config: dict,
    platform: PlatformInfo,
    weight_bytes: int,
    kv_cache_bytes: int,
    available_vram: int,
    details: dict,
) -> Optional[CheckResult]:
    """
    Check if BitsAndBytes on-the-fly 4-bit quantization can make the model fit.

    BnB quantizes FP16/BF16 weights to NF4 at load time (~4x weight reduction).
    Requirements:
      - Model must NOT already be quantized (can't double-quantize)
      - GPU compute capability >= 7.0 (Turing+)
      - Enough system RAM to hold full-precision weights during loading
      - Quantized size must fit in VRAM

    Returns CheckResult(AUTOFIX) if viable, None otherwise.
    """
    # 1. Check model is not already quantized
    quant_config = model_config.get("quantization_config", {})
    existing_quant = quant_config.get("quant_method", "").lower()
    if existing_quant:
        # Already quantized — BnB on top would be double-quantizing
        return None

    # Also check if weights on disk look pre-quantized by examining dtype/bpw
    # If torch_dtype is float16/bfloat16 and no quant_config, it's unquantized
    torch_dtype = model_config.get("torch_dtype", "").lower()
    if torch_dtype not in ("float16", "bfloat16", "float32", ""):
        # Unusual dtype — might be pre-quantized in a non-standard way
        return None

    # 2. Check GPU compute capability >= 7.0 (BnB requirement)
    if platform.gpu_compute_capability is None:
        return None
    if platform.gpu_compute_capability < (7, 0):
        return None

    # 3. Check system RAM can hold full-precision weights during BnB loading
    # BnB loads full FP16 weights into CPU RAM, then quantizes layer-by-layer
    # Need at least 1.2x the weight size in available RAM
    ram_available = platform.cpu_ram_bytes * 0.8  # Reserve 20% for OS
    if weight_bytes > ram_available:
        return None

    # 4. Estimate quantized memory usage
    # BnB NF4: ~4x reduction on weights, KV cache stays same size
    bnb_weight_bytes = weight_bytes / 4
    cuda_context_bytes = 1 * 1024 * 1024 * 1024  # 1GB
    bnb_activation_overhead = int((bnb_weight_bytes + kv_cache_bytes) * 0.20)
    bnb_total = bnb_weight_bytes + kv_cache_bytes + cuda_context_bytes + bnb_activation_overhead

    if bnb_total > available_vram:
        # Even with BnB, model won't fit
        return None

    # BnB quantization is viable!
    details["bnb_quantize"] = True
    details["bnb_estimated_vram"] = int(bnb_total)
    details["bnb_weight_bytes"] = int(bnb_weight_bytes)

    return CheckResult(
        check_id=CheckId.OOM_ESTIMATE,
        severity=CheckSeverity.AUTOFIX,
        message=f"Weights ({weight_bytes/1e9:.1f}GB) exceed {available_vram/1e9:.1f}GB VRAM. "
        f"Auto-enabling BitsAndBytes NF4 on-the-fly quantization "
        f"(~{bnb_weight_bytes/1e9:.1f}GB quantized weights + "
        f"{kv_cache_bytes/1e9:.1f}GB KV cache ≈ {bnb_total/1e9:.1f}GB total). "
        f"Note: loading will be slower as full-precision weights are quantized at startup.",
        auto_fixed=True,
        details=details,
    )

def check_oom_estimate(
    snapshot_path: "Path",
    model_config: dict,
    platform: "PlatformInfo",
    gpu_memory_utilization: float,
    max_model_len: "Optional[int]",
    kv_offload_max_ram_fraction: float = 0.5,
) -> "Optional[CheckResult]":
    """
    Check #4: Estimate if model will fit in available memory.

    Implements a tiered offloading strategy (vLLM v0.15):

    1. If weights fit but KV cache doesn't:
       a. Try kv_offloading_size (async DMA, minimal perf impact)
       b. Fall back to context length capping (handled by check_context_length_mismatch)
    2. If weights slightly exceed VRAM (<30% over):
       a. cpu_offload_gb + enforce_eager
       b. BnB quantization fallback
    3. If weights significantly exceed VRAM (>=30% over):
       a. BnB quantization first (full GPU speed > full quality at terrible speed)
       b. cpu_offload_gb as last resort
    4. Nothing fits → BLOCK
    """
    estimated_bytes, details = estimate_model_memory(
        snapshot_path, model_config, max_model_len
    )

    if platform.has_gpu:
        usable = int(platform.gpu_memory_bytes * gpu_memory_utilization)
        # Get current free VRAM
        try:
            import torch
            free_vram, _ = torch.cuda.mem_get_info()
            available = min(usable, free_vram)
        except Exception:
            available = usable
        memory_type = "GPU VRAM"
    else:
        available = int(platform.cpu_ram_bytes * 0.8)  # Reserve 20% for OS
        memory_type = "RAM"

    details["available_bytes"] = available
    details["memory_type"] = memory_type

    if available == 0:
        return None

    ratio = estimated_bytes / available
    weight_bytes = details.get("weight_bytes", 0)
    kv_cache_bytes = details.get("kv_cache_bytes", 0)

    if ratio > 1.2:  # More than 20% over available
        # ─── Determine what's causing the overflow ───
        #
        # Case A: Weights fit in VRAM, KV cache is the bottleneck
        # Case B: Weights themselves exceed VRAM
        vram_for_weights = available - kv_cache_bytes
        weight_ratio = weight_bytes / available if available > 0 else float("inf")

        # ─── Case A: Weights fit, KV cache is the problem ───
        if platform.has_gpu and weight_bytes <= available * 0.95:
            # Try KV cache offloading first (best option — async, minimal perf hit)
            kv_result = _try_kv_offload(
                model_config=model_config,
                platform=platform,
                kv_cache_bytes=kv_cache_bytes,
                available_vram=available,
                weight_bytes=weight_bytes,
                max_model_len=max_model_len or 4096,
                kv_offload_max_ram_fraction=kv_offload_max_ram_fraction,
                details=details,
            )
            if kv_result is not None:
                return kv_result

            # KV offloading rejected (RAM budget exceeded) — fall through.
            # check_context_length_mismatch will cap context length later.
            # Return a WARN so the user knows what happened.
            if details.get("kv_offload_rejected"):
                return CheckResult(
                    check_id=CheckId.OOM_ESTIMATE,
                    severity=CheckSeverity.WARN,
                    message=f"KV cache ({kv_cache_bytes/1e9:.1f}GB) exceeds VRAM but "
                    f"KV offloading would need {details.get('kv_offload_needed_gb', '?')}GB RAM "
                    f"(budget: {details.get('kv_offload_ram_budget_gb', '?')}GB = "
                    f"{kv_offload_max_ram_fraction:.0%} of system RAM). "
                    f"Context length will be capped instead. "
                    f"Increase VLLM_KV_OFFLOAD_MAX_RAM_FRACTION to allow more RAM usage.",
                    details=details,
                )

        # ─── Case B: Weights exceed VRAM ───
        if platform.has_gpu and vram_for_weights > 0 and weight_bytes > vram_for_weights:
            offload_bytes = weight_bytes - vram_for_weights
            offload_gb = int((offload_bytes / 1e9) + 1)  # Round up + buffer
            weight_overshoot = offload_bytes / weight_bytes  # fraction of weights to offload

            # Decision: BnB first if >30% of weights need offloading,
            # cpu_offload first if <=30% (moderate penalty is acceptable)
            prefer_bnb = weight_overshoot > 0.30

            if prefer_bnb:
                # ─── Strategy: BnB quantization (preferred for large overshoot) ───
                bnb_result = _try_bnb_quantization(
                    model_config, platform, weight_bytes, kv_cache_bytes, available, details,
                )
                if bnb_result is not None:
                    return bnb_result

                # BnB not viable — fall through to cpu_offload as last resort

            # ─── Strategy: cpu_offload_gb ───
            if platform.cpu_ram_bytes > (offload_gb * 1e9 * 1.5):
                # Check UVA / pinned memory support
                pin_memory_ok = False

                if platform.is_wsl:
                    # WSL does NOT support CUDA pinned memory (UVA).
                    pin_memory_ok = False
                else:
                    try:
                        import torch
                        t = torch.empty(1, pin_memory=True)
                        del t
                        pin_memory_ok = True
                    except Exception:
                        pass

                if pin_memory_ok:
                    details["cpu_offload_gb"] = offload_gb
                    details["weight_overshoot_pct"] = round(weight_overshoot * 100, 1)
                    # Also signal enforce_eager when cpu_offload is active.
                    # CUDA graphs + CPU offloading can cause issues on tight VRAM.
                    details["force_enforce_eager"] = True
                    return CheckResult(
                        check_id=CheckId.OOM_ESTIMATE,
                        severity=CheckSeverity.AUTOFIX,
                        message=f"Weights ({weight_bytes/1e9:.1f}GB) + KV cache ({kv_cache_bytes/1e9:.1f}GB) "
                        f"exceed {available/1e9:.1f}GB VRAM. Offloading {offload_gb}GB weights to RAM "
                        f"({weight_overshoot:.0%} of weights). enforce_eager enabled for stability.",
                        auto_fixed=True,
                        details=details,
                    )
                else:
                    # Pin memory not available — try BnB if we haven't already
                    if not prefer_bnb:
                        bnb_result = _try_bnb_quantization(
                            model_config, platform, weight_bytes, kv_cache_bytes, available, details,
                        )
                        if bnb_result is not None:
                            return bnb_result

                    # Nothing works — BLOCK
                    reason = ("WSL does not support CUDA pinned memory (UVA)"
                              if platform.is_wsl else
                              "CUDA pinned memory (UVA) is not available in this environment")
                    return CheckResult(
                        check_id=CheckId.OOM_ESTIMATE,
                        severity=CheckSeverity.BLOCK,
                        message=f"Weights ({weight_bytes/1e9:.1f}GB) + KV cache ({kv_cache_bytes/1e9:.1f}GB) "
                        f"exceed {available/1e9:.1f}GB VRAM. CPU offloading is not possible: {reason}. "
                        f"BitsAndBytes on-the-fly quantization is also not viable.",
                        suggestion="Use a smaller/more quantized model that fits in GPU VRAM. "
                        f"You have ~{available/1e9:.1f}GB usable VRAM — look for models under "
                        f"~{available/1e9 * 0.8:.0f}GB (e.g. 3-4B parameter quantized models).",
                        details=details,
                    )

            else:
                # Not enough CPU RAM for offloading — try BnB if we haven't
                if not prefer_bnb:
                    bnb_result = _try_bnb_quantization(
                        model_config, platform, weight_bytes, kv_cache_bytes, available, details,
                    )
                    if bnb_result is not None:
                        return bnb_result

            # ─── Strategy: BnB as last resort when cpu_offload was preferred ───
            if not prefer_bnb:
                # We already tried cpu_offload and it failed, BnB is our last shot
                pass
            else:
                # We preferred BnB but it failed, cpu_offload also failed
                pass

        # Can't cpu_offload — try BnB as final fallback
        if platform.has_gpu:
            bnb_result = _try_bnb_quantization(
                model_config, platform, weight_bytes, kv_cache_bytes, available, details,
            )
            if bnb_result is not None:
                return bnb_result

        return CheckResult(
            check_id=CheckId.OOM_ESTIMATE,
            severity=CheckSeverity.BLOCK,
            message=f"Model requires ~{estimated_bytes / 1e9:.1f}GB but only "
            f"~{available / 1e9:.1f}GB {memory_type} available "
            f"({ratio:.1f}x over limit). Cannot offload to CPU.",
            suggestion="Use a smaller model or more aggressive quantization that "
            f"fits within ~{available / 1e9 * 0.8:.0f}GB.",
            details=details,
        )
    elif ratio > 0.85:  # Warning zone
        return CheckResult(
            check_id=CheckId.OOM_ESTIMATE,
            severity=CheckSeverity.WARN,
            message=f"Model may be tight on memory (~{ratio:.0%} of available "
            f"{memory_type}). Consider reducing max_model_len.",
            details=details,
        )

    return None


def check_context_length_mismatch(
    model_config: dict,
    max_model_len: Optional[int],
    platform: PlatformInfo,
    auto_cap_context: bool,
) -> Optional[CheckResult]:
    """
    Check #5: Check if context length exceeds model capability or memory.
    Auto-cap if enabled.
    """
    max_pos_embed = model_config.get(
        "max_position_embeddings",
        model_config.get("max_sequence_length", 4096),
    )

    # If user didn't set max_model_len, check if default is reasonable
    if max_model_len is None:
        # Models with huge context windows (128K+) often cause OOM
        # when vLLM tries to allocate KV cache for full length
        if max_pos_embed > 32768:
            # Auto-cap to a reasonable default
            reasonable_cap = 8192
            if platform.has_gpu and platform.gpu_memory_bytes:
                # Allow more context on larger GPUs
                gpu_gb = platform.gpu_memory_bytes / 1e9
                if gpu_gb >= 24:
                    reasonable_cap = 16384
                elif gpu_gb >= 16:
                    reasonable_cap = 12288

            if auto_cap_context:
                return CheckResult(
                    check_id=CheckId.CONTEXT_LENGTH_MISMATCH,
                    severity=CheckSeverity.AUTOFIX,
                    message=f"Model has max_position_embeddings={max_pos_embed} which "
                    f"may cause OOM. Auto-capping to {reasonable_cap}.",
                    auto_fixed=True,
                    details={
                        "original": max_pos_embed,
                        "capped_to": reasonable_cap,
                    },
                )
            else:
                return CheckResult(
                    check_id=CheckId.CONTEXT_LENGTH_MISMATCH,
                    severity=CheckSeverity.WARN,
                    message=f"Model has max_position_embeddings={max_pos_embed}. "
                    f"This may cause OOM if vLLM tries to allocate full KV cache.",
                    suggestion=f"Set VLLM_MAX_MODEL_LEN={reasonable_cap} or lower.",
                )

    # If user explicitly set max_model_len higher than model supports
    if max_model_len and max_model_len > max_pos_embed:
        if auto_cap_context:
            return CheckResult(
                check_id=CheckId.CONTEXT_LENGTH_MISMATCH,
                severity=CheckSeverity.AUTOFIX,
                message=f"Requested max_model_len={max_model_len} exceeds model's "
                f"max_position_embeddings={max_pos_embed}. Capping to {max_pos_embed}.",
                auto_fixed=True,
                details={"requested": max_model_len, "capped_to": max_pos_embed},
            )
        else:
            return CheckResult(
                check_id=CheckId.CONTEXT_LENGTH_MISMATCH,
                severity=CheckSeverity.BLOCK,
                message=f"Requested context length {max_model_len} exceeds model "
                f"maximum {max_pos_embed}.",
                suggestion=f"Set VLLM_MAX_MODEL_LEN to {max_pos_embed} or lower.",
            )

    return None


def check_cpu_dtype_incompatible(
    model_config: dict,
    is_cpu: bool,
    dtype: str,
) -> Optional[CheckResult]:
    """
    Check #6: float16 is not efficient on CPU, auto-fix to bfloat16.
    """
    if not is_cpu:
        return None

    model_dtype = model_config.get("torch_dtype", "float32")

    # If user didn't specify dtype and model uses float16
    if dtype == "auto" and model_dtype == "float16":
        return CheckResult(
            check_id=CheckId.CPU_DTYPE_INCOMPATIBLE,
            severity=CheckSeverity.AUTOFIX,
            message="Model uses float16 but vLLM CPU backend has unstable float16 "
            "support. Auto-switching to bfloat16.",
            auto_fixed=True,
            details={"original_dtype": model_dtype, "fixed_dtype": "bfloat16"},
        )

    return None


def check_vllm_v1_cpu_incompatible(is_cpu: bool) -> "Optional[CheckResult]":
    """
    Check #8: Verify CPU backend compatibility with the current vLLM version.

    As of vLLM 0.11+, the V0 engine has been completely removed and
    AsyncLLMEngine is just an alias for AsyncLLM (V1). The V1 engine
    gained CPU support in vLLM ~0.13+. Setting VLLM_USE_V1=0 no longer
    has any effect — there is no V0 to fall back to.
    """
    import os

    if not is_cpu:
        return None

    try:
        import vllm
        version = getattr(vllm, "__version__", "0.0.0")
        parts = version.split(".")
        major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0

        if major == 0 and minor < 13:
            return CheckResult(
                check_id=CheckId.VLLM_V1_CPU_INCOMPATIBLE,
                severity=CheckSeverity.WARN,
                message=f"Running vLLM {version} on CPU. CPU support in the V1 "
                f"engine was stabilized in vLLM 0.13+. You may encounter "
                f"issues. Consider upgrading vLLM.",
                suggestion="Upgrade to vLLM >= 0.13 for stable CPU support.",
            )
    except (ImportError, ValueError, IndexError):
        pass

    return None


def check_gated_model_no_token(
    model_id: str,
    hf_token: Optional[str],
    is_cached: bool,
) -> Optional[CheckResult]:
    """
    Check #9: Detect gated models that require authentication.
    Only checks if model is not already cached.
    """
    if is_cached:
        return None  # Already have the model, no need to check

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

        api = HfApi(token=hf_token if hf_token else None)
        info = api.model_info(model_id)

        # Check if gated
        if getattr(info, "gated", False):
            if not hf_token:
                return CheckResult(
                    check_id=CheckId.GATED_MODEL_NO_TOKEN,
                    severity=CheckSeverity.BLOCK,
                    message=f"Model '{model_id}' is gated and requires authentication.",
                    suggestion="Set HUGGING_FACE_HUB_TOKEN environment variable "
                    f"with a token that has accepted the model's license at "
                    f"https://huggingface.co/{model_id}",
                )
            # Token provided but might not have accepted license
            # This will be caught during actual download

    except GatedRepoError:
        return CheckResult(
            check_id=CheckId.GATED_MODEL_NO_TOKEN,
            severity=CheckSeverity.BLOCK,
            message=f"Model '{model_id}' is gated. Access denied.",
            suggestion="Accept the model license on HuggingFace and set "
            "HUGGING_FACE_HUB_TOKEN environment variable.",
        )
    except RepositoryNotFoundError:
        return CheckResult(
            check_id=CheckId.GATED_MODEL_NO_TOKEN,
            severity=CheckSeverity.BLOCK,
            message=f"Model '{model_id}' not found on HuggingFace Hub.",
            suggestion="Check the model ID for typos.",
        )
    except Exception:
        pass  # Network issues, etc. - continue with load attempt

    return None


def check_gpu_compute_cap_too_low(
    model_config: dict,
    quantization: Optional[str],
    platform: PlatformInfo,
) -> Optional[CheckResult]:
    """
    Check #10: Check if GPU compute capability supports the quantization method.
    Also checks bfloat16 compatibility.
    """
    if not platform.has_gpu or not platform.gpu_compute_capability:
        return None

    sm = platform.gpu_compute_capability

    # Quantization minimum compute capabilities
    quant_min_sm = {
        "fp8": ((8, 9), "Ada Lovelace (RTX 40xx)"),
        "mxfp4": ((9, 0), "Hopper (H100)"),
        "nvfp4": ((9, 0), "Hopper (H100)"),
        "marlin": ((8, 0), "Ampere (RTX 30xx)"),
        "awq_marlin": ((8, 0), "Ampere (RTX 30xx)"),
        "gptq_marlin": ((8, 0), "Ampere (RTX 30xx)"),
        "exl2": ((8, 0), "Ampere (RTX 30xx)"),
    }

    quant = (quantization or "").lower()
    config_quant = (
        model_config.get("quantization_config", {}).get("quant_method", "").lower()
    )
    detected = quant or config_quant

    if detected in quant_min_sm:
        min_sm, arch_name = quant_min_sm[detected]
        if sm < min_sm:
            return CheckResult(
                check_id=CheckId.GPU_COMPUTE_CAP_TOO_LOW,
                severity=CheckSeverity.BLOCK,
                message=f"Quantization '{detected}' requires {arch_name} "
                f"(SM {min_sm[0]}.{min_sm[1]}+) but your {platform.gpu_name} "
                f"has SM {sm[0]}.{sm[1]}.",
                suggestion="Use a different quantization (AWQ, GPTQ) or "
                "an unquantized model.",
                details={
                    "required_sm": min_sm,
                    "actual_sm": sm,
                    "quant_method": detected,
                },
            )

    # bfloat16 requires SM 8.0+ (Ampere)
    model_dtype = model_config.get("torch_dtype", "float16")
    if model_dtype == "bfloat16" and sm < (8, 0):
        return CheckResult(
            check_id=CheckId.GPU_COMPUTE_CAP_TOO_LOW,
            severity=CheckSeverity.AUTOFIX,
            message=f"bfloat16 requires Ampere+ (SM 8.0) but GPU has SM {sm[0]}.{sm[1]}. "
            "Auto-switching to float16.",
            auto_fixed=True,
            details={"original_dtype": "bfloat16", "fixed_dtype": "float16"},
        )

    return None


def check_moe_on_cpu(model_config: dict, is_cpu: bool) -> Optional[CheckResult]:
    """
    Check #12: MoE models are very slow on CPU.
    """
    if not is_cpu:
        return None

    num_experts = model_config.get(
        "num_local_experts", model_config.get("num_experts", 0)
    )

    if num_experts > 1:
        active_experts = model_config.get("num_experts_per_tok", 2)
        return CheckResult(
            check_id=CheckId.MOE_ON_CPU,
            severity=CheckSeverity.WARN,
            message=f"MoE model with {num_experts} experts ({active_experts} active). "
            "All expert weights must stay in RAM, and CPU expert routing is "
            "unoptimized. Expect very slow inference.",
            suggestion="Consider using a dense model variant or switch to GPU.",
            details={"num_experts": num_experts, "active_experts": active_experts},
        )

    return None


def check_enforce_eager_low_vram(
    snapshot_path: Path,
    model_config: dict,
    platform: PlatformInfo,
    gpu_memory_utilization: float,
    enforce_eager: bool,
) -> Optional[CheckResult]:
    """
    Check #13: Enable enforce_eager on low VRAM GPUs when model uses >70% VRAM.
    CUDAGraph pre-allocation can push tight models into OOM.
    """
    if not platform.has_gpu or enforce_eager:
        return None

    gpu_gb = platform.gpu_memory_bytes / 1e9

    # Only applies to GPUs <= 16GB
    if gpu_gb > 16:
        return None

    estimated_bytes, _ = estimate_model_memory(snapshot_path, model_config)
    usable = platform.gpu_memory_bytes * gpu_memory_utilization

    if usable == 0:
        return None

    ratio = estimated_bytes / usable

    if ratio > 0.70:
        return CheckResult(
            check_id=CheckId.ENFORCE_EAGER_LOW_VRAM,
            severity=CheckSeverity.AUTOFIX,
            message=f"Model uses ~{ratio:.0%} of {gpu_gb:.0f}GB VRAM. "
            "Auto-enabling enforce_eager to avoid CUDA graph memory overhead.",
            auto_fixed=True,
            details={"vram_ratio": ratio, "gpu_gb": gpu_gb},
        )

    return None


def check_missing_tokenizer(
    snapshot_path: Optional[Path],
    resolved_tokenizer: Optional[str],
) -> Optional[CheckResult]:
    """
    Check #7: Verify tokenizer is available.
    """
    if snapshot_path is None:
        return None

    # Check for tokenizer files
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
        "vocab.txt",
        "spiece.model",
    ]

    has_tokenizer = any((snapshot_path / f).exists() for f in tokenizer_files)

    if not has_tokenizer and not resolved_tokenizer:
        return CheckResult(
            check_id=CheckId.MISSING_TOKENIZER,
            severity=CheckSeverity.BLOCK,
            message="Model is missing tokenizer files and no external tokenizer "
            "was resolved.",
            suggestion="Set VLLM_TOKENIZER to the base model ID "
            "(e.g., 'meta-llama/Llama-3.1-8B-Instruct').",
        )
    elif not has_tokenizer and resolved_tokenizer:
        return CheckResult(
            check_id=CheckId.MISSING_TOKENIZER,
            severity=CheckSeverity.AUTOFIX,
            message=f"Model missing tokenizer. Using external: {resolved_tokenizer}",
            auto_fixed=True,
            details={"resolved_tokenizer": resolved_tokenizer},
        )

    return None


# ════════════════════════════════════════════════════════════════════════════
# Main Entry Points
# ════════════════════════════════════════════════════════════════════════════


async def preflight_check(
    model_id: str,
    config: ManagerConfig,
    resolved_tokenizer: Optional[str] = None,
) -> PreflightResult:
    """
    Run all preflight checks before model loading.

    Args:
        model_id: HuggingFace model ID or local path
        config: Manager configuration
        resolved_tokenizer: Tokenizer resolved by tokenizer_resolver (if any)

    Returns:
        PreflightResult with pass/fail status and engine overrides
    """
    result = PreflightResult(model=model_id)
    platform = get_platform()
    result.platform = platform

    # Get model snapshot path
    snapshot_path = _get_snapshot_path(model_id)
    is_cached = snapshot_path is not None

    # Check for gated model (only if not cached)
    gated_check = check_gated_model_no_token(model_id, config.hf_token, is_cached)
    if gated_check:
        result.checks.append(gated_check)
        if gated_check.severity == CheckSeverity.BLOCK:
            result.blocked = True
            result.block_reason = gated_check.message
            result.passed = False
            return result

    # If not cached, can't do other checks
    if snapshot_path is None:
        return result

    # Load config.json
    model_config = _load_model_config(snapshot_path)
    if model_config is None:
        result.checks.append(
            CheckResult(
                check_id=CheckId.NO_CONFIG,
                severity=CheckSeverity.WARN,
                message="Could not read config.json - skipping some preflight checks.",
            )
        )
        result.warnings.append(
            "Could not read config.json - skipping some preflight checks."
        )
        model_config = {}

    # Preserve model config for capabilities detection
    result.model_config = model_config

    # Pre-compute effective max_model_len BEFORE running checks.
    # This ensures OOM estimate uses the capped context length, not the
    # raw max_position_embeddings (which can be 128K-262K and produce
    # wildly inflated KV cache estimates).
    effective_max_model_len = config.vllm_max_model_len
    if effective_max_model_len is None:
        max_pos_embed = model_config.get(
            "max_position_embeddings",
            model_config.get("max_sequence_length", 4096),
        )
        if max_pos_embed > 32768:
            # Same logic as check_context_length_mismatch
            reasonable_cap = 8192
            if platform.has_gpu and platform.gpu_memory_bytes:
                gpu_gb = platform.gpu_memory_bytes / 1e9
                if gpu_gb >= 24:
                    reasonable_cap = 16384
                elif gpu_gb >= 16:
                    reasonable_cap = 12288
            effective_max_model_len = reasonable_cap

    # Run all checks
    checks_to_run = [
        # BLOCK checks first
        check_gpu_required_quantization(
            model_id, model_config, config.vllm_quantization, config.is_cpu
        ),
        check_cpu_all_quant_unsupported(
            model_id, model_config, config.vllm_quantization, config.is_cpu
        ),
        check_unsupported_architecture(model_config),
        check_gpu_compute_cap_too_low(model_config, config.vllm_quantization, platform),
        check_oom_estimate(
            snapshot_path,
            model_config,
            platform,
            config.vllm_gpu_memory_utilization,
            effective_max_model_len,
            kv_offload_max_ram_fraction=getattr(
                config, "vllm_kv_offload_max_ram_fraction", 0.5
            ),
        ),
        # AUTOFIX checks
        check_vllm_v1_cpu_incompatible(config.is_cpu),
        check_context_length_mismatch(
            model_config,
            config.vllm_max_model_len,
            platform,
            getattr(config, "vllm_auto_cap_context", True),
        ),
        check_cpu_dtype_incompatible(
            model_config,
            config.is_cpu,
            getattr(config, "vllm_dtype", "auto"),
        ),
        check_enforce_eager_low_vram(
            snapshot_path,
            model_config,
            platform,
            config.vllm_gpu_memory_utilization,
            getattr(config, "vllm_enforce_eager", False),
        ),
        check_trust_remote_code(
            model_config, getattr(config, "vllm_trust_remote_code", True)
        ),
        check_missing_tokenizer(snapshot_path, resolved_tokenizer),
        # WARN checks
        check_moe_on_cpu(model_config, config.is_cpu),
    ]

    # Process results
    engine_overrides: dict = {}

    for check in checks_to_run:
        if check is None:
            continue

        result.checks.append(check)

        if check.severity == CheckSeverity.BLOCK:
            result.blocked = True
            result.block_reason = check.message
            result.passed = False
            return result  # Stop on first block

        elif check.severity == CheckSeverity.WARN:
            result.warnings.append(check.message)

        elif check.severity == CheckSeverity.AUTOFIX:
            result.warnings.append(f"[auto-fix] {check.message}")

            # Apply auto-fixes to engine_overrides
            if check.check_id == CheckId.TRUST_REMOTE_CODE:
                engine_overrides["trust_remote_code"] = True

            elif check.check_id == CheckId.CONTEXT_LENGTH_MISMATCH:
                capped_to = check.details.get("capped_to")
                if capped_to:
                    engine_overrides["max_model_len"] = capped_to

            elif check.check_id == CheckId.CPU_DTYPE_INCOMPATIBLE:
                engine_overrides["dtype"] = "bfloat16"

            elif check.check_id == CheckId.VLLM_V1_CPU_INCOMPATIBLE:
                # In newer vLLM (0.12+), V0 was removed. CPU backend
                # may work differently. Flag for logging only.
                engine_overrides["cpu_backend_warning"] = True

            elif check.check_id == CheckId.ENFORCE_EAGER_LOW_VRAM:
                engine_overrides["enforce_eager"] = True

            elif check.check_id == CheckId.GPU_COMPUTE_CAP_TOO_LOW:
                # bfloat16 -> float16 fallback
                if check.details.get("fixed_dtype"):
                    engine_overrides["dtype"] = check.details["fixed_dtype"]

            elif check.check_id == CheckId.OOM_ESTIMATE:
                # CPU offload for models that don't fit in VRAM
                if check.details.get("cpu_offload_gb"):
                    engine_overrides["cpu_offload_gb"] = check.details["cpu_offload_gb"]
                    # Force enforce_eager when cpu_offload is active for stability
                    if check.details.get("force_enforce_eager"):
                        engine_overrides["enforce_eager"] = True
                # BitsAndBytes on-the-fly quantization fallback
                elif check.details.get("bnb_quantize"):
                    engine_overrides["quantization"] = "bitsandbytes"
                    engine_overrides["load_format"] = "bitsandbytes"
            elif check.check_id == CheckId.KV_OFFLOAD_ENABLED:
                kv_size = check.details.get("kv_offload_size_gb")
                if kv_size:
                    engine_overrides["kv_offloading_size"] = kv_size
                    engine_overrides["kv_offloading_backend"] = "native"

    result.engine_overrides = engine_overrides
    return result


def classify_engine_error(error: Exception) -> CheckResult:
    """
    Map runtime engine errors to preflight-style check results.
    Used for errors that slip past preflight.
    """
    msg = str(error).lower()

    if "out of memory" in msg or "oom" in msg:
        return CheckResult(
            check_id=CheckId.OOM_RUNTIME,
            severity=CheckSeverity.BLOCK,
            message="Out of GPU memory during model loading.",
            suggestion="Reduce gpu_memory_utilization, use quantization, "
            "or try a smaller model.",
        )

    if "no available memory for the cache blocks" in msg:
        return CheckResult(
            check_id=CheckId.OOM_KV_CACHE,
            severity=CheckSeverity.BLOCK,
            message="Insufficient memory for KV cache blocks.",
            suggestion="Reduce VLLM_MAX_MODEL_LEN or increase GPU_MEMORY_UTILIZATION.",
        )

    if "not supported" in msg and "architecture" in msg:
        return CheckResult(
            check_id=CheckId.UNSUPPORTED_ARCHITECTURE,
            severity=CheckSeverity.BLOCK,
            message="Model architecture not supported by vLLM.",
            suggestion="Check vLLM documentation for supported models.",
        )

    if "trust_remote_code" in msg:
        return CheckResult(
            check_id=CheckId.TRUST_REMOTE_CODE,
            severity=CheckSeverity.BLOCK,
            message="Model requires trust_remote_code=True.",
            suggestion="Set VLLM_TRUST_REMOTE_CODE=true.",
        )

    if any(k in msg for k in ["mxfp4", "flashinfer", "marlin", "triton"]):
        return CheckResult(
            check_id=CheckId.GPU_REQUIRED_QUANTIZATION,
            severity=CheckSeverity.BLOCK,
            message="Quantization requires GPU CUDA kernels.",
            suggestion="Use an unquantized model or switch to GPU mode.",
        )

    if "tokenizer" in msg and ("none" in msg or "not found" in msg):
        return CheckResult(
            check_id=CheckId.MISSING_TOKENIZER,
            severity=CheckSeverity.BLOCK,
            message="Tokenizer not found.",
            suggestion="Set VLLM_TOKENIZER to the base model ID.",
        )

    if "cuda" in msg and "ptx" in msg:
        return CheckResult(
            check_id=CheckId.CUDA_TOOLKIT_MISMATCH,
            severity=CheckSeverity.BLOCK,
            message="CUDA toolkit version mismatch.",
            suggestion="Install cuda-compat package or update NVIDIA drivers.",
        )

    if "vllm_use_v1" in msg and "not supported" in msg:
        return CheckResult(
            check_id=CheckId.VLLM_V1_CPU_INCOMPATIBLE,
            severity=CheckSeverity.BLOCK,
            message="vLLM V1 engine not supported on this platform.",
            suggestion="Set VLLM_USE_V1=0.",
        )

    if "compute capability" in msg or "minimum capability" in msg:
        return CheckResult(
            check_id=CheckId.GPU_COMPUTE_CAP_TOO_LOW,
            severity=CheckSeverity.BLOCK,
            message="GPU compute capability too low for this model/quantization.",
            suggestion="Use a different quantization or upgrade GPU.",
        )

    if "bitsandbytes" in msg and ("not support" in msg or "not compatible" in msg):
        return CheckResult(
            check_id=CheckId.OOM_ESTIMATE,
            severity=CheckSeverity.BLOCK,
            message="BitsAndBytes quantization failed: model architecture or "
            "GPU is not compatible with BitsAndBytes.",
            suggestion="Use a pre-quantized model (AWQ or GPTQ) that fits in "
            "GPU VRAM, or use a smaller model.",
        )

    if "bfloat16 is only supported" in msg:
        return CheckResult(
            check_id=CheckId.CPU_DTYPE_INCOMPATIBLE,
            severity=CheckSeverity.BLOCK,
            message="bfloat16 not supported on this platform.",
            suggestion="Set VLLM_DTYPE=float16.",
        )

    if "graph.replay" in msg or "cudagraph" in msg.replace(" ", ""):
        return CheckResult(
            check_id=CheckId.CUDA_GRAPH_ERROR,
            severity=CheckSeverity.BLOCK,
            message="CUDA graph error during execution.",
            suggestion="Try setting VLLM_ENFORCE_EAGER=1.",
        )

    if "cannot import name" in msg or "no module named" in msg:
        # Model's custom code (trust_remote_code) is incompatible with
        # the installed transformers/torch version. Common case: model
        # uses old import paths like PreTrainedConfig vs PretrainedConfig.
        error_str = str(error)  # preserve original case for the message
        return CheckResult(
            check_id=CheckId.INCOMPATIBLE_REMOTE_CODE,
            severity=CheckSeverity.BLOCK,
            message=f"Model's custom code is incompatible with installed packages: {error_str}",
            suggestion="The model author needs to update their code. "
            "Try a different model, or report the issue to the model's repository.",
        )

    if "uva" in msg or ("pin" in msg and "memory" in msg):
        return CheckResult(
            check_id=CheckId.OOM_ESTIMATE,
            severity=CheckSeverity.BLOCK,
            message="CPU offloading failed: CUDA pinned memory (UVA) is not "
            "available in this environment.",
            suggestion="Use a smaller/more quantized model that fits in GPU VRAM, "
            "or ensure Docker is configured with --gpus all and "
            "nvidia-container-toolkit is properly installed.",
        )

    # Generic fallback
    return CheckResult(
        check_id=CheckId.UNKNOWN_ENGINE_ERROR,
        severity=CheckSeverity.BLOCK,
        message=f"Engine initialization failed: {error}",
        suggestion="Check logs for details.",
    )


def get_recovery_hint(check_id: CheckId) -> str:
    """Get a user-friendly recovery hint for a check ID."""
    hints = {
        CheckId.GPU_REQUIRED_QUANTIZATION: "Use a non-quantized or GPTQ/AWQ model",
        CheckId.CPU_ALL_QUANT_UNSUPPORTED: "Use an unquantized model (bfloat16) or switch to Ollama",
        CheckId.GPU_COMPUTE_CAP_TOO_LOW: "Use a model with less demanding quantization (AWQ/GPTQ) or upgrade GPU",
        CheckId.OOM_ESTIMATE: "Use a smaller model or reduce max_model_len",
        CheckId.OOM_RUNTIME: "Use a smaller model or reduce max_model_len",
        CheckId.OOM_KV_CACHE: "Reduce VLLM_MAX_MODEL_LEN or increase GPU_MEMORY_UTILIZATION",
        CheckId.UNSUPPORTED_ARCHITECTURE: "Check vLLM supported models list",
        CheckId.TRUST_REMOTE_CODE: "Set VLLM_TRUST_REMOTE_CODE=1",
        CheckId.MISSING_TOKENIZER: "Set VLLM_TOKENIZER to the base model ID",
        CheckId.CUDA_TOOLKIT_MISMATCH: "Install cuda-compat package or update NVIDIA drivers",
        CheckId.CUDA_GRAPH_ERROR: "Try setting VLLM_ENFORCE_EAGER=1",
        CheckId.INCOMPATIBLE_REMOTE_CODE: "Model's custom code is incompatible with installed packages. Try a different model.",
        CheckId.VLLM_V1_CPU_INCOMPATIBLE: "Running on CPU, V1 engine auto-disabled",
        CheckId.GATED_MODEL_NO_TOKEN: "Set HUGGING_FACE_HUB_TOKEN with a valid token",
        CheckId.MOE_ON_CPU: "Consider using a dense model for CPU inference",
        CheckId.KV_OFFLOAD_ENABLED: "KV cache offloading to CPU auto-enabled",
    }
    return hints.get(check_id, "Try a different model")
