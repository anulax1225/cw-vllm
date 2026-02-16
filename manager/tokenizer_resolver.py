"""
Tokenizer resolver for community-quantized models.

When a quantized model is missing tokenizer files (vocab.json, tokenizer.json, etc.),
this module detects the issue and resolves the base model to use as tokenizer source.

Used by ModelManager before creating AsyncEngineArgs.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, scan_cache_dir, snapshot_download

logger = logging.getLogger("vllm-manager")

# ── Files that indicate a model has its own tokenizer ───────────────

TOKENIZER_INDICATOR_FILES = [
    "tokenizer.json",       # HuggingFace fast tokenizer
    "tokenizer.model",      # SentencePiece (LLaMA, Mistral, etc.)
    "vocab.json",           # BPE vocabulary (GPT-2 family)
    "spiece.model",         # SentencePiece alternative name
]

# ── config.json fields that may point to the base model ─────────────

BASE_MODEL_CONFIG_KEYS = [
    "_name_or_path",               # Set by transformers on save — most reliable
    "base_model",                  # Some quantizers set this explicitly
    "base_model_name_or_path",     # AutoGPTQ convention
    "model_name_or_path",          # Some fine-tuning tools
]

# ── Known quantizer orgs whose models always need external tokenizer ─

KNOWN_QUANTIZER_ORGS = [
    "TheBloke",
    "TechxGenus",
    "turboderp",       # exl2 quants
    "LoneStriker",
    "bartowski",
]

# ── Quantization suffixes to strip when guessing base model ─────────

QUANT_SUFFIXES = re.compile(
    r'[-_]('
    r'AWQ|GPTQ|GGUF|EXL2|FP8|BNB|NF4|'
    r'\d+bit(?:[-_]g\d+)?|'           # "6bit", "6bit-g128"
    r'Int[48]|'                        # "Int4", "Int8"
    r'Q[2-8]_[KM0-9_]+|'              # GGUF quant names "Q4_K_M"
    r'[Qq]uant(?:ized)?'
    r')(?:[-_].*)?$',
    re.IGNORECASE
)

# ── Tokenizer files to download from base model ────────────────────

TOKENIZER_DOWNLOAD_PATTERNS = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "spiece.model",
    "added_tokens.json",
    "chat_template.json",
]

# ── Model name → canonical HuggingFace org ──────────────────────────

ORG_PATTERNS = {
    "llama": "meta-llama",
    "codellama": "meta-llama",
    "qwen": "Qwen",
    "deepseek": "deepseek-ai",
    "mistral": "mistralai",
    "mixtral": "mistralai",
    "phi-": "microsoft",
    "gemma": "google",
    "hermes": "NousResearch",
    "yi-": "01-ai",
    "starcoder": "bigcode",
    "internlm": "internlm",
    "chatglm": "THUDM",
    "baichuan": "baichuan-inc",
    "falcon": "tiiuae",
    "mpt-": "mosaicml",
    "stablelm": "stabilityai",
    "command": "CohereForAI",
}


def resolve_tokenizer(model_id: str, hf_token: Optional[str] = None) -> Optional[str]:
    """
    Check if a model needs an external tokenizer.
    Returns the base model ID to use as tokenizer, or None if the model
    has its own tokenizer files.

    Call this BEFORE creating AsyncEngineArgs. If it returns a string,
    pass that as the `tokenizer` argument.
    """
    snapshot_path = _get_snapshot_path(model_id)
    if snapshot_path is None:
        logger.warning(f"Model {model_id} not found in cache, skipping tokenizer check")
        return None

    # Step 1: Check if model already has tokenizer files
    if _has_tokenizer_files(snapshot_path):
        logger.debug(f"{model_id} has its own tokenizer files")
        return None

    logger.warning(f"{model_id} is missing tokenizer files, resolving base model...")

    # Step 2: Check config.json for base model reference
    base = _resolve_from_config(snapshot_path, model_id)
    if base:
        logger.info(f"Resolved tokenizer from config.json: {model_id} -> {base}")
        return base

    # Step 3: Try name-based heuristics
    base = _resolve_from_name(model_id)
    if base:
        # Validate the guessed base model actually exists on HF Hub
        if _validate_model_exists(base, hf_token):
            logger.info(f"Resolved tokenizer from name heuristics: {model_id} -> {base}")
            return base
        else:
            logger.warning(f"Guessed base model '{base}' does not exist on HF Hub")

    # Step 4: Give up with a helpful error
    logger.error(
        f"Cannot resolve tokenizer for {model_id}. "
        f"This is likely a community-quantized model missing tokenizer files. "
        f"Set VLLM_TOKENIZER=<org>/<base-model-name> to specify the tokenizer source."
    )
    return None


def ensure_tokenizer_available(tokenizer_model: str, hf_token: Optional[str] = None) -> str:
    """
    Make sure the tokenizer model files are available locally.
    Downloads only the tokenizer files (not full weights) if missing.
    Returns the model ID (unchanged).
    """
    snapshot = _get_snapshot_path(tokenizer_model)

    if snapshot and _has_tokenizer_files(snapshot):
        return tokenizer_model  # Already cached

    logger.info(f"Downloading tokenizer files from {tokenizer_model}...")

    snapshot_download(
        tokenizer_model,
        token=hf_token if hf_token else None,
        allow_patterns=TOKENIZER_DOWNLOAD_PATTERNS,
    )

    logger.info(f"Tokenizer files downloaded for {tokenizer_model}")
    return tokenizer_model


# ── Internal helpers ────────────────────────────────────────────────

def _get_snapshot_path(model_id: str) -> Optional[Path]:
    """Find the latest snapshot path for a model in HF cache."""
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_id and repo.repo_type == "model":
                # Get the latest revision's snapshot path
                revisions = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)
                if revisions:
                    return revisions[0].snapshot_path
    except Exception as e:
        logger.warning(f"Failed to scan cache for {model_id}: {e}")
    return None


def _has_tokenizer_files(snapshot_path: Path) -> bool:
    """Check if the model snapshot contains tokenizer files."""
    for filename in TOKENIZER_INDICATOR_FILES:
        if (snapshot_path / filename).exists():
            return True

    # Also check: tokenizer_config.json might reference a local tokenizer
    config_path = snapshot_path / "tokenizer_config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                tok_config = json.load(f)
            # If it references a tokenizer_file that exists locally, we're good
            if "tokenizer_file" in tok_config:
                ref = tok_config["tokenizer_file"]
                if (snapshot_path / ref).exists():
                    return True
        except (json.JSONDecodeError, KeyError):
            pass

    return False


def _resolve_from_config(snapshot_path: Path, model_id: str) -> Optional[str]:
    """Try to find the base model from config.json fields."""
    config_path = snapshot_path / "config.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path) as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    for key in BASE_MODEL_CONFIG_KEYS:
        value = config.get(key)
        if not value or not isinstance(value, str):
            continue

        # Skip if it points to itself or a local path
        if value == model_id:
            continue
        if value.startswith("/") or value.startswith("./"):
            continue

        # Must look like an org/model HF repo ID
        if "/" in value and not value.startswith("http"):
            return value

    return None


def _resolve_from_name(model_id: str) -> Optional[str]:
    """
    Heuristic: try to guess the base model from the quantized model name.

    Examples:
        "petergilani/Qwen3-Coder-Next-6bit-g128" -> "Qwen/Qwen3-Coder-Next"
        "TheBloke/Llama-2-7B-Chat-AWQ"           -> "meta-llama/Llama-2-7B-Chat"
        "TechxGenus/Qwen2.5-72B-Instruct-GPTQ-Int4" -> "Qwen/Qwen2.5-72B-Instruct"
    """
    if "/" not in model_id:
        return None

    org, name = model_id.split("/", 1)

    # Strip quantization suffixes from the model name
    base_name = QUANT_SUFFIXES.sub("", name).rstrip("-_")

    if base_name == name:
        # No quant suffix detected — can't infer
        return None

    # If the org is a known quantizer, we need to guess the original org
    if org in KNOWN_QUANTIZER_ORGS:
        guessed_org = _guess_original_org(base_name)
        if guessed_org:
            return f"{guessed_org}/{base_name}"
        return None

    # If the org is NOT a known quantizer (e.g., "petergilani"),
    # the base model probably lives under the canonical org.
    guessed_org = _guess_original_org(base_name)
    if guessed_org:
        return f"{guessed_org}/{base_name}"

    return None


def _guess_original_org(model_name: str) -> Optional[str]:
    """
    Guess the HuggingFace org for a model based on its name.
    Returns None if can't determine.
    """
    name_lower = model_name.lower()

    for pattern, org in ORG_PATTERNS.items():
        if pattern in name_lower:
            return org

    return None


def _validate_model_exists(model_id: str, hf_token: Optional[str] = None) -> bool:
    """Check if a model exists on HuggingFace Hub."""
    try:
        api = HfApi(token=hf_token if hf_token else None)
        api.model_info(model_id)
        return True
    except Exception:
        return False
