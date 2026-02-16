"""
HuggingFace Cache Manager.

Manages models in the HuggingFace cache directory.
Provides Ollama-compatible listing, info, and deletion.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, scan_cache_dir

logger = logging.getLogger("vllm-manager")


class HFCacheManager:
    """Manages models in the HuggingFace cache directory."""

    def __init__(self, hf_token: Optional[str] = None):
        self.api = HfApi(token=hf_token if hf_token else None)
        self.cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    def list_models(self) -> list[dict]:
        """List all complete downloaded models in the HF cache (Ollama /api/tags format)."""
        try:
            cache_info = scan_cache_dir()
        except Exception as e:
            logger.warning(f"Failed to scan cache: {e}")
            return []

        models = []
        for repo in cache_info.repos:
            if repo.repo_type != "model":
                continue

            # Only include complete models (verified against manifest if exists)
            if not self.is_cached(repo.repo_id):
                continue

            total_size = repo.size_on_disk

            models.append({
                "name": repo.repo_id,
                "model": repo.repo_id,
                "modified_at": max(
                    (rev.last_modified for rev in repo.revisions),
                    default=0,
                ),
                "size": total_size,
                "digest": str(repo.repo_path),
                "details": {
                    "family": self._detect_family(repo.repo_id),
                    "parameter_size": self._estimate_param_size(repo.repo_id),
                    "quantization_level": self._detect_quantization(repo.repo_id),
                },
            })

        return models

    def get_model_info(self, model_id: str) -> dict:
        """Get detailed info about a downloaded model (Ollama /api/show format)."""
        cache_info = scan_cache_dir()
        local_info = None
        for repo in cache_info.repos:
            if repo.repo_id == model_id:
                local_info = repo
                break

        if local_info is None:
            raise ValueError(f"Model {model_id} not found in cache")

        # Try to read config.json for detailed info
        config = {}
        for revision in local_info.revisions:
            config_path = revision.snapshot_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                break

        return {
            "modelfile": "",  # Ollama compat — not applicable
            "parameters": json.dumps(config, indent=2),
            "template": "",
            "details": {
                "parent_model": "",
                "format": config.get("model_type", "unknown"),
                "family": self._detect_family(model_id),
                "families": [self._detect_family(model_id)],
                "parameter_size": self._estimate_param_size(model_id),
                "quantization_level": self._detect_quantization(model_id),
            },
            "model_info": {
                "general.architecture": config.get("model_type", "unknown"),
                "general.parameter_count": config.get("num_parameters"),
                "context_length": (
                    config.get("max_position_embeddings")
                    or config.get("max_sequence_length")
                    or config.get("seq_length")
                ),
                "hidden_size": config.get("hidden_size"),
                "num_layers": (
                    config.get("num_hidden_layers")
                    or config.get("num_layers")
                ),
                "vocab_size": config.get("vocab_size"),
            },
            "modified_at": max(
                (rev.last_modified for rev in local_info.revisions),
                default=None,
            ),
        }

    def delete_model(self, model_id: str) -> bool:
        """Delete a model from the HuggingFace cache."""
        cache_info = scan_cache_dir()

        for repo in cache_info.repos:
            if repo.repo_id == model_id:
                delete_strategy = cache_info.delete_revisions(
                    *(rev.commit_hash for rev in repo.revisions)
                )
                delete_strategy.execute()
                logger.info(f"Deleted model {model_id}, freed {delete_strategy.expected_freed_size_str}")

                # Also delete manifest
                manifest_path = self._get_manifest_path(model_id)
                if manifest_path.exists():
                    manifest_path.unlink()
                    logger.debug(f"Deleted manifest for {model_id}")

                return True

        return False

    def _get_manifest_path(self, model_id: str) -> Path:
        """Get path to manifest file for a model."""
        safe_name = model_id.replace("/", "__")
        return self.cache_dir / ".vllm-manifests" / f"{safe_name}.json"

    def is_cached(self, model_id: str) -> bool:
        """Check if model exists AND is complete (verified against manifest)."""
        try:
            cache_info = scan_cache_dir()
            repo_exists = any(
                r.repo_id == model_id and r.repo_type == "model"
                for r in cache_info.repos
            )
            if not repo_exists:
                return False

            # If manifest exists, verify all files are present
            manifest_path = self._get_manifest_path(model_id)
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)

                for repo in cache_info.repos:
                    if repo.repo_id == model_id and repo.repo_type == "model":
                        for rev in repo.revisions:
                            snapshot = rev.snapshot_path
                            # Verify all manifest files exist
                            all_exist = all(
                                (snapshot / filename).exists()
                                for filename in manifest["files"]
                            )
                            if all_exist:
                                return True
                return False  # Manifest exists but files incomplete

            # No manifest = legacy download, trust that it's complete
            return True
        except Exception:
            return False

    @staticmethod
    def _detect_family(model_id: str) -> str:
        """Detect model family from model ID."""
        name = model_id.lower()
        families = {
            "llama": "llama", "qwen": "qwen", "deepseek": "deepseek",
            "mistral": "mistral", "mixtral": "mistral", "phi": "phi",
            "gemma": "gemma", "hermes": "hermes", "granite": "granite",
        }
        for key, family in families.items():
            if key in name:
                return family
        return "unknown"

    @staticmethod
    def _estimate_param_size(model_id: str) -> str:
        """Extract parameter size from model name (e.g., '8B', '70B')."""
        match = re.search(r'(\d+\.?\d*)[Bb]', model_id)
        return f"{match.group(1)}B" if match else "unknown"

    @staticmethod
    def _detect_quantization(model_id: str) -> str:
        """Detect quantization method from model name."""
        name = model_id.lower()
        for q in ["awq", "gptq", "gguf", "fp8", "int4", "int8", "bnb"]:
            if q in name:
                return q.upper()
        return "FP16"
