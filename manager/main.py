"""
vLLM Manager v6 — Embedded Engine with Ollama Semantics.

Single-process architecture: FastAPI app directly embeds AsyncLLMEngine.
No subprocess, no HTTP proxy — direct Python calls to vLLM.
"""

import asyncio
import logging
import signal
import sys
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from cache_manager import HFCacheManager
from config import ManagerConfig
from model_manager import ModelManager
from preflight import PreflightError, get_recovery_hint, preflight_check
from pull_manager import PullManager, PullState
from tokenizer_resolver import resolve_tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("vllm-manager")


# ─── FastAPI Application ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize managers on startup, cleanup on shutdown."""
    config = ManagerConfig()
    app.state.config = config
    app.state.model_mgr = ModelManager(config)
    app.state.pull_mgr = PullManager(hf_token=config.hf_token)
    app.state.cache_mgr = HFCacheManager(hf_token=config.hf_token)

    # Log configuration
    logger.info("=" * 60)
    logger.info("vLLM Manager v6 — Embedded Engine")
    logger.info("=" * 60)
    logger.info(f"  Manager port: {config.manager_port}")
    logger.info(f"  CPU mode: {config.is_cpu}")
    logger.info(f"  Keep-alive: {config.keep_alive_seconds}s")
    logger.info(f"  HF token configured: {'yes' if config.hf_token else 'no'}")
    logger.info("=" * 60)

    # List cached models
    try:
        models = app.state.cache_mgr.list_models()
        model_names = [m["name"] for m in models]
        logger.info(f"Models in cache: {model_names if model_names else '(none)'}")
    except Exception as e:
        logger.warning(f"Failed to list cached models: {e}")

    # Don't load default model — models load on-demand (Ollama behavior)
    logger.info("Manager ready. Models load on-demand via inference requests.")

    yield

    # Shutdown
    logger.info("Shutting down manager...")
    try:
        await app.state.model_mgr.unload()
    except Exception as e:
        logger.error(f"Error unloading model: {e}")
    try:
        app.state.pull_mgr.shutdown()
    except Exception as e:
        logger.error(f"Error shutting down pull manager: {e}")
    logger.info("Manager shutdown complete.")


app = FastAPI(title="vLLM Manager v6", lifespan=lifespan)


# ─── Request/Response Models ───────────────────────────────────────

class PullRequest(BaseModel):
    name: str
    stream: bool = True


class ShowRequest(BaseModel):
    name: str


class DeleteRequest(BaseModel):
    name: str


class ValidateRequest(BaseModel):
    name: str


# ─── Inference: /v1/chat/completions (OpenAI-compatible) ───────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions.
    Auto-loads/switches model based on request body (Ollama behavior).

    Uses vLLM's native parsers (via OutputParser) for reasoning extraction
    and tool call parsing on raw engine output.
    """
    mgr: ModelManager = request.app.state.model_mgr
    cache: HFCacheManager = request.app.state.cache_mgr
    body = await request.json()

    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="'model' field is required")

    keep_alive = body.pop("keep_alive", None)
    stream = body.get("stream", False)
    messages = body.get("messages", [])

    if not messages:
        raise HTTPException(status_code=400, detail="'messages' field is required")

    # Verify model is in cache
    if not cache.is_cached(model): pass
        # raise HTTPException(
        #     status_code=404,
        #     detail=f"Model '{model}' not found in cache. Pull it first with POST /api/pull."
        # )

    # Ensure model is loaded (auto-switch if different)
    try:
        await mgr.ensure_loaded(model, keep_alive=keep_alive)
    except PreflightError as e:
        # Return structured error for preflight failures
        logger.error(f"Preflight blocked model: {e.result.block_reason}")
        check = e.result.checks[0] if e.result.checks else None
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": e.result.block_reason,
                    "type": "model_load_error",
                    "code": check.check_id.value if check else "UNKNOWN",
                    "suggestion": check.suggestion if check else None,
                    "checks": [c.to_dict() for c in e.result.checks],
                }
            },
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=503, detail=f"Failed to load model {model}: {e}")

    if mgr.engine is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    # Generate completion (unified path with native output parsing)
    try:
        response = await mgr.chat_completion(
            messages=messages,
            temperature=body.get("temperature", 0.7),
            max_tokens=body.get("max_tokens", 2048),
            stream=stream,
            tools=body.get("tools"),
            tool_choice=body.get("tool_choice", "auto"),
            top_p=body.get("top_p", 1.0),
            frequency_penalty=body.get("frequency_penalty", 0.0),
            presence_penalty=body.get("presence_penalty", 0.0),
            stop=body.get("stop"),
        )

        # Handle keep_alive=0 (unload after request)
        if keep_alive == 0:
            asyncio.create_task(mgr.unload())

        if stream:
            return StreamingResponse(
                response,
                media_type="text/event-stream",
            )
        else:
            return response

    except Exception as e:
        logger.error(f"Inference error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_loaded_models(request: Request):
    """OpenAI-compatible model listing."""
    mgr: ModelManager = request.app.state.model_mgr
    cache: HFCacheManager = request.app.state.cache_mgr

    models = []
    loaded = mgr.current_model

    for m in cache.list_models():
        models.append({
            "id": m["name"],
            "object": "model",
            "owned_by": "local",
            "loaded": m["name"] == loaded,
        })

    return {"object": "list", "data": models}


# ─── Model Management (Ollama-compatible) ──────────────────────────

@app.get("/api/tags")
async def list_models(request: Request):
    """List downloaded models (Ollama-compatible)."""
    cache: HFCacheManager = request.app.state.cache_mgr
    mgr: ModelManager = request.app.state.model_mgr

    models = cache.list_models()

    # Annotate which model is currently loaded
    for m in models:
        m["loaded"] = m["name"] == mgr.current_model

    return {"models": models}


@app.post("/api/pull")
async def pull_model(req: PullRequest, request: Request):
    """
    Pull (download) a model from HuggingFace Hub.
    Non-blocking: download runs in background, inference continues.
    Streams NDJSON progress (Ollama-compatible).
    """
    pull_mgr: PullManager = request.app.state.pull_mgr

    job = await pull_mgr.pull(req.name)

    if job.state == PullState.COMPLETED:
        return {"status": "success"}

    if job.state == PullState.FAILED:
        raise HTTPException(status_code=400, detail=job.error)

    if req.stream:
        return StreamingResponse(
            pull_mgr.subscribe(req.name),
            media_type="application/x-ndjson",
        )
    else:
        # Wait for completion
        while job.state in (PullState.QUEUED, PullState.DOWNLOADING):
            await asyncio.sleep(1)
            job = pull_mgr.get_pull_status(req.name)
            if job is None:
                break

        if job and job.state == PullState.FAILED:
            raise HTTPException(status_code=400, detail=job.error)

        return {"status": "success"}


@app.get("/api/pull/status/{model_id:path}")
async def pull_status(model_id: str, request: Request):
    """Check pull progress (reconnect-friendly)."""
    pull_mgr: PullManager = request.app.state.pull_mgr
    job = pull_mgr.get_pull_status(model_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"No pull found for {model_id}")

    return job.to_status_dict()


@app.delete("/api/pull/{model_id:path}")
async def cancel_pull(model_id: str, request: Request):
    """Cancel an active pull."""
    pull_mgr: PullManager = request.app.state.pull_mgr

    if await pull_mgr.cancel(model_id):
        return {"status": "cancelled"}
    else:
        raise HTTPException(status_code=404, detail=f"No active pull for {model_id}")


@app.post("/api/show")
async def show_model(req: ShowRequest, request: Request):
    """Model information (Ollama-compatible)."""
    cache: HFCacheManager = request.app.state.cache_mgr
    try:
        return cache.get_model_info(req.name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/validate")
async def validate_model(req: ValidateRequest, request: Request):
    """
    Dry-run preflight validation without loading the model.
    Returns validation results and predicted engine overrides.
    """
    config: ManagerConfig = request.app.state.config
    cache: HFCacheManager = request.app.state.cache_mgr

    # Check if model is in cache
    if not cache.is_cached(req.name):
        return JSONResponse(
            status_code=404,
            content={
                "valid": False,
                "error": f"Model '{req.name}' not in cache. Pull it first with POST /api/pull.",
            },
        )

    # Resolve tokenizer
    tokenizer = resolve_tokenizer(req.name, hf_token=config.hf_token)

    # Run preflight
    try:
        result = await preflight_check(
            model_id=req.name,
            config=config,
            resolved_tokenizer=tokenizer,
        )
    except Exception as e:
        logger.error(f"Preflight check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"valid": False, "error": f"Preflight check failed: {e}"},
        )

    status_code = 200 if result.passed else 422
    return JSONResponse(status_code=status_code, content=result.to_dict())


@app.delete("/api/delete")
async def delete_model(req: DeleteRequest, request: Request):
    """Delete model from cache (Ollama-compatible)."""
    mgr: ModelManager = request.app.state.model_mgr
    cache: HFCacheManager = request.app.state.cache_mgr

    # Can't delete loaded model
    if mgr.current_model == req.name and mgr.state == "ready":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete currently loaded model '{req.name}'. "
                   f"Wait for keep_alive to expire, or load a different model first.",
        )

    if not cache.delete_model(req.name):
        raise HTTPException(status_code=404, detail=f"Model '{req.name}' not found")

    return {"status": "success"}


# ─── Health & Status ───────────────────────────────────────────────

@app.get("/health")
async def health(request: Request):
    """Health check with preflight info."""
    mgr: ModelManager = request.app.state.model_mgr
    status = mgr.get_status()

    if status["state"] == "ready":
        response = {"status": "ok", "model": status["model"]}
        # Include preflight info if available
        if mgr._last_preflight:
            pf = mgr._last_preflight
            response["preflight"] = {
                "passed": pf.passed,
                "warnings": pf.warnings,
                "engine_overrides": pf.engine_overrides,
            }
        return response
    elif status["state"] == "loading":
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "model": status["model"]},
        )
    elif status["state"] == "idle":
        return {"status": "idle", "message": "No model loaded (will auto-load on first request)"}
    else:
        # Error state - include preflight info if it was a preflight failure
        error_response = {"status": status["state"], "error": status["error"]}
        if mgr._last_preflight and mgr._last_preflight.blocked:
            pf = mgr._last_preflight
            check = pf.checks[0] if pf.checks else None
            error_response["preflight_blocked"] = True
            error_response["check_id"] = check.check_id.value if check else None
            error_response["suggestion"] = check.suggestion if check else None
            error_response["hint"] = get_recovery_hint(check.check_id) if check else None
        return JSONResponse(status_code=503, content=error_response)


@app.get("/api/status")
async def full_status(request: Request):
    """Detailed status of engine, pulls, and cache."""
    mgr: ModelManager = request.app.state.model_mgr
    pull_mgr: PullManager = request.app.state.pull_mgr
    cache: HFCacheManager = request.app.state.cache_mgr

    return {
        "engine": mgr.get_status(),
        "pulls": pull_mgr.get_active_pulls(),
        "cache": {
            "models": [m["name"] for m in cache.list_models()],
            "count": len(cache.list_models()),
        },
    }


# ─── Signal Handlers ───────────────────────────────────────────────

def handle_signal(signum, frame):
    """Handle termination signals gracefully."""
    sig_name = signal.Signals(signum).name
    logger.info(f"Received {sig_name}, initiating graceful shutdown...")
    sys.exit(0)


# ─── Entry Point ───────────────────────────────────────────────────

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        import uvicorn
    except ImportError as e:
        logger.error(f"Failed to import uvicorn: {e}")
        logger.error("Install with: pip install uvicorn[standard]")
        sys.exit(1)

    config = ManagerConfig()

    try:
        logger.info(f"Starting uvicorn on 0.0.0.0:{config.manager_port}")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=config.manager_port,
            log_level="info",
            access_log=True,
        )
    except Exception as e:
        logger.error(f"Failed to start uvicorn: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
