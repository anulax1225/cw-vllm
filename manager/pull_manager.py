"""
Pull Manager — Background model downloads.

Non-blocking downloads: inference continues while models download.
Features:
- Deduplication: concurrent pulls for same model → single job
- Progress streaming via async queues
- Reconnection support via GET /api/pull/status/{model}
- Cancellation support
"""

import asyncio
import json
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Optional

from huggingface_hub import HfApi, hf_hub_download, scan_cache_dir

logger = logging.getLogger("vllm-manager")


class PullState(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PullJob:
    """Represents a model download job."""
    model_id: str
    state: PullState = PullState.QUEUED
    total_bytes: int = 0
    downloaded_bytes: int = 0
    files_total: int = 0
    files_done: int = 0
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None
    _subscribers: list = field(default_factory=list)
    _future: Optional[object] = None
    _files: list = field(default_factory=list)  # File metadata for progress tracking

    @property
    def progress_pct(self) -> float:
        """Calculate download progress percentage."""
        if self.total_bytes == 0:
            return 0
        return round(self.downloaded_bytes / self.total_bytes * 100, 1)

    def to_ndjson_line(self) -> str:
        """Produce one NDJSON progress line (Ollama-compatible format)."""
        if self.state == PullState.FAILED:
            return json.dumps({"error": self.error}) + "\n"

        if self.state == PullState.COMPLETED:
            return json.dumps({"status": "success"}) + "\n"

        if self.state == PullState.CANCELLED:
            return json.dumps({"status": "cancelled"}) + "\n"

        return json.dumps({
            "status": f"downloading {self.model_id[0:5]}... - {self._files[self.files_done if len(self._files) > self.files_done else len(self._files) - 1].rfilename}" if self.state == PullState.DOWNLOADING
                      else f"pulling manifest for {self.model_id}",
            "total": self.total_bytes,
            "completed": self.downloaded_bytes,
            "files_total": self.files_total,
            "files_done": self.files_done,
        }) + "\n"

    def to_status_dict(self) -> dict:
        """Return status dict for /api/pull/status endpoint."""
        return {
            "model": self.model_id,
            "state": self.state.value,
            "progress": self.progress_pct,
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "files_total": self.files_total,
            "files_done": self.files_done,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


def get_file_size(sibling) -> int:
    """Get file size, checking both direct size and LFS size.

    HuggingFace stores large file sizes in the `lfs` attribute, not `size`.
    """
    # Direct size attribute
    if sibling.size:
        return sibling.size

    # LFS (Large File Storage) - used for model weights
    if hasattr(sibling, 'lfs') and sibling.lfs:
        if isinstance(sibling.lfs, dict):
            return sibling.lfs.get('size', 0)
        else:
            return getattr(sibling.lfs, 'size', 0)

    return 0


class PullManager:
    """
    Manages model downloads in the background.
    Inference continues uninterrupted during pulls.
    """

    def __init__(self, hf_token: Optional[str] = None, max_workers: int = 2):
        self.hf_token = hf_token
        self.api = HfApi(token=hf_token if hf_token else None)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._active: dict[str, PullJob] = {}
        self._history: deque[PullJob] = deque(maxlen=20)
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def pull(self, model_id: str) -> PullJob:
        """
        Start pulling a model. Returns immediately.
        If already pulling this model, returns existing job (deduplication).
        """
        async with self._lock:
            # Dedup: return existing active pull
            if model_id in self._active:
                existing = self._active[model_id]
                if existing.state in (PullState.QUEUED, PullState.DOWNLOADING):
                    logger.info(f"Pull already active for {model_id}, deduplicating")
                    return existing

            # Check if already downloaded
            if self._is_cached(model_id):
                job = PullJob(model_id=model_id, state=PullState.COMPLETED)
                return job

            # Validate model exists on HF Hub before starting download
            try:
                # files_metadata=True is required to get file sizes
                model_info = self.api.model_info(model_id, files_metadata=True)
                siblings = model_info.siblings or []
                total_size = sum(get_file_size(s) for s in siblings)
                files_total = len(siblings)
                logger.info(f"[PULL] Model info: {files_total} files, {total_size:,} bytes total")

                # Debug: log first few files to verify size retrieval
                for s in siblings[:3]:
                    size = get_file_size(s)
                    logger.debug(f"[PULL] File sample: {s.rfilename} = {size:,} bytes (lfs={getattr(s, 'lfs', None)})")

                # Save manifest for verification (ensures incomplete downloads are detected)
                manifest = {
                    "model_id": model_id,
                    "files": [s.rfilename for s in siblings],
                    "total_bytes": total_size,
                    "files_total": files_total,
                }
                manifest_path = self._get_manifest_path(model_id)
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f)
                logger.debug(f"[PULL] Saved manifest to {manifest_path}")
            except Exception as e:
                logger.error(f"[PULL] Failed to get model info: {e}")
                job = PullJob(model_id=model_id, state=PullState.FAILED, error=str(e))
                return job

            # Create job and start background download
            job = PullJob(
                model_id=model_id,
                state=PullState.QUEUED,
                total_bytes=total_size,
                files_total=files_total,
                _files=siblings,  # Store for progress tracking
            )
            self._active[model_id] = job

            # Store loop reference for thread-safe notifications
            self._loop = asyncio.get_event_loop()
            job._future = self._loop.run_in_executor(
                self._executor,
                self._download_sync,
                job,
            )

            # Start heartbeat loop for SSE keep-alive during large file downloads
            job._heartbeat_task = asyncio.create_task(self._heartbeat_loop(job))

            # Fire-and-forget cleanup when done
            asyncio.ensure_future(self._wait_and_cleanup(model_id, job))

            logger.info(f"Started background pull for {model_id}")
            return job

    def _download_sync(self, job: PullJob) -> None:
        """Synchronous download running in thread pool with progress tracking."""
        job.state = PullState.DOWNLOADING
        logger.info(f"[PULL] Starting download: {job.model_id} ({job.files_total} files)")
        self._notify(job)

        try:
            # Get existing snapshot path to check for already downloaded files
            snapshot_path = None
            try:
                cache_info = scan_cache_dir()
                for repo in cache_info.repos:
                    if repo.repo_id == job.model_id and repo.repo_type == "model":
                        for rev in repo.revisions:
                            snapshot_path = rev.snapshot_path
                            break
                        break
            except Exception:
                pass  # No existing cache, will download everything

            # Use file list from job (populated in pull())
            for i, file_info in enumerate(job._files):
                if job.state == PullState.CANCELLED:
                    logger.info(f"[PULL] Cancelled: {job.model_id}")
                    return

                filename = file_info.rfilename
                file_size = get_file_size(file_info)

                # Check if file already exists (resume support)
                if snapshot_path and (snapshot_path / filename).exists():
                    job.files_done = i + 1
                    job.downloaded_bytes += file_size
                    logger.info(f"[PULL] Skipping already downloaded: {filename} ({file_size:,} bytes)")
                    self._notify(job)
                    continue

                logger.info(f"[PULL] Downloading file {i+1}/{job.files_total}: {filename} ({file_size:,} bytes)")

                # Download file
                hf_hub_download(
                    repo_id=job.model_id,
                    filename=filename,
                    token=self.hf_token if self.hf_token else None,
                )

                # Update progress
                job.files_done = i + 1
                job.downloaded_bytes += file_size
                logger.info(f"[PULL] Progress: {job.progress_pct}% ({job.files_done}/{job.files_total} files)")
                self._notify(job)

            job.state = PullState.COMPLETED
            job.completed_at = time.time()
            job.downloaded_bytes = job.total_bytes  # Ensure exact match
            logger.info(f"Pull complete: {job.model_id}")

        except Exception as e:
            job.state = PullState.FAILED
            job.error = str(e)
            job.completed_at = time.time()
            logger.error(f"Pull failed for {job.model_id}: {e}")

            # Delete manifest on failure to allow retry
            try:
                manifest_path = self._get_manifest_path(job.model_id)
                if manifest_path.exists():
                    manifest_path.unlink()
                    logger.debug(f"[PULL] Deleted manifest for failed download: {job.model_id}")
            except Exception:
                pass

        self._notify(job)

    def _notify(self, job: PullJob) -> None:
        """Push update to all subscribers (for NDJSON streaming).

        Thread-safe: uses call_soon_threadsafe for cross-thread queue updates.
        """
        line = job.to_ndjson_line()
        subscriber_count = len(job._subscribers)
        logger.debug(
            f"[NOTIFY] {job.model_id}: state={job.state.value}, "
            f"progress={job.progress_pct}%, subscribers={subscriber_count}"
        )

        if subscriber_count == 0:
            logger.debug(f"[NOTIFY] No subscribers for {job.model_id}")
            return

        for queue in job._subscribers[:]:
            try:
                # Thread-safe queue update via event loop
                if self._loop and self._loop.is_running():
                    logger.debug(f"[NOTIFY] Using call_soon_threadsafe for {job.model_id}")
                    self._loop.call_soon_threadsafe(self._safe_put, queue, line)
                else:
                    logger.debug(f"[NOTIFY] Direct put_nowait for {job.model_id}")
                    queue.put_nowait(line)
            except Exception as e:
                logger.warning(f"[NOTIFY] Failed to notify: {e}")

    @staticmethod
    def _safe_put(queue: asyncio.Queue, line: str) -> None:
        """Put item in queue, ignoring if full."""
        try:
            queue.put_nowait(line)
            logger.debug(f"[SSE] Queued: {line.strip()[:80]}...")
        except asyncio.QueueFull:
            logger.warning("[SSE] Queue full, dropping message")

    async def _heartbeat_loop(self, job: PullJob) -> None:
        """Send periodic heartbeats to all subscribers during download."""
        try:
            while job.state == PullState.DOWNLOADING:
                await asyncio.sleep(5)  # Send heartbeat every 5 seconds
                if job.state != PullState.DOWNLOADING:
                    break
                # Notify with current progress
                self._notify(job)
        except asyncio.CancelledError:
            pass

    async def _wait_and_cleanup(self, model_id: str, job: PullJob) -> None:
        """Wait for download to finish, then move from active to history."""
        try:
            await asyncio.wrap_future(job._future)
        except Exception:
            pass
        finally:
            # Cancel heartbeat task
            if hasattr(job, '_heartbeat_task') and job._heartbeat_task:
                job._heartbeat_task.cancel()
                try:
                    await job._heartbeat_task
                except asyncio.CancelledError:
                    pass

        async with self._lock:
            if model_id in self._active:
                del self._active[model_id]
                self._history.append(job)

    async def subscribe(self, model_id: str) -> AsyncIterator[str]:
        """
        Subscribe to pull progress for NDJSON streaming.
        Yields progress lines until pull completes.
        """
        logger.info(f"[SSE] New subscriber for {model_id}")

        async with self._lock:
            job = self._active.get(model_id)

        if job is None:
            logger.warning(f"[SSE] No active pull for {model_id}")
            yield json.dumps({"error": "No active pull for this model"}) + "\n"
            return

        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        job._subscribers.append(queue)
        logger.info(f"[SSE] Subscriber added for {model_id}, total subscribers: {len(job._subscribers)}")

        # Send current state immediately
        initial_line = job.to_ndjson_line()
        logger.info(f"[SSE] Sending initial state: {initial_line.strip()[:80]}...")
        yield initial_line

        try:
            while True:
                try:
                    line = await asyncio.wait_for(queue.get(), timeout=10)
                    logger.info(f"[SSE] Yielding to client: {line.strip()[:80]}...")
                    yield line

                    # Check if this is a terminal message (success, error, cancelled)
                    # Don't check job.state - the final message might still be in queue
                    if '"status": "success"' in line or '"error":' in line or '"status": "cancelled"' in line:
                        logger.info(f"[SSE] Terminal message sent, closing stream")
                        break
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    logger.debug(f"[SSE] Sending heartbeat for {model_id}")
                    yield json.dumps({
                        "status": f"downloading {job.model_id}",
                        "total": job.total_bytes,
                        "completed": job.downloaded_bytes,
                    }) + "\n"
        finally:
            logger.info(f"[SSE] Subscriber disconnected from {model_id}")
            if queue in job._subscribers:
                job._subscribers.remove(queue)

    async def cancel(self, model_id: str) -> bool:
        """Cancel an active pull."""
        async with self._lock:
            job = self._active.get(model_id)
            if job and job._future and not job._future.done():
                job._future.cancel()
                job.state = PullState.CANCELLED
                job.completed_at = time.time()
                self._notify(job)
                logger.info(f"Cancelled pull for {model_id}")
                return True
            return False

    def get_pull_status(self, model_id: str) -> Optional[PullJob]:
        """Get status of an active or recently completed pull."""
        if model_id in self._active:
            return self._active[model_id]
        for job in reversed(self._history):
            if job.model_id == model_id:
                return job
        return None

    def get_active_pulls(self) -> dict[str, dict]:
        """Get all active pulls for /api/status endpoint."""
        return {
            mid: {
                "state": job.state.value,
                "progress": job.progress_pct,
                "total_bytes": job.total_bytes,
            }
            for mid, job in self._active.items()
        }

    def _get_manifest_path(self, model_id: str) -> Path:
        """Get path to manifest file for a model."""
        safe_name = model_id.replace("/", "__")
        manifest_dir = Path.home() / ".cache" / "huggingface" / "hub" / ".vllm-manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        return manifest_dir / f"{safe_name}.json"

    def _is_cached(self, model_id: str) -> bool:
        """Check if model is fully downloaded by verifying against manifest."""
        try:
            cache_info = scan_cache_dir()
            snapshot_path = None
            for repo in cache_info.repos:
                if repo.repo_id == model_id and repo.repo_type == "model":
                    for rev in repo.revisions:
                        snapshot_path = rev.snapshot_path
                        break
                    break

            if snapshot_path is None:
                return False  # Not in cache at all

            # If manifest exists, verify all files are present
            manifest_path = self._get_manifest_path(model_id)
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)

                for filename in manifest["files"]:
                    if not (snapshot_path / filename).exists():
                        return False  # Manifest exists but files incomplete

            # Either all manifest files exist, or no manifest (legacy) = cached
            return True
        except Exception:
            return False

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)
