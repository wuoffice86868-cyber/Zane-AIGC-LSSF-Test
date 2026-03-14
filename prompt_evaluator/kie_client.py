"""KIE API client for Seedream and Seedance model generation.

Wraps the api.kie.ai REST API for:
- Seedream 4.5: text-to-image generation
- Seedance 1.5 Pro: text/image-to-video generation

Includes task creation, polling with exponential backoff, result
download, and cumulative cost tracking.

Usage:
    from prompt_evaluator.kie_client import KieClient

    client = KieClient(api_key="...")

    # Generate an image
    result = client.generate_image("A hotel pool at sunset")

    # Generate a video from image
    result = client.generate_video(
        prompt="Single continuous shot. Camera pushes forward...",
        image_url="https://...",
    )

    # Check spending
    print(client.stats)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen, ProxyHandler, build_opener
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.kie.ai"
CREATE_TASK_PATH = "/api/v1/jobs/createTask"
QUERY_TASK_PATH = "/api/v1/jobs/recordInfo"

MODEL_SEEDREAM = "seedream/4.5-text-to-image"
# Kie.ai API model identifier (required by their endpoint)
MODEL_SEEDANCE = "bytedance/seedance-1.5-pro"

# Task states
STATE_WAITING = "waiting"
STATE_QUEUING = "queuing"
STATE_GENERATING = "generating"
STATE_SUCCESS = "success"
STATE_FAIL = "fail"

TERMINAL_STATES = {STATE_SUCCESS, STATE_FAIL}
ACTIVE_STATES = {STATE_WAITING, STATE_QUEUING, STATE_GENERATING}

# Polling defaults
DEFAULT_POLL_INTERVAL = 3.0      # seconds (initial)
DEFAULT_POLL_MAX_INTERVAL = 30.0  # seconds (cap after backoff)
DEFAULT_POLL_TIMEOUT = 600.0     # seconds (10 min max wait)
BACKOFF_FACTOR = 1.5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Result of a completed generation task."""

    task_id: str
    model: str
    state: str
    result_urls: List[str] = field(default_factory=list)
    cost_time_ms: int = 0
    create_time: int = 0
    complete_time: int = 0
    error_message: str = ""

    @property
    def success(self) -> bool:
        return self.state == STATE_SUCCESS

    @property
    def cost_time_sec(self) -> float:
        return self.cost_time_ms / 1000.0


@dataclass
class ClientStats:
    """Cumulative usage statistics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_cost_time_ms: int = 0
    image_generations: int = 0
    video_generations: int = 0

    @property
    def total_cost_time_sec(self) -> float:
        return self.total_cost_time_ms / 1000.0

    def summary(self) -> str:
        return (
            f"KIE API Stats: {self.total_requests} requests "
            f"({self.successful_requests} ok, {self.failed_requests} failed) | "
            f"Images: {self.image_generations}, Videos: {self.video_generations} | "
            f"Total gen time: {self.total_cost_time_sec:.1f}s"
        )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class KieAPIError(Exception):
    """Raised when the KIE API returns an error."""

    def __init__(self, message: str, status_code: int = 0, response: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class KieTimeoutError(KieAPIError):
    """Raised when polling exceeds the timeout."""
    pass


class KieBudgetError(KieAPIError):
    """Raised when a budget limit would be exceeded."""
    pass


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class KieClient:
    """Client for the KIE (api.kie.ai) generation API.

    Args:
        api_key: API key for Bearer authentication.
            If *None*, reads from ``KIE_API_KEY`` env var or the
            credentials file at ``~/.openclaw/workspace/.credentials/kie-api.json``.
        base_url: Override the API base URL.
        max_requests: Hard cap on total API calls (safety guard).
            Set to 0 for unlimited.
        callback_url: Optional callback URL for async notifications.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = BASE_URL,
        max_requests: int = 50,
        callback_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or self._resolve_api_key()
        if not self.api_key:
            raise KieAPIError("No API key provided and none found in env or credentials file")

        self.base_url = base_url.rstrip("/")
        self.max_requests = max_requests
        self.callback_url = callback_url
        self.stats = ClientStats()

    # ------------------------------------------------------------------
    # Key resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_api_key() -> Optional[str]:
        """Try env var, then credentials file."""
        key = os.environ.get("KIE_API_KEY")
        if key:
            return key

        cred_path = Path.home() / ".openclaw" / "workspace" / ".credentials" / "kie-api.json"
        if cred_path.exists():
            try:
                data = json.loads(cred_path.read_text())
                return data.get("api_key")
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        """Make an authenticated HTTP request to the KIE API.
        
        Bypasses system proxy (localhost:3128 may be intermittently down)
        and retries on transient connection errors.
        """
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = json.dumps(body).encode("utf-8") if body else None
        req = Request(url, data=data, headers=headers, method=method)
        
        # Build opener that bypasses proxy
        opener = build_opener(ProxyHandler({}))
        
        max_retries = 3
        last_error: Optional[Exception] = None
        
        for attempt in range(max_retries):
            try:
                with opener.open(req, timeout=30) as resp:
                    resp_body = resp.read().decode("utf-8")
                    result = json.loads(resp_body)

                    if result.get("code") != 200:
                        raise KieAPIError(
                            f"API error: {result.get('msg', 'unknown')}",
                            status_code=result.get("code", 0),
                            response=resp_body,
                        )
                    return result

            except HTTPError as e:
                body_text = ""
                try:
                    body_text = e.read().decode("utf-8")
                except Exception:
                    pass
                raise KieAPIError(
                    f"HTTP {e.code}: {e.reason}",
                    status_code=e.code,
                    response=body_text,
                ) from e
            except URLError as e:
                last_error = e
                logger.warning(
                    "Connection error on attempt %d/%d: %s",
                    attempt + 1, max_retries, e.reason,
                )
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
        
        raise KieAPIError(f"Connection error after {max_retries} retries: {last_error.reason}") from last_error

    # ------------------------------------------------------------------
    # Budget guard
    # ------------------------------------------------------------------

    def _check_budget(self) -> None:
        """Raise KieBudgetError if max_requests would be exceeded."""
        if self.max_requests > 0 and self.stats.total_requests >= self.max_requests:
            raise KieBudgetError(
                f"Budget limit reached: {self.stats.total_requests}/{self.max_requests} "
                f"requests used. Call reset_budget() or increase max_requests to continue."
            )

    def reset_budget(self, new_max: Optional[int] = None) -> None:
        """Reset the request counter (not stats). Optionally set new max."""
        if new_max is not None:
            self.max_requests = new_max
        # Stats are cumulative, but the budget gate checks total_requests
        # so we don't reset stats — just allow more headroom by raising max

    # ------------------------------------------------------------------
    # Task creation
    # ------------------------------------------------------------------

    def _create_task(self, model: str, input_params: dict) -> str:
        """Create a generation task and return the task ID."""
        self._check_budget()

        payload: Dict[str, Any] = {
            "model": model,
            "input": input_params,
        }
        if self.callback_url:
            payload["callBackUrl"] = self.callback_url

        result = self._request("POST", CREATE_TASK_PATH, payload)
        task_id = result.get("data", {}).get("taskId", "")
        if not task_id:
            raise KieAPIError("No taskId in response", response=json.dumps(result))

        self.stats.total_requests += 1
        logger.info("Created task %s (model=%s)", task_id, model)
        return task_id

    # ------------------------------------------------------------------
    # Task polling
    # ------------------------------------------------------------------

    def _poll_task(
        self,
        task_id: str,
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_interval: float = DEFAULT_POLL_MAX_INTERVAL,
        timeout: float = DEFAULT_POLL_TIMEOUT,
    ) -> TaskResult:
        """Poll a task until it reaches a terminal state.

        Uses exponential backoff starting at ``poll_interval`` and capped
        at ``max_interval``.
        """
        start = time.monotonic()
        interval = poll_interval

        while True:
            elapsed = time.monotonic() - start
            if elapsed > timeout:
                raise KieTimeoutError(
                    f"Task {task_id} did not complete within {timeout}s"
                )

            result = self._query_task(task_id)
            if result.state in TERMINAL_STATES:
                return result

            logger.debug(
                "Task %s state=%s, waiting %.1fs (elapsed %.1fs)",
                task_id, result.state, interval, elapsed,
            )
            time.sleep(interval)
            interval = min(interval * BACKOFF_FACTOR, max_interval)

    def _query_task(self, task_id: str) -> TaskResult:
        """Query the status of a single task."""
        result = self._request("GET", f"{QUERY_TASK_PATH}?taskId={task_id}")
        data = result.get("data", {})

        # Parse resultJson
        result_urls: List[str] = []
        result_json_str = data.get("resultJson", "")
        if result_json_str:
            try:
                rj = json.loads(result_json_str)
                result_urls = rj.get("resultUrls", [])
            except json.JSONDecodeError:
                pass

        return TaskResult(
            task_id=data.get("taskId", task_id),
            model=data.get("model", ""),
            state=data.get("state", "unknown"),
            result_urls=result_urls,
            cost_time_ms=data.get("costTime", 0),
            create_time=data.get("createTime", 0),
            complete_time=data.get("completeTime", 0),
        )

    # ------------------------------------------------------------------
    # Public: Image generation
    # ------------------------------------------------------------------

    def generate_image(
        self,
        prompt: str,
        *,
        aspect_ratio: str = "16:9",
        quality: str = "basic",
        poll: bool = True,
        timeout: float = DEFAULT_POLL_TIMEOUT,
    ) -> TaskResult:
        """Generate an image using Seedream 4.5.

        Args:
            prompt: Text description of the desired image.
            aspect_ratio: Output aspect ratio (e.g. "1:1", "16:9", "9:16").
            quality: Quality preset ("basic" or "hd").
            poll: If True, block until the task completes.
            timeout: Max seconds to wait when polling.

        Returns:
            ``TaskResult`` with URLs in ``result_urls``.
        """
        input_params = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "quality": quality,
        }

        task_id = self._create_task(MODEL_SEEDREAM, input_params)
        self.stats.image_generations += 1

        if not poll:
            return TaskResult(task_id=task_id, model=MODEL_SEEDREAM, state=STATE_WAITING)

        result = self._poll_task(task_id, timeout=timeout)
        self._update_stats(result)
        return result

    # ------------------------------------------------------------------
    # Public: Video generation
    # ------------------------------------------------------------------

    def generate_video(
        self,
        prompt: str,
        *,
        image_url: Optional[str] = None,
        image_urls: Optional[List[str]] = None,
        aspect_ratio: str = "16:9",
        resolution: str = "1080p",
        duration: int = 8,
        fixed_lens: bool = False,
        poll: bool = True,
        timeout: float = DEFAULT_POLL_TIMEOUT,
    ) -> TaskResult:
        """Generate a video using Seedance 1.5 Pro.

        Args:
            prompt: Cinematography prompt for the video.
            image_url: Single input image URL (convenience shortcut).
            image_urls: List of input image URLs (0-2). Takes priority
                over ``image_url`` if both are provided.
            aspect_ratio: Output aspect ratio.
            resolution: Video resolution ("720p" or "1080p").
            duration: Video duration in seconds (only 8 accepted for 1.5 Pro).
            fixed_lens: Lock the camera (no camera movement).
            poll: If True, block until the task completes.
            timeout: Max seconds to wait when polling.

        Returns:
            ``TaskResult`` with video URLs in ``result_urls``.
        """
        # Resolve input URLs
        urls: List[str] = []
        if image_urls is not None:
            urls = image_urls[:2]
        elif image_url is not None:
            urls = [image_url]

        input_params: Dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "duration": str(duration),
            "fixed_lens": fixed_lens,
            "generate_audio": False,  # ALWAYS false per constraint
        }
        if urls:
            input_params["input_urls"] = urls

        task_id = self._create_task(MODEL_SEEDANCE, input_params)
        self.stats.video_generations += 1

        if not poll:
            return TaskResult(task_id=task_id, model=MODEL_SEEDANCE, state=STATE_WAITING)

        result = self._poll_task(task_id, timeout=timeout)
        self._update_stats(result)
        return result

    # ------------------------------------------------------------------
    # Public: Task management
    # ------------------------------------------------------------------

    def query_task(self, task_id: str) -> TaskResult:
        """Query the current status of a task (non-blocking)."""
        return self._query_task(task_id)

    def wait_for_task(
        self,
        task_id: str,
        *,
        timeout: float = DEFAULT_POLL_TIMEOUT,
    ) -> TaskResult:
        """Wait for a task to complete (blocking with backoff)."""
        result = self._poll_task(task_id, timeout=timeout)
        self._update_stats(result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_stats(self, result: TaskResult) -> None:
        """Update cumulative stats from a completed task."""
        if result.success:
            self.stats.successful_requests += 1
        else:
            self.stats.failed_requests += 1
        self.stats.total_cost_time_ms += result.cost_time_ms

    # ------------------------------------------------------------------
    # LLMClient protocol compatibility
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Satisfy the ``LLMClient`` protocol for use with PromptOptimizer.

        This generates a video from the prompt and returns the first
        result URL.  NOT intended for LLM text generation — use this
        only when the optimizer needs to evaluate video outputs.
        """
        result = self.generate_video(prompt, **kwargs)
        if result.result_urls:
            return result.result_urls[0]
        return ""
