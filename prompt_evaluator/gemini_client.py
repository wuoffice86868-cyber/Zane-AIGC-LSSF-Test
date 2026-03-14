"""Gemini API clients for QC evaluation and LLM generation.

Uses the new google.genai SDK (replaces deprecated google.generativeai).

Provides:
- GeminiVideoQC: Evaluates video quality using Gemini Vision (multimodal)
- GeminiLLM: Text generation client implementing LLMClient protocol
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# QC Prompt — this is the core of video quality evaluation
# ---------------------------------------------------------------------------

HOTEL_VIDEO_QC_PROMPT = """\
You are an expert video quality evaluator for AI-generated hotel/resort marketing clips.

You will be shown a short (5-10 second) AI-generated video clip intended for use in hotel marketing materials. Your job is to evaluate it rigorously, as if you were the final human reviewer before this goes on a hotel's website or social media.

Evaluate across these dimensions:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. CRITICAL DEFECTS (auto-fail)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Any ONE of these present → pass = false

- action_loop: Visible temporal looping — water, fire, smoke, or any motion repeats in an obvious cycle. Look for the exact same ripple/flame/particle pattern recurring.
- face_morphing: Human faces distort, melt, gain/lose features, or shift between identities. Even subtle nose/eye drift counts.
- object_morphing: Solid objects (furniture, architecture, vehicles) change shape, merge, split, or wobble unnaturally. Edges that breathe or pulse.
- structural_collapse: Architectural elements (walls, pillars, railings, pool edges, door frames) warp, bend, or lose geometric integrity.
- sky_anomaly: For OUTDOOR scenes: sky/clouds move unnaturally fast, smear, or exhibit non-physical behavior. (Indoor scenes: skip this check.)
- scene_jump: Abrupt cut or scene transition — the video should be a single continuous shot with no cuts.
- extreme_blur: Motion blur so heavy that the scene becomes unrecognizable for >1 second.
- watermark: Any visible watermark, logo overlay, or text stamp.
- human_distortion: Human body proportions change, limbs stretch/shrink, fingers multiply, clothing merges with skin.
- physics_violation: Gross physics violations — water flowing upward, objects floating without cause, gravity reversed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. MINOR ISSUES (note but don't auto-fail)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- slight_flicker: Brief lighting inconsistency (<0.5s)
- edge_softness: Mild softness at frame edges
- color_drift: Subtle color temperature shift during the clip
- speed_inconsistency: Camera or object motion speed changes unexpectedly
- texture_shimmer: Fine textures (grass, fabric, stone) shimmer or crawl slightly
- depth_inconsistency: Depth of field changes unexpectedly
- reflection_error: Reflections in water/mirrors don't match the scene correctly

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. AESTHETIC QUALITY (1-10)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Evaluate as a short-form video viewer scrolling their feed. Would this stop your thumb?

- Composition and framing — does it feel cinematic or amateur?
- Lighting quality — warm, inviting, moody? Or flat and lifeless?
- Color palette — cohesive and scroll-stopping? Or washed out / oversaturated?
- "Vibe check" — does the clip evoke a feeling (luxury, relaxation, wanderlust)?
- First-frame impact — would the opening frame make you pause scrolling?
- Professional polish — does this look like it belongs on @luxuryhotels or a random upload?

1-3: Instant scroll-past. Obvious AI, no visual appeal.
4-5: You'd notice something is off. Wouldn't save or share.
6-7: Decent for social media. Might stop scrolling briefly.
8-9: Would save and share. Genuinely appealing content.
10: Would go viral. Indistinguishable from professional hotel footage.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. MOTION QUALITY (1-10)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rate as someone watching on a phone screen:
- Camera move: smooth and cinematic, or jittery / robotic?
- Living elements (water, curtains, candles, steam): do they feel alive and natural, or uncanny?
- Speed: does it feel intentional and calming, or awkwardly slow / rushed?
- Static elements: truly still, or subtly drifting/breathing? (Even slight drift is distracting on repeat viewing)
- Overall feel: does the motion add to the mood, or distract from it?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. PROMPT ADHERENCE (1-10)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If you can infer the intended prompt from the video content, rate how well the video matches what was likely requested:
- Camera move type matches intent?
- Scene elements match description?
- Mood/atmosphere matches intent?
(If you cannot infer the prompt, score 5 as neutral.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. SCROLL-STOP FACTOR (1-10)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The scroll-stop test — purely about viewer engagement potential:
- Does the first 0.5 second grab attention?
- Is there a visual "wow" moment?
- Would this make someone comment "where is this?" or "I need to go here"?
- Does it create FOMO or wanderlust?
- Is the pacing right for short-form (not boring, not chaotic)?

1-3: Nobody stops scrolling. Forgettable.
4-5: Mildly interesting. No engagement.
6-7: Would make some people pause.
8-9: Strong engagement potential. Saveable content.
10: Viral potential. Dream destination vibes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a JSON object, no other text:

{
    "pass": true/false,
    "confidence": 0.0-1.0,
    "aesthetic_score": 1-10,
    "motion_score": 1-10,
    "prompt_adherence_score": 1-10,
    "human_present": true/false,
    "auto_fail_triggered": ["rule_name", ...],
    "minor_issues": ["issue_name", ...],
    "scroll_stop_score": 1-10,
    "scene_type_detected": "pool/room/lobby/spa/restaurant/exterior/bathroom/beach/other",
    "summary": "One sentence: what works and what doesn't"
}

IMPORTANT:
- Be STRICT on critical defects. Hotel brands have zero tolerance for morphing faces or warped architecture.
- Be FAIR on aesthetics. AI video is not yet photorealistic — judge relative to current SOTA, not real footage.
- Confidence should reflect how certain you are of your pass/fail decision (0.5 = coin flip, 1.0 = absolutely sure).
"""


# ---------------------------------------------------------------------------
# Gemini Video QC Client (new google.genai SDK)
# ---------------------------------------------------------------------------

class GeminiVideoQC:
    """Video QC evaluator using Gemini Vision (multimodal).

    Args:
        api_key: Google AI API key. Falls back to GEMINI_API_KEY env var.
        model: Gemini model name. Default: gemini-2.5-flash.
        qc_prompt: Custom QC evaluation prompt.
        max_retries: Retry count on API failures.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = "gemini-2.5-flash",
        qc_prompt: str = HOTEL_VIDEO_QC_PROMPT,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("No API key. Pass api_key= or set GEMINI_API_KEY.")
        self.model_name = model
        self.qc_prompt = qc_prompt
        self.max_retries = max_retries

        from google import genai
        
        # Temporarily remove proxy env vars — the local proxy (localhost:3128)
        # is intermittently unavailable and google.genai SDK picks it up
        self._saved_proxy_env: Dict[str, str] = {}
        for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                     "all_proxy", "ALL_PROXY"):
            if var in os.environ:
                self._saved_proxy_env[var] = os.environ.pop(var)
        
        self._client = genai.Client(api_key=self.api_key)
        
        # Restore proxy env vars (other tools may need them)
        os.environ.update(self._saved_proxy_env)

    def evaluate(self, video_url: str, **kwargs: Any) -> Dict[str, Any]:
        """Evaluate a video URL for quality.

        Downloads the video, uploads to Gemini, runs multimodal QC.
        """
        import requests
        import tempfile
        from google import genai

        last_error = None
        
        # Temporarily remove proxy env vars for all network calls in this method
        saved_proxy = {}
        for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                     "all_proxy", "ALL_PROXY"):
            if var in os.environ:
                saved_proxy[var] = os.environ.pop(var)

        for attempt in range(self.max_retries + 1):
            try:
                # Download video to temp file
                logger.info("Downloading video for QC: %s", video_url[:80])
                resp = requests.get(video_url, timeout=60, proxies={"http": None, "https": None})
                resp.raise_for_status()

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    f.write(resp.content)
                    tmp_path = f.name

                # Upload to Gemini File API
                logger.info("Uploading video to Gemini File API...")
                video_file = self._client.files.upload(
                    file=tmp_path,
                    config={"mime_type": "video/mp4"},
                )

                # Wait for processing
                while video_file.state.name == "PROCESSING":
                    time.sleep(2)
                    video_file = self._client.files.get(name=video_file.name)

                if video_file.state.name == "FAILED":
                    raise RuntimeError(f"Gemini file processing failed: {video_file.state}")

                # Generate QC evaluation
                logger.info("Running Gemini QC evaluation...")
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=[self.qc_prompt, video_file],
                    config={
                        "temperature": 0.2,
                        "max_output_tokens": 2048,
                        "response_mime_type": "application/json",
                    },
                )

                # Clean up uploaded file
                try:
                    self._client.files.delete(name=video_file.name)
                except Exception:
                    pass

                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

                # Restore proxy env vars before returning
                os.environ.update(saved_proxy)
                return self._parse_response(response.text, video_url)

            except Exception as e:
                last_error = e
                logger.warning(
                    "QC attempt %d/%d failed: %s",
                    attempt + 1, self.max_retries + 1, e,
                )
                # Clean up temp file on error
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                continue

        # Restore proxy env vars
        os.environ.update(saved_proxy)
        
        logger.error("QC failed after %d attempts: %s", self.max_retries + 1, last_error)
        return self._error_result(video_url, str(last_error))

    @staticmethod
    def _parse_response(text: str, video_url: str) -> Dict[str, Any]:
        """Parse Gemini JSON response."""
        cleaned = text.strip()
        # Strip markdown fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[-1].strip() == "```":
                cleaned = "\n".join(lines[1:-1])
            else:
                cleaned = "\n".join(lines[1:])
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: find outermost braces
            brace_depth = 0
            start_idx = None
            data = None
            for i, ch in enumerate(cleaned):
                if ch == '{':
                    if brace_depth == 0:
                        start_idx = i
                    brace_depth += 1
                elif ch == '}':
                    brace_depth -= 1
                    if brace_depth == 0 and start_idx is not None:
                        try:
                            data = json.loads(cleaned[start_idx:i + 1])
                            break
                        except json.JSONDecodeError:
                            start_idx = None
                            continue

            if data is None:
                data = GeminiVideoQC._extract_fields_regex(cleaned)
                if data is None:
                    return GeminiVideoQC._error_result(
                        video_url, f"JSON parse failed: {cleaned[:100]}"
                    )

        # Normalize to QCResult-compatible format
        return {
            "pass": bool(data.get("pass", True)),
            "confidence": float(data.get("confidence", 0.5)),
            "human_present": bool(data.get("human_present", False)),
            "auto_fail_triggered": list(data.get("auto_fail_triggered", [])),
            "minor_issues": list(data.get("minor_issues", [])),
            "summary": str(data.get("summary", "")),
            "aesthetic_score": int(data.get("aesthetic_score", 5)),
            "motion_score": int(data.get("motion_score", 5)),
            "prompt_adherence_score": int(data.get("prompt_adherence_score", 5)),
            "scroll_stop_score": int(data.get("scroll_stop_score", 5)),
            "scene_type_detected": str(data.get("scene_type_detected", "other")),
        }

    @staticmethod
    def _extract_fields_regex(text: str) -> Optional[Dict[str, Any]]:
        """Last-resort: extract individual fields via regex."""
        result: Dict[str, Any] = {}

        pass_m = re.search(r'"pass"\s*:\s*(true|false)', text, re.I)
        if pass_m:
            result["pass"] = pass_m.group(1).lower() == "true"

        for field in ["confidence", "aesthetic_score", "motion_score", "prompt_adherence_score", "scroll_stop_score"]:
            m = re.search(rf'"{field}"\s*:\s*([\d.]+)', text)
            if m:
                result[field] = float(m.group(1))

        bool_m = re.search(r'"human_present"\s*:\s*(true|false)', text, re.I)
        if bool_m:
            result["human_present"] = bool_m.group(1).lower() == "true"

        summary_m = re.search(r'"summary"\s*:\s*"([^"]*)', text)
        if summary_m:
            result["summary"] = summary_m.group(1)

        for arr_field in ["auto_fail_triggered", "minor_issues"]:
            arr_m = re.search(rf'"{arr_field}"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if arr_m:
                result[arr_field] = re.findall(r'"([^"]*)"', arr_m.group(1))

        scene_m = re.search(r'"scene_type_detected"\s*:\s*"([^"]*)"', text)
        if scene_m:
            result["scene_type_detected"] = scene_m.group(1)

        if "pass" in result:
            return result
        return None

    @staticmethod
    def _error_result(video_url: str, error: str) -> Dict[str, Any]:
        return {
            "pass": True,
            "confidence": 0.0,
            "human_present": False,
            "auto_fail_triggered": [],
            "minor_issues": [],
            "summary": f"QC error (safe pass): {error[:100]}",
            "aesthetic_score": 5,
            "motion_score": 5,
            "prompt_adherence_score": 5,
            "scroll_stop_score": 5,
            "scene_type_detected": "other",
        }


# ---------------------------------------------------------------------------
# Gemini LLM Client (for optimizer / prompt generation)
# ---------------------------------------------------------------------------

class GeminiLLM:
    """Text generation client using Gemini.

    Args:
        api_key: Google AI API key.
        model: Gemini model for text generation. Default: gemini-2.5-flash.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = "gemini-2.5-flash",
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("No API key. Pass api_key= or set GEMINI_API_KEY.")

        from google import genai
        
        # Temporarily remove proxy env vars
        saved_proxy = {}
        for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                     "all_proxy", "ALL_PROXY"):
            if var in os.environ:
                saved_proxy[var] = os.environ.pop(var)
        self._client = genai.Client(api_key=self.api_key)
        os.environ.update(saved_proxy)
        
        self._model_name = model

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt text.
            **kwargs: Optional: temperature (float), max_tokens (int),
                thinking_budget (int, 0=disabled, default 0 for speed).

        Returns:
            Generated text string.
        """
        from google.genai import types

        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 4096)
        thinking_budget = kwargs.get("thinking_budget", 0)  # Disable thinking by default

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        )

        # Temporarily remove proxy env vars for the API call
        saved_proxy = {}
        for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                     "all_proxy", "ALL_PROXY"):
            if var in os.environ:
                saved_proxy[var] = os.environ.pop(var)
        
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=config,
            )
            return response.text.strip()
        finally:
            os.environ.update(saved_proxy)
