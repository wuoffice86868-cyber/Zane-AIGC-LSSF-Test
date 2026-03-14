"""Test Gemini QC on the 4 videos we already generated.

This validates:
1. Gemini API key works
2. Video upload + evaluation flow works
3. QC prompt produces useful structured scores
"""

import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Load Gemini key
cred_path = "/home/node/.openclaw/workspace/.gemini_credentials.json"
with open(cred_path) as f:
    creds = json.load(f)
api_key = creds["api_key"]

# Our 4 previously generated videos
videos = [
    {
        "sample_id": "eval_0001",
        "scene_type": "pool",
        "prompt": "Single continuous shot. Wide shot of the pool area. The camera pushes slowly forward. Water ripples softly in the warm light. Sky and clouds remain completely still. The pool edge stays anchored in frame.",
        "video_url": "https://tempfile.aiquickdraw.com/r/69175be0e417887827bf9d94c14e0a18_1773219364_q06htd16.mp4",
    },
    {
        "sample_id": "eval_0002",
        "scene_type": "room",
        "prompt": "Single continuous shot. Medium shot of the hotel room. The camera pushes slowly forward toward the window. Curtains sway gently. The bed frame stays fixed in place.",
        "video_url": "https://tempfile.aiquickdraw.com/r/28a604c61a33411e316526eeb38d5365_1773219522_n4i6zrkl.mp4",
    },
    {
        "sample_id": "eval_0003",
        "scene_type": "lobby",
        "prompt": "Single continuous shot. Wide shot of the hotel lobby. The camera pushes slowly forward across the marble floor. The chandelier stays anchored above. Reflections on the floor remain steady.",
        "video_url": "https://tempfile.aiquickdraw.com/r/568e737ef9d6213d1ceb874f648f1601_1773219707_cmtob1ze.mp4",
    },
    {
        "sample_id": "eval_0004",
        "scene_type": "spa",
        "prompt": "Single continuous shot. Medium shot of the spa room. The camera pushes slowly forward. Candle flames flicker gently. Stone walls stay perfectly still.",
        "video_url": "https://tempfile.aiquickdraw.com/r/e5e63852ad6b970b932c00593efdc391_1773219905_1yo5qqd5.mp4",
    },
]

from prompt_evaluator.gemini_client import GeminiVideoQC

# Initialize with comprehensive QC prompt
qc = GeminiVideoQC(api_key=api_key, model="gemini-2.5-flash")

results = []
for v in videos:
    logger.info("=" * 60)
    logger.info("Evaluating %s (%s)", v["sample_id"], v["scene_type"])
    logger.info("Prompt: %s", v["prompt"][:80])
    logger.info("Video: %s", v["video_url"])

    try:
        result = qc.evaluate(v["video_url"])
        result["sample_id"] = v["sample_id"]
        result["scene_type"] = v["scene_type"]
        result["prompt_used"] = v["prompt"]
        results.append(result)

        logger.info("Result: pass=%s, confidence=%.2f", result["pass"], result["confidence"])
        logger.info("  aesthetic=%s, motion=%s, adherence=%s",
                     result.get("aesthetic_score"), result.get("motion_score"),
                     result.get("prompt_adherence_score"))
        logger.info("  auto_fail: %s", result.get("auto_fail_triggered", []))
        logger.info("  minor: %s", result.get("minor_issues", []))
        logger.info("  summary: %s", result.get("summary", ""))
    except Exception as e:
        logger.error("FAILED: %s", e)
        results.append({"sample_id": v["sample_id"], "error": str(e)})

# Save results
out_path = "eval_results/gemini_qc_live_test.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

logger.info("\n" + "=" * 60)
logger.info("All results saved to %s", out_path)

# Summary
passed = sum(1 for r in results if r.get("pass", False))
logger.info("Pass rate: %d/%d", passed, len(results))
for r in results:
    if "error" not in r:
        logger.info("  %s (%s): pass=%s, aesthetic=%s, motion=%s",
                     r["sample_id"], r["scene_type"], r["pass"],
                     r.get("aesthetic_score"), r.get("motion_score"))
