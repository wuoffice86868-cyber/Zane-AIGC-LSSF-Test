"""Full autonomous pipeline: generate → QC → score → analyze → optimize.

This is the main overnight run script. It:
1. Generates videos for multiple scenes using different system prompts
2. Runs Gemini QC immediately on each video (before URL expires)
3. Computes reward scores from QC results
4. Analyzes prompt features vs reward correlation
5. Runs OPRO optimization to suggest improved system prompts

Budget-aware: tracks all API calls and stops if budget exceeded.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from prompt_evaluator.kie_client import KieClient
from prompt_evaluator.gemini_client import GeminiVideoQC, GeminiLLM
from prompt_evaluator.reward_calculator import RewardCalculator
from prompt_evaluator.prompt_analyzer import PromptAnalyzer
from prompt_evaluator.optimizer import PromptOptimizer
from prompt_evaluator.models import (
    EvalSample, InputInfo, OutputInfo, PromptInfo, QCResult, RewardBreakdown,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Load credentials
with open("/home/node/.openclaw/workspace/.gemini_credentials.json") as f:
    GEMINI_KEY = json.load(f)["api_key"]

# Budget limits
MAX_KIE_REQUESTS = 40  # ~20 scenes (1 image + 1 video each)
MAX_GEMINI_EVALS = 25

OUTPUT_DIR = Path("eval_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Scene definitions — diverse set for good coverage
# ---------------------------------------------------------------------------

SCENES = [
    # Pool variations
    {"type": "pool", "desc": "Luxury infinity pool overlooking ocean at sunset",
     "img_prompt": "A luxury infinity pool overlooking tropical ocean at golden sunset, crystal clear turquoise water reflecting golden light, palm trees, lounge chairs, professional hotel photography, 9:16 portrait"},
    {"type": "pool", "desc": "Indoor heated pool with glass ceiling",
     "img_prompt": "An indoor heated hotel pool with glass atrium ceiling, steam rising from warm water, modern architecture, ambient blue lighting, professional hotel photography, 9:16 portrait"},

    # Room variations
    {"type": "room", "desc": "Suite with panoramic city view at night",
     "img_prompt": "A luxury hotel suite with king bed, panoramic floor-to-ceiling windows showing city lights at night, warm ambient lighting, elegant decor, professional hotel photography, 9:16 portrait"},
    {"type": "room", "desc": "Tropical villa bedroom with garden view",
     "img_prompt": "A tropical resort villa bedroom, open sliding doors to lush garden, white canopy bed, natural wood furniture, morning sunlight streaming in, professional hotel photography, 9:16 portrait"},

    # Lobby variations
    {"type": "lobby", "desc": "Modern minimalist hotel lobby",
     "img_prompt": "A modern minimalist hotel lobby, clean concrete and wood design, large art installation, floor to ceiling windows, natural light, fresh flowers at reception, professional hotel photography, 9:16 portrait"},

    # Spa
    {"type": "spa", "desc": "Outdoor spa with mountain view",
     "img_prompt": "An outdoor hotel spa area with hot tub, mountain view backdrop, wooden deck, candles, fluffy white robes on hooks, misty morning atmosphere, professional hotel photography, 9:16 portrait"},

    # Restaurant
    {"type": "restaurant", "desc": "Rooftop restaurant at twilight",
     "img_prompt": "A luxury hotel rooftop restaurant at twilight, elegant table settings with candles, city skyline view, warm string lights overhead, professional hotel photography, 9:16 portrait"},

    # Exterior
    {"type": "exterior", "desc": "Hotel facade with fountain at entrance",
     "img_prompt": "A grand luxury hotel exterior entrance, classical facade with columns, ornamental fountain in driveway, manicured hedges, warm evening lighting, professional hotel photography, 9:16 portrait"},

    # Beach
    {"type": "beach", "desc": "Private beach cabana setup",
     "img_prompt": "A private hotel beach with white sand, luxury cabanas with flowing white curtains, turquoise ocean, palm trees, golden hour light, professional resort photography, 9:16 portrait"},

    # Bathroom
    {"type": "bathroom", "desc": "Marble bathroom with freestanding tub",
     "img_prompt": "A luxury hotel bathroom, white marble surfaces, freestanding copper bathtub by window, monstera plant, warm ambient lighting, rolled white towels, professional hotel photography, 9:16 portrait"},
]

# Two system prompt versions to compare
SYSTEM_PROMPTS = {
    "v1": open("system_prompts/hotel_v1.txt").read(),
    "v2_diverse": open("system_prompts/hotel_v2_diverse.txt").read(),
}

# ---------------------------------------------------------------------------
# Cinematography prompt generation via Gemini
# ---------------------------------------------------------------------------

def generate_cinematography_prompt(llm: GeminiLLM, system_prompt: str, scene: dict) -> str:
    """Use Gemini to generate a cinematography prompt given the system prompt and scene."""
    user_msg = f"Generate a cinematography prompt for this scene: {scene['desc']}\nScene type: {scene['type']}"
    full_prompt = f"{system_prompt}\n\n{user_msg}"
    result = llm.generate(full_prompt, temperature=0.7, max_tokens=200)
    # Clean up — sometimes Gemini wraps in quotes or adds explanation
    result = result.strip().strip('"').strip("'")
    # Remove any leading labels like "Cinematography Prompt:" etc
    for prefix in ["Cinematography Prompt:", "Prompt:", "Output:"]:
        if result.startswith(prefix):
            result = result[len(prefix):].strip()
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline():
    logger.info("=" * 70)
    logger.info("FULL PIPELINE RUN — %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 70)

    # Init clients
    kie = KieClient(max_requests=MAX_KIE_REQUESTS)
    qc = GeminiVideoQC(api_key=GEMINI_KEY, model="gemini-2.5-flash")
    llm = GeminiLLM(api_key=GEMINI_KEY, model="gemini-2.5-flash")
    reward_calc = RewardCalculator()
    analyzer = PromptAnalyzer()

    all_samples = []
    all_results = []
    gemini_eval_count = 0

    for sp_name, system_prompt in SYSTEM_PROMPTS.items():
        logger.info("\n" + "=" * 50)
        logger.info("SYSTEM PROMPT: %s", sp_name)
        logger.info("=" * 50)

        for scene in SCENES:
            if kie.stats.total_requests >= MAX_KIE_REQUESTS - 2:
                logger.warning("Approaching KIE budget limit, stopping generation")
                break
            if gemini_eval_count >= MAX_GEMINI_EVALS:
                logger.warning("Approaching Gemini eval limit, stopping")
                break

            sample_id = f"{sp_name}_{scene['type']}_{int(time.time())}"
            logger.info("\n--- %s | %s ---", sample_id, scene['desc'][:50])

            result_entry = {
                "sample_id": sample_id,
                "system_prompt_version": sp_name,
                "scene_type": scene["type"],
                "scene_description": scene["desc"],
            }

            try:
                # Step 1: Generate cinematography prompt
                logger.info("Generating cinematography prompt...")
                cine_prompt = generate_cinematography_prompt(llm, system_prompt, scene)
                result_entry["generated_prompt"] = cine_prompt
                logger.info("Prompt: %s", cine_prompt)

                # Step 2: Generate image
                logger.info("Generating image...")
                img_result = kie.generate_image(
                    scene["img_prompt"],
                    aspect_ratio="9:16",
                )
                if not img_result.success or not img_result.result_urls:
                    result_entry["error"] = f"Image gen failed: {img_result.state}"
                    logger.error(result_entry["error"])
                    all_results.append(result_entry)
                    continue

                img_url = img_result.result_urls[0]
                result_entry["image_url"] = img_url
                result_entry["image_task_id"] = img_result.task_id
                logger.info("Image: %s", img_url[:80])

                # Step 3: Generate video
                logger.info("Generating video...")
                vid_result = kie.generate_video(
                    prompt=cine_prompt,
                    image_url=img_url,
                    aspect_ratio="9:16",
                    duration=8,
                )
                if not vid_result.success or not vid_result.result_urls:
                    result_entry["error"] = f"Video gen failed: {vid_result.state}"
                    logger.error(result_entry["error"])
                    all_results.append(result_entry)
                    continue

                vid_url = vid_result.result_urls[0]
                result_entry["video_url"] = vid_url
                result_entry["video_task_id"] = vid_result.task_id
                logger.info("Video: %s", vid_url[:80])

                # Step 4: QC immediately (before URL expires!)
                logger.info("Running Gemini QC...")
                qc_dict = qc.evaluate(vid_url)
                gemini_eval_count += 1
                result_entry["qc_result"] = qc_dict
                logger.info("QC: pass=%s, aesthetic=%s, motion=%s, scroll_stop=%s, confidence=%.2f",
                           qc_dict.get("pass"), qc_dict.get("aesthetic_score"),
                           qc_dict.get("motion_score"), qc_dict.get("scroll_stop_score"),
                           qc_dict.get("confidence", 0))
                if qc_dict.get("auto_fail_triggered"):
                    logger.info("  Auto-fail: %s", qc_dict["auto_fail_triggered"])
                if qc_dict.get("minor_issues"):
                    logger.info("  Minor: %s", qc_dict["minor_issues"])
                logger.info("  Summary: %s", qc_dict.get("summary", ""))

                # Step 5: Compute reward using multi-dimensional scoring
                # Normalize Gemini's 1-10 scores to 0-1 for the reward calc
                aesthetic_norm = qc_dict.get("aesthetic_score", 5) / 10.0
                motion_norm = qc_dict.get("motion_score", 5) / 10.0
                adherence_norm = qc_dict.get("prompt_adherence_score", 5) / 10.0

                reward = reward_calc.calculate(
                    qc_dict,
                    aesthetic=aesthetic_norm,
                    clip_score=adherence_norm,  # using adherence as clip proxy
                    motion_score=motion_norm,
                )
                result_entry["reward"] = {
                    "total_score": reward.total_score,
                    "breakdown": reward.breakdown,
                }
                logger.info("Reward: %.1f/100", reward.total_score)

                # Step 6: Extract prompt features
                features = analyzer.extract_features(cine_prompt)
                result_entry["features"] = features.model_dump()

                # Build EvalSample for later analysis
                qc_model = QCResult(**{
                    "pass": qc_dict.get("pass", True),
                    "confidence": qc_dict.get("confidence", 0.5),
                    "auto_fail_triggered": qc_dict.get("auto_fail_triggered", []),
                    "minor_issues": qc_dict.get("minor_issues", []),
                    "summary": qc_dict.get("summary", ""),
                })
                sample = EvalSample(
                    sample_id=sample_id,
                    input=InputInfo(image_url=img_url, scene_type=scene["type"]),
                    prompt=PromptInfo(
                        system_prompt_version=sp_name,
                        generated_prompt=cine_prompt,
                    ),
                    output=OutputInfo(video_url=vid_url, video_duration_sec=8.0),
                    qc_result=qc_model,
                )
                all_samples.append(sample)

            except Exception as e:
                result_entry["error"] = str(e)
                logger.error("Pipeline error: %s", e, exc_info=True)

            all_results.append(result_entry)

            # Brief pause between scenes to avoid rate limits
            time.sleep(2)

    # ---------------------------------------------------------------------------
    # Post-processing: Analysis + Optimization
    # ---------------------------------------------------------------------------

    logger.info("\n" + "=" * 70)
    logger.info("POST-PROCESSING")
    logger.info("=" * 70)

    # Save raw results
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    raw_path = OUTPUT_DIR / f"full_pipeline_{ts}.json"
    with open(raw_path, "w") as f:
        json.dump({
            "run_info": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system_prompts": list(SYSTEM_PROMPTS.keys()),
                "scenes_count": len(SCENES),
                "total_samples": len(all_results),
                "kie_stats": kie.stats.summary(),
                "gemini_evals": gemini_eval_count,
            },
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    logger.info("Raw results saved to %s", raw_path)

    # Analyze
    if len(all_samples) >= 3:
        logger.info("\n--- Prompt Feature Correlation ---")
        corr = analyzer.analyze_correlation(all_samples, use_human_score=False)
        if corr.feature_importance:
            for feat, r_val in list(corr.feature_importance.items())[:10]:
                direction = "↑" if r_val > 0 else "↓"
                logger.info("  %s: r=%+.3f %s", feat, r_val, direction)
        if corr.high_reward_patterns:
            logger.info("High-score patterns: %s", corr.high_reward_patterns)
        if corr.low_reward_patterns:
            logger.info("Low-score patterns: %s", corr.low_reward_patterns)

        # Per-system-prompt breakdown
        logger.info("\n--- Per-System-Prompt Performance ---")
        from collections import defaultdict
        sp_scores = defaultdict(list)
        for r in all_results:
            if "reward" in r and "error" not in r:
                sp_scores[r["system_prompt_version"]].append(r["reward"]["total_score"])

        for sp_name, scores in sp_scores.items():
            avg = sum(scores) / len(scores)
            pass_rate = sum(1 for r in all_results
                          if r.get("system_prompt_version") == sp_name
                          and r.get("qc_result", {}).get("pass", True)) / max(len(scores), 1)
            logger.info("  %s: avg_reward=%.1f, pass_rate=%.0f%%, n=%d",
                        sp_name, avg, pass_rate * 100, len(scores))

        # Per-scene-type breakdown
        logger.info("\n--- Per-Scene-Type Performance ---")
        scene_scores = defaultdict(list)
        for r in all_results:
            if "reward" in r and "error" not in r:
                scene_scores[r["scene_type"]].append(r["reward"]["total_score"])
        for st, scores in sorted(scene_scores.items()):
            avg = sum(scores) / len(scores)
            logger.info("  %s: avg_reward=%.1f, n=%d", st, avg, len(scores))

    # Run OPRO optimization
    if len(all_samples) >= 5:
        logger.info("\n--- OPRO Optimization ---")
        optimizer = PromptOptimizer(
            llm_client=llm,
            reward_calculator=reward_calc,
            prompt_analyzer=analyzer,
        )

        for sp_name, system_prompt in SYSTEM_PROMPTS.items():
            logger.info("\nOptimizing: %s", sp_name)
            try:
                improved = optimizer.suggest_improvement(
                    system_prompt,
                    all_samples,
                    top_k=5,
                    bottom_k=3,
                    temperature=1.0,
                )
                # Save improved prompt
                improved_path = Path("system_prompts") / f"{sp_name}_improved_{ts}.txt"
                with open(improved_path, "w") as f:
                    f.write(improved)
                logger.info("Improved prompt saved to %s", improved_path)
                logger.info("First 200 chars: %s", improved[:200])
            except Exception as e:
                logger.error("Optimization failed for %s: %s", sp_name, e)

    # Generate summary report
    report_path = OUTPUT_DIR / f"report_{ts}.md"
    report_lines = [
        f"# Pipeline Run Report — {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Total samples: {len(all_results)}",
        f"Successful: {sum(1 for r in all_results if 'error' not in r)}",
        f"Failed: {sum(1 for r in all_results if 'error' in r)}",
        f"KIE API: {kie.stats.summary()}",
        f"Gemini QC evals: {gemini_eval_count}",
        "",
        "## Per-Sample Results",
        "",
    ]
    for r in all_results:
        status = "✗" if "error" in r else ("✓" if r.get("qc_result", {}).get("pass", True) else "⚠")
        reward_str = f"{r['reward']['total_score']:.1f}" if "reward" in r else "N/A"
        report_lines.append(f"- {status} **{r['sample_id']}** [{r['scene_type']}] reward={reward_str}")
        if r.get("generated_prompt"):
            report_lines.append(f"  - Prompt: {r['generated_prompt'][:100]}...")
        if r.get("qc_result", {}).get("auto_fail_triggered"):
            report_lines.append(f"  - Auto-fail: {r['qc_result']['auto_fail_triggered']}")
        if r.get("qc_result", {}).get("summary"):
            report_lines.append(f"  - QC: {r['qc_result']['summary']}")
        if r.get("error"):
            report_lines.append(f"  - Error: {r['error']}")
        report_lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    logger.info("\nReport saved to %s", report_path)

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("Total API calls: KIE=%d, Gemini QC=%d", kie.stats.total_requests, gemini_eval_count)
    logger.info("=" * 70)


if __name__ == "__main__":
    run_pipeline()
