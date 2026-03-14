"""Round 2: Test OPRO-improved system prompts against originals.

Focused run: 6 high-failure scenes × 2 improved prompts = 12 samples.
Compares improved prompts against Round 1 baselines.
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
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("round2_run.log"),
    ],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from prompt_evaluator.kie_client import KieClient
from prompt_evaluator.gemini_client import GeminiVideoQC, GeminiLLM
from prompt_evaluator.reward_calculator import RewardCalculator
from prompt_evaluator.prompt_analyzer import PromptAnalyzer
from prompt_evaluator.optimizer import PromptOptimizer
from prompt_evaluator.models import EvalSample, InputInfo, OutputInfo, PromptInfo, QCResult

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

with open("/home/node/.openclaw/workspace/.gemini_credentials.json") as f:
    GEMINI_KEY = json.load(f)["api_key"]

MAX_KIE_REQUESTS = 30
MAX_GEMINI_EVALS = 15

OUTPUT_DIR = Path("eval_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Focus on scenes that had highest failure rates in Round 1
FOCUS_SCENES = [
    {"type": "room", "desc": "Suite with panoramic city view at night",
     "img_prompt": "A luxury hotel suite with king bed, panoramic floor-to-ceiling windows showing city lights at night, warm ambient lighting, elegant decor, professional hotel photography, 9:16 portrait"},
    {"type": "lobby", "desc": "Modern minimalist hotel lobby",
     "img_prompt": "A modern minimalist hotel lobby, clean concrete and wood design, large art installation, floor to ceiling windows, natural light, fresh flowers at reception, professional hotel photography, 9:16 portrait"},
    {"type": "bathroom", "desc": "Marble bathroom with freestanding tub",
     "img_prompt": "A luxury hotel bathroom, white marble surfaces, freestanding copper bathtub by window, monstera plant, warm ambient lighting, rolled white towels, professional hotel photography, 9:16 portrait"},
    {"type": "restaurant", "desc": "Rooftop restaurant at twilight",
     "img_prompt": "A luxury hotel rooftop restaurant at twilight, elegant table settings with candles, city skyline view, warm string lights overhead, professional hotel photography, 9:16 portrait"},
    {"type": "exterior", "desc": "Hotel facade with fountain at entrance",
     "img_prompt": "A grand luxury hotel exterior entrance, classical facade with columns, ornamental fountain in driveway, manicured hedges, warm evening lighting, professional hotel photography, 9:16 portrait"},
    {"type": "beach", "desc": "Private beach cabana setup",
     "img_prompt": "A private hotel beach with white sand, luxury cabanas with flowing white curtains, turquoise ocean, palm trees, golden hour light, professional resort photography, 9:16 portrait"},
]

# Load improved prompts from OPRO Round 1
SYSTEM_PROMPTS = {}
prompt_dir = Path("system_prompts")
for f in sorted(prompt_dir.glob("*_improved_*.txt")):
    key = f.stem  # e.g. "v1_improved_20260311_105049"
    SYSTEM_PROMPTS[key] = f.read_text()

if not SYSTEM_PROMPTS:
    logger.error("No improved system prompts found!")
    sys.exit(1)

logger.info("Loaded %d improved system prompts: %s", len(SYSTEM_PROMPTS), list(SYSTEM_PROMPTS.keys()))


def generate_cinematography_prompt(llm, system_prompt, scene):
    user_msg = f"Generate a cinematography prompt for this scene: {scene['desc']}\nScene type: {scene['type']}"
    full_prompt = f"{system_prompt}\n\n{user_msg}"
    result = llm.generate(full_prompt, temperature=0.7, max_tokens=200)
    result = result.strip().strip('"').strip("'")
    for prefix in ["Cinematography Prompt:", "Prompt:", "Output:"]:
        if result.startswith(prefix):
            result = result[len(prefix):].strip()
    return result


def run_round2():
    logger.info("=" * 70)
    logger.info("ROUND 2: IMPROVED PROMPTS — %s", datetime.now(timezone.utc).isoformat())
    logger.info("=" * 70)

    kie = KieClient(max_requests=MAX_KIE_REQUESTS)
    qc = GeminiVideoQC(api_key=GEMINI_KEY, model="gemini-2.5-flash")
    llm = GeminiLLM(api_key=GEMINI_KEY, model="gemini-2.5-flash")
    reward_calc = RewardCalculator()
    analyzer = PromptAnalyzer()

    all_results = []
    gemini_eval_count = 0

    for sp_name, system_prompt in SYSTEM_PROMPTS.items():
        logger.info("\n" + "=" * 50)
        logger.info("SYSTEM PROMPT: %s", sp_name)
        logger.info("=" * 50)

        for scene in FOCUS_SCENES:
            if kie.stats.total_requests >= MAX_KIE_REQUESTS - 2:
                logger.warning("Approaching KIE budget limit, stopping")
                break
            if gemini_eval_count >= MAX_GEMINI_EVALS:
                logger.warning("Approaching Gemini eval limit, stopping")
                break

            sample_id = f"r2_{sp_name[:10]}_{scene['type']}_{int(time.time())}"
            logger.info("\n--- %s | %s ---", sample_id, scene['desc'][:50])

            entry = {
                "sample_id": sample_id,
                "round": 2,
                "system_prompt_version": sp_name,
                "scene_type": scene["type"],
                "scene_description": scene["desc"],
            }

            try:
                # Generate cinematography prompt
                cine_prompt = generate_cinematography_prompt(llm, system_prompt, scene)
                entry["generated_prompt"] = cine_prompt
                logger.info("Prompt: %s", cine_prompt)

                # Generate image
                img_result = kie.generate_image(scene["img_prompt"], aspect_ratio="9:16")
                if not img_result.success or not img_result.result_urls:
                    entry["error"] = f"Image gen failed: {img_result.state}"
                    logger.error(entry["error"])
                    all_results.append(entry)
                    continue
                entry["image_url"] = img_result.result_urls[0]
                entry["image_task_id"] = img_result.task_id

                # Generate video
                vid_result = kie.generate_video(
                    prompt=cine_prompt,
                    image_url=img_result.result_urls[0],
                    aspect_ratio="9:16",
                    duration=8,
                )
                if not vid_result.success or not vid_result.result_urls:
                    entry["error"] = f"Video gen failed: {vid_result.state}"
                    logger.error(entry["error"])
                    all_results.append(entry)
                    continue
                entry["video_url"] = vid_result.result_urls[0]
                entry["video_task_id"] = vid_result.task_id

                # QC immediately
                qc_dict = qc.evaluate(vid_result.result_urls[0])
                gemini_eval_count += 1
                entry["qc_result"] = qc_dict
                logger.info("QC: pass=%s, aesthetic=%s, motion=%s, scroll_stop=%s",
                           qc_dict.get("pass"), qc_dict.get("aesthetic_score"),
                           qc_dict.get("motion_score"), qc_dict.get("scroll_stop_score"))
                if qc_dict.get("auto_fail_triggered"):
                    logger.info("  Auto-fail: %s", qc_dict["auto_fail_triggered"])

                # Reward
                aesthetic_norm = qc_dict.get("aesthetic_score", 5) / 10.0
                motion_norm = qc_dict.get("motion_score", 5) / 10.0
                adherence_norm = qc_dict.get("prompt_adherence_score", 5) / 10.0
                reward = reward_calc.calculate(
                    qc_dict,
                    aesthetic=aesthetic_norm,
                    clip_score=adherence_norm,
                    motion_score=motion_norm,
                )
                entry["reward"] = {
                    "total_score": reward.total_score,
                    "breakdown": reward.breakdown,
                }
                logger.info("Reward: %.1f/100", reward.total_score)

                # Features
                features = analyzer.extract_features(cine_prompt)
                entry["features"] = features.model_dump()

            except Exception as e:
                entry["error"] = str(e)
                logger.error("Error: %s", e, exc_info=True)

            all_results.append(entry)
            time.sleep(2)

    # ---------------------------------------------------------------------------
    # Save + Compare
    # ---------------------------------------------------------------------------
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Save raw
    raw_path = OUTPUT_DIR / f"round2_{ts}.json"
    with open(raw_path, "w") as f:
        json.dump({
            "run_info": {
                "round": 2,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system_prompts": list(SYSTEM_PROMPTS.keys()),
                "focus_scenes": [s["type"] for s in FOCUS_SCENES],
                "total_samples": len(all_results),
                "kie_requests": kie.stats.total_requests,
                "gemini_evals": gemini_eval_count,
            },
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    logger.info("\nRaw results saved to %s", raw_path)

    # Comparison report
    logger.info("\n" + "=" * 70)
    logger.info("ROUND 2 SUMMARY")
    logger.info("=" * 70)

    from collections import defaultdict
    sp_scores = defaultdict(list)
    sp_pass = defaultdict(lambda: [0, 0])  # [pass_count, total]
    scene_scores = defaultdict(list)

    for r in all_results:
        if "reward" in r and "error" not in r:
            sp_scores[r["system_prompt_version"]].append(r["reward"]["total_score"])
            sp_pass[r["system_prompt_version"]][1] += 1
            if r.get("qc_result", {}).get("pass", True):
                sp_pass[r["system_prompt_version"]][0] += 1
            scene_scores[r["scene_type"]].append(r["reward"]["total_score"])

    logger.info("\nPer-Prompt Performance:")
    for sp, scores in sp_scores.items():
        avg = sum(scores) / len(scores)
        p, t = sp_pass[sp]
        logger.info("  %s: avg=%.1f, pass=%d/%d (%.0f%%), n=%d", sp, avg, p, t, p/max(t,1)*100, len(scores))

    logger.info("\nPer-Scene Performance:")
    for st, scores in sorted(scene_scores.items()):
        avg = sum(scores) / len(scores)
        logger.info("  %s: avg=%.1f, n=%d", st, avg, len(scores))

    # Load Round 1 for comparison
    r1_path = OUTPUT_DIR / "full_pipeline_20260311_105049.json"
    if r1_path.exists():
        r1_data = json.load(open(r1_path))
        r1_scores = defaultdict(list)
        for r in r1_data["results"]:
            if "reward" in r and "error" not in r:
                r1_scores[r["scene_type"]].append(r["reward"]["total_score"])

        logger.info("\nRound 1 vs Round 2 (per scene):")
        for st in sorted(set(list(scene_scores.keys()) + list(r1_scores.keys()))):
            r1_avg = sum(r1_scores[st]) / len(r1_scores[st]) if r1_scores[st] else 0
            r2_avg = sum(scene_scores[st]) / len(scene_scores[st]) if scene_scores[st] else 0
            delta = r2_avg - r1_avg
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            logger.info("  %s: R1=%.1f → R2=%.1f (%s%.1f)", st, r1_avg, r2_avg, arrow, delta)

    # Generate OPRO Round 2 improvements
    if len(all_results) >= 5:
        logger.info("\n--- OPRO Round 2 Optimization ---")
        optimizer = PromptOptimizer(
            llm_client=llm,
            reward_calculator=reward_calc,
            prompt_analyzer=analyzer,
        )

        # Build EvalSamples from round 2
        samples = []
        for r in all_results:
            if "error" not in r and "qc_result" in r:
                qc_r = r["qc_result"]
                samples.append(EvalSample(
                    sample_id=r["sample_id"],
                    input=InputInfo(image_url=r.get("image_url", ""), scene_type=r["scene_type"]),
                    prompt=PromptInfo(system_prompt_version=r["system_prompt_version"], generated_prompt=r.get("generated_prompt", "")),
                    output=OutputInfo(video_url=r.get("video_url", ""), video_duration_sec=8.0),
                    qc_result=QCResult(**{
                        "pass": qc_r.get("pass", True),
                        "confidence": qc_r.get("confidence", 0.5),
                        "auto_fail_triggered": qc_r.get("auto_fail_triggered", []),
                        "minor_issues": qc_r.get("minor_issues", []),
                        "summary": qc_r.get("summary", ""),
                    }),
                ))

        # Use the best-performing improved prompt as base
        best_sp = max(sp_scores.items(), key=lambda x: sum(x[1])/len(x[1]))
        logger.info("Best prompt this round: %s (avg=%.1f)", best_sp[0], sum(best_sp[1])/len(best_sp[1]))

        try:
            improved = optimizer.suggest_improvement(
                SYSTEM_PROMPTS[best_sp[0]],
                samples,
                top_k=5,
                bottom_k=3,
                temperature=0.8,
            )
            improved_path = prompt_dir / f"r2_improved_{ts}.txt"
            with open(improved_path, "w") as f:
                f.write(improved)
            logger.info("Round 2 improved prompt saved to %s", improved_path)
        except Exception as e:
            logger.error("OPRO failed: %s", e)

    report_path = OUTPUT_DIR / f"round2_report_{ts}.md"
    report_lines = [
        f"# Round 2 Report — {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Focus: High-failure scenes from Round 1 with OPRO-improved prompts",
        f"Total samples: {len(all_results)}",
        f"Successful: {sum(1 for r in all_results if 'error' not in r)}",
        f"KIE requests: {kie.stats.total_requests}",
        f"Gemini evals: {gemini_eval_count}",
        "",
    ]
    for r in all_results:
        status = "✗" if "error" in r else ("✓" if r.get("qc_result", {}).get("pass", True) else "⚠")
        reward_str = f"{r['reward']['total_score']:.1f}" if "reward" in r else "N/A"
        report_lines.append(f"- {status} **{r['sample_id']}** [{r['scene_type']}] reward={reward_str}")
        if r.get("qc_result", {}).get("auto_fail_triggered"):
            report_lines.append(f"  - Auto-fail: {r['qc_result']['auto_fail_triggered']}")
        if r.get("qc_result", {}).get("summary"):
            report_lines.append(f"  - {r['qc_result']['summary'][:150]}")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    logger.info("Report saved to %s", report_path)

    logger.info("\n" + "=" * 70)
    logger.info("ROUND 2 COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_round2()
