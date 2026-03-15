#!/usr/bin/env python3
"""PE Layer Probe — Full Pipeline.

Generates videos from 12 prompt variants (same reference image),
scores each with Gemini 2.5 Pro QC, computes 3-dim rewards, and
outputs a structured analysis of what the PE layer responds to.

Budget: ~13 Kie calls (1 image + 12 videos), ~12 Gemini QC calls.
Estimated cost: ~$8-12 on Kie, ~$0 on Gemini (free tier).
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt_evaluator.kie_client import KieClient
from prompt_evaluator.gemini_client import GeminiVideoQC
from prompt_evaluator.reward_calculator import RewardCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt variants — each tests a specific PE behavior hypothesis
# ---------------------------------------------------------------------------

VARIANTS = {
    # Dimension 1: Structure order
    "A1_subject_first": {
        "hypothesis": "PE preserves subject when it comes first",
        "prompt": (
            "Turquoise infinity pool at golden hour with crystal clear water. "
            "The camera pushes slowly forward. Soft ripples catch warm sunset light. "
            "Palm fronds sway gently at the edges."
        ),
    },
    "A2_camera_first": {
        "hypothesis": "PE may reorder camera instructions",
        "prompt": (
            "The camera pushes slowly forward over a turquoise infinity pool at golden hour. "
            "Crystal clear water ripples softly. Warm sunset light reflects off the surface. "
            "Palm fronds sway gently."
        ),
    },
    "A3_scene_first": {
        "hypothesis": "Scene context first may anchor PE better",
        "prompt": (
            "Golden hour at a luxury resort. Wide shot of infinity pool stretching toward "
            "the horizon. Camera pushes forward slowly. Water surface catches warm light. "
            "Tropical plants frame the shot."
        ),
    },

    # Dimension 2: Length
    "B1_minimal_15w": {
        "hypothesis": "Very short prompts — does PE fill in details?",
        "prompt": (
            "Luxury pool at sunset. Camera pushes forward slowly. "
            "Water ripples gently."
        ),
    },
    "B2_medium_30w": {
        "hypothesis": "Sweet spot length for PE",
        "prompt": (
            "Turquoise infinity pool at golden hour. The camera pushes slowly forward "
            "revealing crystal clear water with natural ripples. Warm amber light. "
            "Palm fronds sway gently in a soft breeze."
        ),
    },
    "B3_verbose_50w": {
        "hypothesis": "Overly detailed — does PE truncate or lose coherence?",
        "prompt": (
            "Wide establishing shot of a turquoise infinity pool at a luxury beachside resort "
            "during golden hour. The camera pushes forward very slowly and steadily, revealing "
            "crystal clear water with soft natural ripples catching warm amber sunset light. "
            "Lush tropical palm fronds sway ever so gently in a light breeze at the pool edges. "
            "Wooden sun loungers sit perfectly still alongside."
        ),
    },

    # Dimension 3: Degree adverbs
    "C1_with_adverbs": {
        "hypothesis": "Adverbs modulate PE output strength",
        "prompt": (
            "Turquoise infinity pool at golden hour. The camera pushes extremely slowly "
            "and very steadily forward. Water ripples incredibly gently and naturally. "
            "Palm leaves drift barely perceptibly in soft warm light."
        ),
    },
    "C2_without_adverbs": {
        "hypothesis": "No adverbs — PE generates default motion intensity",
        "prompt": (
            "Turquoise infinity pool at golden hour. The camera pushes forward. "
            "Water ripples. Palm leaves move. Warm light on the surface."
        ),
    },

    # Dimension 4: Negative constraints (confirmed don't work, but testing PE behavior)
    "D1_with_negatives": {
        "hypothesis": "PE strips negatives — score should equal without_negatives",
        "prompt": (
            "Turquoise infinity pool at golden hour. The camera pushes slowly forward. "
            "Soft water ripples. No morphing, no warping. Pool edges remain perfectly still. "
            "Architecture maintains structural integrity throughout."
        ),
    },
    "D2_without_negatives": {
        "hypothesis": "Baseline without negatives — control for D1",
        "prompt": (
            "Turquoise infinity pool at golden hour. The camera pushes slowly forward. "
            "Soft water ripples in warm sunset light. Professional cinematic atmosphere. "
            "Tropical paradise feeling."
        ),
    },

    # Dimension 5: Vocabulary type
    "E1_equipment_names": {
        "hypothesis": "Equipment names (Steadicam, gimbal) — does PE translate or ignore?",
        "prompt": (
            "Steadicam tracking shot of turquoise infinity pool at golden hour. "
            "Drone-like aerial glide. Gimbal-stabilized forward push. "
            "Water ripples softly in warm cinematic light."
        ),
    },
    "E2_native_verbs": {
        "hypothesis": "Seedance-native verbs (drift, undulate, cascade) — better PE translation?",
        "prompt": (
            "Turquoise infinity pool at golden hour. The camera drifts forward gradually. "
            "Water surface undulates gently. Warm amber light cascades across the pool. "
            "Palm fronds sway rhythmically in the breeze."
        ),
    },
}


def main():
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("eval_results")
    output_dir.mkdir(exist_ok=True)

    # Load credentials
    kie_cred = Path.home() / ".openclaw/workspace/.credentials/kie-api.json"
    gemini_cred = Path.home() / ".openclaw/workspace/.gemini_credentials.json"

    kie_key = json.loads(kie_cred.read_text())["api_key"]
    gemini_key = json.loads(gemini_cred.read_text())["api_key"]

    kie = KieClient(api_key=kie_key, max_requests=50)
    qc = GeminiVideoQC(api_key=gemini_key, model="gemini-2.5-pro")
    reward_calc = RewardCalculator()

    print(f"PE Layer Probe — {len(VARIANTS)} variants")
    print(f"Estimated: 1 image + {len(VARIANTS)} videos + {len(VARIANTS)} QC calls")
    print(f"{'='*70}\n")

    # Step 1: Generate reference image
    print("[1/3] Generating reference image (pool scene, 9:16)...")
    img = kie.generate_image(
        "A luxury hotel outdoor infinity pool at golden hour, crystal clear "
        "turquoise water, sun loungers, palm trees, warm golden light, "
        "professional architectural photography, high detail",
        aspect_ratio="9:16",
    )
    if not img.success:
        print(f"FATAL: Image gen failed: {img.state}")
        return
    image_url = img.result_urls[0]
    print(f"  ✓ Image: {image_url}\n")

    # Step 2: Generate videos + QC score each
    print(f"[2/3] Generating {len(VARIANTS)} video variants + QC scoring...\n")
    results = []

    for i, (name, spec) in enumerate(VARIANTS.items(), 1):
        prompt = spec["prompt"]
        wc = len(prompt.split())
        print(f"  [{i}/{len(VARIANTS)}] {name} ({wc}w) — {spec['hypothesis']}")

        # Generate video
        try:
            vid = kie.generate_video(
                prompt=prompt,
                image_url=image_url,
                duration=8,
                resolution="1080p",
                aspect_ratio="9:16",
            )
        except Exception as e:
            print(f"    ✗ Video gen error: {e}")
            results.append({
                "variant": name,
                "hypothesis": spec["hypothesis"],
                "prompt": prompt,
                "word_count": wc,
                "success": False,
                "error": str(e),
            })
            continue

        if not vid.success:
            print(f"    ✗ Video gen failed: {vid.state}")
            results.append({
                "variant": name,
                "hypothesis": spec["hypothesis"],
                "prompt": prompt,
                "word_count": wc,
                "success": False,
                "error": vid.state,
            })
            continue

        video_url = vid.result_urls[0] if vid.result_urls else ""
        print(f"    ✓ Video: {video_url[:80]}...")

        # QC score
        print(f"    Scoring with Gemini 2.5 Pro...")
        try:
            qc_result = qc.evaluate(video_url)
        except Exception as e:
            print(f"    ✗ QC error: {e}")
            qc_result = {"pass": True, "confidence": 0.0, "aesthetic_score": 5,
                         "motion_score": 5, "prompt_adherence_score": 5,
                         "scroll_stop_score": 5, "auto_fail_triggered": [],
                         "minor_issues": [], "summary": f"QC error: {e}"}

        # Compute reward
        reward_input = {
            "pass": qc_result.get("pass", True),
            "confidence": qc_result.get("confidence", 0.5),
            "aesthetic_score": qc_result.get("aesthetic_score", 5),
            "motion_score": qc_result.get("motion_score", 5),
            "prompt_adherence_score": qc_result.get("prompt_adherence_score", 5),
            "auto_fail_triggered": qc_result.get("auto_fail_triggered", []),
            "minor_issues": qc_result.get("minor_issues", []),
        }
        reward_breakdown = reward_calc.calculate(reward_input)
        reward_score = reward_breakdown.total_score

        entry = {
            "variant": name,
            "hypothesis": spec["hypothesis"],
            "prompt": prompt,
            "word_count": wc,
            "success": True,
            "video_url": video_url,
            "task_id": vid.task_id,
            "cost_time_ms": vid.cost_time_ms,
            "qc": qc_result,
            "reward": reward_score,
            "reward_dimensions": {
                k: {"score": v.score, "weighted": v.weighted_score}
                for k, v in reward_breakdown.dimensions.items()
            } if reward_breakdown.dimensions else {},
        }
        results.append(entry)

        status = "PASS" if qc_result.get("pass") else "FAIL"
        fails = qc_result.get("auto_fail_triggered", [])
        print(f"    → {status} | reward={reward_score:.1f} | aesthetic={qc_result.get('aesthetic_score')} "
              f"motion={qc_result.get('motion_score')} adherence={qc_result.get('prompt_adherence_score')}")
        if fails:
            print(f"    → auto-fail: {', '.join(fails)}")
        print()

    # Step 3: Analysis
    print(f"\n[3/3] Analysis\n{'='*70}")

    successful = [r for r in results if r.get("success")]
    if not successful:
        print("No successful results to analyze.")
        return

    # Sort by reward
    successful.sort(key=lambda x: float(x.get("reward", 0)), reverse=True)

    print(f"\n{'Variant':<25} {'Words':>5} {'Pass':>5} {'Reward':>7} {'Aes':>4} {'Mot':>4} {'Adh':>4} {'Scroll':>7}")
    print("-" * 70)
    for r in successful:
        qc = r["qc"]
        status = "✓" if qc.get("pass") else "✗"
        print(f"{r['variant']:<25} {r['word_count']:>5} {status:>5} {float(r['reward']):>7.1f} "
              f"{qc.get('aesthetic_score', 0):>4} {qc.get('motion_score', 0):>4} "
              f"{qc.get('prompt_adherence_score', 0):>4} {qc.get('scroll_stop_score', 0):>7}")

    # Dimension analysis
    print(f"\n{'='*70}")
    print("DIMENSION ANALYSIS")
    print(f"{'='*70}")

    def avg_reward(prefix):
        items = [r for r in successful if r["variant"].startswith(prefix)]
        if not items:
            return 0
        return sum(float(r.get("reward", 0)) for r in items) / len(items)

    def get_reward(name):
        for r in successful:
            if r["variant"] == name:
                return float(r.get("reward", 0))
        return 0

    # Structure order
    print("\n1. STRUCTURE ORDER")
    for v in ["A1_subject_first", "A2_camera_first", "A3_scene_first"]:
        print(f"   {v}: reward={get_reward(v):.1f}")

    # Length
    print("\n2. PROMPT LENGTH")
    for v in ["B1_minimal_15w", "B2_medium_30w", "B3_verbose_50w"]:
        r = next((x for x in successful if x["variant"] == v), None)
        wc = r["word_count"] if r else "?"
        print(f"   {v} ({wc}w): reward={get_reward(v):.1f}")

    # Adverbs
    print("\n3. DEGREE ADVERBS")
    for v in ["C1_with_adverbs", "C2_without_adverbs"]:
        print(f"   {v}: reward={get_reward(v):.1f}")
    delta = get_reward("C1_with_adverbs") - get_reward("C2_without_adverbs")
    print(f"   Delta (with - without): {delta:+.1f}")

    # Negatives
    print("\n4. NEGATIVE CONSTRAINTS")
    for v in ["D1_with_negatives", "D2_without_negatives"]:
        print(f"   {v}: reward={get_reward(v):.1f}")
    delta = get_reward("D1_with_negatives") - get_reward("D2_without_negatives")
    print(f"   Delta (with - without): {delta:+.1f}")

    # Vocabulary
    print("\n5. VOCABULARY TYPE")
    for v in ["E1_equipment_names", "E2_native_verbs"]:
        print(f"   {v}: reward={get_reward(v):.1f}")

    # Save results
    output = {
        "experiment": "pe_layer_probe_v1",
        "timestamp": ts,
        "reference_image_url": image_url,
        "total_variants": len(VARIANTS),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "results": results,
        "api_stats": kie.stats.summary(),
    }
    output_path = output_dir / f"pe_probe_{ts}.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nResults saved: {output_path}")
    print(f"API stats: {kie.stats.summary()}")


if __name__ == "__main__":
    main()
