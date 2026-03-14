#!/usr/bin/env python3
"""Probe Seedance PE (Prompt Engineering) Layer Behavior.

Seedance 1.0 tech report (arXiv:2506.09113) reveals an internal Qwen2.5-14B
PE layer that rewrites ALL user prompts before they hit the diffusion model.

This script systematically tests what prompt structures/vocabulary survive
the PE rewrite by generating videos from varied prompt formats and
analyzing which ones produce the best quality results.

Test dimensions:
  1. Structure order: subject-first vs camera-first vs scene-first
  2. Vocabulary: Seedance-native verbs vs generic verbs vs equipment names
  3. Length: 15 words vs 30 words vs 50 words
  4. Specificity: vague vs detailed scene descriptions
  5. Degree adverbs: with vs without ("slowly", "gently")
  6. Negative constraints: with vs without ("no morphing", "avoid warping")

Each test generates a video from the same reference image with different
prompt variants, scores them with Gemini QC, and compares using the
3-dim reward model.

Usage:
    python3 scripts/probe_pe_layer.py --scene pool --variants 6 --dry-run
    python3 scripts/probe_pe_layer.py --scene pool --variants 6
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt_evaluator.kie_client import KieClient
from prompt_evaluator.reward_calculator import RewardCalculator
from prompt_evaluator.models import RewardDimension

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt variants for PE layer probing
# ---------------------------------------------------------------------------

POOL_SCENE_IMAGE = None  # Will be generated once, reused for all variants

PROMPT_VARIANTS = {
    "subject_first": (
        "Turquoise infinity pool at golden hour with crystal clear water. "
        "The camera pushes slowly forward. Soft ripples catch warm sunset light. "
        "Palm fronds sway gently at the edges."
    ),
    "camera_first": (
        "The camera pushes slowly forward over a turquoise infinity pool at golden hour. "
        "Crystal clear water ripples softly. Warm sunset light reflects off the surface. "
        "Palm fronds sway gently."
    ),
    "scene_first": (
        "Golden hour at a luxury resort. Wide shot of infinity pool stretching toward "
        "the horizon. Camera pushes forward slowly. Water surface catches warm light. "
        "Tropical plants frame the shot."
    ),
    "minimal_15w": (
        "Luxury pool at sunset. Camera pushes forward slowly. "
        "Water ripples gently."
    ),
    "detailed_50w": (
        "Wide establishing shot of a turquoise infinity pool at a luxury beachside resort "
        "during golden hour. The camera pushes forward very slowly and steadily, revealing "
        "crystal clear water with soft natural ripples catching warm amber sunset light. "
        "Lush tropical palm fronds sway ever so gently in a light breeze at the pool edges. "
        "Wooden sun loungers sit perfectly still."
    ),
    "with_adverbs": (
        "Turquoise infinity pool at golden hour. The camera pushes extremely slowly "
        "and steadily forward. Water ripples very gently and naturally. "
        "Palm leaves drift barely perceptibly in soft warm light."
    ),
    "without_adverbs": (
        "Turquoise infinity pool at golden hour. The camera pushes forward. "
        "Water ripples. Palm leaves move. Warm light on the surface."
    ),
    "with_negatives": (
        "Turquoise infinity pool at golden hour. The camera pushes slowly forward. "
        "Water ripples softly. No morphing, no warping. Pool edges stay perfectly still. "
        "Architecture maintains structural integrity."
    ),
    "without_negatives": (
        "Turquoise infinity pool at golden hour. The camera pushes slowly forward. "
        "Water ripples softly. Warm sunset light plays on the surface. "
        "Tropical atmosphere, professional cinematic quality."
    ),
    "equipment_names": (
        "Steadicam tracking shot of turquoise infinity pool at golden hour. "
        "Drone-like aerial perspective. Gimbal-stabilized movement. "
        "Water ripples softly in warm light."
    ),
    "native_verbs": (
        "Turquoise infinity pool at golden hour. The camera drifts forward gradually. "
        "Water surface undulates gently. Warm amber light cascades across the pool. "
        "Palm fronds sway in the breeze."
    ),
    "v3_template": (
        "Turquoise infinity pool at golden hour, crystal clear water with soft natural "
        "ripples. The camera pushes slowly forward, revealing sun loungers and tropical "
        "palms bathed in warm amber sunset light."
    ),
}


def load_credentials():
    """Load Kie API key."""
    cred_path = Path.home() / ".openclaw/workspace/.credentials/kie-api.json"
    if cred_path.exists():
        return json.load(open(cred_path))["api_key"]
    return os.environ.get("KIE_API_KEY", "")


def run_probe(args):
    api_key = load_credentials()
    if not api_key:
        print("ERROR: No Kie API key found")
        return

    kie = KieClient(api_key=api_key, max_requests=50)
    calc = RewardCalculator()

    # Select variants
    variants = list(PROMPT_VARIANTS.items())
    if args.variants:
        variants = variants[:args.variants]

    if args.dry_run:
        print(f"\nDRY RUN — would test {len(variants)} prompt variants:")
        for name, prompt in variants:
            wc = len(prompt.split())
            print(f"  {name:20s} ({wc} words): {prompt[:80]}...")
        print(f"\nEstimated cost: {len(variants) + 1} API calls (1 image + {len(variants)} videos)")
        return

    # Step 1: Generate reference image (once)
    print("Generating reference image...")
    img_result = kie.generate_image(
        "A luxury hotel outdoor infinity pool at golden hour, crystal clear turquoise "
        "water, sun loungers, palm trees, warm golden light, professional photography",
        aspect_ratio="9:16",
    )
    if not img_result.success:
        print(f"ERROR: Image generation failed: {img_result.state}")
        return
    image_url = img_result.result_urls[0]
    print(f"  Image: {image_url}")

    # Step 2: Generate video for each prompt variant
    results = []
    for name, prompt in variants:
        print(f"\nTesting: {name} ({len(prompt.split())} words)")
        print(f"  Prompt: {prompt[:100]}...")

        vid_result = kie.generate_video(
            prompt=prompt,
            image_url=image_url,
            duration=8,
            resolution="1080p",
            aspect_ratio="9:16",
        )

        entry = {
            "variant": name,
            "prompt": prompt,
            "word_count": len(prompt.split()),
            "video_url": "",
            "video_task_id": vid_result.task_id,
            "success": vid_result.success,
            "cost_time_ms": vid_result.cost_time_ms,
        }

        if vid_result.success and vid_result.result_urls:
            entry["video_url"] = vid_result.result_urls[0]
            print(f"  ✓ Video: {entry['video_url']}")
        else:
            print(f"  ✗ Failed: {vid_result.state}")

        results.append(entry)

    # Step 3: Save results
    output_dir = Path("eval_results")
    output_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"pe_probe_{ts}.json"

    data = {
        "experiment": "pe_layer_probe",
        "timestamp": ts,
        "image_url": image_url,
        "variants_tested": len(results),
        "results": results,
        "note": "Videos need Gemini QC scoring separately (QC not run in this script to save time)",
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\nSaved results to {output_path}")

    # Summary
    success_count = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}")
    print(f"PE Layer Probe Complete: {success_count}/{len(results)} videos generated")
    print(f"API calls used: {kie.stats.total_requests}")
    print(f"Next step: run Gemini QC on each video to score quality differences")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe Seedance PE layer behavior")
    parser.add_argument("--scene", default="pool", help="Scene type to test")
    parser.add_argument("--variants", type=int, default=6, help="Number of variants to test")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without generating")
    args = parser.parse_args()
    run_probe(args)
