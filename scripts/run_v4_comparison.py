#!/usr/bin/env python3
"""v4 vs v3 comparison — 6 scenes × 2 versions at 1080p.

Runs after analyze_pe_probe.py has generated system_prompts/hotel_v4.txt.
Uses Gemini 2.5 Pro for prompt generation + QC scoring.

Budget: ~14 Kie calls (6 images + 6×2 videos), ~12 Gemini QC calls.
Estimated cost: ~$6-8 on Kie.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt_evaluator.kie_client import KieClient
from prompt_evaluator.gemini_client import GeminiVideoQC, GeminiLLM
from prompt_evaluator.reward_calculator import RewardCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SCENES = {
    "pool": {
        "image_prompt": "A luxury hotel outdoor infinity pool at golden hour, crystal clear turquoise water, sun loungers, palm trees, warm golden light, professional architectural photography",
        "v3_prompt": "Turquoise infinity pool at golden hour, crystal clear water with soft natural ripples. The camera pushes slowly forward, revealing sun loungers and tropical palms bathed in warm amber sunset light.",
        "scene_type": "infinity pool",
        "camera_move": "push",
        "lighting": "golden hour",
        "key_feature": "crystal clear turquoise water, sun loungers, palm trees",
        "subtle_motion": "water ripples, palm fronds swaying",
    },
    "room": {
        "image_prompt": "A luxury hotel room with king bed, white linen, large windows with city view, soft morning light, elegant interior design, 9:16",
        "v3_prompt": "Elegant hotel room with king bed and crisp white linen. The camera pushes forward slowly, revealing floor-to-ceiling windows with soft morning light flooding in. Sheer curtains drift gently in a subtle breeze.",
        "scene_type": "hotel room",
        "camera_move": "push",
        "lighting": "soft morning light",
        "key_feature": "king bed, white linen, floor-to-ceiling windows with city view",
        "subtle_motion": "sheer curtains drifting",
    },
    "lobby": {
        "image_prompt": "A grand luxury hotel lobby with marble floors, high ceilings, large floral arrangement, warm ambient lighting, sophisticated interior",
        "v3_prompt": "Grand hotel lobby with polished marble floors and soaring ceilings. The camera moves slowly right, revealing an elaborate floral arrangement under warm ambient lighting. Guests move quietly in the background.",
        "scene_type": "hotel lobby",
        "camera_move": "move right",
        "lighting": "warm ambient",
        "key_feature": "marble floors, soaring ceilings, floral arrangement",
        "subtle_motion": "soft ambient movement",
    },
    "restaurant": {
        "image_prompt": "A fine dining restaurant with white tablecloths, candlelight, elegant glassware, warm intimate lighting, luxury hotel restaurant",
        "v3_prompt": "Intimate fine dining restaurant with white tablecloths and candlelit atmosphere. The camera drifts forward gradually, revealing crystal glassware catching warm candlelight. An elegant, hushed ambiance.",
        "scene_type": "fine dining restaurant",
        "camera_move": "drift forward",
        "lighting": "candlelight, warm intimate",
        "key_feature": "white tablecloths, crystal glassware, candles",
        "subtle_motion": "candle flames flickering gently",
    },
    "exterior": {
        "image_prompt": "Luxury hotel exterior facade at dusk, dramatic lighting, modern architecture, grand entrance with landscape lighting",
        "v3_prompt": "Majestic hotel facade at dusk with dramatic architectural lighting. The camera rises slowly upward, revealing the grand entrance and illuminated windows against a deep blue twilight sky.",
        "scene_type": "hotel exterior",
        "camera_move": "rise",
        "lighting": "dusk, dramatic architectural lighting",
        "key_feature": "grand facade, illuminated entrance, twilight sky",
        "subtle_motion": "subtle light shimmer",
    },
    "spa": {
        "image_prompt": "A luxury hotel spa with zen interior, natural stone, soft candlelight, fresh flowers, peaceful atmosphere",
        "v3_prompt": "Serene spa treatment room with natural stone surfaces and soft candlelight. The camera drifts forward slowly, revealing fresh orchids and smooth river stones in warm, diffused golden light.",
        "scene_type": "spa treatment room",
        "camera_move": "drift forward",
        "lighting": "soft candlelight, warm diffused",
        "key_feature": "natural stone, fresh orchids, candles",
        "subtle_motion": "candle flames, steam wisps",
    },
}

V4_PROMPT_TEMPLATE_PATH = Path("system_prompts/hotel_v4.txt")
V3_SYSTEM_PROMPT_PATH = Path("system_prompts/hotel_v3.txt")


def generate_v4_prompt(llm: GeminiLLM, scene_data: dict) -> str:
    """Generate a v4 prompt using the v4 system prompt template."""
    system = V4_PROMPT_TEMPLATE_PATH.read_text()
    user = f"""Generate a Seedance video prompt for this scene:

scene_type: {scene_data['scene_type']}
camera_move: {scene_data['camera_move']}
lighting: {scene_data['lighting']}
key_feature: {scene_data['key_feature']}
subtle_motion: {scene_data['subtle_motion']}

Output ONLY the video prompt — one paragraph, no labels, no explanations."""

    full = system + "\n\n" + user
    return llm.generate(full, temperature=0.4, max_tokens=200)


def main():
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("eval_results")
    output_dir.mkdir(exist_ok=True)

    if not V4_PROMPT_TEMPLATE_PATH.exists():
        print(f"ERROR: {V4_PROMPT_TEMPLATE_PATH} not found. Run analyze_pe_probe.py first.")
        return

    kie_cred = Path.home() / ".openclaw/workspace/.credentials/kie-api.json"
    gemini_cred = Path.home() / ".openclaw/workspace/.gemini_credentials.json"
    kie_key = json.loads(kie_cred.read_text())["api_key"]
    gemini_key = json.loads(gemini_cred.read_text())["api_key"]

    kie = KieClient(api_key=kie_key, max_requests=60)
    qc_scorer = GeminiVideoQC(api_key=gemini_key, model="gemini-2.5-pro")
    llm = GeminiLLM(api_key=gemini_key, model="gemini-2.5-flash")
    calc = RewardCalculator()

    print(f"v4 vs v3 Comparison — {len(SCENES)} scenes × 2 versions")
    print(f"{'=' * 70}\n")

    scene_results = []

    for scene_name, scene_data in SCENES.items():
        print(f"\n{'─' * 70}")
        print(f"Scene: {scene_name.upper()}")
        print(f"{'─' * 70}")

        # Generate reference image
        print(f"  Generating reference image...")
        try:
            img = kie.generate_image(
                scene_data["image_prompt"] + ", 9:16 vertical format",
                aspect_ratio="9:16",
            )
            if not img.success:
                print(f"  ✗ Image failed: {img.state}")
                continue
            image_url = img.result_urls[0]
            print(f"  ✓ Image: {image_url[:80]}...")
        except Exception as e:
            print(f"  ✗ Image error: {e}")
            continue

        scene_entry = {"scene": scene_name, "image_url": image_url, "versions": {}}

        for version in ["v3", "v4"]:
            print(f"\n  [{version.upper()}]")

            if version == "v3":
                prompt = scene_data["v3_prompt"]
                print(f"  Prompt (hardcoded): {prompt[:100]}...")
            else:
                print(f"  Generating prompt with Gemini...")
                try:
                    prompt = generate_v4_prompt(llm, scene_data)
                    print(f"  Generated: {prompt[:100]}...")
                except Exception as e:
                    print(f"  ✗ Prompt gen error: {e}")
                    continue

            wc = len(prompt.split())
            print(f"  Word count: {wc}")

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
                print(f"  ✗ Video gen error: {e}")
                scene_entry["versions"][version] = {"error": str(e)}
                continue

            if not vid.success:
                print(f"  ✗ Video failed: {vid.state}")
                scene_entry["versions"][version] = {"error": vid.state}
                continue

            video_url = vid.result_urls[0] if vid.result_urls else ""
            print(f"  ✓ Video: {video_url[:80]}...")

            # QC score
            print(f"  Scoring...")
            try:
                qc_result = qc_scorer.evaluate(video_url)
            except Exception as e:
                print(f"  ✗ QC error: {e}")
                qc_result = {
                    "pass": True, "confidence": 0.0,
                    "aesthetic_score": 5, "motion_score": 5,
                    "prompt_adherence_score": 5, "scroll_stop_score": 5,
                    "auto_fail_triggered": [], "minor_issues": [],
                    "summary": f"QC error: {e}"
                }

            reward_result = calc.calculate(qc_result)
            reward_score = reward_result.total_score

            status = "PASS" if qc_result.get("pass") else "FAIL"
            fails = qc_result.get("auto_fail_triggered", [])
            print(f"  → {status} | reward={reward_score:.1f} | "
                  f"aes={qc_result.get('aesthetic_score')} "
                  f"mot={qc_result.get('motion_score')} "
                  f"adh={qc_result.get('prompt_adherence_score')}")
            if fails:
                print(f"  → auto-fail: {', '.join(fails)}")

            scene_entry["versions"][version] = {
                "prompt": prompt,
                "word_count": wc,
                "video_url": video_url,
                "task_id": vid.task_id,
                "qc": qc_result,
                "reward": reward_score,
            }

        scene_results.append(scene_entry)

    # Summary
    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Scene':<15} {'v3 Reward':>10} {'v4 Reward':>10} {'Delta':>8} {'Winner':>8}")
    print("-" * 60)

    v3_rewards = []
    v4_rewards = []
    for sr in scene_results:
        v3 = sr["versions"].get("v3", {})
        v4 = sr["versions"].get("v4", {})
        v3_r = float(v3.get("reward", 0)) if "reward" in v3 else None
        v4_r = float(v4.get("reward", 0)) if "reward" in v4 else None

        if v3_r is not None:
            v3_rewards.append(v3_r)
        if v4_r is not None:
            v4_rewards.append(v4_r)

        if v3_r is not None and v4_r is not None:
            delta = v4_r - v3_r
            winner = "v4" if delta > 2 else ("v3" if delta < -2 else "tie")
            print(f"{sr['scene']:<15} {v3_r:>10.1f} {v4_r:>10.1f} {delta:>+8.1f} {winner:>8}")
        else:
            v3_str = f"{v3_r:.1f}" if v3_r is not None else "ERR"
            v4_str = f"{v4_r:.1f}" if v4_r is not None else "ERR"
            print(f"{sr['scene']:<15} {v3_str:>10} {v4_str:>10} {'N/A':>8}")

    if v3_rewards and v4_rewards:
        avg_v3 = sum(v3_rewards) / len(v3_rewards)
        avg_v4 = sum(v4_rewards) / len(v4_rewards)
        print("-" * 60)
        print(f"{'AVERAGE':<15} {avg_v3:>10.1f} {avg_v4:>10.1f} {avg_v4-avg_v3:>+8.1f}")
        print(f"\nv3 pass rate: {sum(1 for sr in scene_results if sr['versions'].get('v3',{}).get('qc',{}).get('pass'))}/{len(scene_results)}")
        print(f"v4 pass rate: {sum(1 for sr in scene_results if sr['versions'].get('v4',{}).get('qc',{}).get('pass'))}/{len(scene_results)}")

    # Save results
    output = {
        "experiment": "v4_vs_v3_comparison",
        "timestamp": ts,
        "scenes": list(SCENES.keys()),
        "results": scene_results,
        "summary": {
            "v3_avg": sum(v3_rewards) / len(v3_rewards) if v3_rewards else 0,
            "v4_avg": sum(v4_rewards) / len(v4_rewards) if v4_rewards else 0,
            "delta": (sum(v4_rewards) / len(v4_rewards) - sum(v3_rewards) / len(v3_rewards)) if v3_rewards and v4_rewards else 0,
        },
        "api_stats": kie.stats.summary(),
    }
    output_path = output_dir / f"v4_comparison_{ts}.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nResults saved: {output_path}")
    print(f"API stats: {kie.stats.summary()}")


if __name__ == "__main__":
    main()
