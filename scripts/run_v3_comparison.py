#!/usr/bin/env python3
"""V3 vs V1 system prompt comparison test.

V3: subject+action+scene+camera structure, rich visual descriptions, 
    degree adverbs, no negative constraints, no stability anchors.
V1: "Single continuous shot" format, stability anchors, negative avoidance rules.

Tests 6 scenes. For each scene, generates video with both v1 and v3 style prompts,
runs Gemini QC, computes reward, compares.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, '/home/node/.openclaw/workspace/prompt_evaluator')

from prompt_evaluator.kie_client import KieClient
from prompt_evaluator.gemini_client import GeminiVideoQC
from prompt_evaluator.reward_calculator import RewardCalculator

# --- Load credentials ---
with open(os.path.expanduser('~/.openclaw/workspace/.credentials/kie-api.json')) as f:
    kie_key = json.load(f)['api_key']

gemini_cred_paths = [
    '/data/workspace/.gemini_credentials.json',
    os.path.expanduser('~/.openclaw/workspace/.credentials/gemini.json'),
    os.path.expanduser('~/.openclaw/workspace/.gemini_credentials.json'),
]
gemini_key = None
for p in gemini_cred_paths:
    if os.path.exists(p):
        with open(p) as f:
            gemini_key = json.load(f).get('api_key')
        if gemini_key:
            break

if not gemini_key:
    print("ERROR: No Gemini credentials found")
    sys.exit(1)

kie = KieClient(api_key=kie_key, max_requests=30)
qc = GeminiVideoQC(api_key=gemini_key)
reward_calc = RewardCalculator()

# --- Image prompts (same for both versions — fair comparison) ---
image_prompts = {
    "pool": "A luxury hotel infinity pool at golden hour, turquoise water merging with ocean horizon, palm trees, warm sunset light, professional hotel photography, 9:16 vertical",
    "room": "A luxury penthouse hotel bedroom, silk curtains, morning sunlight through floor-to-ceiling windows, ocean view, crisp white bedding, professional hotel photography, 9:16 vertical",
    "lobby": "A grand hotel lobby, polished marble floors, warm amber lighting, orchid arrangements, elegant reception desk, professional hotel photography, 9:16 vertical",
    "spa": "A tropical hotel spa, volcanic stone bath among ferns, steam rising, golden afternoon light through bamboo canopy, professional hotel photography, 9:16 vertical",
    "restaurant": "An open-air hotel terrace restaurant at dusk, candlelit tables with white linen, purple sky over calm sea, warm lamp glow, professional hotel photography, 9:16 vertical",
    "beach": "A pristine white sand beach with draped cabana, sheer fabric panels, golden sunset, long shadows on sand, professional hotel photography, 9:16 vertical",
}

# --- V1 prompts (old structure: "Single continuous shot", stability anchors) ---
v1_prompts = {
    "pool": "Single continuous shot. Wide shot of the infinity pool at golden hour. The camera pushes slowly forward along the pool edge. Water ripples softly in warm light. Sky and clouds remain completely still.",
    "room": "Single continuous shot. Medium shot of the penthouse bedroom. The camera pushes slowly forward toward the window. Silk curtains sway gently in the breeze. The bed frame stays firmly anchored in place.",
    "lobby": "Single continuous shot. Wide shot of the grand hotel lobby. The camera pushes slowly forward across the marble floor. The chandelier stays anchored above. Reflections on the floor remain steady.",
    "spa": "Single continuous shot. Medium shot of the volcanic stone bath. The camera pushes slowly forward. Thin steam rises gently above the rocks. Stone walls stay perfectly still.",
    "restaurant": "Single continuous shot. Medium shot of the terrace restaurant at dusk. The camera moves slowly left along the tables. Candle flames flicker gently. The railing stays fixed in frame.",
    "beach": "Single continuous shot. Wide shot of the beach cabana. The camera rises slowly. Sheer fabric panels sway gently in the breeze. Sky and clouds remain completely still.",
}

# --- V3 prompts (new structure: subject+details+motion+camera, no negatives) ---
v3_prompts = {
    "pool": "Luxury infinity pool merges with golden ocean horizon, palm fronds casting warm shadows on turquoise water, light ripples dancing across the surface. Camera slowly pushes forward along the pool edge.",
    "room": "Sunlit penthouse bedroom with silk curtains gently swaying, warm morning light streaming through floor-to-ceiling windows revealing a vast ocean below. Camera gradually pulls back from the window.",
    "lobby": "Grand hotel lobby with polished marble floors reflecting warm amber sconces, a single orchid arrangement at the reception desk, soft shadows pooling in the archways. Camera smoothly pans left across the space.",
    "spa": "Secluded stone bath nestled among lush tropical ferns, thin steam rising slowly above dark volcanic rocks, golden afternoon light filtering through a bamboo canopy. Camera gently pushes forward toward the water.",
    "restaurant": "Open-air terrace restaurant at dusk, candle flames softly flickering on linen tablecloths, deep purple sky above a calm sea, warm lamplight glowing along the balcony railing. Camera slowly drifts left.",
    "beach": "Pristine white sand beach with a draped cabana, sheer fabric panels swaying in a sea breeze, golden sunset light painting long shadows across undisturbed sand. Camera gradually rises above the shoreline.",
}

scenes = ["pool", "room", "lobby", "spa", "restaurant", "beach"]

results = []
total_start = time.time()


def run_one(scene_type: str, prompt_version: str, video_prompt: str, image_url: str) -> dict:
    """Generate video + QC + reward for one prompt."""
    entry = {
        "scene": scene_type,
        "version": prompt_version,
        "prompt": video_prompt,
        "word_count": len(video_prompt.split()),
        "image_url": image_url,
        "video_url": "",
        "qc": {},
        "reward": 0.0,
        "error": "",
    }

    # Generate video
    try:
        vid = kie.generate_video(
            prompt=video_prompt,
            image_url=image_url,
            duration=8,
            resolution="1080p",
            aspect_ratio="9:16",
        )
        if not vid.success or not vid.result_urls:
            entry["error"] = f"Video gen failed: {vid.state}"
            print(f"  Video gen FAILED: {vid.state}")
            return entry
        entry["video_url"] = vid.result_urls[0]
        entry["video_cost_ms"] = vid.cost_time_ms
        print(f"  Video: {vid.result_urls[0]}")
    except Exception as e:
        entry["error"] = f"Video error: {e}"
        print(f"  Video ERROR: {e}")
        return entry

    # QC
    try:
        qc_result = qc.evaluate(entry["video_url"])
        entry["qc"] = qc_result
        passed = qc_result.get("pass", False)
        ae = qc_result.get("aesthetic_score", "?")
        mo = qc_result.get("motion_score", "?")
        ad = qc_result.get("prompt_adherence_score", "?")
        ss = qc_result.get("scroll_stop_score", "?")
        fails = qc_result.get("auto_fail_triggered", [])
        status = "PASS" if passed else "FAIL"
        print(f"  QC: {status} | aesthetic={ae} motion={mo} adherence={ad} scroll_stop={ss}")
        if fails:
            print(f"  Auto-fails: {fails}")
        summary = qc_result.get("summary", "")
        if summary:
            print(f"  Summary: {summary[:150]}")
    except Exception as e:
        entry["error"] = f"QC error: {e}"
        print(f"  QC ERROR: {e}")
        entry["qc"] = {"pass": True, "confidence": 0.0}

    # Reward
    try:
        reward_bd = reward_calc.calculate(entry["qc"])
        entry["reward"] = reward_bd.total_score
        print(f"  Reward: {reward_bd.total_score:.1f}")
    except Exception as e:
        print(f"  Reward ERROR: {e}")
        entry["reward"] = 0.0

    return entry


# --- Main loop: generate shared images, then test both prompt versions ---
for scene_type in scenes:
    print(f"\n{'='*70}")
    print(f"SCENE: {scene_type}")
    print(f"{'='*70}")

    # Generate reference image (shared between v1 and v3)
    img_prompt = image_prompts[scene_type]
    print(f"Generating image...")
    try:
        img_result = kie.generate_image(prompt=img_prompt, aspect_ratio="9:16")
        if not img_result.success or not img_result.result_urls:
            print(f"Image gen FAILED for {scene_type}, skipping")
            continue
        image_url = img_result.result_urls[0]
        print(f"Image: {image_url}")
    except Exception as e:
        print(f"Image ERROR for {scene_type}: {e}, skipping")
        continue

    # V1
    print(f"\n--- V1 (baseline) ---")
    print(f"Prompt ({len(v1_prompts[scene_type].split())}w): {v1_prompts[scene_type]}")
    r1 = run_one(scene_type, "v1", v1_prompts[scene_type], image_url)
    results.append(r1)

    # V3
    print(f"\n--- V3 (new structure) ---")
    print(f"Prompt ({len(v3_prompts[scene_type].split())}w): {v3_prompts[scene_type]}")
    r3 = run_one(scene_type, "v3", v3_prompts[scene_type], image_url)
    results.append(r3)


# --- Summary ---
elapsed = time.time() - total_start
print(f"\n{'='*70}")
print(f"V3 vs V1 COMPARISON — {len(results)} samples in {elapsed:.0f}s")
print(f"{'='*70}")

v1_results = [r for r in results if r["version"] == "v1" and r["video_url"]]
v3_results = [r for r in results if r["version"] == "v3" and r["video_url"]]

if v1_results:
    v1_avg = sum(r["reward"] for r in v1_results) / len(v1_results)
    v1_pass = sum(1 for r in v1_results if r["qc"].get("pass", False))
    print(f"V1: avg reward = {v1_avg:.1f} | pass rate = {v1_pass}/{len(v1_results)}")
else:
    v1_avg = 0
    print("V1: no results")

if v3_results:
    v3_avg = sum(r["reward"] for r in v3_results) / len(v3_results)
    v3_pass = sum(1 for r in v3_results if r["qc"].get("pass", False))
    print(f"V3: avg reward = {v3_avg:.1f} | pass rate = {v3_pass}/{len(v3_results)}")
else:
    v3_avg = 0
    print("V3: no results")

if v1_avg and v3_avg:
    delta = v3_avg - v1_avg
    print(f"Delta (V3 - V1): {delta:+.1f}")

print(f"\nPer-scene:")
for scene_type in scenes:
    v1 = next((r for r in results if r["scene"] == scene_type and r["version"] == "v1"), None)
    v3 = next((r for r in results if r["scene"] == scene_type and r["version"] == "v3"), None)
    v1_r = f"{v1['reward']:.0f}" if v1 and v1["video_url"] else "FAIL"
    v3_r = f"{v3['reward']:.0f}" if v3 and v3["video_url"] else "FAIL"
    v1_p = "PASS" if v1 and v1["qc"].get("pass") else "FAIL"
    v3_p = "PASS" if v3 and v3["qc"].get("pass") else "FAIL"
    print(f"  {scene_type:12s} | V1: {v1_r:>5s} ({v1_p}) | V3: {v3_r:>5s} ({v3_p})")

# Save
output_path = f"/home/node/.openclaw/workspace/prompt_evaluator/eval_results/v3_comparison_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
with open(output_path, 'w') as f:
    json.dump({
        "test": "v3_vs_v1_comparison",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": elapsed,
        "summary": {
            "v1_avg_reward": v1_avg,
            "v3_avg_reward": v3_avg,
            "delta": v3_avg - v1_avg if v1_avg and v3_avg else None,
            "v1_count": len(v1_results),
            "v3_count": len(v3_results),
        },
        "results": results,
    }, f, indent=2, default=str)
print(f"\nSaved: {output_path}")
