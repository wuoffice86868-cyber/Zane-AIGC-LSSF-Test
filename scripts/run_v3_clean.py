#!/usr/bin/env python3
"""Clean V3 vs V1 comparison — robust error handling, retries, and progress output."""

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.path.insert(0, '/home/node/.openclaw/workspace/prompt_evaluator')

from prompt_evaluator.kie_client import KieClient
from prompt_evaluator.gemini_client import GeminiVideoQC
from prompt_evaluator.reward_calculator import RewardCalculator

# --- Credentials ---
with open(os.path.expanduser('~/.openclaw/workspace/.credentials/kie-api.json')) as f:
    kie_key = json.load(f)['api_key']

for p in ['/data/workspace/.gemini_credentials.json',
          os.path.expanduser('~/.openclaw/workspace/.credentials/gemini.json'),
          os.path.expanduser('~/.openclaw/workspace/.gemini_credentials.json')]:
    if os.path.exists(p):
        with open(p) as f:
            gemini_key = json.load(f).get('api_key')
        if gemini_key:
            break
else:
    print("ERROR: No Gemini credentials"); sys.exit(1)

kie = KieClient(api_key=kie_key, max_requests=50)
qc = GeminiVideoQC(api_key=gemini_key, model="gemini-2.5-pro")  # Pro for better QC
reward_calc = RewardCalculator()

# --- Prompts ---
image_prompts = {
    "pool": "A luxury hotel infinity pool at golden hour, turquoise water merging with ocean horizon, palm trees, warm sunset light, professional hotel photography, 9:16 vertical",
    "room": "A luxury penthouse hotel bedroom, silk curtains, morning sunlight through floor-to-ceiling windows, ocean view, crisp white bedding, professional hotel photography, 9:16 vertical",
    "lobby": "A grand hotel lobby, polished marble floors, warm amber lighting, orchid arrangements, elegant reception desk, professional hotel photography, 9:16 vertical",
    "spa": "A tropical hotel spa, volcanic stone bath among ferns, steam rising, golden afternoon light through bamboo canopy, professional hotel photography, 9:16 vertical",
    "restaurant": "An open-air hotel terrace restaurant at dusk, candlelit tables with white linen, purple sky over calm sea, warm lamp glow, professional hotel photography, 9:16 vertical",
    "beach": "A pristine white sand beach with draped cabana, sheer fabric panels, golden sunset, long shadows on sand, professional hotel photography, 9:16 vertical",
}

# V1: old structure — "Single continuous shot", stability anchors, prescriptive
v1_prompts = {
    "pool": "Single continuous shot. Wide shot of the infinity pool at golden hour. The camera pushes slowly forward along the pool edge. Water ripples softly in warm light. Sky and clouds remain completely still.",
    "room": "Single continuous shot. Medium shot of the penthouse bedroom. The camera pushes slowly forward toward the window. Silk curtains sway gently in the breeze. The bed frame stays firmly anchored in place.",
    "lobby": "Single continuous shot. Wide shot of the grand hotel lobby. The camera pushes slowly forward across the marble floor. The chandelier stays anchored above. Reflections on the floor remain steady.",
    "spa": "Single continuous shot. Medium shot of the volcanic stone bath. The camera pushes slowly forward. Thin steam rises gently above the rocks. Stone walls stay perfectly still.",
    "restaurant": "Single continuous shot. Medium shot of the terrace restaurant at dusk. The camera moves slowly left along the tables. Candle flames flicker gently. The railing stays fixed in frame.",
    "beach": "Single continuous shot. Wide shot of the beach cabana. The camera rises slowly. Sheer fabric panels sway gently in the breeze. Sky and clouds remain completely still.",
}

# V3: new structure — subject+details+motion+camera, rich descriptions, no negatives
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
errors = []
total_start = time.time()

def run_one(scene: str, version: str, prompt: str, image_url: str) -> dict:
    entry = {
        "scene": scene, "version": version, "prompt": prompt,
        "word_count": len(prompt.split()), "image_url": image_url,
        "video_url": "", "qc": {}, "reward": 0.0, "error": "",
    }

    # Generate video (retry once on failure)
    for attempt in range(2):
        try:
            vid = kie.generate_video(
                prompt=prompt, image_url=image_url,
                duration=8, resolution="1080p", aspect_ratio="9:16",
            )
            if vid.success and vid.result_urls:
                entry["video_url"] = vid.result_urls[0]
                entry["video_cost_ms"] = vid.cost_time_ms
                print(f"  Video OK: {vid.result_urls[0][:80]}...")
                break
            else:
                print(f"  Video gen attempt {attempt+1} failed: {vid.state}")
                if attempt == 0:
                    time.sleep(5)
        except Exception as e:
            print(f"  Video gen attempt {attempt+1} error: {e}")
            if attempt == 0:
                time.sleep(5)

    if not entry["video_url"]:
        entry["error"] = "Video generation failed after retries"
        return entry

    # QC (GeminiVideoQC already has internal retries)
    try:
        qc_result = qc.evaluate(entry["video_url"])
        entry["qc"] = qc_result
        p = "PASS" if qc_result.get("pass") else "FAIL"
        ae = qc_result.get("aesthetic_score", "?")
        mo = qc_result.get("motion_score", "?")
        ss = qc_result.get("scroll_stop_score", "?")
        fails = qc_result.get("auto_fail_triggered", [])
        print(f"  QC: {p} | aes={ae} mot={mo} scroll={ss} | fails={fails}")
        print(f"  Summary: {qc_result.get('summary', '')[:120]}")
    except Exception as e:
        entry["error"] = f"QC error: {e}"
        print(f"  QC ERROR: {e}")
        return entry

    # Reward
    try:
        reward_bd = reward_calc.calculate(entry["qc"])
        entry["reward"] = reward_bd.total_score
        print(f"  Reward: {reward_bd.total_score:.1f}")
    except Exception as e:
        print(f"  Reward ERROR: {e}")

    return entry


print(f"Starting V3 vs V1 comparison — {len(scenes)} scenes × 2 versions = {len(scenes)*2} videos")
print(f"Using Gemini 2.5 Pro for QC, Seedance 1.5 Pro @ 1080p 9:16")
print(f"Time: {datetime.now(timezone.utc).isoformat()}")
print()

for i, scene in enumerate(scenes):
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(scenes)}] SCENE: {scene}")
    print(f"{'='*60}")

    # Generate shared reference image
    print("Generating reference image...")
    try:
        img = kie.generate_image(prompt=image_prompts[scene], aspect_ratio="9:16")
        if not img.success or not img.result_urls:
            print(f"  Image gen FAILED, skipping scene")
            errors.append(f"{scene}: image gen failed")
            continue
        image_url = img.result_urls[0]
        print(f"  Image: {image_url[:80]}...")
    except Exception as e:
        print(f"  Image ERROR: {e}, skipping scene")
        errors.append(f"{scene}: {e}")
        continue

    # V1
    print(f"\n  --- V1 (baseline) [{len(v1_prompts[scene].split())}w] ---")
    r1 = run_one(scene, "v1", v1_prompts[scene], image_url)
    results.append(r1)

    # Small delay between API calls
    time.sleep(2)

    # V3
    print(f"\n  --- V3 (new) [{len(v3_prompts[scene].split())}w] ---")
    r3 = run_one(scene, "v3", v3_prompts[scene], image_url)
    results.append(r3)

    # Progress summary
    done_v1 = [r for r in results if r["version"] == "v1" and r["video_url"]]
    done_v3 = [r for r in results if r["version"] == "v3" and r["video_url"]]
    if done_v1:
        avg1 = sum(r["reward"] for r in done_v1) / len(done_v1)
        print(f"\n  Running V1 avg: {avg1:.1f} ({len(done_v1)} samples)")
    if done_v3:
        avg3 = sum(r["reward"] for r in done_v3) / len(done_v3)
        print(f"  Running V3 avg: {avg3:.1f} ({len(done_v3)} samples)")

    time.sleep(2)


# --- Final Summary ---
elapsed = time.time() - total_start
print(f"\n{'='*60}")
print(f"FINAL RESULTS — {len(results)} samples in {elapsed:.0f}s")
print(f"{'='*60}")

v1_ok = [r for r in results if r["version"] == "v1" and r["video_url"]]
v3_ok = [r for r in results if r["version"] == "v3" and r["video_url"]]

v1_avg = sum(r["reward"] for r in v1_ok) / len(v1_ok) if v1_ok else 0
v3_avg = sum(r["reward"] for r in v3_ok) / len(v3_ok) if v3_ok else 0
v1_pass = sum(1 for r in v1_ok if r["qc"].get("pass"))
v3_pass = sum(1 for r in v3_ok if r["qc"].get("pass"))

print(f"V1: avg={v1_avg:.1f} | pass={v1_pass}/{len(v1_ok)}")
print(f"V3: avg={v3_avg:.1f} | pass={v3_pass}/{len(v3_ok)}")
if v1_avg and v3_avg:
    print(f"Delta (V3-V1): {v3_avg - v1_avg:+.1f}")

# Aesthetic comparison
v1_aes = [r["qc"].get("aesthetic_score", 0) for r in v1_ok if r["qc"]]
v3_aes = [r["qc"].get("aesthetic_score", 0) for r in v3_ok if r["qc"]]
if v1_aes and v3_aes:
    print(f"\nAesthetic avg: V1={sum(v1_aes)/len(v1_aes):.1f} V3={sum(v3_aes)/len(v3_aes):.1f}")

# Scroll-stop comparison
v1_ss = [r["qc"].get("scroll_stop_score", 0) for r in v1_ok if r["qc"]]
v3_ss = [r["qc"].get("scroll_stop_score", 0) for r in v3_ok if r["qc"]]
if v1_ss and v3_ss:
    print(f"Scroll-stop avg: V1={sum(v1_ss)/len(v1_ss):.1f} V3={sum(v3_ss)/len(v3_ss):.1f}")

print(f"\nPer-scene breakdown:")
for scene in scenes:
    v1 = next((r for r in results if r["scene"] == scene and r["version"] == "v1"), None)
    v3 = next((r for r in results if r["scene"] == scene and r["version"] == "v3"), None)
    v1_s = f"{v1['reward']:.0f}" if v1 and v1["video_url"] else "ERR"
    v3_s = f"{v3['reward']:.0f}" if v3 and v3["video_url"] else "ERR"
    v1_p = "P" if v1 and v1["qc"].get("pass") else "F"
    v3_p = "P" if v3 and v3["qc"].get("pass") else "F"
    v1_ae = v1["qc"].get("aesthetic_score", "-") if v1 and v1["qc"] else "-"
    v3_ae = v3["qc"].get("aesthetic_score", "-") if v3 and v3["qc"] else "-"
    print(f"  {scene:12s} | V1: {v1_s:>4s}({v1_p}) aes={v1_ae} | V3: {v3_s:>4s}({v3_p}) aes={v3_ae}")

if errors:
    print(f"\nErrors: {errors}")

# Save
out = f"/home/node/.openclaw/workspace/prompt_evaluator/eval_results/v3_clean_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
with open(out, 'w') as f:
    json.dump({
        "test": "v3_vs_v1_clean",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": elapsed,
        "qc_model": "gemini-2.5-pro",
        "video_model": "seedance-1.5-pro",
        "resolution": "1080p",
        "aspect_ratio": "9:16",
        "summary": {
            "v1_avg": v1_avg, "v3_avg": v3_avg,
            "delta": v3_avg - v1_avg if v1_avg and v3_avg else None,
            "v1_pass_rate": f"{v1_pass}/{len(v1_ok)}", "v3_pass_rate": f"{v3_pass}/{len(v3_ok)}",
            "v1_aesthetic_avg": sum(v1_aes)/len(v1_aes) if v1_aes else 0,
            "v3_aesthetic_avg": sum(v3_aes)/len(v3_aes) if v3_aes else 0,
        },
        "results": results,
        "errors": errors,
    }, f, indent=2, default=str)
print(f"\nSaved: {out}")
print(f"\nAPI usage: {kie.stats.summary()}")
