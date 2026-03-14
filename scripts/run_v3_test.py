#!/usr/bin/env python3
"""V3 system prompt test — 6 scenes with manually crafted v3-structure prompts."""

import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, '/home/node/.openclaw/workspace/prompt_evaluator')

from prompt_evaluator.kie_client import KieClient
from prompt_evaluator.gemini_client import GeminiVideoQC
from prompt_evaluator.reward_calculator import RewardCalculator

with open(os.path.expanduser('~/.openclaw/workspace/.credentials/kie-api.json')) as f:
    kie_key = json.load(f)['api_key']

with open('/data/workspace/.gemini_credentials.json') as f:
    gemini_key = json.load(f)['api_key']

kie = KieClient(api_key=kie_key)
qc = GeminiVideoQC(api_key=gemini_key)
reward_calc = RewardCalculator()

# V3 prompts: subject+action+scene+camera order, NO negative constraints, NO stability anchors
# Based on Seedance official guide: degree adverbs matter, rich visual description, no "avoid X"
scenes = [
    {
        "type": "pool",
        "v3_prompt": "Luxury infinity pool merges with golden ocean horizon, palm fronds casting warm shadows on turquoise water, light ripples dancing across the surface. Camera slowly pushes forward along the pool edge.",
    },
    {
        "type": "room",
        "v3_prompt": "Sunlit penthouse bedroom with silk curtains gently swaying, warm morning light streaming through floor-to-ceiling windows revealing a vast ocean below. Camera gradually pulls back from the window.",
    },
    {
        "type": "lobby",
        "v3_prompt": "Grand hotel lobby with polished marble floors reflecting warm amber sconces, a single orchid arrangement at the reception desk, soft shadows pooling in the archways. Camera smoothly pans left across the space.",
    },
    {
        "type": "spa",
        "v3_prompt": "Secluded stone bath nestled among lush tropical ferns, thin steam rising slowly above dark volcanic rocks, golden afternoon light filtering through a bamboo canopy. Camera gently pushes forward toward the water.",
    },
    {
        "type": "restaurant",
        "v3_prompt": "Open-air terrace restaurant at dusk, candle flames softly flickering on linen tablecloths, deep purple sky above a calm sea, warm lamplight glowing along the balcony railing. Camera slowly drifts left.",
    },
    {
        "type": "beach",
        "v3_prompt": "Pristine white sand beach with a draped cabana, sheer fabric panels swaying in a sea breeze, golden sunset light painting long shadows across undisturbed sand. Camera gradually rises above the shoreline.",
    },
]

results = []
total_start = time.time()

for scene in scenes:
    scene_type = scene["type"]
    prompt = scene["v3_prompt"]
    print(f"\n{'='*60}")
    print(f"Scene: {scene_type}")
    print(f"Prompt ({len(prompt.split())} words): {prompt}")
    print(f"{'='*60}")

    # 1. Generate image
    try:
        img_result = kie.generate_image(prompt=prompt, aspect_ratio="9:16")
        if img_result.state != 'success':
            print(f"Image gen failed: {img_result}")
            continue
        image_url = img_result.result_urls[0]
        print(f"Image: {image_url}")
    except Exception as e:
        print(f"ERROR image: {e}")
        continue

    # 2. Generate video
    try:
        vid_result = kie.generate_video(
            prompt=prompt,
            image_url=image_url,
            duration=8,
            resolution="1080p",
        )
        if vid_result.state != 'success':
            print(f"Video gen failed: {vid_result}")
            continue
        video_url = vid_result.result_urls[0]
        print(f"Video: {video_url}")
    except Exception as e:
        print(f"ERROR video: {e}")
        continue

    # 3. QC
    try:
        qc_result = qc.evaluate_video(video_url=video_url, original_prompt=prompt)
        pf = qc_result.get('pass_fail', 'N/A')
        ae = qc_result.get('aesthetic_score', 'N/A')
        mo = qc_result.get('motion_score', 'N/A')
        ad = qc_result.get('adherence_score', 'N/A')
        print(f"QC: {pf} | aesthetic={ae} motion={mo} adherence={ad}")
        if qc_result.get('auto_fail_triggered'):
            issues = [i.get('type', i) if isinstance(i, dict) else i for i in qc_result.get('issues', [])]
            print(f"Auto-fails: {issues}")
        print(f"Summary: {str(qc_result.get('summary', ''))[:200]}")
    except Exception as e:
        print(f"ERROR QC: {e}")
        qc_result = {"pass_fail": "error", "error": str(e)}

    # 4. Reward
    try:
        reward = reward_calc.calculate(qc_result)
    except Exception as e:
        print(f"ERROR reward: {e}")
        reward = 0
    print(f"Reward: {reward}")

    results.append({
        "scene": scene_type,
        "prompt": prompt,
        "image_url": image_url,
        "video_url": video_url,
        "qc": qc_result,
        "reward": reward,
    })

# Summary
elapsed = time.time() - total_start
print(f"\n{'='*60}")
print(f"V3 RESULTS — {len(results)} samples in {elapsed:.0f}s")
print(f"{'='*60}")
if results:
    avg_reward = sum(r['reward'] for r in results) / len(results)
    passes = sum(1 for r in results if r['qc'].get('pass_fail') == 'PASS')
    print(f"Avg reward: {avg_reward:.1f} | Pass rate: {passes}/{len(results)}")
    print(f"V1 baseline avg was 70.8 | Delta: {avg_reward - 70.8:+.1f}")
    print()
    for r in results:
        status = "✓" if r['qc'].get('pass_fail') == 'PASS' else "✗"
        print(f"  {status} {r['scene']}: reward={r['reward']:.1f} | {r['video_url']}")

output_path = f"/home/node/.openclaw/workspace/prompt_evaluator/eval_results/v3_test_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved: {output_path}")
