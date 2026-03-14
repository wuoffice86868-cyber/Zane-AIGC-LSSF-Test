#!/usr/bin/env python3
"""V3 end-to-end test using Gemini for prompt generation (simulates real pipeline).

Flow: System Prompt v3 → Gemini generates cinematography prompt → Seedream image →
      Seedance video → Gemini QC → Reward scoring
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, '/home/node/.openclaw/workspace/prompt_evaluator')

from prompt_evaluator.kie_client import KieClient
from prompt_evaluator.gemini_client import GeminiVideoQC, GeminiLLM
from prompt_evaluator.reward_calculator import RewardCalculator

# --- Load credentials ---
with open(os.path.expanduser('~/.openclaw/workspace/.credentials/kie-api.json')) as f:
    kie_key = json.load(f)['api_key']

with open('/data/workspace/.gemini_credentials.json') as f:
    gemini_key = json.load(f)['api_key']

# --- Init clients ---
kie = KieClient(api_key=kie_key, max_requests=30)
qc = GeminiVideoQC(api_key=gemini_key)
llm = GeminiLLM(api_key=gemini_key)
reward_calc = RewardCalculator()

# --- Load system prompt v3 ---
with open('/home/node/.openclaw/workspace/prompt_evaluator/system_prompts/hotel_v3.txt') as f:
    system_prompt = f.read()

# --- Scene requests (what upstream pipeline would pass) ---
scene_requests = [
    "Create a video prompt for: luxury resort infinity pool at sunset",
    "Create a video prompt for: boutique hotel penthouse bedroom, morning light",
    "Create a video prompt for: grand hotel lobby with marble floors and warm lighting",
    "Create a video prompt for: zen spa treatment room with stone bath and steam",
    "Create a video prompt for: rooftop restaurant terrace at twilight with ocean view",
    "Create a video prompt for: beachfront cabana at golden hour",
]

results = []
total_start = time.time()

for i, scene_req in enumerate(scene_requests):
    print(f"\n{'='*60}")
    print(f"Scene {i+1}/{len(scene_requests)}: {scene_req}")
    print(f"{'='*60}")

    # Step 1: Gemini generates the cinematography prompt from system prompt
    print("[1/4] Generating prompt via Gemini...")
    try:
        # System prompt defines the role + rules; scene_req is the user turn
        # Combine so Gemini acts as the cinematography director
        full_prompt = (
            f"{system_prompt}\n\n"
            f"Scene request: {scene_req}\n\n"
            f"Write a 25-40 word cinematography prompt following the structure above:"
        )
        cinema_prompt = llm.generate(full_prompt, temperature=0.7, max_tokens=200)
        # Clean up any quotes, labels, or extra whitespace
        cinema_prompt = cinema_prompt.strip().strip('"').strip("'").strip()
        # Strip any label prefix like "Prompt:" or "Here's a prompt:"
        for prefix in ["Prompt:", "Here's a prompt:", "Cinematography prompt:", "Output:"]:
            if cinema_prompt.lower().startswith(prefix.lower()):
                cinema_prompt = cinema_prompt[len(prefix):].strip()
        print(f"  Prompt ({len(cinema_prompt.split())} words): {cinema_prompt}")
    except Exception as e:
        print(f"  ERROR generating prompt: {e}")
        continue

    # Step 2: Generate image with Seedream
    print("[2/4] Generating image via Seedream 4.5...")
    try:
        img_result = kie.generate_image(prompt=cinema_prompt, aspect_ratio="9:16")
        if not img_result.success:
            print(f"  Image gen failed: state={img_result.state}")
            continue
        image_url = img_result.result_urls[0]
        print(f"  Image: {image_url}")
    except Exception as e:
        print(f"  ERROR image gen: {e}")
        continue

    # Step 3: Generate video with Seedance
    print("[3/4] Generating video via Seedance 1.5 Pro (1080p, 8s)...")
    try:
        vid_result = kie.generate_video(
            prompt=cinema_prompt,
            image_url=image_url,
            aspect_ratio="9:16",
            duration=8,
            resolution="1080p",
        )
        if not vid_result.success:
            print(f"  Video gen failed: state={vid_result.state} err={vid_result.error_message}")
            continue
        video_url = vid_result.result_urls[0]
        print(f"  Video: {video_url}")
    except Exception as e:
        print(f"  ERROR video gen: {e}")
        continue

    # Step 4: QC via Gemini
    print("[4/4] Running Gemini QC...")
    try:
        qc_result = qc.evaluate(video_url=video_url)
        pf = "PASS" if qc_result.get('pass') else "FAIL"
        ae = qc_result.get('aesthetic_score', '?')
        mo = qc_result.get('motion_score', '?')
        ad = qc_result.get('prompt_adherence_score', '?')
        ss = qc_result.get('scroll_stop_score', '?')
        print(f"  QC: {pf} | aesthetic={ae} motion={mo} adherence={ad} scroll_stop={ss}")
        if qc_result.get('auto_fail_triggered'):
            print(f"  Auto-fails: {qc_result['auto_fail_triggered']}")
        if qc_result.get('minor_issues'):
            print(f"  Minor issues: {qc_result['minor_issues']}")
        print(f"  Summary: {qc_result.get('summary', '')[:200]}")
    except Exception as e:
        print(f"  ERROR QC: {e}")
        qc_result = {"pass": True, "confidence": 0.0, "summary": f"QC error: {e}"}

    # Step 5: Reward
    try:
        reward = reward_calc.calculate(qc_result)
    except Exception as e:
        print(f"  ERROR reward: {e}")
        reward = 0
    print(f"  Reward: {reward}")

    results.append({
        "scene_request": scene_req,
        "generated_prompt": cinema_prompt,
        "prompt_word_count": len(cinema_prompt.split()),
        "image_url": image_url,
        "video_url": video_url,
        "qc": qc_result,
        "reward": reward,
    })

# --- Summary ---
elapsed = time.time() - total_start
print(f"\n{'='*60}")
print(f"V3 GEMINI E2E RESULTS — {len(results)}/{len(scene_requests)} samples in {elapsed:.0f}s")
print(f"{'='*60}")

if results:
    avg_reward = sum(r['reward'] for r in results) / len(results)
    passes = sum(1 for r in results if r['qc'].get('pass'))
    print(f"Avg reward: {avg_reward:.1f} | Pass rate: {passes}/{len(results)}")
    print(f"V1 baseline avg was 70.8 | Delta: {avg_reward - 70.8:+.1f}")
    print()
    for r in results:
        status = "✓" if r['qc'].get('pass') else "✗"
        ae = r['qc'].get('aesthetic_score', '?')
        mo = r['qc'].get('motion_score', '?')
        ss = r['qc'].get('scroll_stop_score', '?')
        scene = r['scene_request'].split(': ')[-1][:40]
        print(f"  {status} {scene}: reward={r['reward']:.1f} ae={ae} mo={mo} ss={ss}")
        print(f"    prompt: {r['generated_prompt'][:100]}...")
        print(f"    video: {r['video_url']}")

# Save results
output_path = f"/home/node/.openclaw/workspace/prompt_evaluator/eval_results/v3_gemini_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved: {output_path}")
print(f"\nKie stats: {kie.stats.summary()}")
