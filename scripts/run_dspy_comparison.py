#!/usr/bin/env python3
"""Run a head-to-head comparison: v3 baseline vs DSPy-improved template.

Generates videos for 6 scenes using both templates, scores with Gemini QC,
and outputs a comparison table.

This is the real test: does DSPy actually improve video quality?
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from prompt_evaluator.kie_client import KieClient
from prompt_evaluator.gemini_client import GeminiVideoQC, GeminiLLM
from prompt_evaluator.reward_calculator import RewardCalculator
from prompt_evaluator.dspy_optimizer import SceneInput, SEEDANCE_CONSTRAINTS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCENES = [
    SceneInput(
        scene_description="luxury hotel infinity pool at golden sunset",
        main_subject="infinity pool with calm turquoise water",
        foreground="two rattan lounge chairs",
        background="ocean horizon at golden hour",
        camera_move="push",
        camera_direction="forward toward the pool edge",
        shot_size="wide",
        lighting="golden sunset light, warm tones",
        subtle_motion=["water ripples softly", "palm fronds sway gently"],
        stable_element="pool edge tiles",
    ),
    SceneInput(
        scene_description="boutique hotel room with king bed",
        main_subject="king bed with white linens and throw pillows",
        foreground="bedside lamp on wooden nightstand",
        background="floor-to-ceiling window with city skyline",
        camera_move="push",
        camera_direction="forward toward the bed",
        shot_size="medium",
        lighting="warm afternoon sunlight from window",
        subtle_motion=["sheer curtains sway gently"],
        stable_element="wooden furniture",
    ),
    SceneInput(
        scene_description="grand hotel lobby with high ceilings",
        main_subject="reception desk with marble counter",
        foreground="floral arrangement on side table",
        background="grand staircase with wrought iron railing",
        camera_move="pull",
        camera_direction="back revealing the full lobby",
        shot_size="wide",
        lighting="soft ambient chandelier light",
        subtle_motion=[],
        stable_element="marble floor and columns",
    ),
    SceneInput(
        scene_description="hotel rooftop restaurant at golden hour",
        main_subject="candlelit table for two with white tablecloth",
        foreground="wine glasses catching light",
        background="city skyline at dusk",
        camera_move="circle around",
        camera_direction="slowly around the table",
        shot_size="medium",
        lighting="warm golden hour mixed with candlelight",
        subtle_motion=["candle flame flickers gently"],
        stable_element="table and chairs",
    ),
    SceneInput(
        scene_description="hotel spa with freestanding bathtub",
        main_subject="freestanding marble bathtub",
        foreground="row of candles along the tub edge",
        background="floor-to-ceiling window with garden view",
        camera_move="push",
        camera_direction="forward toward the bathtub",
        shot_size="close-up",
        lighting="soft diffused daylight",
        subtle_motion=["steam rises gently", "candle flames flicker"],
        stable_element="marble tub and tile floor",
    ),
    SceneInput(
        scene_description="tropical beach cabana at sunset",
        main_subject="private beach cabana with white curtains",
        foreground="sandy path with footprints",
        background="calm ocean with warm sunset reflections",
        camera_move="push",
        camera_direction="forward along the sandy path",
        shot_size="wide",
        lighting="warm sunset backlighting through curtains",
        subtle_motion=["curtains billow gently", "waves lap softly"],
        stable_element="cabana frame",
    ),
]

# v3 template (our current best baseline)
V3_TEMPLATE = """You are a cinematography prompt writer for Seedance 1.5 Pro video generation.

Given a structured scene JSON, output a natural language video prompt following these rules:

STRUCTURE (4-section, this order):
1. Subject + key visual details (what the viewer sees first)
2. Scene context + lighting (set the mood)
3. Motion elements (what moves naturally — water, fabric, flame, steam)
4. Camera movement with degree adverbs (slowly, gently, gradually)

RULES:
- 25-40 words total
- Start with the subject, NOT "Single continuous shot"
- Always include a degree adverb with camera movement (slowly, gently, gradually)
- Use Seedance camera verbs: pushes, pulls back, circles around, moves left/right, pans, rises, tilts, drifts
- Do NOT use negative constraints ("no warping", "stays still", "remains fixed") — Seedance ignores these
- Do NOT use equipment names (gimbal, steadicam, drone, crane)
- Do NOT add stability anchors — they waste words and don't work
- Focus on DESCRIBING what's beautiful, not on constraining what could go wrong
- For complex scenes (lobby, chandelier, crowds): simplify — describe fewer elements, focus on one clear subject

OUTPUT: Just the prompt text, nothing else.
"""


def load_credentials():
    """Load API credentials."""
    kie_path = os.path.expanduser("~/.openclaw/workspace/.kie_credentials")
    gemini_path = os.path.expanduser("~/.openclaw/workspace/.gemini_credentials.json")
    
    kie_key = ""
    if os.path.exists(kie_path):
        with open(kie_path) as f:
            data = json.load(f)
            kie_key = data.get("api_key", "")
    
    gemini_key = ""
    if os.path.exists(gemini_path):
        with open(gemini_path) as f:
            data = json.load(f)
            gemini_key = data.get("api_key", "")
    
    return kie_key, gemini_key


def generate_dspy_improved_template(gemini_key: str) -> str:
    """Use DSPy TemplateImproverModule to generate an improved template."""
    import dspy
    from prompt_evaluator.dspy_optimizer import TemplateImproverModule
    
    lm = dspy.LM("gemini/gemini-2.5-flash", api_key=gemini_key)
    dspy.configure(lm=lm)
    
    module = TemplateImproverModule()
    
    # Use our actual test data as evidence
    success_patterns = """Score 94: Wide shot of a luxury hotel pool at golden hour, calm turquoise water. The camera pushes slowly forward past rattan chairs, water ripples softly under warm sunset light.
Score 94: Close-up of boutique hotel bed with white linens. The camera drifts right gently along the headboard, afternoon sun streaming through sheer curtains. Fabric sways softly.
Score 92: Medium shot of hotel exterior at dusk. The camera rises slowly along the glass facade, city lights glowing warm in the background.
Score 90: Freestanding marble bathtub in soft daylight. The camera pushes gently forward, steam rising, candle flames flickering. Warm diffused light fills the space.
Score 86: Private beach cabana at sunset, white curtains catching warm light. The camera drifts slowly along the sandy path, waves lapping gently, ocean glowing amber."""
    
    failure_patterns = """Score 42: Wide shot of a grand hotel lobby with crystal chandelier. The camera pushes forward into the space, chandelier light shimmering. | Fails: ['object_morphing', 'structural_collapse'] — chandelier crystals warped, ceiling geometry collapsed
Score 49: Tropical beach with palm trees and lounging guests. The camera pans right along the shoreline, waves crashing. | Fails: ['object_morphing', 'action_loop'] — palm trees morphed, waves repeated identically
Score 42: Rooftop restaurant at sunset, candlelit table. The camera circles around the table, city skyline behind. | Fails: ['object_morphing', 'structural_collapse'] — furniture warped during orbit
Score 55: Hotel suite with complex furniture arrangement. The camera pushes through the room. | Fails: ['structural_collapse'] — ceiling and window frames deformed
Score 56: Marble bathroom with mirror and fixtures. The camera moves along the vanity. | Fails: ['reflection_error', 'object_morphing'] — mirror reflection inconsistent, faucet morphed"""
    
    result = module(
        current_template=V3_TEMPLATE,
        success_patterns=success_patterns,
        failure_patterns=failure_patterns,
        model_constraints=SEEDANCE_CONSTRAINTS,
    )
    
    return result.improved_template


def generate_prompt_with_template(template: str, scene: SceneInput, gemini_key: str) -> str:
    """Generate a video prompt using the given template + scene input via Gemini."""
    import dspy
    from prompt_evaluator.dspy_optimizer import VideoPromptModule
    
    lm = dspy.LM("gemini/gemini-2.5-flash", api_key=gemini_key)
    dspy.configure(lm=lm)
    
    module = VideoPromptModule(use_cot=False)  # No CoT for speed
    result = module(
        scene_json=scene.to_json(),
        template_instructions=template,
    )
    return result.video_prompt


def run_comparison():
    """Run the full comparison."""
    kie_key, gemini_key = load_credentials()
    
    if not kie_key or not gemini_key:
        print("ERROR: Missing API credentials")
        return
    
    print(f"Start time: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
    print()
    
    # Step 1: Generate DSPy-improved template
    print("=" * 60)
    print("STEP 1: Generating DSPy-improved template...")
    print("=" * 60)
    
    dspy_template = generate_dspy_improved_template(gemini_key)
    print(f"Generated template ({len(dspy_template)} chars)")
    print(dspy_template[:500])
    print("...")
    print()
    
    # Save it
    with open("prompts/dspy_improved_template.txt", "w") as f:
        f.write(dspy_template)
    
    # Step 2: Generate prompts for all scenes using both templates
    print("=" * 60)
    print("STEP 2: Generating prompts (v3 baseline + DSPy improved)")
    print("=" * 60)
    
    v3_prompts = []
    dspy_prompts = []
    
    for i, scene in enumerate(SCENES):
        print(f"\nScene {i+1}: {scene.scene_description}")
        
        # v3
        p_v3 = generate_prompt_with_template(V3_TEMPLATE, scene, gemini_key)
        v3_prompts.append(p_v3)
        print(f"  v3:   {p_v3[:100]}...")
        
        # DSPy
        p_dspy = generate_prompt_with_template(dspy_template, scene, gemini_key)
        dspy_prompts.append(p_dspy)
        print(f"  dspy: {p_dspy[:100]}...")
    
    # Step 3: Generate images for each scene (shared between both versions)
    print()
    print("=" * 60)
    print("STEP 3: Generating source images (6 scenes)")
    print("=" * 60)
    
    kie = KieClient(api_key=kie_key)
    images = []
    
    for i, scene in enumerate(SCENES):
        print(f"\nScene {i+1}: generating image...")
        img_prompt = f"Professional architectural photography of {scene.scene_description}, 9:16 vertical, photorealistic, high detail"
        result = kie.generate_image(img_prompt, aspect_ratio="9:16")
        
        if result.success and result.result_urls:
            images.append(result.result_urls[0])
            print(f"  ✓ {result.result_urls[0][:80]}...")
        else:
            images.append("")
            print(f"  ✗ Failed: {result.state}")
    
    # Step 4: Generate videos (12 total: 6 scenes × 2 versions)
    print()
    print("=" * 60)
    print("STEP 4: Generating videos (12 total)")
    print("=" * 60)
    
    results = []
    
    for i, scene in enumerate(SCENES):
        if not images[i]:
            print(f"\nScene {i+1}: skipped (no image)")
            results.append({"scene": scene.scene_description, "v3": None, "dspy": None})
            continue
        
        print(f"\nScene {i+1}: {scene.scene_description}")
        scene_result = {"scene": scene.scene_description}
        
        for version, prompt in [("v3", v3_prompts[i]), ("dspy", dspy_prompts[i])]:
            print(f"  {version}: generating video...")
            vid = kie.generate_video(
                prompt=prompt,
                image_url=images[i],
                duration=8,
                resolution="1080p",
                aspect_ratio="9:16",
            )
            
            if vid.success and vid.result_urls:
                scene_result[version] = {
                    "prompt": prompt,
                    "video_url": vid.result_urls[0],
                }
                print(f"  {version}: ✓ {vid.result_urls[0][:60]}...")
            else:
                scene_result[version] = {"prompt": prompt, "video_url": "", "error": vid.state}
                print(f"  {version}: ✗ {vid.state}")
        
        results.append(scene_result)
    
    # Step 5: Score all videos with Gemini QC
    print()
    print("=" * 60)
    print("STEP 5: Scoring videos with Gemini QC (2.5 Pro)")
    print("=" * 60)
    
    qc = GeminiVideoQC(api_key=gemini_key, model="gemini-2.5-pro")
    reward_calc = RewardCalculator()
    
    for i, r in enumerate(results):
        print(f"\nScene {i+1}: {r['scene']}")
        
        for version in ["v3", "dspy"]:
            if r.get(version) and r[version].get("video_url"):
                print(f"  {version}: scoring...")
                try:
                    qc_result = qc.evaluate_video(
                        video_url=r[version]["video_url"],
                        original_prompt=r[version]["prompt"],
                    )
                    reward = reward_calc.calculate_from_qc_dict(qc_result)
                    r[version]["qc"] = qc_result
                    r[version]["reward"] = reward.total_score
                    r[version]["pass"] = qc_result.get("pass", False)
                    print(f"  {version}: reward={reward.total_score:.1f} pass={qc_result.get('pass', '?')}")
                except Exception as e:
                    r[version]["reward"] = 0
                    r[version]["error"] = str(e)
                    print(f"  {version}: ERROR {e}")
    
    # Step 6: Summary
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Scene':<35} {'v3':>8} {'DSPy':>8} {'Delta':>8}")
    print("-" * 65)
    
    v3_scores = []
    dspy_scores = []
    
    for r in results:
        scene = r["scene"][:34]
        v3_score = r.get("v3", {}).get("reward", 0) if r.get("v3") else 0
        dspy_score = r.get("dspy", {}).get("reward", 0) if r.get("dspy") else 0
        delta = dspy_score - v3_score
        
        v3_scores.append(v3_score)
        dspy_scores.append(dspy_score)
        
        sign = "+" if delta > 0 else ""
        print(f"{scene:<35} {v3_score:>7.1f} {dspy_score:>7.1f} {sign}{delta:>7.1f}")
    
    v3_avg = sum(v3_scores) / len(v3_scores) if v3_scores else 0
    dspy_avg = sum(dspy_scores) / len(dspy_scores) if dspy_scores else 0
    delta_avg = dspy_avg - v3_avg
    
    print("-" * 65)
    sign = "+" if delta_avg > 0 else ""
    print(f"{'AVERAGE':<35} {v3_avg:>7.1f} {dspy_avg:>7.1f} {sign}{delta_avg:>7.1f}")
    
    # Save full results
    with open("eval_results/dspy_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nFull results saved to eval_results/dspy_comparison.json")
    print(f"End time: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")


if __name__ == "__main__":
    run_comparison()
