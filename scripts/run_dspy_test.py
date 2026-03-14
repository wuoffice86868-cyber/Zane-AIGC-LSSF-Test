#!/usr/bin/env python3
"""Test DSPy optimizer with real APIs.

This is a lightweight test: use DSPy's prompt generation module
with Gemini, but DON'T generate videos (expensive). Instead, test:
1. Does DSPy + Gemini produce valid prompts from scene inputs?
2. Can the critique module analyze prompt quality?
3. Can the template improver suggest changes?

Video generation + scoring will be tested separately once
we confirm the DSPy→Gemini pipeline works.
"""

import json
import os
import sys
import time

# Ensure project is importable
sys.path.insert(0, os.path.dirname(__file__))

from prompt_evaluator.dspy_optimizer import (
    SceneInput,
    VideoPromptOptimizer,
    SEEDANCE_CONSTRAINTS,
    HAS_DSPY,
)

def load_gemini_key():
    """Load Gemini API key from credentials file."""
    paths = [
        os.path.expanduser("~/.openclaw/workspace/.gemini_credentials.json"),
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                data = json.load(f)
                return data.get("api_key", "")
    return os.environ.get("GEMINI_API_KEY", "")


def test_prompt_generation():
    """Test 1: Generate prompts from structured scene inputs using DSPy + Gemini."""
    import dspy
    
    api_key = load_gemini_key()
    if not api_key:
        print("ERROR: No Gemini API key found")
        return False
    
    # Configure DSPy with Gemini
    lm = dspy.LM("gemini/gemini-2.5-flash", api_key=api_key)
    dspy.configure(lm=lm)
    
    # Import the module
    from prompt_evaluator.dspy_optimizer import VideoPromptModule
    
    module = VideoPromptModule(use_cot=True)
    
    # Load the cinematography template
    template_path = os.path.join(os.path.dirname(__file__), "prompts", "gemini_cinematography_prompt.txt")
    template = ""
    if os.path.exists(template_path):
        with open(template_path) as f:
            template = f.read()
    
    # Test scenes
    scenes = [
        SceneInput(
            scene_description="luxury hotel infinity pool at sunset",
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
            scene_description="boutique hotel lobby with high ceilings",
            main_subject="grand reception desk with marble counter",
            foreground="floral arrangement on side table",
            background="staircase with wrought iron railing",
            camera_move="pull",
            camera_direction="back revealing the full lobby",
            shot_size="medium",
            lighting="soft ambient light from chandeliers",
            subtle_motion=[],
            stable_element="marble floor and columns",
        ),
        SceneInput(
            scene_description="hotel spa treatment room",
            main_subject="massage table with white linens",
            foreground="row of candles along the edge",
            background="bamboo wall with soft backlight",
            camera_move="circle around",
            camera_direction="counterclockwise around the table",
            shot_size="close-up",
            lighting="soft warm candlelight",
            subtle_motion=["candle flames flicker gently", "steam rises softly"],
            stable_element="bamboo wall",
        ),
    ]
    
    print("=" * 60)
    print("TEST 1: Prompt Generation (DSPy + Gemini)")
    print("=" * 60)
    
    results = []
    for i, scene in enumerate(scenes):
        print(f"\nScene {i+1}: {scene.scene_description}")
        try:
            start = time.time()
            prediction = module(
                scene_json=scene.to_json(),
                template_instructions=template or "Generate a 30-45 word cinematic video prompt for Seedance.",
            )
            elapsed = time.time() - start
            
            prompt = prediction.video_prompt
            word_count = len(prompt.split())
            
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Words: {word_count}")
            print(f"  Prompt: {prompt}")
            
            # Basic validation
            issues = []
            if word_count < 15:
                issues.append("too short (<15 words)")
            if word_count > 60:
                issues.append("too long (>60 words)")
            if not any(cam in prompt.lower() for cam in ["camera", "pushes", "pulls", "circles", "moves", "pans", "rises", "tilts", "drifts"]):
                issues.append("no camera movement detected")
            
            if issues:
                print(f"  Issues: {', '.join(issues)}")
            else:
                print(f"  ✓ Valid")
            
            results.append({
                "scene": scene.scene_description,
                "prompt": prompt,
                "word_count": word_count,
                "time": elapsed,
                "valid": len(issues) == 0,
                "issues": issues,
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "scene": scene.scene_description,
                "prompt": "",
                "error": str(e),
                "valid": False,
            })
    
    valid = sum(1 for r in results if r.get("valid"))
    print(f"\nResults: {valid}/{len(results)} valid prompts")
    return results


def test_critique():
    """Test 2: Critique a prompt based on quality scores."""
    import dspy
    
    api_key = load_gemini_key()
    lm = dspy.LM("gemini/gemini-2.5-flash", api_key=api_key)
    dspy.configure(lm=lm)
    
    from prompt_evaluator.dspy_optimizer import CritiqueModule
    
    module = CritiqueModule()
    
    # Mock a bad result
    scene = SceneInput(
        scene_description="hotel lobby with grand chandelier",
        main_subject="crystal chandelier",
        camera_move="push",
        camera_direction="forward into the lobby",
        shot_size="wide",
        lighting="ambient chandelier light",
        subtle_motion=["crystal reflections shimmer"],
        stable_element="marble floor",
    )
    
    bad_prompt = "Single continuous shot. Wide shot of a grand hotel lobby. The camera pushes forward toward the crystal chandelier, light shimmering through the crystals. Marble floor stays perfectly still."
    
    scores = {
        "aesthetic": 3,
        "motion": 4,
        "adherence": 9,
        "pass": False,
    }
    fails = ["object_morphing", "structural_collapse"]
    
    print("\n" + "=" * 60)
    print("TEST 2: Prompt Critique")
    print("=" * 60)
    print(f"Prompt: {bad_prompt}")
    print(f"Scores: {scores}")
    print(f"Auto-fails: {fails}")
    
    try:
        start = time.time()
        result = module(
            video_prompt=bad_prompt,
            scene_json=scene.to_json(),
            quality_scores=json.dumps(scores),
            auto_fails=json.dumps(fails),
        )
        elapsed = time.time() - start
        
        print(f"\nCritique ({elapsed:.1f}s):")
        print(result.critique)
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_template_improvement():
    """Test 3: Improve template based on evidence."""
    import dspy
    
    api_key = load_gemini_key()
    lm = dspy.LM("gemini/gemini-2.5-flash", api_key=api_key)
    dspy.configure(lm=lm)
    
    from prompt_evaluator.dspy_optimizer import TemplateImproverModule
    
    module = TemplateImproverModule()
    
    # Load current template
    template_path = os.path.join(os.path.dirname(__file__), "prompts", "gemini_cinematography_prompt.txt")
    template = ""
    if os.path.exists(template_path):
        with open(template_path) as f:
            template = f.read()
    
    success_patterns = """Score 94: Wide shot of a luxury hotel pool at golden hour, calm turquoise water. The camera pushes slowly forward past rattan chairs, water ripples softly under warm sunset light.
Score 92: Close-up of boutique hotel bed with white linens. The camera drifts right gently along the headboard, afternoon sun streaming through sheer curtains.
Score 90: Medium shot of a modern hotel exterior at dusk. The camera rises slowly along the glass facade, city lights emerging in the background."""
    
    failure_patterns = """Score 42: Wide shot of a grand hotel lobby. The camera pushes forward toward the crystal chandelier, light shimmering through the crystals. | Fails: ['object_morphing', 'structural_collapse']
Score 49: Wide shot of a tropical beach with palm trees and lounging guests. The camera pans right along the shoreline, waves crashing gently. | Fails: ['object_morphing', 'action_loop']
Score 42: Medium shot of a rooftop restaurant at sunset. The camera circles around a candlelit table for two, city skyline glowing behind. | Fails: ['object_morphing', 'structural_collapse']"""
    
    print("\n" + "=" * 60)
    print("TEST 3: Template Improvement")
    print("=" * 60)
    
    try:
        start = time.time()
        result = module(
            current_template=template[:2000],  # Truncate for context window
            success_patterns=success_patterns,
            failure_patterns=failure_patterns,
            model_constraints=SEEDANCE_CONSTRAINTS,
        )
        elapsed = time.time() - start
        
        improved = result.improved_template
        print(f"\nImproved template ({elapsed:.1f}s, {len(improved)} chars):")
        print(improved[:1500])
        if len(improved) > 1500:
            print(f"... [{len(improved) - 1500} more chars]")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


if __name__ == "__main__":
    if not HAS_DSPY:
        print("ERROR: DSPy not installed")
        sys.exit(1)
    
    print("DSPy Video Prompt Optimizer - Live Test")
    print(f"DSPy version: {__import__('dspy').__version__}")
    print()
    
    # Test 1: Prompt generation
    gen_results = test_prompt_generation()
    
    # Test 2: Critique
    critique_ok = test_critique()
    
    # Test 3: Template improvement
    improve_ok = test_template_improvement()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if gen_results:
        valid = sum(1 for r in gen_results if r.get("valid"))
        print(f"Prompt generation: {valid}/{len(gen_results)} valid")
    print(f"Critique: {'✓' if critique_ok else '✗'}")
    print(f"Template improvement: {'✓' if improve_ok else '✗'}")
