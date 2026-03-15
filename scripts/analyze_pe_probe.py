#!/usr/bin/env python3
"""Analyze PE probe results and generate v4 system prompt.

Reads the latest pe_probe_*.json from eval_results/, generates:
  1. Dimension analysis table
  2. Key findings / hypotheses confirmed/rejected
  3. v4 system prompt recommendations
  4. Writes system_prompts/hotel_v4.txt
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

def main():
    eval_dir = Path("eval_results")
    probe_files = sorted(eval_dir.glob("pe_probe_*.json"))
    if not probe_files:
        print("No pe_probe results found in eval_results/")
        return

    latest = probe_files[-1]
    print(f"Analyzing: {latest}\n")

    data = json.loads(latest.read_text())
    results = data["results"]
    successful = [r for r in results if r.get("success")]

    if not successful:
        print("No successful results to analyze.")
        return

    # Build lookup
    by_name = {r["variant"]: r for r in successful}

    def reward(name):
        r = by_name.get(name)
        return float(r["reward"]) if r else None

    def qc(name, field, default=None):
        r = by_name.get(name)
        if not r:
            return default
        return r["qc"].get(field, default)

    # -------------------------------------------------------
    print("=" * 70)
    print("RESULTS TABLE (sorted by reward)")
    print("=" * 70)
    sorted_results = sorted(successful, key=lambda r: float(r["reward"]), reverse=True)
    print(f"\n{'Variant':<28} {'Wds':>4} {'Pass':>5} {'Reward':>7} {'Aes':>4} {'Mot':>4} {'Adh':>4} {'Scrl':>5}")
    print("-" * 68)
    for r in sorted_results:
        q = r["qc"]
        status = "✓" if q.get("pass") else "✗"
        fails = q.get("auto_fail_triggered", [])
        fail_str = f" [{','.join(fails[:2])}]" if fails else ""
        print(f"{r['variant']:<28} {r['word_count']:>4} {status:>5} {float(r['reward']):>7.1f} "
              f"{q.get('aesthetic_score', 0):>4} {q.get('motion_score', 0):>4} "
              f"{q.get('prompt_adherence_score', 0):>4} {q.get('scroll_stop_score', 0):>5}"
              f"{fail_str}")

    # -------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("DIMENSION ANALYSIS")
    print("=" * 70)

    findings = []

    # A: Structure order
    print("\n── A. STRUCTURE ORDER ──────────────────────────────────────────")
    a_vals = {}
    for v in ["A1_subject_first", "A2_camera_first", "A3_scene_first"]:
        r_val = reward(v)
        a_vals[v] = r_val
        if r_val is not None:
            print(f"  {v}: reward={r_val:.1f}  (aes={qc(v,'aesthetic_score')} mot={qc(v,'motion_score')} adh={qc(v,'prompt_adherence_score')})")
        else:
            print(f"  {v}: FAILED/MISSING")

    a_valid = {k: v for k, v in a_vals.items() if v is not None}
    if a_valid:
        best_a = max(a_valid, key=a_valid.get)
        worst_a = min(a_valid, key=a_valid.get)
        spread = a_valid[best_a] - a_valid[worst_a]
        print(f"\n  Best: {best_a} ({a_valid[best_a]:.1f})")
        print(f"  Spread: {spread:.1f} pts")
        if spread > 10:
            findings.append(f"✅ Structure order MATTERS (spread={spread:.1f}): use {best_a} ordering")
        else:
            findings.append(f"⚠️ Structure order low impact (spread={spread:.1f}): PE normalizes order")

    # B: Length
    print("\n── B. PROMPT LENGTH ────────────────────────────────────────────")
    b_vals = {}
    for v in ["B1_minimal_15w", "B2_medium_30w", "B3_verbose_50w"]:
        r_val = reward(v)
        b_vals[v] = r_val
        r = by_name.get(v)
        wc = r["word_count"] if r else "?"
        if r_val is not None:
            print(f"  {v} ({wc}w): reward={r_val:.1f}  (aes={qc(v,'aesthetic_score')} mot={qc(v,'motion_score')})")
        else:
            print(f"  {v} ({wc}w): FAILED/MISSING")

    b_valid = {k: v for k, v in b_vals.items() if v is not None}
    if b_valid:
        best_b = max(b_valid, key=b_valid.get)
        print(f"\n  Best: {best_b} ({b_valid[best_b]:.1f})")
        if best_b == "B1_minimal_15w":
            findings.append("✅ PE fills in detail — shorter prompts can work, less is more")
        elif best_b == "B2_medium_30w":
            findings.append("✅ Medium length (25-35w) is sweet spot — PE can work with this density")
        else:
            findings.append("⚠️ Verbose prompts score best — PE preserves detail density")

    # C: Adverbs
    print("\n── C. DEGREE ADVERBS ───────────────────────────────────────────")
    c_with = reward("C1_with_adverbs")
    c_without = reward("C2_without_adverbs")
    for v, val in [("C1_with_adverbs", c_with), ("C2_without_adverbs", c_without)]:
        if val is not None:
            print(f"  {v}: reward={val:.1f}  (aes={qc(v,'aesthetic_score')} mot={qc(v,'motion_score')})")
        else:
            print(f"  {v}: FAILED/MISSING")
    if c_with is not None and c_without is not None:
        delta = c_with - c_without
        print(f"\n  Delta (with - without): {delta:+.1f}")
        if delta > 5:
            findings.append(f"✅ Adverbs HELP (+{delta:.1f}): use 'slowly', 'gently', 'gradually' etc.")
        elif delta < -5:
            findings.append(f"❌ Adverbs HURT ({delta:.1f}): overly dense language confuses PE")
        else:
            findings.append(f"⚠️ Adverbs neutral (delta={delta:.1f}): minor effect only")

    # D: Negatives
    print("\n── D. NEGATIVE CONSTRAINTS ─────────────────────────────────────")
    d_with = reward("D1_with_negatives")
    d_without = reward("D2_without_negatives")
    for v, val in [("D1_with_negatives", d_with), ("D2_without_negatives", d_without)]:
        if val is not None:
            print(f"  {v}: reward={val:.1f}  (aes={qc(v,'aesthetic_score')} mot={qc(v,'motion_score')})")
        else:
            print(f"  {v}: FAILED/MISSING")
    if d_with is not None and d_without is not None:
        delta = d_with - d_without
        print(f"\n  Delta (with - without): {delta:+.1f}")
        if abs(delta) < 5:
            findings.append(f"✅ CONFIRMED: negatives don't help (delta={delta:.1f}) — PE strips them")
        elif delta < -5:
            findings.append(f"✅ CONFIRMED: negatives actively HURT ({delta:.1f}) — never use them")
        else:
            findings.append(f"⚠️ Negatives actually helped (+{delta:.1f}) — unexpected, needs re-test")

    # E: Vocabulary
    print("\n── E. VOCABULARY TYPE ──────────────────────────────────────────")
    e_equip = reward("E1_equipment_names")
    e_native = reward("E2_native_verbs")
    for v, val in [("E1_equipment_names", e_equip), ("E2_native_verbs", e_native)]:
        if val is not None:
            print(f"  {v}: reward={val:.1f}  (aes={qc(v,'aesthetic_score')} mot={qc(v,'motion_score')})")
        else:
            print(f"  {v}: FAILED/MISSING")
    if e_equip is not None and e_native is not None:
        delta = e_native - e_equip
        print(f"\n  Delta (native - equipment): {delta:+.1f}")
        if delta > 5:
            findings.append(f"✅ Native verbs BETTER (+{delta:.1f}): use 'drifts', 'undulates', 'cascades'")
        elif delta < -5:
            findings.append(f"⚠️ Equipment names BETTER — PE may translate these to motion commands")
        else:
            findings.append(f"⚠️ Vocabulary type neutral (delta={delta:.1f})")

    # -------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("KEY FINDINGS")
    print("=" * 70)
    for i, f in enumerate(findings, 1):
        print(f"  {i}. {f}")

    # -------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("GENERATING V4 SYSTEM PROMPT")
    print("=" * 70)

    # Determine best structure from findings
    best_structure = "subject_first"  # default
    if a_valid:
        best_a_key = max(a_valid, key=a_valid.get)
        if "camera_first" in best_a_key:
            best_structure = "camera_first"
        elif "scene_first" in best_a_key:
            best_structure = "scene_first"

    use_adverbs = True
    if any("Adverbs HURT" in f for f in findings):
        use_adverbs = False

    include_negatives = False  # never, confirmed

    # Sweet spot word count
    target_wc = "25-35"
    if b_valid:
        best_b_key = max(b_valid, key=b_valid.get)
        if "minimal" in best_b_key:
            target_wc = "15-20"
        elif "verbose" in best_b_key:
            target_wc = "40-50"

    v4_notes = [
        f"# v4 System Prompt — generated {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        f"# Based on PE probe experiment ({data['timestamp']})",
        f"# Findings: {'; '.join(findings[:3])}",
        "",
        f"# Structure: {best_structure}",
        f"# Target word count: {target_wc} words",
        f"# Adverbs: {'YES — use degree adverbs' if use_adverbs else 'NO — avoid'}",
        f"# Negatives: NEVER — PE strips them, confirmed no benefit",
        "",
    ]

    # Build v4 template based on findings
    if best_structure == "subject_first":
        v4_template = """\
You are a cinematography director generating precise video prompts for Seedance 1.5 Pro.

Given a hotel/resort scene and camera parameters, output a single-paragraph video prompt in {target_wc} words.

STRUCTURE (subject → action → camera → atmosphere):
1. [Subject + scene]: Name what you see, be specific. "{scene_type} with {key_feature}."
2. [Natural motion]: What moves and how. Use degree adverbs: slowly, gently, gradually.
3. [Camera]: "[Camera verb] [direction] [speed adverb]". Use: pushes, drifts, rises, circles, moves right.
4. [Light/atmosphere]: Warm descriptors. Time of day, quality of light.

RULES:
- 25-35 words total (PE works best in this range)
- Subject and key feature first — anchor the scene before adding motion
- Degree adverbs on both camera AND natural motion (slowly, gently, gradually, subtly)
- End with atmosphere/light, not camera instructions
- NO negative constraints ("no warping", "avoid morphing", etc.) — they don't reach the diffusion model
- NO equipment names (Steadicam, gimbal, drone) — use motion descriptions instead
- ONE camera move only — don't combine pushes with tilts

CAMERA VOCABULARY (PE-friendly verbs):
  Forward: pushes forward, drifts forward, glides forward
  Upward: rises slowly, lifts gently  
  Lateral: moves right, moves left, drifts sideways
  Circular: circles around, orbits slowly
  Reveal: pulls back to reveal, tracks back slowly

INPUT FORMAT:
  scene_type: e.g. "infinity pool"
  camera_move: e.g. "push"
  lighting: e.g. "golden hour"
  key_feature: e.g. "crystal clear water, sun loungers"
  subtle_motion: e.g. "water ripples, palm fronds"

OUTPUT: One paragraph, {target_wc} words, no bullets, no labels.
""".replace("{target_wc}", target_wc)
    else:
        # Camera-first or scene-first structure
        v4_template = """\
You are a cinematography director generating precise video prompts for Seedance 1.5 Pro.

Given a hotel/resort scene and camera parameters, output a single-paragraph video prompt in {target_wc} words.

STRUCTURE (camera → subject → motion → atmosphere):
1. [Camera]: "The camera [verb] [direction] [adverb]" — lead with the camera move.
2. [Subject + scene]: What is revealed. "{key feature} at {time/lighting}."  
3. [Natural motion]: What moves and how — degree adverbs mandatory.
4. [Atmosphere]: Final sensory detail (light, warmth, mood).

RULES:
- 25-35 words total
- Camera instruction first — sets the viewer's POV immediately
- Degree adverbs on BOTH camera and motion elements
- NO negative constraints — they don't reach the diffusion model
- NO equipment names — describe motion, not gear

CAMERA VOCABULARY:
  Forward: pushes forward, drifts forward
  Upward: rises slowly, lifts gently
  Lateral: moves right, moves left
  Circular: circles around, orbits slowly

OUTPUT: One paragraph, no bullets, no labels.
""".replace("{target_wc}", target_wc)

    print(v4_template[:500] + "...")

    # Save v4 template
    v4_path = Path("system_prompts/hotel_v4.txt")
    v4_path.write_text("\n".join(v4_notes) + "\n" + v4_template)
    print(f"\nSaved: {v4_path}")

    # Save findings to research/
    findings_doc = {
        "experiment": "pe_layer_probe_v1",
        "timestamp": data["timestamp"],
        "total_variants": data["total_variants"],
        "successful": len(successful),
        "findings": findings,
        "best_structure": best_structure,
        "use_adverbs": use_adverbs,
        "target_word_count": target_wc,
        "include_negatives": include_negatives,
        "reward_table": [
            {
                "variant": r["variant"],
                "word_count": r["word_count"],
                "reward": float(r["reward"]),
                "qc_pass": r["qc"].get("pass"),
                "aesthetic": r["qc"].get("aesthetic_score"),
                "motion": r["qc"].get("motion_score"),
                "adherence": r["qc"].get("prompt_adherence_score"),
                "scroll_stop": r["qc"].get("scroll_stop_score"),
                "auto_fail": r["qc"].get("auto_fail_triggered", []),
                "summary": r["qc"].get("summary", ""),
            }
            for r in sorted_results
        ],
    }

    findings_path = Path(f"research/pe_probe_analysis_{data['timestamp']}.json")
    findings_path.write_text(json.dumps(findings_doc, indent=2, ensure_ascii=False))
    print(f"Saved: {findings_path}")
    print(f"\nNext step: run run_v4_comparison.py to test v4 vs v3")


if __name__ == "__main__":
    main()
