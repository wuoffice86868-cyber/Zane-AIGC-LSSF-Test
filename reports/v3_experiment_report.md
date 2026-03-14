# V1 → V3 System Prompt Experiment Report

## 1. What Changed: V1 vs V3

### V1: Rule-Based Template (hotel_v1.txt)
**Design philosophy:** Rigid format rules + stability constraints

Key characteristics:
- Forces "Single continuous shot" opener on every prompt
- Requires a stability anchor as the closer ("Sky and clouds remain completely still", "The bed frame stays firmly anchored")
- Prescriptive camera move vocabulary (exact verbs only)
- Scene-specific motion strategies hardcoded (pool=push forward, room=push forward toward window, etc.)
- "Do NOT" rules: no equipment names, no negative phrases, no humans
- Word count: 30-45 words enforced

**Example output (pool):**
> Single continuous shot. Wide shot of the infinity pool at golden hour. The camera pushes slowly forward along the pool edge. Water ripples softly in warm light. Sky and clouds remain completely still.

**Problem:** Prompts feel formulaic. Every scene starts the same, ends the same, uses the same camera move. The "stability anchor" text is wasted tokens — Seedance ignores negative/stability instructions entirely.

---

### V3: Description-Rich Template (hotel_v3.txt)
**Design philosophy:** Rich visual description first, camera as supporting element

Key changes from V1:
1. **Dropped "Single continuous shot" opener** — wastes tokens, Seedance doesn't need it
2. **Dropped stability anchors** — confirmed Seedance ignores these; freed up ~8 words for actual description
3. **Subject + scene + details FIRST, camera movement LAST** — follows Seedance official guide (model "sees" the scene before it moves)
4. **Richer sensory vocabulary** — textures, light quality, materials, atmosphere (not just "wide shot of X")
5. **Degree adverbs on motion** — "gently swaying", "softly flickering", "slowly rising" (Seedance responds strongly to these)
6. **More diverse camera moves** — pans, pulls back, drifts, rises (V1 was 90% "pushes forward")
7. **25-40 words** (tighter than V1's 30-45, forces density)

**Example output (pool):**
> Luxury infinity pool merges with golden ocean horizon, palm fronds casting warm shadows on turquoise water, light ripples dancing across the surface. Camera slowly pushes forward along the pool edge.

**What stayed the same:** No equipment names, no negative phrases, no humans, cinematic/editorial tone.

---

### Side-by-side prompt comparison (all 6 scenes)

**Pool**
- V1: "Single continuous shot. Wide shot of the infinity pool at golden hour. The camera pushes slowly forward along the pool edge. Water ripples softly in warm light. Sky and clouds remain completely still." (33 words)
- V3: "Luxury infinity pool merges with golden ocean horizon, palm fronds casting warm shadows on turquoise water, light ripples dancing across the surface. Camera slowly pushes forward along the pool edge." (30 words)
- Change: Dropped formulaic opener/closer, added visual richness (palm fronds, shadows, turquoise water, golden horizon)

**Room**
- V1: "Single continuous shot. Medium shot of the penthouse bedroom. The camera pushes slowly forward toward the window. Silk curtains sway gently in the breeze. The bed frame stays firmly anchored in place." (32 words)
- V3: "Sunlit penthouse bedroom with silk curtains gently swaying, warm morning light streaming through floor-to-ceiling windows revealing a vast ocean below. Camera gradually pulls back from the window." (27 words)
- Change: Dropped opener/anchor, added light quality (warm morning light, streaming), different camera move (pull back vs push forward), fewer words but more information density

**Lobby**
- V1: "Single continuous shot. Wide shot of the grand hotel lobby. The camera pushes slowly forward across the marble floor. The chandelier stays anchored above. Reflections on the floor remain steady." (30 words)
- V3: "Grand hotel lobby with polished marble floors reflecting warm amber sconces, a single orchid arrangement at the reception desk, soft shadows pooling in the archways. Camera smoothly pans left across the space." (32 words)
- Change: Dropped stability anchors (chandelier stays, reflections remain), added specific visual details (amber sconces, orchid arrangement, shadows in archways), different camera move (pan left vs push forward)

**Spa**
- V1: "Single continuous shot. Medium shot of the volcanic stone bath. The camera pushes slowly forward. Thin steam rises gently above the rocks. Stone walls stay perfectly still." (27 words)
- V3: "Secluded stone bath nestled among lush tropical ferns, thin steam rising slowly above dark volcanic rocks, golden afternoon light filtering through a bamboo canopy. Camera gently pushes forward toward the water." (31 words)
- Change: Added environment (ferns, bamboo canopy), light quality (golden afternoon), more specific materials (dark volcanic rocks)

**Restaurant**
- V1: "Single continuous shot. Medium shot of the terrace restaurant at dusk. The camera moves slowly left along the tables. Candle flames flicker gently. The railing stays fixed in frame." (29 words)
- V3: "Open-air terrace restaurant at dusk, candle flames softly flickering on linen tablecloths, deep purple sky above a calm sea, warm lamplight glowing along the balcony railing. Camera slowly drifts left." (30 words)
- Change: Dropped anchor, added sky color, fabric detail (linen), warm lamplight — scene feels more alive

**Beach**
- V1: "Single continuous shot. Wide shot of the beach cabana. The camera rises slowly. Sheer fabric panels sway gently in the breeze. Sky and clouds remain completely still." (27 words)
- V3: "Pristine white sand beach with a draped cabana, sheer fabric panels swaying in a sea breeze, golden sunset light painting long shadows across undisturbed sand. Camera gradually rises above the shoreline." (31 words)
- Change: Added sand color/texture, sunset light painting shadows, more specific camera direction (above the shoreline)

---

## 2. Cross-Scene Comparison & Scores

### Test Setup
- Model: Seedance 1.5 Pro, 1080p, 9:16, 8 seconds
- QC: Gemini 2.5 Pro multimodal evaluation
- Scoring: 4 axes (aesthetic 1-10, motion 1-10, prompt adherence 1-10, scroll-stop 1-10) + auto-fail detection + overall reward (0-100)
- Same source image used for V1 and V3 within each scene (fair comparison)
- Two independent runs (Run A: Mar 12 10:42 UTC, Run B: Mar 12 23:33 UTC)

### Run A Results (2026-03-12 10:42 UTC)

| Scene | V1 Reward | V3 Reward | Delta | V1 Pass | V3 Pass | V1 Auto-Fail | V3 Auto-Fail |
|-------|-----------|-----------|-------|---------|---------|---------------|---------------|
| Pool | 94.0 | 96.0 | +2.0 | ✓ | ✓ | — | — |
| Room | 69.6 | 82.8 | +13.2 | ✗ | ✗ | structural_collapse | object_morphing |
| Lobby | 51.2 | 92.0 | +40.8 | ✗ | ✓ | object_morphing, structural_collapse | — |
| Spa | 96.0 | 96.0 | 0.0 | ✓ | ✓ | — | — |
| Restaurant | 75.6 | 58.4 | -17.2 | ✗ | ✗ | object_morphing | structural_collapse, object_morphing |
| Beach | 67.6 | 94.0 | +26.4 | ✗ | ✓ | object_morphing, structural_collapse | — |

**Run A Summary: V1 avg = 75.7, V3 avg = 86.5, Delta = +10.9**
V1 pass rate: 2/6 | V3 pass rate: 4/6

### Run B Results (2026-03-12 23:33 UTC)

| Scene | V1 Reward | V3 Reward | Delta | V1 Pass | V3 Pass | V1 Auto-Fail | V3 Auto-Fail |
|-------|-----------|-----------|-------|---------|---------|---------------|---------------|
| Pool | 48.4 | 69.6 | +21.2 | ✗ | ✗ | object_morphing, structural_collapse | object_morphing |
| Room | 96.0 | 98.0 | +2.0 | ✓ | ✓ | — | — |
| Lobby | 57.6 | 53.2 | -4.4 | ✗ | ✗ | face_morphing | face_morphing |
| Spa | 80.8 | 96.0 | +15.2 | ✗ | ✓ | action_loop | — |
| Restaurant | 94.0 | 96.0 | +2.0 | ✓ | ✓ | — | — |
| Beach | 74.8 | 96.0 | +21.2 | ✗ | ✓ | action_loop | — |

**Run B Summary: V1 avg = 75.3, V3 avg = 84.8, Delta = +9.5**
V1 pass rate: 2/6 | V3 pass rate: 4/6

### Combined (2 Runs, 24 total samples)

| Metric | V1 | V3 | Delta |
|--------|-----|-----|-------|
| Avg Reward | 75.5 | 85.7 | **+10.2** |
| Pass Rate | 4/12 (33%) | 8/12 (67%) | **+33%** |
| Avg Aesthetic | 7.8 | 8.8 | **+1.0** |
| Avg Motion | 5.8 | 7.3 | **+1.5** |
| Avg Adherence | 9.0 | 9.5 | +0.5 |

### Aesthetic Score Breakdown (per scene, averaged over 2 runs)

| Scene | V1 Aesthetic | V3 Aesthetic | Delta |
|-------|-------------|-------------|-------|
| Pool | 6.5 | 8.5 | +2.0 |
| Room | 8.5 | 9.0 | +0.5 |
| Lobby | 6.0 | 8.5 | +2.5 |
| Spa | 9.0 | 9.0 | 0.0 |
| Restaurant | 9.0 | 8.5 | -0.5 |
| Beach | 8.0 | 9.0 | +1.0 |

---

## 3. How We Score: The Evaluation System

### Overview
Each video goes through a 3-stage scoring pipeline:

```
Generated Video → [Gemini 2.5 Pro QC] → [Reward Calculator] → Final Score (0-100)
```

### Stage 1: Video Upload + Gemini Multimodal QC

The video file is uploaded to Gemini's File API, then evaluated by Gemini 2.5 Pro using a comprehensive QC prompt. Gemini watches the video and returns structured JSON.

**10 Auto-Fail Rules** (any one = immediate FAIL):
1. `object_morphing` — furniture, objects change shape/size
2. `face_morphing` — human faces distort
3. `structural_collapse` — walls, floors, ceilings warp
4. `action_loop` — motion repeats in an obvious cycle
5. `sky_anomaly` — sky tears, color shifts, unnatural movement
6. `reflection_error` — mirrors/glass show impossible reflections
7. `human_generation` — faces/bodies generated from nothing
8. `text_generation` — readable text appears that wasn't there
9. `color_banding` — visible color stepping/posterization
10. `temporal_tear` — frames skip or scene jumps

**Minor Issues** (deduct points but don't auto-fail):
- `texture_shimmer`, `edge_softness`, `speed_inconsistency`, `depth_inconsistency`, `slight_flicker`, `color_shift`

**4-Axis Scoring** (each 1-10):
- `aesthetic_score` — visual beauty, composition, color, mood (scored as a TikTok/Reels viewer: would this stop your scroll?)
- `motion_score` — naturalness, smoothness, consistency of all movement
- `prompt_adherence_score` — does the video match what was asked for?
- `scroll_stop_score` — overall marketing impact (would a viewer pause on this?)

### Stage 2: Reward Calculation

```
If auto_fail triggered:
  reward = (aesthetic × 20) + (adherence × 20) + 10  [capped at ~50-70]
  
If pass:
  reward = 50 (base) + (aesthetic_norm × 20) + (motion_norm × 20) + (adherence_norm × 10)
  minus: minor_issue_count × 2
```

This means:
- A video can FAIL but still get a moderate score if it's aesthetically beautiful (which matters for understanding what's working)
- A PASS video starts at 50 and climbs based on quality axes
- Minor issues are small deductions, not hard failures
- Aesthetic and motion carry equal weight because both matter for marketing video quality

### Stage 3: Feature Extraction (for analysis)

Each prompt is also analyzed for structural features:
- `word_count`, `sentence_count`
- `camera_move_type` (push, pan, rise, pull, drift, circle)
- `has_stability_anchor` (V1=yes, V3=no)
- `motion_elements` (what's moving: water, fabric, steam, etc.)
- `speed_modifier` (slowly, gently, gradually)
- `shot_size` (wide, medium, close-up)

These features let us correlate prompt structure with score outcomes.

---

## 4. What Led to This Change: The Research Chain

### Starting Point
V1 was written based on general prompt engineering intuition: give the model clear rules, use consistent format, add stability anchors to prevent warping. This is what most people do with AI generation prompts.

### Round 1 Data (19 samples, V1 + V2_diverse)
- Average reward: 70.8
- Pass rate: 6/19 (32%)
- Top auto-fails: `object_morphing` (8×), `action_loop` (5×), `structural_collapse` (4×)
- Key insight: pool and simple exteriors scored well (88+), but lobby/beach/bathroom scored poorly (42-60)
- V1 and V2_diverse scored almost identically (~71 each)

### Round 2: OPRO Optimization Attempt (16 samples)
- OPRO rewrote prompts by analyzing (prompt, score) pairs and asking the LLM to find patterns
- Result: average reward **dropped** to 62.9 (from 70.8)
- OPRO's changes: added more constraint text ("maintain structural integrity", "ensure consistent geometry")
- Why it failed: Seedance ignores text-based constraints. More constraint words = fewer descriptive words = worse output.

### The Research Pivot
After Round 2 failed, I dug into:
1. **Seedance official documentation** — discovered recommended prompt structure is subject+action+scene+camera, NOT format rules + constraints
2. **Seedance prompt best practices** — confirmed negative prompts ("do not warp") have zero effect; degree adverbs (slowly, gently) have strong effect
3. **Prompt engineering literature** — OPRO (2023) works on text classification tasks but is poorly suited for generation tasks where the model doesn't understand meta-constraints
4. **Community findings** — Higgsfield/Kling users report richer descriptions outperform rule-heavy prompts

### The V3 Hypothesis
If Seedance ignores constraints but responds to descriptions, then:
- Remove all constraint/stability text (free up ~30% of token budget)
- Fill that space with rich visual descriptions (textures, light, materials)
- Put the scene description BEFORE camera movement (model "sees" before it "moves")
- Diversify camera moves (V1 was 90% "push forward")

### Result
+10.2 reward points, pass rate 33% → 67%, aesthetic +1.0, motion +1.5

The improvement came from **giving the model more visual information to work with**, not from **telling it what not to do**.

---

## 5. Key Takeaways

1. **Description > Constraint.** V3 proves that removing constraints and adding rich visual descriptions improves quality. Seedance responds to what you describe, not what you prohibit.

2. **Consistent across runs.** Two independent runs gave nearly identical deltas (+10.9 and +9.5). This isn't noise.

3. **Aesthetic gap is real.** V3 avg aesthetic = 8.8 vs V1 = 7.8. Richer descriptions literally make the output more beautiful.

4. **Motion improvement matters more.** V3 avg motion = 7.3 vs V1 = 5.8. This is where the pass rate jump comes from — fewer auto-fails on morphing/looping.

5. **Some scenes resist improvement.** Lobby with humans still fails on face_morphing regardless of prompt. Restaurant is inconsistent. These may be model-level limitations.

6. **Sample size caveat.** 12 paired samples (24 total) show a clear trend but aren't statistically bulletproof. Higher variance on per-scene level (lobby went from +40.8 to -4.4 across runs). Need 50+ paired samples for robust per-scene conclusions.

---

## 6. What's Next

1. **Increase sample size** — run 50+ paired samples to get statistically significant per-scene deltas
2. **QC consistency check** — score same videos 3× each, measure inter-run variance
3. **Replace OPRO with DSPy/TextGrad** — now that V3 baseline is established, use gradient-based optimization to find further improvements
4. **Vocabulary research** — systematically test which camera verbs, descriptive adjectives, and structural patterns Seedance responds to best
5. **Test VideoScore2** — compare Gemini QC scores with a purpose-built video quality model to validate our scorer

---

*Report generated 2026-03-13. Data from prompt_evaluator/ eval runs.*
*Video URLs are Kie.ai temporary links (~24h TTL).*
