# Seedance Prompt Vocabulary Research

> Compiled from official Seedance docs, third-party guides, and our own testing data.
> Last updated: 2026-03-12

## 1. Confirmed Camera Verbs (Seedance 1.0/1.5 Pro responds accurately)

From official JoyPix guide (based on Seedance official documentation):

| Verb | Description | Confidence |
|------|-------------|------------|
| push | Camera moves toward subject | ✅ Official, confirmed |
| pull | Camera moves away from subject | ✅ Official, confirmed |
| pan (left/right) | Camera rotates horizontally | ✅ Official, confirmed |
| move (left/right) | Camera translates horizontally | ✅ Official, confirmed |
| orbit / surround | Camera circles around subject | ✅ Official, confirmed |
| follow | Camera tracks a moving subject | ✅ Official, confirmed |
| rise / lift | Camera moves vertically upward | ✅ Official, confirmed |
| lower | Camera moves vertically down | ✅ Official, confirmed |
| zoom | Camera zoom in/out | ✅ Official, confirmed |
| shake | Camera shake/handheld effect | ✅ Official, confirmed |

### Additional verbs from Seedance 2.0 guides (may or may not apply to 1.5 Pro):

| Verb | Description | Source |
|------|-------------|--------|
| dolly-in / dolly-out | More cinematic push/pull | Redreamality guide |
| track left/right | Horizontal dolly alongside subject | Redreamality guide |
| crane up/down | Vertical camera movement | Redreamality guide |
| 360 orbit | Full rotation | Redreamality guide |
| handheld | Subtle shake, documentary feel | Redreamality guide |
| gimbal | Smooth stabilized motion | Redreamality guide |
| dolly zoom | Hitchcock vertigo effect | Redreamality guide |
| whip pan | Rapid horizontal sweep | Redreamality guide |
| aerial sweep / drone shot | High altitude panoramic | Redreamality guide |

### Shot size vocabulary (confirmed for 1.0/1.5 Pro):

- extreme close-up
- close-up / close shot
- medium shot
- full shot
- wide shot / long shot
- macro shot

### Angle vocabulary:

- low-angle shot
- high-angle shot
- aerial shot
- underwater shot
- eye level (default if unspecified)

## 2. Prompt Structure: Consensus Across Sources

### Official 1.5 Pro (I2V): Subject + Motion + Background + Camera
- For I2V: reduce/avoid static descriptions (model gets scene from image)
- Focus on WHAT MOVES and HOW

### Official 1.5 Pro (T2V): Subject + Motion + Scene + Shot/Style
- Core elements: subject + motion + scene are mandatory
- Shot, style, atmosphere are additive quality boosters

### Seedance 2.0 formula: Subject + Action + Scene + Lighting + Camera + Style + Quality + Constraints
- More complex, but 2.0 can handle longer prompts (30-200 words)
- Template card format: Subject / Camera / Style / Constraints

### Higgsfield's recommendation (for stable output):
1. Composition (framing, angle, background) — first priority
2. Main character (description, clothing, action)
3. Camera movement (action sequence, focus)
4. Overall mood (cinematic, documentary, etc.)

> "Ordering reduces contradictions and helps the model 'lock' the frame before inventing motion"

### VEED's 6-part framework:
1. Subject definition
2. Motion description
3. Camera work
4. Environment
5. Lighting/Atmosphere
6. Style

### Our v1 (old, problematic):
"Single continuous shot" → shot size → scene → camera move → stability anchor
Problems: started with structural phrase, ended with negative instruction, too constrained

### Our v3 (current):
Subject+details → motion element → camera move
Improvements: rich description first, no negatives, degree adverbs

## 3. Key Findings: What Works and What Doesn't

### ✅ CONFIRMED EFFECTIVE

1. **Degree adverbs strongly affect output**
   - "slowly", "gently", "gradually", "smoothly" — for calm/marketing content
   - "quickly", "violently", "with large amplitude", "powerfully" — for action
   - "Appropriately exaggerate the degree" enhances expressiveness
   - Official: "clear → the model cannot obtain degree of motion from input, so it must be clear in the prompt"

2. **One shot, one verb** (Seedance 2.0 official)
   - Multiple motion verbs in a single shot confuse the model
   - For complex motion: use "shot cut" / "lens switch" for multi-shot
   - Our data: all our videos use single camera move, which is correct

3. **Shot size words work reliably**
   - "Close-up of...", "Medium shot of...", "Wide shot of..."
   - Close-up + pan "feels unnatural" per 2.0 guide
   - Wide + fast pan "causes ghosting" per 2.0 guide

4. **Rich visual descriptions improve quality**
   - Texture, material, color, light quality → model renders better
   - "The model will expand the prompt" based on descriptions
   - Too generic = model fills in randomly

5. **Style anchors work**
   - "cinematic", "film grain", "editorial", "documentary"
   - 2D/3D style words: voxel, pixel, felt, clay, illustration
   - Film references: "Japanese fresh style", "film noir"

6. **Prompt length matters**
   - 1.5 Pro I2V: shorter is better (model reads image)
   - 1.5 Pro T2V: more detail needed (no image reference)
   - 2.0: 30-200 words optimal range
   - Under 30: insufficient info; over 200: model ignores details

### ❌ CONFIRMED INEFFECTIVE

1. **Negative prompts do NOT work** (official, all versions)
   - "no warping", "avoid morphing" = ignored
   - 2.0 guide: "Seedance does NOT support negative prompts"
   - Use positive constraint statements instead (2.0 only)
   - Our data: v2 added negative constraints, scores unchanged

2. **Stability anchors** (our v1's "[element] stays still")
   - These ARE negative-style instructions
   - "Sky and clouds remain completely still" — model ignores this
   - Our data: dropping them in v3 did not worsen scores

3. **Equipment names** (gimbal, steadicam, drone, crane)
   - Some guides say use them (2.0), others say avoid
   - For 1.5 Pro: safer to use the basic verb list above
   - Exception: "gimbal" and "handheld" confirmed in 2.0

### ⚠️ UNCERTAIN / VERSION-DEPENDENT

1. **Positive constraint statements** (e.g., "Maintain face consistency")
   - 2.0 guide says these work
   - 1.5 Pro: unconfirmed. Need to test.
   - "No distortion" is still a negative; "maintain" is positive

2. **Quality suffix** ("4K Ultra HD, rich details, sharp clarity")
   - 2.0 guide recommends appending to every prompt
   - 1.5 Pro: unknown effect. We use Kie's resolution parameter.
   - Might be worth A/B testing

3. **Timed beats** ("1-5s: [action]. 6-10s: [action]")
   - Confirmed for 2.0
   - 1.5 Pro: likely not supported (no evidence)

## 4. Implications for Our System Prompt

### Changes to make in v3/v4:

1. **Word count target: 30-45 words for I2V** (image provides context)
   - Current v3 is 25-40, roughly right
   - T2V would need more (50-80 words)

2. **Structure order (for I2V):**
   ```
   [motion verb + subject action] + [key visual detail] + [camera movement with adverb]
   ```
   - DON'T start with "Single continuous shot" — wastes tokens on structural phrase
   - DON'T start with camera move — scene should come first
   - Start with what the viewer SEES, then how the camera MOVES

3. **Camera vocabulary: stick to confirmed 1.5 Pro verbs**
   - push, pull, pan, move, orbit, follow, rise, lower, zoom
   - Always pair with degree adverb: "slowly pushes", "gently pans"

4. **Motion elements: keep ONE**
   - Water ripple, fabric sway, steam rise, candle flicker, light play
   - These are safe because they're small, contained motion
   - Avoid: crowds, complex reflections, many small objects (chandelier crystals)

5. **Drop these entirely:**
   - "Single continuous shot" — unnecessary structural phrase
   - All stability anchors — confirmed ineffective
   - All negative instructions — confirmed ineffective
   - Equipment names — stick to basic verbs

6. **Add these:**
   - Degree adverbs on EVERYTHING: "soft golden light", "gently swaying curtains"
   - Material/texture words: "polished marble", "crisp white linen", "volcanic stone"
   - Atmosphere words: "warm", "serene", "intimate" — helps model set mood

## 5. Analysis: Does Image Analysis → Seedream Enhancement Add Value?

### What the pre-processing pipeline does:
1. CLIP + YOLO + aesthetic scoring (~2.5s/image)
2. Gemini image analysis → structured JSON (scene, subject, defects, faces, etc.)
3. Seedream enhancement (outpainting, aspect ratio conversion, defect fix, color grading)

### Argument FOR (adds value):
- Aspect ratio conversion is essential (16:9 source → 9:16 vertical)
- Defect/watermark removal prevents artifacts propagating to video
- Face detection + removal prevents morphing (faces in Seedance are risky)
- Color normalization gives Seedance a cleaner starting point
- Structured analysis feeds directly into cinematography prompt (camera move selection based on scene type)

### Argument AGAINST (overhead):
- Seedream is another API call ($) + latency
- Enhancement might introduce its OWN artifacts (AI-enhanced images can look synthetic)
- If source images are already high quality + correct aspect ratio, enhancement is unnecessary
- CLIP/YOLO scores may not correlate with video output quality

### Test design to evaluate:
- Take 10 images, run pipeline WITH and WITHOUT Seedream enhancement
- Same cinematography prompt, same Seedance settings
- Compare: reward scores, specific artifact types, Gemini QC breakdown
- Hypothesis: enhancement helps most for low-quality/wrong-ratio sources, negligible for good sources

### Our current limitation:
- We don't have the full pre-processing pipeline (CLIP+YOLO+aesthetic scoring)
- We CAN test: same image → with/without Seedream enhancement → compare video quality
- Need Leo to confirm if he wants us to build/test this

## 6. Comparison With Our Previous Prompt Versions

### v1 (baseline, avg reward 70.8):
```
Single continuous shot. [Shot size] shot of [scene]. The camera [move] [slowly/gradually/gently] [direction]. [subtle motion]. [stability anchor].
```
Issues: structural opener, stability anchors (ineffective), rigid camera vocabulary

### v2 (diverse, avg reward ~71):
```
Same as v1 but with varied camera moves per scene
```
Issues: same structural problems, only changed camera move type

### v3 (restructured, testing):
```
[Subject + rich visual description], [subtle motion element], [camera move with adverb].
```
Improvements: no structural opener, no negatives, richer descriptions, subject-first

### Proposed v4 (based on this research):
```
[Shot size]. [Subject in scene with material/texture/light description]. The camera [confirmed 1.5 Pro verb] [degree adverb] [direction], [what's revealed]. [One subtle motion element with degree adverb].
```
Example: "Medium shot of a luxury infinity pool at golden hour, turquoise water reflecting warm amber light, palm shadows stretching across polished stone deck. The camera pushes slowly forward along the pool edge, revealing the ocean horizon. Water ripples softly."

Key differences from v3:
- Adds shot size back (confirmed effective)
- More specific material/texture words
- "The camera [verb]" uses confirmed 1.5 Pro vocabulary only
- Degree adverbs on both camera and motion
- No structural phrases, no anchors, no negatives

## 7. Sources

1. JoyPix — "How to write Seedance 1.0 Pro prompt" (official Seedance docs mirror)
2. Redreamality — "Seedance 2.0 Complete Prompt Engineering Playbook"
3. ImagineArt — "Seedance 2.0 Prompt Guide With 70 Ready-To-Use Prompts"
4. Seedance15.net — "Seedance 1.5 Pro AI Video Generator Prompt Guide"
5. VEED — "Seedance 1.0 Prompting Guide: Get Better Results in 6 Steps"
6. Higgsfield — "Seedance 1.5 Pro on Higgsfield: A Practical Creator Guide"
7. ByteDance Seed — Official Seedance 1.5 Pro blog
8. Our own data: 35 video samples, 10 scenes, 3 prompt versions
