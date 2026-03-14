# Seedance Prompt Vocabulary Research

> Compiled from official BytePlus docs, Replicate docs, CreateVision guide, Reddit r/PromptEngineering, and our own testing (35+ video samples).
> Last updated: 2026-03-12

## Core Prompt Formula

**Official (BytePlus docs for Seedance 1.0-lite, applicable to 1.5 Pro):**
```
Prompt = Subject + Movement, Background + Movement, Camera + Movement
```

**Higgsfield practical guide:**
```
Composition/Angle → Main Character → Camera Movement → Mood
```

**Our v3 (informed by official + testing):**
```
Subject + Visual Details + Subtle Motion + Camera Movement
```

## What the Official Docs Explicitly Say

1. **"Negative prompt words do not work"** — The model does not respond to negative prompts. Don't say what shouldn't happen; say what should.
2. **"Key degree adverbs must be clear"** — fast, large amplitude, slowly, gently. These strongly affect output.
3. **"Simple and direct"** — Use simple words and sentence structures. The model expands prompts internally.
4. **"Follow the picture"** (i2v) — Prompt should not contradict the input image.
5. **"Minimize description of still/unchanged parts"** — Focus on what moves. Don't waste words on static elements.
6. **Camera switching** — Use "Cut to" / "Camera cut to" / "Camera switching" for multi-shot. Must describe the new scene.
7. **Multiple consecutive actions** — List in chronological order: `subject 1 + movement 1 + movement 2`

## Camera Vocabulary (Confirmed Working)

### Movement Verbs (from official docs + Replicate + CreateVision)
| Verb | Notes |
|------|-------|
| pushes forward / push in | Most reliable for hotel content |
| pulls back / pull out | Good for reveals |
| pans left / pans right | Horizontal rotation |
| tilts up / tilts down | Vertical rotation |
| dolly in / dolly out | Same as push/pull but cinematography term |
| truck left / truck right | Lateral translation (different from pan) |
| orbits / circles around | Orbital movement around subject |
| crane up / crane down | Vertical translation |
| zoom in / zoom out | Lens zoom (different from dolly) |
| follows / tracking shot | Camera follows subject movement |
| dolly zoom | Push+zoom combo (Vertigo effect) — CreateVision confirms works |
| handheld | Slight shake, documentary feel |

### Speed Modifiers (critical — official docs emphasize these)
- slowly, gently, gradually, smoothly, subtly → **calm/hotel tone**
- quickly, rapidly, fast → **action/energy**
- "very slowly" → works, even slower

### Shot Size Terms
- close-up, extreme close-up, medium shot, medium close-up, wide shot, establishing shot
- All confirmed in official examples and Replicate docs

## What Works Well (from our 35 samples + external guides)

### High-success patterns (reward 88+):
1. **Single clear subject** — "infinity pool", "king bed", "modern exterior"
2. **One motion element** — water ripples, fabric sway, steam, candle flame
3. **One camera move** — preferably push or pull
4. **Warm/ambient lighting** — "golden hour", "warm sunlight", "soft morning light"
5. **25-40 word range** — enough detail for guidance, not so much the model ignores parts
6. **Subject-first structure** — describe what you see before how the camera moves

### Low-success patterns (reward <60):
1. **Complex interior geometry** — chandeliers, ornate lobbies, multiple reflections
2. **Multiple motion elements** — water + flags + people + vehicles
3. **Crowds or multiple humans** — face/body morphing
4. **Glass/mirror surfaces** — reflection errors
5. **"Single continuous shot" opener** — wastes 3 words, doesn't help model
6. **Stability anchors** — "sky stays still", "furniture anchored" → negative instructions, ignored
7. **Equipment names** — gimbal, steadicam, drone (from our v1 avoidance list; not necessarily bad per Replicate docs which include these)

## Key Insight: Equipment Names

Our v1 template banned equipment names (gimbal, steadicam, drone, crane). BUT:
- Official BytePlus docs use "crane" in camera movement
- Replicate docs mention "dolly zoom", "tracking shot"
- CreateVision confirms "dolly", "crane", "truck" all work

**Conclusion**: The ban was wrong. Equipment-derived camera terms (dolly, crane, truck) are VALID vocabulary that Seedance understands. We should use them when appropriate. The ban should only apply to irrelevant technical terms (sensor size, bitrate, codec).

## Key Insight: Description of Still Parts

Official docs say **"minimize description of still/unchanged parts"**. Our v1 and Leo's template both spend words on stability anchors ("Furniture stays perfectly still", "Sky and clouds remain completely still"). This is:
1. Wasted words (model ignores negative/stability instructions)
2. Contradicts official guidance (don't describe what doesn't move)
3. Takes word budget away from describing motion elements richly

## Key Insight: Prompt Expansion

Official docs reveal that Seedance **internally expands prompts** based on the input image + text. The model adds scene context, character features, environmental details. This means:
- We don't need to over-describe what's already in the image
- We should focus on what the model CAN'T infer: motion, camera, mood
- Redundant descriptions waste the word budget on information the model already has

## Revised Vocabulary Recommendations

### Use freely:
- Camera movement verbs: pushes, pulls, pans, tilts, dollies, trucks, orbits, cranes, follows, tracks, zooms
- Speed adverbs: slowly, gently, gradually, smoothly, subtly
- Shot sizes: close-up, medium shot, wide shot
- Lighting words: golden hour, warm, soft, ambient, natural, morning, sunset, dusk
- Motion elements: ripples, sways, flickers, drifts, rises, flows, glows, shimmers

### Use carefully:
- "Cut to" / "Camera switching" — only for intentional multi-shot
- Multiple subjects — each additional subject increases morphing risk
- Speed words like "quickly", "rapidly" — can cause artifacts in hotel content
- Human subjects — face morphing risk; if used, keep face small or away from camera

### Avoid:
- Negative instructions: "no", "don't", "avoid", "without", "remains still"
- Technical parameters: resolution, fps, aspect ratio, codec, bitrate
- Over-description of static elements visible in the input image
- More than one camera movement per prompt
- Long compound sentences (>50 words)

## Structure Recommendation for v4

Based on all research:
```
[Subject with key visual detail], [one motion element with degree adverb], [lighting/atmosphere]. Camera [movement verb] [speed adverb] [direction].
```

Example:
```
Luxury infinity pool merging with the ocean horizon, gentle ripples dancing across turquoise water in golden sunset light. Camera slowly dollies forward along the pool edge.
```

Word count: 25 words. Subject-first. One motion. One camera. Rich but concise.

## Next Steps
1. Test v4 prompts built with this vocabulary against v1/v3
2. Specifically test: dolly vs pushes, crane vs rises, truck vs moves left
3. Test removing ALL stability anchors and measuring effect
4. Test different sentence structures (comma-separated vs period-separated)
5. Test word count sweet spot (20/30/40/50 words)
