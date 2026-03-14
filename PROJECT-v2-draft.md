# Prompt Evaluation & Optimization System — V2 Draft

> Source of truth for the project. Pending Leo's review before replacing V1.
> Last updated: 2026-03-14

---

## Changes from V1 & Reasons

This section summarizes what changed between PROJECT.md V1 and this V2, and why.

### Structural Changes

1. **Separated "What We're Building" from "What We're Referencing"** (§4, §5)
   - V1 mixed tools we're implementing with academic references and benchmarks in the same tables/sections
   - V2 has explicit separation: §4 covers our infrastructure, §5 covers research landscape as reference
   - *Reason*: Leo flagged this — calling T2V-Turbo-v2, ViCLIP, PickScore "modules" was misleading

2. **Added Infrastructure Audit** (§6)
   - V1 listed module status but didn't assess what needs rebuilding
   - V2 has module-by-module verdict: keep/rewrite/delete with specific reasons
   - *Reason*: Before planning next phases, need honest assessment of what we have

3. **Rewrote Roadmap** (§7)
   - V1 had 4 phases loosely defined
   - V2 has 5 phases with concrete tasks, deliverables, dependencies, and time estimates
   - *Reason*: Need actionable plan, not aspirational bullets

### Factual Corrections

4. **Seedance PE Layer Discovery** (§3.3)
   - V1 didn't mention Seedance's internal Prompt Engineering module
   - V2 documents that Seedance 1.0 has a Qwen2.5-14B PE module that rewrites ALL incoming prompts
   - *Reason*: This changes our optimization target fundamentally. Found in Seedance tech report (arXiv:2506.09113). Explains why negative prompts/stability anchors don't work.

5. **DSPy/TextGrad Not Suitable** (§5.2)
   - V1 positioned DSPy as "our primary choice" and TextGrad as backup
   - V2 documents that neither has any video generation validation. VPO and PhyPrompt are the correct domain-specific alternatives.
   - *Reason*: Horizontal research revealed DSPy/TextGrad are text reasoning tools, not T2V optimization tools. No papers, no benchmarks, no evidence they work for our use case.

6. **GRPO as Dominant Paradigm** (§5.2)
   - V1 didn't mention GRPO/Flow-GRPO
   - V2 documents it as THE convergence point for generation model alignment (200+ papers, SJTU survey arXiv:2603.06623)
   - *Reason*: The field moved fast in 2025-2026. Both PhyPrompt and VideoScore2 use GRPO.

7. **Evaluation Metrics Horizontal Comparison** (§5.1)
   - V1 mentioned VideoScore and VBench briefly
   - V2 has full landscape: ViCLIP, PickScore, HPSv2.1, VQAScore, VideoScore2, VisionReward, ETVA, VBench 2.0, Video-Bench — with specific strengths, limitations, and our-use-case fit assessment
   - *Reason*: Leo asked for this research. Needed to evaluate whether each tool is suitable before committing to any.

### Removed/Deprecated

8. **OPRO marked for deletion** (§6)
   - V1 described OPRO as "functional but underperforming"
   - V2 marks it for complete removal — 422 lines of dead code that actively made results worse (-8 points)
   - *Reason*: Ran 2 rounds, confirmed harmful. Replaced by DSPy critique logic (kept), with VPO-inspired architecture as next target.

9. **5-section prompt → 4-section canonical**
   - V1 described 5-section as current with a note about v3 dropping sections
   - V2 treats 4-section as canonical: [Shot Size] + [Scene] + [Camera Move] + [Subtle Motion]
   - *Reason*: Leo confirmed. Stability anchor and sky instruction sections proven ineffective.

---

## 1. Background & Motivation

In our AIGC video pipeline, the Prompt is the core link between input and output. A good prompt → stable, natural video. A bad prompt → jitter, action loops, background distortion.

### The Four Problems

1. **No standards**: What is a "good video"? Different people judge differently.
2. **No data**: After modifying a prompt, did it get better or worse? Only gut feeling.
3. **No accumulation**: A good prompt for pools doesn't transfer to lobbies. Experience stays in heads.
4. **No closed loop**: Without data, no systematic improvement — just trial and error.

### Core Insight

From Anthropic's eval research: you're not evaluating a single model — you're evaluating the entire pipeline (structured input + model + scorer) working together. Even with Seedance fixed, we can continuously improve by optimizing Prompt structure and QC standards.

### Strategy

1. Build evaluation infrastructure (structured prompt + auto QC + data accumulation)
2. Use evaluation data to drive optimization
3. Build while producing — evaluation is continuous infrastructure, pipeline runs → data accumulates → analysis and training begin when volume is sufficient

---

## 2. Production Pipeline Architecture

### Main Pipeline

```
Scraped Image
  → CLIP + YOLO + Aesthetic Scoring (~2.5s/image, 1004 images across 45 folders)
  → Image Analysis JSON (structured: scene, subject, position, aspect, outpainting, preservation, lighting, defects, faces, text)
  → Gemini + gemini_enhancement_prompt_9x16.txt
  → Seedream 4.5 (enhancement / outpainting / aspect-ratio conversion)
  → Enhanced Image (9:16 vertical)
  → Gemini + gemini_cinematography_prompt.txt (analysis JSON → 30-45 word natural language prompt)
  → Seedance 1.5 Pro (1080p, 8s, 9:16)
  → Video
```

**Key architectural fact**: There are TWO Gemini calls before video generation, not one. Image analysis feeds structured data INTO the cinematography prompt generator.

### QC Side Path

```
Video → Gemini QC → Pass?
                      ├─ Yes → Keep
                      └─ No  → Delete + Restore Original Image
                    → Lark Base (all results logged, data accumulates passively)
```

### Three Production Prompt Templates

All stored in `prompts/` with complete original text:

**1. gemini_image_analysis.txt** (187 lines)
Input: image → Output: structured JSON
Fields: scene_description, main_subject, subject_position, current/target aspect, extend_top/bottom/left/right (texture descriptions), preserve[], lighting, color_style, defects[], faces[] (position/quality/action: keep|remove person), text[] (content/action: preserve|remove), remove[]

Key design: face `action: "remove person"` means inpaint entire person (not just face repair). Text `action: "preserve"` for signage, remove for watermarks.

**2. gemini_enhancement_prompt_9x16.txt** (208 lines)
Input: analysis JSON → Output: Seedream enhancement instructions
5-step pipeline: Reframing → Removals → Lighting & Color → Preservation → Negative Constraints
4 complete examples + 10 rules. Key rule: result must look like well-exposed photograph, never AI art.

**3. gemini_cinematography_prompt.txt** (224 lines)
Input: analysis JSON → Output: 30-45 word Seedance prompt
8-section build structure: #0 shot size → #1 scene context → #2 camera movement → #3 what's revealed → #4 lighting → #5 subtle motion → #6 stability anchor → #7 sky/time realism
5 complete input→output examples. Seedance camera language table included.

**Note**: V3 system prompt evolution dropped #6 and #7 (stability anchor and sky instruction — Seedance ignores them). Canonical structure is now 4-section. See §3.2.

---

## 3. Key Technical Findings

### 3.1 Seedance Prompt Best Practices (tested + confirmed)

**Effective**:
- Subject + Action + Scene + Camera structure (official recommendation)
- Degree adverbs: "slowly", "gently", "gradually" (strong model response)
- Single camera move per prompt (multi-move → geometric distortion)
- Rich visual descriptions over constraint lists

**Ineffective**:
- Negative prompts: "no warping", "avoid morphing" (zero effect, confirmed)
- Stability anchors: "sky remains still", "furniture stays fixed" (ignored)
- "Single continuous shot" as opener (doesn't help)
- Constraint-heavy prompts (OPRO's approach, made things worse)

**Untested (candidates for investigation)**:
- Temporal segmentation: "first 2 seconds... then..."
- Few-shot in-prompt: embedding good examples
- Detail density sweet spot (current: 30-45 words — optimal?)

### 3.2 V3 System Prompt Results

4-section structure: [Shot Size] + [Scene] + [Camera Move] + [Subtle Motion]

| Scene | v1 Reward | v3 Reward | Delta |
|-------|-----------|-----------|-------|
| pool | 45.2 | 58.4 | +13.2 |
| room | 94.0 | 94.0 | 0 |
| lobby | 92.0 | 69.6 | -22.4 |
| spa | 69.6 | 90.0 | +20.4 |
| restaurant | 57.2 | 92.0 | +34.8 |
| beach | 69.6 | 86.0 | +16.4 |
| **Average** | **71.3** | **81.7** | **+10.4** |

V3 wins 4/6 scenes. Lobby regression needs investigation.

### 3.3 Seedance Internal PE Layer (NEW — critical finding)

Source: Seedance 1.0 Tech Report (arXiv:2506.09113, June 2025)

Seedance uses a **Qwen2.5-14B model as an internal "Prompt Engineering" (PE) module** that rewrites ALL user prompts into dense video captions before the diffusion model sees them. Trained in two stages: SFT on (user prompt → dense caption) pairs, then RL to fix hallucination.

**Implications**:
- Our prompt gets REWRITTEN before generation. We don't control what the diffusion model actually sees.
- Negative prompts ignored → PE strips/rewrites them. Expected behavior.
- Stability anchors ignored → PE decides what to keep. Expected behavior.
- Prompt format matters less than we thought → PE normalizes into dense caption format.
- **Our optimization target should shift**: not "write perfect Seedance prompt" but "write prompt that PE translates well into intended video"

### 3.4 Scene Complexity as Dominant Factor

From 35 video samples across 10 scenes:
- Simple scenes (pool, single room, exterior): consistently reward 88+
- Complex scenes (lobby with chandeliers, beach with people, bathroom with reflections): consistently reward 42-60
- Prompt version has secondary effect (v3 improved +10.4 avg) but scene complexity dominates

Most common auto-fail triggers: object_morphing (8×), action_loop (5×), structural_collapse (4×). These are model-level limitations of Seedance 1.5 Pro, not prompt-fixable.

---

## 4. What We've Built (Our Infrastructure)

### 4.1 `prompt_evaluator/` Python Module

6,137 lines, pip installable, 142 tests passing.

```
prompt_evaluator/
├── models.py            — Pydantic data models
├── reward_calculator.py — Multi-dimensional reward scoring
├── prompt_analyzer.py   — Feature extraction + correlation
├── optimizer.py         — OPRO optimizer (deprecated, to be removed)
├── dspy_optimizer.py    — DSPy-based optimizer (critique + improve logic)
├── calibration.py       — QC vs human label calibration
├── kie_client.py        — Kie.ai API (Seedream + Seedance)
├── gemini_client.py     — Gemini QC (multimodal) + LLM
├── qc_client.py         — QC client interface
├── pipeline.py          — End-to-end orchestrator
└── tests/               — 142 unit tests
```

### 4.2 API Clients (working, live-tested)

**KieClient**: Seedream 4.5 + Seedance 1.5 Pro. Budget tracking, 1080p default, 9:16 aspect ratio, 8-second duration, `generate_audio: false`. ~59 API calls made, $20.31 spent.

**GeminiVideoQC**: Video upload to Gemini File API → multimodal evaluation with comprehensive QC prompt (10 auto-fail rules + 7 minor issues + aesthetic/motion/adherence 1-10 scoring). Currently on Gemini 2.5 Pro for QC, 2.5 Flash for prompt generation.

### 4.3 System Prompt Versions

- **v1** (hotel_v1.txt): 5-section, "Single continuous shot" opener, negative constraints. Avg reward: ~71
- **v2** (hotel_v2_diverse.txt): v1 with more camera variety. Avg reward: ~71 (no improvement)
- **v3** (hotel_v3.txt): 4-section, subject-first, no negative constraints, richer descriptions. Avg reward: **81.7 (+10.4)**

### 4.4 Experiment Data

- 3 test rounds: Round 1 (19 samples, baseline), Round 2 (16 samples, OPRO improved → worse), V3 comparison (12 samples, 6 scenes × 2 versions)
- All raw data in `eval_results/` as timestamped JSON
- Video URLs from Kie.ai are temporary (~24h TTL) — evaluation data persisted in JSON

---

## 5. Research Landscape (Reference, Not Implementation)

### 5.1 Evaluation / Scoring Tools

**Single-dimension metrics** (each measures ONE aspect):
- **ViCLIP** (ICLR 2024) — Video-text alignment. Best CLIP-family metric for video (trained on InternVid 7.1M clips). Useful for alignment checking, but doesn't measure morphing/physics.
- **PickScore** (NeurIPS 2023) — Image preference. 500K real user preferences. Strong aesthetic scorer for keyframes, but image-only.
- **HPSv2.1** (2024) — Image preference. 798K annotated choices, 4 style categories. Similar to PickScore, image-only.
- **VQAScore** (NeurIPS 2024) — Compositional alignment via QA. SOTA for text-image alignment, extends to video. Better than CLIP/PickScore on compositional prompts.

**Multi-dimension video evaluators** (measure multiple aspects):
- **VideoScore2** (TIGER-Lab, 2025) — 3 dimensions: visual quality, text-video alignment, physical/common-sense consistency. Chain-of-thought rationales. Based on Qwen2-VL, GRPO-trained. HF Space available (no GPU needed). **Best candidate to complement/replace our Gemini QC.**
- **VisionReward** (Tsinghua/Zhipu, 2024-2026) — Multi-dimensional preference model for image AND video. Surpasses VideoScore by 17.2%. Used by VPO as reward model. Interpretable (linear-weighted fine-grained judgments). **Best candidate for optimization reward signal.**

**Benchmark suites** (define WHAT to measure, not scorers themselves):
- **VBench 2.0** (CVPR 2025) — 5 dimensions, 18 sub-dimensions. Distinguishes "superficial faithfulness" from "intrinsic faithfulness" (physics/commonsense). Industry standard.
- **Video-Bench** (CVPR 2025) — Human-aligned benchmark using MLLMs as judges.
- **ETVA** (ICCV 2025) — QA-based alignment evaluation. SOTA correlation with human judgment.

**Assessment for our use case**: Single-dimension metrics (ViCLIP, PickScore, HPSv2.1) are useful as COMPONENTS of a composite reward signal but insufficient alone. Our failures are multi-dimensional (morphing + looping + physics). VideoScore2 and VisionReward are the strongest standalone candidates.

### 5.2 Prompt Optimization Methods

**General-purpose (not video-specific)**:
- **OPRO** (DeepMind, ICLR 2024) — Sort (prompt, score) pairs, ask LLM to find patterns. We tested: made things WORSE. Confirmed outdated.
- **DSPy** (Stanford, 2024-2026) — Framework for composing/compiling LLM pipelines. 32K+ GitHub stars, most mature prompt optimization framework. **BUT**: designed for text reasoning tasks (QA, classification). Zero evidence of T2V use. Useful concepts (modular signatures, trainable modules) but not the right framework for video generation.
- **TextGrad** (Stanford, 2024) — LLM-as-differentiable-engine for text gradients. Elegant concept, but all validation is on text tasks (coding, QA). No T2V evidence. 3 LLM calls per step (expensive).

**Video-specific**:
- **VPO** (ICCV 2025, Tsinghua) — Video Prompt Optimization. SFT + DPO on LLaMA3-8B with VisionReward as reward model. Dual feedback: text-level (grammar, safety, detail) + video-level (visual quality). Open source. **Most directly relevant existing method for us.**
- **PhyPrompt** (arXiv March 2026, Northwestern + Dolby) — GRPO + dynamic reward curriculum for physical plausibility. 7B model beats GPT-4o (+3.8%) and DeepSeek-V3 (+2.2%, 100× larger). Zero-shot cross-architecture transfer. **Directly targets our #1 failure mode (morphing/physics).**
- **Prompt-A-Video** (ICCV 2025) — Reward-guided prompt evolution + SFT. Superseded by VPO.

**Underlying paradigm**:
- **GRPO/Flow-GRPO** (2025-2026) — Group Relative Policy Optimization for generation models. 200+ papers since mid-2025. SJTU survey (arXiv:2603.06623). THE convergence point for generation alignment. Both PhyPrompt and VideoScore2 use GRPO.

**Assessment**: DSPy/TextGrad are wrong tools. VPO + PhyPrompt are the right references. The critique-based improvement logic we extracted from our DSPy implementation is sound (keep it), but the DSPy framework dependency should go.

### 5.3 Validation Case: T2V-Turbo-v2

Proves the closed-loop approach works: VBench 85.13, beat Gen-3/Kling.

3 reward models combined: HPSv2.1 (visual quality) + ViCLIP (text alignment) + PickScore (aesthetic).

Key takeaway: **Reward signal DESIGN matters more than raw training data.** Well-designed multi-signal reward can improve 50% while 10× more data might only improve 10%.

---

## 6. Infrastructure Audit

### What to Keep

| Module | Lines | Verdict | Reason |
|--------|-------|---------|--------|
| models.py | 165 | Keep + update | Good foundation, needs ImageAnalysisResult model, configurable SceneType, structured QC scores |
| kie_client.py | 514 | Keep | Works, live-tested, budget tracking. Add GeneratorProtocol interface, local file caching |
| calibration.py | 308 | Keep, defer | Sound implementation, unused (no human labels yet). Low priority |
| prompt_analyzer.py | 262 | Keep + update | Good feature extraction. Update vocabulary, remove deprecated features (sky_freeze, single_continuous_shot) |

### What to Rewrite

| Module | Lines | Verdict | Reason |
|--------|-------|---------|--------|
| reward_calculator.py | 256 | Rewrite core | 4-dimension model is ad-hoc. Should align to industry standard (visual quality / text alignment / physical consistency). Need pluggable scorer interface. Remove hand-tuned heuristics. |
| gemini_client.py | 457 | Refactor | QC prompt hardcoded (300+ lines in source). Should externalize to config. Needs abstract ScorerProtocol so VideoScore2/VisionReward can plug in. Add consistency checking (3-run median). |
| dspy_optimizer.py | 695 | Extract logic | The 3-module architecture (generate/critique/improve) is sound. Extract from DSPy wrapper into standalone Python. Add VPO's dual feedback + PhyPrompt's dynamic curriculum concept. |
| pipeline.py | 819 | Major refactor | Too monolithic. Decompose into generation/evaluation/optimization sub-pipelines. Remove hotel-specific hardcoding. Wire in full production flow (image analysis → enhancement → cinematography). |
| qc_client.py | 249 | Merge | Good protocol definition, but real implementation is in gemini_client.py. Consolidate into evaluation/ subpackage. |

### What to Delete

| Module | Lines | Verdict | Reason |
|--------|-------|---------|--------|
| optimizer.py | 422 | Delete | OPRO. Tested, made things worse (-8 points). Hardcodes wrong assumptions. 100% dead code. |

### Scripts to Consolidate

11 one-off experiment scripts → 3 canonical runners: run_evaluation.py, run_optimization.py, run_comparison.py.

---

## 7. Phased Roadmap

### Phase 1: Foundation Cleanup (3-5 days)

**Goal**: Clean codebase with correct abstractions.

- Delete optimizer.py (OPRO)
- Restructure into subpackages: models/, evaluation/, generation/, optimization/, analysis/, pipeline/
- Define abstract interfaces: ScorerProtocol, GeneratorProtocol, RewardProtocol
- Externalize all prompts to config files
- Remove hotel-specific hardcoding (make domain-configurable)
- Update models: add ImageAnalysisResult, structured QC scores, remove dead fields
- Consolidate scripts, update tests

**Deliverable**: Refactored codebase, all tests passing, pip installable.

### Phase 2: Evaluation Layer Upgrade (3-5 days)

**Goal**: Multi-scorer evaluation with cross-validation.

- Implement VideoScore2Client (via HF Space API)
- Compare VideoScore2 vs Gemini QC on existing 35 videos
- Add ViCLIP-based alignment score as supplementary metric
- Design composite reward: configurable weighted combination of multiple scorers
- Add QC consistency checking (N-run median aggregation)
- Cross-calibrate Gemini vs VideoScore2, generate calibration report

**Deliverable**: Multi-scorer evaluation, cross-calibration report, composite reward function.

### Phase 3: Optimization Engine Rebuild (5-7 days)

**Goal**: Optimizer that demonstrably improves prompts.

- Extract optimization logic from DSPy into standalone Python
- Implement VPO-inspired dual feedback (text-level + video-level)
- Implement PhyPrompt-inspired dynamic reward curriculum
- Test VPO pre-trained model on our inputs
- Seedance PE layer investigation (100 varied prompts → analyze preservation/modification)
- Run 3+ optimization rounds, 30+ samples per round

**Deliverable**: Working optimizer, PE layer analysis, demonstrated improvement.

### Phase 4: Production Integration (3-5 days)

**Goal**: Wire into Leo's production flow.

- Implement ImageAnalyzer class wrapping gemini_image_analysis.txt
- Wire full production flow (image analysis → Seedream → cinematography → Seedance → QC)
- Lark Base integration (coordinate ownership with Leo)
- Accumulate 100+ scored production samples
- A/B testing infrastructure

**Deliverable**: Production-integrated pipeline, 100+ scored samples.

### Phase 5: Closed-Loop Optimization (ongoing)

**Goal**: Self-improving system.

- Continuous optimization loop: production → auto-score → optimize → improve template → deploy
- Domain-specific reward model (if GPU available): fine-tune VisionReward on our data
- Reward hacking monitoring
- Periodic reports (weekly score distributions, monthly system review)

**Deliverable**: Self-improving pipeline, monitoring.

**Timeline**: Phase 1-4 total ~14-22 days to production-ready.

---

## 8. Closed-Loop Architecture (Long-term Vision)

### The Full Loop

```
Generate Video → QC Auto-Evaluation → Data Accumulation → Train Reward Model → Optimize Prompt/Model → Improved Generation → Loop
```

### Three Concrete Directions

1. **QC Data → Reward Model**: Train hotel-scene-specific video scoring model (reference: VideoScore architecture + VBench 2.0 dimensions)
2. **Reward Signal → Model Optimization**: RLHF/DPO to teach generation model "what a good video looks like" (reference: T2V-Turbo-v2, long-term, needs data + GPU)
3. **High-Score Samples → Fine-tune Prompt Generator**: VPO method — use QC high-scoring (video, prompt) pairs to fine-tune prompt generation (lighter weight, faster payoff)

### Current vs Planned

```
[IMPLEMENTED]                              [PLANNED]
Generate → Gemini QC → Data accumulation → Reward Model → RLHF/DPO
                                              ↑                |
                                              └────────────────┘
```

Core insight: **Prompt optimization is the short-term play. Long-term is using Reward signals to drive model evolution itself. The evaluation system's ultimate value is producing data that drives model improvement.**

---

## 9. API & Infrastructure Notes

### API Access
- **Kie.ai**: Seedream 4.5 + Seedance 1.5 Pro. Key in `.credentials/`. $20.31 spent total.
- **Gemini**: AI Studio key. QC on 2.5 Pro, prompt gen on 2.5 Flash. ~70 calls, likely $0 on free tier.
- **Seedance 2.0**: NOT available via API. BytePlus delayed indefinitely (copyright dispute). Third-party wrappers exist but unofficial. Staying on 1.5 Pro — interface swap-compatible when 2.0 opens.

### Constraints
- 1080p, 9:16, 8 seconds, generate_audio: false
- No GPU environment — API-based and HF Space solutions only
- Kie temp URLs expire ~24h — evaluation data persisted in JSON

### Known Gaps

**Have complete original text**: All 3 production prompt templates, integration architecture, all experiment data, Seedance best practices

**Have summary only**: Leo's optimization reasoning process, T2V-Turbo analysis details, research doc screenshot content (extracted, images not persisted)

**Missing**: Original ZIP files (INSTRUCTIONS.md, System Design), 纠正替阿_Prompt评测系统.md, HANDOFF_PROMPT_FOR..., "there is more" follow-up materials

Coverage estimate: ~90% of technical specs, ~75% of design reasoning narratives.

---

## 10. References

Core:
- Seedance 1.0 Tech Report: arXiv:2506.09113 (PE layer disclosure)
- VPO: ICCV 2025, github.com/thu-coai/VPO
- PhyPrompt: arXiv:2603.03505 (March 2026)
- GRPO Survey: arXiv:2603.06623 (March 2026)
- VideoScore2: huggingface.co/TIGER-Lab/VideoScore2
- VisionReward: arXiv:2412.21059, github.com/THUDM/VisionReward

Evaluation:
- ViCLIP / InternVid: arXiv:2307.06942 (ICLR 2024)
- PickScore / Pick-a-Pic: arXiv:2305.01569 (NeurIPS 2023)
- HPSv2: github.com/tgxs002/HPSv2
- VQAScore: NeurIPS 2024
- VBench 2.0: arXiv:2503.21755 (CVPR 2025)
- ETVA: arXiv:2503.16867 (ICCV 2025)

Optimization:
- OPRO: arXiv:2309.03409 (ICLR 2024) — tested, deprecated
- DSPy: dspy.ai — evaluated, not suitable for T2V
- TextGrad: textgrad.com — evaluated, not suitable for T2V
- T2V-Turbo-v2: t2v-turbo-v2.github.io (validation case)

Foundation:
- Anthropic Evaluation Harness: anthropic.com/engineering/demystifying-evals-for-ai-agents
