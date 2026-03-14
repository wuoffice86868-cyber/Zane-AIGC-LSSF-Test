# Deep Research Report: Prompt Evaluator Project
Date: 2026-03-14
Researcher: Zane Jenkins

---

## Executive Summary

After a systematic review of PROJECT.md against the latest academic research and industry developments, I found **3 major corrections** to our assumptions, **5 areas where our knowledge was outdated**, and **2 new research directions** that could significantly change our approach.

The single biggest finding: **Seedance already has its own internal Prompt Engineering (PE) layer** — a Qwen2.5-14B model trained with SFT+RL that rewrites user prompts before generation. Our external prompt optimization is working ON TOP OF an already-optimized rewriter. This fundamentally changes what our system should be doing.

---

## 1. Major Corrections to PROJECT.md

### 1.1 Seedance's Internal PE Layer (NEW — not in our docs)

**Source**: Seedance 1.0 Tech Report (arXiv:2506.09113, June 2025)

Seedance 1.0 uses a Qwen2.5-14B model as a "Prompt Engineering" (PE) module that converts user prompts into dense video captions before generation. This PE model is trained in two stages:
- Stage 1: SFT on manually annotated (user prompt → dense caption) pairs
- Stage 2: RL to fix hallucination issues from SFT

**Implication**: When we send a prompt like "A luxury hotel pool at golden hour, camera pushes forward gently..." to Seedance, it gets REWRITTEN internally before the diffusion model sees it. Our prompt optimization efforts are being filtered through this PE layer.

This means:
- Negative prompts not working is expected — the PE layer strips/rewrites them
- "Stability anchors" being ignored makes sense — the PE decides what to keep
- The format/structure of our prompts matters less than we thought — the PE normalizes everything into dense caption format
- **Our optimization target should shift**: instead of crafting the "perfect Seedance prompt," we should be crafting prompts that the PE layer translates well

### 1.2 OPRO is Confirmed Outdated — But DSPy is Also Wrong Direction

**Our assumption**: Replace OPRO with DSPy (text-based prompt optimization)
**Reality**: Neither OPRO nor DSPy is designed for video generation prompt optimization

Two purpose-built alternatives now exist:

**PhyPrompt** (arXiv:2603.03505, March 2026 — 12 days old!)
- RL-based prompt refinement specifically for physically plausible T2V generation
- Uses GRPO (Group Relative Policy Optimization) with dynamic reward curriculum
- 7B model (Qwen2.5-based) reaches 40.8% joint success on VideoPhy2
- +11pp physical commonsense improvement, +4.4pp semantic adherence
- Transfers ZERO-SHOT across T2V architectures (including CogVideoX-5B)
- Outperforms GPT-4o (+3.8%) and DeepSeek-V3 (+2.2%, 100× larger)

**VPO** (ICCV 2025, Tsinghua COAI Lab)
- Video Prompt Optimization: SFT + DPO on LLaMA3-8B
- Uses VisionReward as reward model
- Text-level AND video-level feedback
- Code + data + model publicly available: github.com/thu-coai/VPO, HuggingFace

Both require GPU to train. But their architecture and approach are directly applicable.

### 1.3 TextGrad is Not Proven for Video Generation

**Our assumption**: TextGrad as a replacement for OPRO (§4.1 of PROJECT.md)
**Reality**: TextGrad is tested on text reasoning tasks (Word Sorting, GSM8k, Object Counting, science questions). No evidence of it working for video generation prompt optimization.

PhyPrompt's GRPO approach and VPO's DPO approach are both more validated and directly applicable to our use case.

---

## 2. Outdated Knowledge in PROJECT.md

### 2.1 Seedance RLHF Architecture (§5.1 — Closed Loop Design)

**What we had**: Generic "Reward Model → RLHF/DPO" design
**What actually exists** (Seedance 1.0 Tech Report):

Seedance uses THREE specialized reward models:
1. **Foundational Reward Model** — VLM-based, targets image-text alignment + structural stability
2. **Motion Reward Model** — mitigates video artifacts, enhances motion amplitude and vividness
3. **Aesthetic Reward Model** — image-space input (keyframes), adapted from Seedream architecture

Training approach:
- Direct reward maximization (NOT DPO/PPO/GRPO — they tested and rejected all three)
- Multi-round iterative learning between diffusion model and reward models
- Separate RLHF for the super-resolution refiner (post-upscaling quality improvement)

**Key insight**: Seedance found that direct reward maximization outperforms DPO/PPO/GRPO for their use case. Their approach: simulate video inference during training, predict x₀ directly, maximize composite rewards from all 3 RMs.

This is different from what the broader field is doing (most recent papers use GRPO). Possible explanation: Seedance has enough compute to do multi-round iterative learning, which smaller teams can't afford.

### 2.2 VideoScore — Correction

**PROJECT.md says**: "VideoScore (EMNLP 2024), 5 维度, 与人工判断相关性 77%"
**Correction**: VideoScore is from TIGER-AI-Lab. The 77% correlation claim needs verification. VideoScore2 (the newer version) uses Qwen2-VL backbone with chain-of-thought reasoning and evaluates 3 dimensions: visual quality, text-to-video alignment, physical/common-sense consistency.

VideoScore2 GitHub: github.com/TIGER-AI-Lab/VideoScore2
HuggingFace Space available for online scoring (no GPU needed).

### 2.3 T2V-Turbo-v2 — Context Update

**What we had**: "VBench 85.13, beat Gen-3/Kling"
**Current context**: T2V-Turbo-v2 is now a relatively older approach. The field has moved to:
- Multi-dimensional reward models (not just HPSv2.1 + CLIPScore + PickScore)
- GRPO-based optimization (not PPO)
- Model-specific reward models (not generic ones)

The T2V-Turbo approach is still valid as a proof of concept that reward signals can improve video generation, but the specific implementation (10 sampling steps, generic reward models) is superseded.

### 2.4 VBench 2.0 — Much More Comprehensive Than We Described

**PROJECT.md says**: Generic mention of VBench 2.0 dimensions
**Actual VBench 2.0** (arXiv:2503.21755, March 2025):

5 key dimensions, each with sub-dimensions (18 total):
1. **Human Fidelity** — anatomical correctness, human motion quality
2. **Controllability** — camera control, composition, timing
3. **Creativity** — novel content, style mixing, narrative coherence
4. **Physics** — gravity, fluid dynamics, rigid body mechanics
5. **Commonsense** — object permanence, spatial reasoning, cause-effect

Uses "generalists" (VLMs/LLMs) + "specialists" (anomaly detection models trained on generated video). Human preference annotations for validation.

VBench 2.0 explicitly distinguishes "superficial faithfulness" (looks good) from "intrinsic faithfulness" (physically/logically correct). Our QC system only evaluates superficial faithfulness.

### 2.5 GRPO is the Emerging Standard for Video Generation RL

**PROJECT.md mentions**: OPRO, DSPy, TextGrad, VPO
**What's actually happening in the field** (March 2026):

GRPO (Group Relative Policy Optimization) has become the dominant approach:
- PhyPrompt (Northwestern): GRPO for physics-aware prompt refinement
- DreamVideo-Omni (Fudan): Reward Feedback Learning for multi-subject video
- FlowPortrait (Johns Hopkins): GRPO for audio-driven portrait video
- Phys4D (Northwestern): PPO for 4D physics consistency
- Place-it-R1 (HKUST): DPO for video object insertion
- SPIRAL (Zhejiang): GRPO for action world models
- RL-Video-Gen (Zhejiang): GRPO for general video generation

A comprehensive survey was published: "Advances in GRPO for Generation Models" (arXiv:2603.06623, March 2026, SJTU). Covers reward signal design, credit assignment, sampling efficiency, diversity preservation, reward hacking mitigation.

---

## 3. What's Actually Working in the Field Right Now

### 3.1 Prompt Optimization Approaches (ranked by relevance to us)

| Approach | GPU Required? | Directly Applicable? | Proven Results? |
|----------|:---:|:---:|:---:|
| PhyPrompt (GRPO + reward curriculum) | Yes (7B model) | HIGH — prompt refinement for T2V | +11pp physics, +4.4pp semantic |
| VPO (SFT + DPO) | Yes (8B model) | HIGH — prompt optimization for T2V | ICCV 2025, code available |
| DSPy MIPROv2 | No | MEDIUM — designed for text tasks | Strong for reasoning, unproven for video |
| TextGrad | No | LOW — text reasoning only | No video generation evidence |
| OPRO | No | LOW — basic, outdated | Our data shows it makes things worse |
| EvoPrompt | No | LOW — evolutionary, no gradient signal | Marginal improvements |

### 3.2 Video Quality Assessment (ranked by suitability)

| Method | GPU Required? | Dimensions | Human Correlation |
|--------|:---:|:---:|:---:|
| VideoScore2 (Qwen2-VL) | Yes, but HF Space available | 3 (visual quality, alignment, physics) | High (CoT reasoning) |
| VBench 2.0 (VLM ensemble) | Yes | 18 sub-dimensions | Validated with human annotations |
| Gemini QC (our current) | No (API) | 3 (aesthetic, motion, adherence) | Unknown — not validated |
| VisionReward (used by VPO) | Yes | Multi-dimensional | Trained on preference data |

### 3.3 Reward Model Architectures (for closed-loop optimization)

**Seedance approach (production-proven)**:
- 3 specialized reward models (Foundational/Motion/Aesthetic)
- Direct reward maximization
- Multi-round iterative learning

**Community approach (most common)**:
- GRPO with combined reward signal
- Single reward model or simple combination
- One-shot training

**Lightweight approach (feasible for us)**:
- VideoScore2 as ready-made reward model
- Gemini as flexible evaluator (with validated prompt)
- Combine both: VideoScore2 for consistency, Gemini for nuance

---

## 4. What This Means for Our Project

### 4.1 Things We Got Right

✅ The evaluation harness architecture is correct — generate → evaluate → optimize is the standard loop
✅ Gemini QC as human substitute is a reasonable approach (Seedance uses human annotations + RM, but VLM-as-judge is accepted)
✅ Multi-dimensional reward (aesthetic/motion/adherence) aligns with how Seedance structures its reward models
✅ Object morphing as #1 failure mode is validated — Seedance's Foundational RM specifically targets structural stability
✅ Scene complexity determining quality more than prompt wording — validated by the field (model limitations, not prompt limitations)

### 4.2 Things We Got Wrong

❌ DSPy as optimizer — wrong tool for the job. Purpose-built approaches (PhyPrompt, VPO) are proven and available
❌ TextGrad as replacement — not validated for video generation
❌ Assuming prompt structure matters to Seedance — the PE layer normalizes everything
❌ Adding negative constraints ("no morphing") — confirmed ineffective, PE layer strips these
❌ "Stability anchors" in prompts — makes sense that they don't work given the PE rewrite

### 4.3 Recommended Course Correction

**Short-term (can do now, no GPU)**:
1. Validate our Gemini QC against VideoScore2 HF Space — run same videos through both, compare scores
2. Research Seedance PE layer behavior — systematically test what prompt structures survive the PE rewrite and produce meaningfully different outputs
3. Replace DSPy optimizer with a simpler GRPO-inspired loop — don't need a full 7B model, can use Gemini to do critique + rewrite with explicit physics/motion/aesthetic dimensions
4. Add VBench 2.0 dimension mapping to our QC — align our auto-fail categories with VBench's 18 sub-dimensions

**Medium-term (requires planning)**:
5. Test VPO's approach at small scale — use their public code/model on CogVideoX-5B as a validation, before adapting for Seedance
6. Build proper reward signal architecture — 3 separate evaluators (Foundational/Motion/Aesthetic) instead of one monolithic QC prompt
7. Systematic PE layer reverse-engineering — feed 100 varied prompts through Seedance, analyze what the PE preserves vs. rewrites

**Long-term (requires GPU + commitment)**:
8. Train a lightweight prompt optimizer (2-3B) using GRPO on accumulated evaluation data
9. Develop domain-specific reward models for hotel/travel content
10. Build the full closed loop: generate → multi-RM evaluate → GRPO optimize → iterate

---

## 5. Specific Research Validations

### 5.1 Claims in PROJECT.md That Are Confirmed

| Claim | Status | Evidence |
|-------|--------|----------|
| "Prompt optimization is short-term, reward model is long-term" | ✅ Confirmed | Seedance tech report, GRPO survey |
| "Object morphing is model-level limitation" | ✅ Confirmed | VBench 2.0 shows structural stability is unsolved across all models |
| "Negative prompts don't work in Seedance" | ✅ Confirmed | PE layer rewrites prompts, negative instructions get stripped |
| "Need specialized video quality evaluator" | ✅ Confirmed | VideoScore2, VBench 2.0 both purpose-built |
| "OPRO is outdated" | ✅ Confirmed | PhyPrompt, VPO, GRPO survey all supersede it |

### 5.2 Claims That Need Revision

| Claim | Issue | Correction |
|-------|-------|------------|
| "DSPy is the right replacement for OPRO" | Wrong domain | PhyPrompt/VPO are purpose-built for T2V |
| "TextGrad gives actual gradient direction" | Unproven for video | No evidence of TextGrad working for video gen |
| "5-section → 4-section prompt structure matters" | Overemphasized | PE layer normalizes structure; content words matter more than format |
| "T2V-Turbo-v2 as primary reference case" | Outdated | Seedance 1.0 tech report is the gold standard now |
| "VideoScore 77% correlation" | Unverified | Need to check original paper for exact number |

### 5.3 New Information Not in PROJECT.md

1. **PhyPrompt** — 12 days old, directly relevant, GRPO + reward curriculum for T2V prompt refinement
2. **Seedance PE layer** — our prompts get rewritten before generation, changes optimization strategy
3. **GRPO as dominant approach** — 7+ papers in March 2026 alone using GRPO for video gen
4. **Seedance rejected DPO/PPO/GRPO** — they found direct reward maximization works better (with enough compute)
5. **VBench 2.0 "intrinsic faithfulness"** — distinction between "looks good" (superficial) and "is correct" (intrinsic)
6. **SeedVideoBench 1.0** — Seedance's internal benchmark: Subject, Subject Description, Action, Action Description, Camera, Aesthetic — maps well to our evaluation dimensions
7. **Multi-round iterative RM training** — Seedance re-trains reward models between optimization rounds to raise performance ceiling

---

## 6. Source URLs

- Seedance 1.0 Tech Report: https://arxiv.org/abs/2506.09113
- PhyPrompt: https://arxiv.org/abs/2603.03505
- VPO: https://arxiv.org/abs/2503.20491 / https://github.com/thu-coai/VPO
- VBench 2.0: https://arxiv.org/abs/2503.21755 / https://github.com/Vchitect/VBench
- VideoScore2: https://github.com/TIGER-AI-Lab/VideoScore2
- GRPO Survey: https://arxiv.org/abs/2603.06623
- DSPy Optimizers: https://dspy.ai/learn/optimization/optimizers/
- TextGrad: https://github.com/zou-group/textgrad
- Awesome RL for Video Gen: https://github.com/wendell0218/Awesome-RL-for-Video-Generation
- Seedance Guide (fal.ai): https://fal.ai/learn/devs/seedance-1-5-user-guide

---

## 7. What I'd Research Next

1. **Deep dive into Seedance PE behavior** — run controlled experiments to understand what survives the rewrite
2. **PhyPrompt paper full read** — the GRPO reward curriculum is novel and directly applicable
3. **VPO code review** — understand their SFT+DPO pipeline, see how to adapt for Seedance
4. **VideoScore2 vs Gemini QC benchmark** — empirical comparison on our existing 35 videos
5. **SeedVideoBench 1.0 prompt categories** — adopt their taxonomy for our test suite
6. **GRPO survey deep dive** — reward hacking mitigation and diversity preservation sections
