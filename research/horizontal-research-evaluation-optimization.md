# Horizontal Research: Evaluation Metrics & Prompt Optimization Methods

Date: 2026-03-14
Researcher: Zane Jenkins

---

## Purpose

Leo asked for a horizontal comparison of evaluation metrics (ViCLIP, PickScore, HPSV2.1) and prompt optimization methods (DSPy, TextGrad, VPO), plus any better alternatives. This doc covers everything found, organized by category.

---

## PART 1: EVALUATION / SCORING METRICS

### Category A: Text-Image Alignment Metrics

These were originally designed for text-to-image evaluation. Some have been extended or adapted for video.

#### 1. CLIPScore (OpenAI, 2021)
- **What it does**: Cosine similarity between CLIP text embedding and CLIP image embedding
- **Strengths**: Simple, fast, widely adopted as baseline
- **Weaknesses**: Poor at compositional prompts (counts, spatial relations, negation). Saturates — struggles to differentiate between good and great. No video understanding (frame-level only).
- **Our use case**: Too coarse for our needs. Can't evaluate motion, temporal consistency, or subtle quality differences.

#### 2. PickScore (Stability AI / Hebrew U, NeurIPS 2023)
- **What it does**: CLIP-based scoring model fine-tuned on Pick-a-Pic dataset (500K+ real user preferences over generated images). Predicts which of two images a human would prefer.
- **Training data**: Real user choices on a web app, not synthetic/annotated data
- **Strengths**: Significantly better than CLIPScore at predicting human preference. Captures aesthetic quality + prompt adherence together. Public model + dataset.
- **Weaknesses**: Image-only — no temporal/motion understanding. Trained on general T2I preferences, not domain-specific (hotels, commercial content). Preference model, not absolute quality scorer — designed for pairwise comparison.
- **Our use case**: Useful as a COMPONENT of our reward signal (aesthetic + adherence dimension for keyframes). NOT sufficient as standalone QC — misses motion artifacts, temporal consistency, morphing issues entirely. Could score individual frames from generated videos to get an aesthetic signal.

#### 3. HPSv2 / HPSv2.1 (Tsinghua, 2023-2024)
- **What it does**: Human Preference Score v2 — CLIP ViT-H fine-tuned on HPD v2 dataset (798K preference choices, 430K images). Predicts human preference across 4 styles: animation, concept-art, painting, photo.
- **Strengths**: Larger training set than PickScore. Style-aware (4 categories). Strong benchmark for T2I model comparison.
- **Weaknesses**: Same fundamental limitation as PickScore — image-only, no temporal understanding. The style categories (animation, concept-art, painting, photo) don't map well to our hotel marketing content. No motion quality assessment.
- **Our use case**: Similar to PickScore — potential keyframe aesthetic scorer, but not standalone QC. Between PickScore and HPSv2, PickScore's training data (real user preferences) may be more relevant than HPSv2's (annotator preferences in controlled setting).

#### 4. VQAScore (CMU, NeurIPS 2024)
- **What it does**: Uses a VQA model to answer "Does this image match this description?" — probability of "yes" answer. Works with any VQA/vision-language model.
- **Strengths**: Significantly outperforms CLIPScore AND PickScore on compositional prompts (spatial, counts, attributes, negation). No training needed — works with off-the-shelf VQA models. Extends to video (frame-level).
- **Weaknesses**: Slower than embedding-based methods (requires model inference per evaluation). Quality is dependent on the backbone VQA model used.
- **Our use case**: Best-in-class for text-video ALIGNMENT checking (does the video match what we asked for?). Could replace our Gemini QC's "adherence" dimension with something more principled. BUT still doesn't capture motion quality or temporal artifacts.

### Category B: Video-Specific Evaluation Metrics

#### 5. ViCLIP (OpenGVLab / Shanghai AI Lab, ICLR 2024)
- **What it does**: Video-text contrastive model trained on InternVid (7.1M video clips, 234M clips total). Extension of CLIP to video domain — encodes full video (not just frames) into embedding space alongside text.
- **Training data**: InternVid — largest video-text dataset at time of release. Videos are captioned using a pipeline of ASR + tag extraction + LLM caption generation.
- **Strengths**: Native video understanding (temporal, not frame-by-frame). Best CLIP-family metric for text-video alignment per VBench evaluation. Public model + dataset.
- **Weaknesses**: Still an embedding metric — cosine similarity has the same saturation issues as CLIP. Doesn't explicitly score morphing, flickering, or other video-specific artifacts. Was trained on general web videos, not AI-generated content — may miss AI-specific failure modes.
- **Our use case**: Best available embedding-based metric for text-VIDEO alignment. Upgrade over using CLIPScore on individual frames. However, for our pipeline the main failures aren't alignment (prompt says pool, we get pool) — they're quality issues (morphing, looping, structural collapse). ViCLIP doesn't help with those.

#### 6. VideoScore (TIGER-Lab, EMNLP 2024)
- **What it does**: Fine-tuned on VideoFeedback (37.6K videos, human annotations). Scores videos on 5 dimensions: visual quality, temporal consistency, dynamic degree, text-video alignment, factual consistency. Based on Mantis (multi-image VLM).
- **Strengths**: First dedicated AI video quality scorer. 5-dimensional (not just alignment). 77% correlation with human judgments.
- **Weaknesses**: Doesn't specifically target AI generation artifacts (morphing, flickering). Superseded by VideoScore2.
- **Our use case**: Superseded — go to VideoScore2.

#### 7. VideoScore2 (TIGER-Lab, 2025)
- **What it does**: Based on Qwen2-VL, trained on VideoFeedback2 (27,168 human-annotated videos). 3 dimensions: visual quality, text-to-video alignment, physical/common-sense consistency. Chain-of-thought rationales for each score.
- **Training pipeline**: SFT on human annotations → GRPO reinforcement learning for robustness
- **Strengths**:
  - Purpose-built for AI-generated video evaluation (exactly our use case)
  - Chain-of-thought rationales (interpretable — we can see WHY it scored low)
  - GRPO-trained (same RL method as PhyPrompt — state of art)
  - 44.35% accuracy on in-domain benchmark (+5.94 over VideoScore)
  - 50.37% average on 4 out-of-domain benchmarks (+4.32)
  - HuggingFace Space available (no GPU needed for inference)
  - Open source: model, code, dataset all public
- **Weaknesses**: Still a general evaluator — not specifically tuned for our domain (hotel marketing, commercial content). Requires GPU for local deployment (Qwen2-VL based). HF Space may have rate limits.
- **Our use case**: STRONGEST CANDIDATE to replace or complement Gemini QC. The 3 dimensions map directly to what we care about: visual quality (morphing/artifacts), alignment (does it match prompt), physics (object behavior). Chain-of-thought gives actionable feedback for the optimization loop.

#### 8. VisionReward (Tsinghua / Zhipu AI, Dec 2024, updated Jan 2026)
- **What it does**: General framework for learning human preferences in BOTH image and video generation. Hierarchical visual assessment → linear weighting → interpretable preference score. Multi-dimensional scoring.
- **Strengths**:
  - Surpasses VideoScore by 17.2% in preference prediction accuracy
  - T2V models using VisionReward achieve 31.6% higher pairwise win rate vs same models using VideoScore
  - Interpretable (linear weighting of fine-grained judgments)
  - Works for both image AND video (could score our Seedream images AND Seedance videos)
  - Used by VPO as the reward model — proven in prompt optimization context
  - Open source: code + datasets at github.com/THUDM/VisionReward
- **Weaknesses**: Newer, less battle-tested than VideoScore2 in diverse benchmarks. GPU required for local inference.
- **Our use case**: VERY STRONG CANDIDATE. The fact that VPO uses VisionReward as its reward model means there's a proven pipeline: VisionReward scores → drive prompt optimization. This is exactly our architecture. Interpretability is a bonus — Leo can understand why something scored low.

### Category C: Benchmark Suites (not scorers, but define what to measure)

#### 9. VBench 2.0 (Shanghai AI Lab, CVPR 2025)
- **What it does**: Comprehensive benchmark suite for video generation. 5 major dimensions, 18 sub-dimensions:
  - Human Fidelity
  - Creativity
  - Controllability
  - Physics (thermotics, geometry, mechanics, optics, material)
  - Commonsense (state change, entity, event)
- **Key distinction**: Separates "superficial faithfulness" (looks right) from "intrinsic faithfulness" (physically/logically correct). This is EXACTLY the gap we found — our high-aesthetic videos can still have garbage physics.
- **Our use case**: Not a scorer we deploy, but the STANDARD for what dimensions to evaluate. Our QC system should be aligned with VBench 2.0's taxonomy. Specifically, we're weak on Physics and Commonsense sub-dimensions.

#### 10. Video-Bench (CVPR 2025)
- **What it does**: Human-aligned video generation benchmark using MLLMs (multimodal LLMs) as judges.
- **Our use case**: Reference for evaluation methodology. Validates using VLMs (like Gemini) as video judges — confirms our approach is sound in principle, but needs proper calibration.

#### 11. ETVA (ICCV 2025)
- **What it does**: Evaluates Text-to-Video Alignment via fine-grained question generation and answering. Multi-agent framework: one agent generates specific questions about what should appear in the video, another answers them by watching the video.
- **Strengths**: 47.16 Kendall's τ and 58.47 Spearman's ρ with human judgment — SOTA for alignment.
- **Our use case**: Interesting methodology (QA-based evaluation). Could inspire how we structure our Gemini QC prompt — instead of "rate this video 1-10", break it into specific questions about what should be there.

---

## PART 2: PROMPT OPTIMIZATION METHODS

### Category A: General-Purpose (not video-specific)

#### 1. DSPy (Stanford, 2023-2026)
- **What it does**: Framework for composing + compiling LLM pipelines, automatically optimizing prompts and weights. Multiple optimizers: BootstrapFewShot, MIPROv2, COPRO.
- **Strengths**: Most mature prompt optimization framework (32K+ GitHub stars). Systematic approach to few-shot example selection, chain-of-thought, module composition. Active development (v3.1.3).
- **Weaknesses for our case**:
  - Designed for TEXT reasoning tasks (QA, classification, summarization)
  - Zero evidence of use in video generation prompt optimization
  - Optimizes for text output quality, not downstream visual output quality
  - No mechanism to incorporate video quality feedback into optimization loop
  - Would need significant custom engineering to work with generate→score→improve loop
- **Verdict**: Wrong tool for our job. The optimization target (text quality) doesn't match our target (video quality). We could force-fit it, but purpose-built tools exist.

#### 2. TextGrad (Stanford, 2024)
- **What it does**: Treats LLMs as differentiable engines — computes "text gradients" (natural language feedback) and backpropagates them to improve prompts. PyTorch-like API.
- **Strengths**: Elegant concept. Gives directional improvement ("change X because Y"), not random exploration. Works on any text optimization problem.
- **Weaknesses for our case**:
  - No video generation validation — all examples are text tasks (coding, QA, molecular optimization)
  - Each optimization step needs 3 LLM calls (forward + loss + backward) — expensive
  - The "gradient" is only as good as the LLM's ability to critique video quality — which is exactly the problem we're trying to solve with better QC
  - No temporal/motion reasoning in the gradient computation
  - COPRO (DSPy's optimizer) was shown to match TextGrad performance with simpler implementation
- **Verdict**: Theoretically applicable but unvalidated for our domain. The core insight (directional feedback > random search) is correct, but TextGrad the framework doesn't add value over simply asking Gemini "what should I change and why" (which we already do in our DSPy critique module).

#### 3. OPRO (Google DeepMind, ICLR 2024)
- **What it does**: Feeds (prompt, score) pairs to LLM, asks it to find patterns and generate better prompts.
- **Our experience**: Ran 2 rounds, made things WORSE (-8 points). Confirmed outdated.
- **Verdict**: Superseded. Don't use.

### Category B: Video-Specific Prompt Optimization

#### 4. VPO — Video Prompt Optimization (Tsinghua COAI, ICCV 2025)
- **What it does**: Optimizes text prompts for T2V models. Two-stage: SFT on high-quality (user prompt → optimized prompt) pairs, then DPO using VisionReward as reward model. Based on LLaMA3-8B.
- **Training**: SFT on curated dataset → multi-feedback DPO (text-level + video-level rewards)
- **Key innovation**: Addresses BOTH text-level quality (grammar, detail, safety) AND video-level quality (what the generated video looks like). Previous methods only did one or the other.
- **Strengths**:
  - Purpose-built for T2V prompt optimization (exactly our use case)
  - Uses VisionReward as reward model — proven human-preference alignment
  - Multi-feedback: text-level safety/accuracy + video-level quality (dual optimization)
  - Open source: code, data, model at github.com/thu-coai/VPO
  - ICCV 2025 accepted — peer-reviewed
- **Weaknesses**:
  - Needs GPU for training (LLaMA3-8B fine-tuning)
  - Trained on general T2V, not domain-specific (hotels/commercial)
  - DPO requires paired preference data (good prompt vs bad prompt for same input)
  - No physics-specific optimization (addressed by PhyPrompt)
- **Our use case**: MOST DIRECTLY RELEVANT existing method. The architecture (LLM + SFT + preference optimization + visual reward model) is exactly what we should build toward. We could: (a) use their pre-trained model as-is for prompt enhancement, (b) fine-tune on our domain data, or (c) adopt their architecture with our own training data.

#### 5. PhyPrompt (Northwestern + Dolby, arXiv March 2026)
- **What it does**: RL-based prompt refinement specifically for physically plausible T2V generation. Two-stage: SFT on physics-focused Chain-of-Thought dataset, then GRPO with dynamic reward curriculum.
- **Key innovation**: Dynamic reward curriculum — initially prioritizes semantic fidelity, then progressively shifts toward physical commonsense. This COMPOSITIONALITY exceeds single-objective training on both metrics.
- **Results**:
  - 7B model (Qwen2.5-based)
  - 40.8% joint success on VideoPhy2 (+8.6pp)
  - Physical commonsense: 55.8% → 66.8% (+11pp)
  - Semantic adherence: 43.4% → 47.8% (+4.4pp)
  - Outperforms GPT-4o (+3.8%) and DeepSeek-V3 (+2.2%, 100× larger)
  - Zero-shot transfer across T2V architectures (Lavie, VideoCrafter2, CogVideoX-5B)
- **Strengths**:
  - GRPO-based (current dominant paradigm for generation alignment)
  - Specifically targets physics — our #1 failure mode (object morphing, structural collapse)
  - Cross-architecture transfer — if it works on CogVideoX, it should work on Seedance prompts
  - Only 7B parameters — much smaller than general LLMs we'd otherwise use
  - Very fresh research (March 2026 — 12 days old)
- **Weaknesses**:
  - Needs GPU for training
  - Focused on physics only — doesn't optimize aesthetic quality or commercial appeal
  - Not yet peer-reviewed (preprint)
  - Tested on open-source models, not proprietary APIs like Seedance
- **Our use case**: STRONG COMPLEMENT to VPO. VPO handles general quality + safety, PhyPrompt handles physics. Together they'd cover most of our failure modes. The GRPO + dynamic curriculum approach could be adapted even without training a full 7B — the concept (start with semantic fidelity, shift to physics) could drive our Gemini-based optimization.

#### 6. Prompt-A-Video (ICCV 2025)
- **What it does**: Prompts video diffusion models via preference-aligned LLM. Reward-guided prompt evolution → SFT → optional DPO.
- **Strengths**: Uses image AND video reward models. Automated prompt pool creation.
- **Weaknesses**: Doesn't account for text-level alignment (noted by VPO authors). Less comprehensive than VPO.
- **Our use case**: Superseded by VPO for our purposes.

### Category C: The Dominant Framework

#### 7. GRPO / Flow-GRPO (2025-2026)
- **What it does**: Group Relative Policy Optimization extended to flow-matching generation models. THE dominant paradigm for aligning generative models with human preferences.
- **Scale**: 200+ papers since mid-2025. SJTU published a comprehensive survey (arXiv:2603.06623, March 2026).
- **Key insight**: Doesn't need a separate value network (unlike PPO). Samples multiple candidates per query, estimates advantages using normalized relative rewards. More stable training, higher sample efficiency.
- **Methodological advances covered in survey**:
  - Reward signal design: sparse → dense
  - Credit assignment: trajectory → step level
  - Sampling efficiency and training acceleration
  - Mode collapse and diversity preservation
  - Reward hacking mitigation
  - Reward model construction
- **Applications across modalities**: T2I, T2V, I2V, speech, 3D, VLA, restoration
- **Our use case**: This is the THEORETICAL FRAMEWORK underneath both PhyPrompt and VideoScore2's training. We don't implement Flow-GRPO directly (needs model internals), but understanding it tells us: (a) the field has converged on GRPO, (b) reward signal design is the key lever, (c) multi-objective curricula work better than single-objective.

---

## PART 3: SYNTHESIS — WHAT SHOULD WE USE?

### For Evaluation / QC Scoring

| Tool | Type | Dimension | GPU? | Our Fit | Role |
|------|------|-----------|------|---------|------|
| CLIPScore | Embedding | Alignment | No | Low | Baseline only |
| PickScore | Preference | Aesthetic+Align | No | Medium | Keyframe aesthetic |
| HPSv2.1 | Preference | Aesthetic | No | Medium | Keyframe aesthetic |
| VQAScore | QA-based | Alignment | Varies | High | Text-video alignment |
| ViCLIP | Video embed | Alignment | Yes | Medium | Video-level alignment |
| VideoScore2 | VLM-based | Quality+Align+Physics | Yes/HF | **Very High** | Primary QC replacement |
| VisionReward | VLM-based | Multi-dim preference | Yes | **Very High** | Reward model for optimization |
| Gemini QC (ours) | Prompt-based | Custom | No | High | Current, good enough |

**Recommendation (tiered)**:
1. **Immediate (no GPU)**: Keep Gemini QC as primary, add VideoScore2 HF Space as validation layer. Compare scores to calibrate Gemini.
2. **Short-term**: Integrate VisionReward as reward signal for optimization loop (it's what VPO was built on — proven pipeline).
3. **Long-term**: Deploy VideoScore2 locally (needs GPU) as primary scorer, use VisionReward for optimization reward, keep Gemini as fallback.

**Why not PickScore/HPSv2.1/ViCLIP as standalone?**
They measure ONE dimension each (aesthetic preference or alignment). Our main failures are multi-dimensional: morphing (visual quality), looping (temporal consistency), physics violations (commonsense). Single-dimension metrics miss most of our problems. They're useful as COMPONENTS of a composite reward signal, not as QC replacements.

### For Prompt Optimization

| Method | Domain | GPU? | Our Fit | Status |
|--------|--------|------|---------|--------|
| OPRO | General | No | ❌ | Tested, failed |
| DSPy | General text | No | Low | Wrong domain |
| TextGrad | General text | No | Low | Wrong domain |
| VPO | T2V specific | Train: Yes | **Very High** | Best match |
| PhyPrompt | T2V physics | Train: Yes | **High** | Physics complement |
| Prompt-A-Video | T2V general | Train: Yes | Medium | Superseded by VPO |
| GRPO | Framework | Train: Yes | Theory | Underlying paradigm |

**Recommendation (tiered)**:
1. **Immediate (no GPU)**: Adopt VPO's ARCHITECTURE conceptually with Gemini as the backbone. Instead of training LLaMA3-8B, use Gemini to: (a) rewrite prompts (SFT equivalent), (b) score video quality (VisionReward equivalent), (c) do preference-based selection (DPO equivalent). Essentially: VPO's logic, Gemini's compute.
2. **Short-term**: Download VPO's pre-trained model and test it directly on our prompt inputs. If it improves Seedance output quality out-of-the-box, use it as-is. It's only 8B, inference on CPU is feasible (slow but possible).
3. **Long-term (needs GPU)**: Fine-tune VPO on our domain data (hotel/commercial content). Add PhyPrompt's dynamic curriculum approach for physics-specific improvement. Use VisionReward as the reward model.

### Key Insight: Seedance's Internal PE Layer Changes Everything

From our deep research: Seedance 1.0 has an internal Qwen2.5-14B PE module that rewrites ALL incoming prompts into dense video captions before generation.

This means:
- ViCLIP/PickScore/HPSv2 measuring prompt↔output alignment is partially measuring the PE layer's rewrite quality, not our prompt quality
- Our optimization target should be: prompts that the PE layer PRESERVES intent from (not prompts that are "perfect Seedance format")
- VPO/PhyPrompt's approach (train an LLM to enhance prompts) is essentially building an EXTERNAL PE layer — interesting redundancy question
- The evaluation should focus on final VIDEO quality, not prompt↔video alignment (since the PE layer mediates between them)

---

## PART 4: ACTION ITEMS

### Research Complete
- [x] ViCLIP — video-text alignment, ICLR 2024, best CLIP-family metric for video
- [x] PickScore — image preference, NeurIPS 2023, strong aesthetic scorer
- [x] HPSv2.1 — image preference, 798K choices, style-aware
- [x] VideoScore2 — AI video quality, 3-dim + CoT, GRPO-trained, HF Space available
- [x] VisionReward — multi-dim preference, 17.2% better than VideoScore, used by VPO
- [x] VQAScore — compositional alignment via QA, state-of-art alignment metric
- [x] ETVA — fine-grained T2V alignment, ICCV 2025
- [x] VBench 2.0 — benchmark taxonomy, 5 dim 18 sub-dim
- [x] DSPy — general text optimization, not suitable for T2V
- [x] TextGrad — text gradient optimization, not suitable for T2V
- [x] VPO — T2V prompt optimization, ICCV 2025, most relevant
- [x] PhyPrompt — physics-focused prompt RL, GRPO-based, March 2026
- [x] Prompt-A-Video — T2V prompt enhancement, superseded by VPO
- [x] GRPO/Flow-GRPO — dominant RL paradigm for generation alignment, 200+ papers

### Next Steps
1. Test VideoScore2 HF Space on our existing videos (compare with Gemini QC scores)
2. Download VPO pre-trained model, test on our prompt inputs
3. Design composite reward: VisionReward (preference) + VideoScore2 (quality) + ViCLIP (alignment)
4. Update prompt_evaluator architecture to support pluggable reward signals
5. Update PROJECT.md with corrected terminology (tools vs modules)

---

## References

- ViCLIP: arxiv.org/abs/2307.06942 (ICLR 2024)
- PickScore: arxiv.org/abs/2305.01569 (NeurIPS 2023)
- HPSv2: github.com/tgxs002/HPSv2
- VideoScore: tiger-ai-lab.github.io/VideoScore (EMNLP 2024)
- VideoScore2: huggingface.co/TIGER-Lab/VideoScore2
- VisionReward: arxiv.org/abs/2412.21059, github.com/THUDM/VisionReward
- VQAScore: linzhiqiu.github.io/papers/vqascore (NeurIPS 2024)
- ETVA: arxiv.org/abs/2503.16867 (ICCV 2025)
- VBench 2.0: arxiv.org/abs/2503.21755 (CVPR 2025)
- Video-Bench: CVPR 2025
- DSPy: dspy.ai (Stanford)
- TextGrad: textgrad.com (Stanford)
- OPRO: arxiv.org/abs/2309.03409 (ICLR 2024)
- VPO: arxiv.org/abs/2503.20491, github.com/thu-coai/VPO (ICCV 2025)
- PhyPrompt: arxiv.org/abs/2603.03505 (March 2026)
- Prompt-A-Video: ICCV 2025
- GRPO Survey: arxiv.org/abs/2603.06623 (March 2026)
- Seedance 1.0 Tech Report: arxiv.org/abs/2506.09113
