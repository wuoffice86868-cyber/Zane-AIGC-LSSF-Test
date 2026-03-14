# Prompt Evaluation & Optimization System

> Source of truth for the project. Updated as we progress.
> Last updated: 2026-03-14

---

## 1. Background & Motivation

In our AIGC video pipeline, the Prompt is the core link between input and output. A good prompt → stable, natural video. A bad prompt → jitter, action loops, background distortion. In a fully automated production line this is critical.

### The Problems We're Solving

- **No standards**: What is a "good video"? Different people judge the same clip differently.
- **No data**: After modifying a prompt, did it get better or worse? We only had manual review of a few results, pure gut feeling.
- **No accumulation**: A good prompt recipe today may not work for a different scene tomorrow. Experience doesn't transfer.
- **No closed loop**: Without data, no systematic improvement — just trial and error.

### Core Insight (Anthropic Evaluation Harness)

From [Anthropic's eval research](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents): When evaluating a system, you're not evaluating a single model — you're evaluating the entire pipeline (structured input + model + scorer) working together.

**Practical implication**: Even with Seedance (the generation model) fixed, we can continuously improve system performance by optimizing Prompt structure and QC standards. The evaluation system itself is a tool for improvement, not just inspection.

### Our Strategy

1. **First**: Build evaluation infrastructure (structured prompt + auto QC + data accumulation)
2. **Then**: Use evaluation data to drive optimization (DSPy auto-optimization, Reward Model training)
3. **Philosophy**: Build while producing — evaluation is continuous infrastructure, not a one-time check. Pipeline runs → data accumulates → once volume is sufficient → analysis and training begin.

---

## 2. Production Pipeline Architecture

### Main Pipeline (生产主线)

```
Image → CLIP+YOLO+Aesthetic Scoring (~2.5s/image)
      → Image Analysis JSON
      → Gemini + Enhancement Prompt Template (gemini_enhancement_prompt_9x16.txt)
      → SeeDream 4.5 (enhancement/outpainting/aspect-ratio conversion)
      → Enhanced Image
      → Gemini + Video Prompt Template (structured JSON → natural language)
      → Seedance 1.5 Pro
      → Video
```

### Image Enhancement Prompt (gemini_enhancement_prompt_9x16.txt)

Converts image analysis JSON into Seedream enhancement instructions via 6-step pipeline:
1. **Reframing** — aspect ratio conversion (16:9 → 9:16), extend with texture descriptions
2. **Removals** — defects, morphed faces → fill with surrounding texture
3. **Lighting & Color** — subtle fixes, film stock grading (KEEP IT SUBTLE)
4. *(step 4 not visible in screenshots)*
5. **Preservation** — what must stay unchanged
6. **Negative Constraints** — "No new text or signs. Natural photograph look."

10 Rules:
1. Natural language only — no JSON output
2. Direct instructions — "Extend the top with..." not "The top should be extended..."
3. Texture descriptions — "white plaster ceiling" not just "ceiling"
4. Always end with preservations
5. Always end with negative constraints
6. Keep it concise — Seedream works better with shorter prompts
7. One paragraph per task
8. Subtle is better — natural photograph, not stylized AI art
9. Start with "9:16 vertical format"
10. End with "No new text or signs"

### Video Prompt Template (structured JSON → Seedance)

Input fields:
- scene_description, main_subject, foreground, background
- camera_move, camera_direction, shot_size
- lighting, subtle_motion, stable_element, sky_instruction

Output format: "Single continuous shot. [shot_size] shot of [scene] in [lighting]. The camera [move + direction]..." (~30-38 words)

Maps to 5-section structure: [镜头类型]+[场景描述]+[相机运动]+[动态元素]+[稳定锚点]

### QC Side Path (评测旁路)

QC runs async, doesn't block production:

```
Video → Gemini QC → Pass?
                      ├─ Yes → Keep
                      └─ No  → Delete + Restore Original Image
                    → Lark Base (all results logged)
```

### Image Pre-Processing Pipeline
- 45 folders, 1004 images processed
- ~2.5 seconds per image (CLIP + YOLO + aesthetic scoring)
- Produces analysis JSON with: scene, main_subject, subject_position, current/target aspect, extend directions, preserve list, lighting, defects, faces (position/quality/action), text detection, remove list

### Data Format (per QC record)

Each QC record contains full context:
- `video_url` — viewable for retrospective review
- `prompt_used` — trace which prompt wording caused which issues
- `qc_pass: bool` (later → int score)
- `auto_fail_reasons: []`
- `minor_issues: []`

**Design goal**: Support downstream analysis — which issues are most common, which prompt structures cause problems, failure rate trends over time.

---

## 3. Built Infrastructure

### 3.1 Structured Prompt Template

| Aspect | Detail |
|--------|--------|
| **Evolution** | Intuitive writing (play director) → Found problems (fixing one defect breaks something else, don't know what you're tuning) → Shift to structured design |
| **Current Implementation** | 5-section template: `[Shot Size] + [Scene] + [Camera Move] + [Subtle Motion] + [Stability]` |
| **Next Step** | DSPy framework for data-driven optimization (2025: MIPROv2 Bayesian optimization, SIMBA hard-sample identification) — from "manual tuning" to "framework auto-finds optimal" |

**Why Structure Matters**: Visual generation prompts are multi-dimensional (composition, lighting, motion...). Without structure, changing one element has uncontrollable ripple effects. Classic case: discovered sky was "breathing", added "sky stays still" — fixed locally but conflicted with other instructions. Structured approach: stability instructions go in [Stability] section, changes don't leak.

**Template Structure**:

```
Prompt Template
├── A: Shot Size      → Close-up / Medium / Wide
├── B: Scene          → Environment + Lighting + Atmosphere
├── C: Camera Move    → pushes / pulls / circles + slowly
├── D: Subtle Motion  → Water ripples softly / 空调 breeze
└── E: Stability      → Furniture stays still
```

**Structure Sources**:
- **Official docs**: Seedance docs specify which keywords work best (e.g., "pushes" >> "moves toward"). These are high-frequency training data words — the docs tell you how to maximize the model's learned capabilities.
- **QC experience**: Official docs are general-purpose; we only make hotel videos with specific needs: (1) mostly static environments (rooms, lobbies, poolside), (2) marketing context → users sensitive to realism, (3) short videos → defects more visible on repeat viewing.
- **Vertical research**: Testing + papers/AIGC forums revealed patterns:
  - Single camera move only: multi-movement (zoom+pan+rotate) causes geometric distortion → template allows only ONE camera_move
  - Slow motion advantage: hotel tone requires "slowly"/"gradually" — fast movement triggers artifacts

### 3.2 Seedance Prompt Best Practices (from official guide + our testing)

Key findings that informed v3 system prompt:
- **Negative prompts don't work**: "no warping", "avoid morphing" etc. have zero effect. Seedance ignores negative instructions.
- **Recommended structure**: Subject + Action + Scene + Camera/Style (not our old "Single continuous shot..." opener)
- **Degree adverbs matter**: "slowly", "gently", "gradually" — model responds strongly to these
- **Camera switch for complex shots**: Don't pack multiple moves into one sentence
- **Description > Constraints**: Model needs rich visual descriptions, not rule lists

### 3.3 Pipeline Integration (Gemini QC → Lark Base)

| Aspect | Detail |
|--------|--------|
| **Pain Point** | Editing prompts to fix local issues often hurts overall quality. Without data, can't attribute cause — was it the change, or was it always there? |
| **Evolution** | Manual review + filter → Gemini auto-scoring → Lark Base auto-insert → Current: every video auto-triggers QC, results written to Lark Base in real-time |
| **Current State** | Pipeline generates video → auto Gemini QC → results (pass/fail + specific reasons) real-time to Lark Base → data accumulates passively with production |
| **Next Step** | From "passive collection" to active use: feed accumulated QC data to DSPy for prompt optimization, or train Reward Model. From "just recording" to "production drives improvement" |

### 3.4 `prompt_evaluator/` Module (Zane's implementation)

Standalone Python module, pip installable (`pip install -e .`):

```
prompt_evaluator/
├── models.py               # Pydantic data models (EvalSample, QCResult, RewardBreakdown, etc.)
├── reward_calculator.py    # Multi-dimensional reward scoring
├── prompt_analyzer.py      # Feature extraction + correlation analysis
├── optimizer.py            # OPRO-style meta-prompt optimizer (to be replaced)
├── calibration.py          # QC vs human label calibration
├── kie_client.py           # Kie.ai API wrapper (Seedream image + Seedance video)
├── gemini_client.py        # GeminiVideoQC (multimodal) + GeminiLLM (text gen)
├── pipeline.py             # EvalPipeline orchestrator
├── qc_client.py            # QC client interface
└── tests/                  # 141 unit tests, all passing
```

**Core classes**:
- `RewardCalculator` — Multi-dimension reward: QC pass/fail + aesthetic + motion + adherence scores. Configurable weights. Supports `fit_weights()` grid search.
- `PromptAnalyzer` — Extracts prompt features (camera move, shot size, word count, stability anchors, motion elements) + Pearson correlation with reward scores.
- `PromptOptimizer` — OPRO-style meta-prompt optimization (functional but underperforming — being replaced, see §4.1).
- `QCCalibrator` — Accuracy/precision/recall/F1 computation, per-rule analysis, per-scene breakdown.

**API Clients**:
- `KieClient` — Seedream 4.5 (text→image) + Seedance 1.5 Pro (text/image→video). Budget tracking, exponential backoff polling, 1080p default, `generate_audio` always false.
- `GeminiVideoQC` — Uploads video to Gemini File API → multimodal QC evaluation with comprehensive prompt (10 auto-fail rules, 7 minor issue checks, 4-axis scoring: aesthetic/motion/adherence/scroll-stop, 1-10 each).
- `GeminiLLM` — Text generation for optimizer, currently on gemini-2.5-flash (QC scoring upgraded to gemini-2.5-pro for better judgment).

### 3.5 System Prompt Versions

| Version | Description | Avg Reward | Notes |
|---------|-------------|------------|-------|
| hotel_v1.txt | Original 5-section template (shot type + scene + camera + motion + stability anchor) | ~71 | Baseline. "Single continuous shot" opener. Negative constraints included. |
| hotel_v2_diverse.txt | v1 with more camera move variety | ~71 | Nearly identical to v1 — confirms prompt wording isn't the bottleneck for simple changes |
| hotel_v3.txt | Rewritten based on Seedance official guide. Subject-first structure, no negative constraints, richer descriptions, 25-40 words | Testing | Structural overhaul. Removed stability anchors. Added degree adverbs guidance. |

---

## 4. Technical Evolution Direction

### 4.1 Prompt Optimization Research

| Method | Details |
|--------|---------|
| **[OPRO](https://github.com/google-deepmind/opro)** | DeepMind 2023. LLM-as-Optimizer: show LLM high/low scoring samples → it summarizes "what makes a good prompt" → generates new candidates → test → feedback → iterate. GSM8K +8%, Big-Bench Hard +50%. Fun finding: "Take a deep breath and work on this step by step" > "Let's think step by step" — optimal prompts can't be intuitively written, require search. **Limitation**: Needs hand-written Meta-Prompt per task, high maintenance. Theory is useful, practice is limited. |
| **[DSPy](https://github.com/stanfordnlp/dspy)** ⭐ | Stanford 2024. Turns prompt engineering into "programming": define modules and signatures, framework auto-handles prompt construction and optimization. Like PyTorch eliminates manual gradient math, DSPy eliminates manual prompt writing. Multiple built-in optimizers (OPRO/MIPO/BootstrapFewShot). **2025 additions**: MIPROv2 (Bayesian, more efficient than random search) / SIMBA (hard sample identification) / GEPA (reflective evolution). **Advantages**: Modular (compose multiple LLM calls) + Testable (like writing unit tests) + Traceable (know what optimization changed). **This is our primary choice.** |
| **[TextGrad](https://github.com/zou-group/textgrad)** | Nature 2024. Uses LLM text feedback as "gradients": when output is bad → LLM analyzes "what went wrong" (compute loss) → "backpropagates" to prompt with targeted fixes (gradient update). LeetCode-Hard +20%, GPT-4o Q&A 78%→92%. **Limitation**: Each optimization round needs multiple LLM calls (generate→evaluate→feedback→modify), costly. Video generation + evaluation time stacks up. **Backup option.** |
| **[VPO](https://openaccess.thecvf.com/content/ICCV2025/papers/Cheng_VPO_A...)** | ICCV 2025. Video Prompt Optimization — the only method specifically for video generation prompt optimization. Core idea: collect human preference data on videos, train prompt optimizer to convert "user intent" into "model-friendly prompts". Outperforms baselines on instruction-following / video quality / overall. **Most relevant academic research to our use case.** |

**Takeaway**: Prompt optimization can be systematic, data-driven, and engineerable — not just intuitive tuning. Our plan: DSPy for engineering implementation + VPO's video-specific thinking + TextGrad as backup. Goal: build a sustainable, iterative optimization pipeline.

### 4.2 Our OPRO Results (what we learned)

Round 1 (baseline, 19 samples across 10 hotel scenes): avg reward = 70.8
Round 2 (OPRO-improved prompts): avg reward = 62.9 (WORSE by -7.9)

**What happened**: OPRO added more constraint words ("maintain structural integrity", negative descriptions) which Seedance ignores or misinterprets. The "optimization" was in the wrong dimension entirely.

**Key insight**: Prompt optimization isn't just about constraint wording. Dimensions to explore:
1. Structure/format — sentence order, segmentation
2. Camera vocabulary — which specific verbs the model responds to best
3. Scene decomposition — whole-scene vs element-by-element descriptions
4. Temporal guidance — "first 2 seconds..." vs single description
5. Negative space — what NOT to say may matter more than what to say
6. Few-shot — show model "good prompt → good result" pairings
7. Style anchoring — known visual style words (cinematic, editorial, ambient)
8. Detail density — too little = uncontrolled, too much = selective ignoring

### 4.3 Video Quality Evaluation Alternatives

| Method | Details |
|--------|---------|
| **Current: Gemini QC** | Custom prompt → Gemini 2.5 Pro (multimodal). Auto-Fail (任一即失败): 人脸变形(五官错位) / 手指错误(6根、融合) / 循环artifact / 穿透问题 — 一眼能看出AI生成的. Minor Issues: 累计≥2则Fail (轻微闪烁, 颜色不一致, 边缘细节丢失) — 单独一个可接受, 累积起来观感变差. **Limitation**: scoring criteria hand-written, not benchmarked; inter-run consistency unvalidated; general-purpose VLM, not video-quality-specialized. |
| **[VideoScore](https://github.com/TIGER-AI-Lab/VideoScore)** | TIGER-AI-Lab, EMNLP 2024. 专为AI视频评测训练, 5维度与人工判断相关性77%. 相比通用VLM, 它见过大量AI视频典型瑕疵, 识别更准确. VideoScore2 (2025) is the follow-up with chain-of-thought explanations, available as HF Space. |
| **[VBench 2.0](https://vchitect.github.io/VBench-project/)** | CVPR 2025. Standard benchmark suite: Human Fidelity, Controllability, Creativity, Physics, Aesthetics. Good for standardized reference, but it's a benchmark not an evaluator — need to implement each dimension's computation. |

**Improvement directions**:
- 专业模型: VideoScore/VideoScore2 补充 Gemini (验证专业模型效果)
- 自训练: 用运行pipeline积累的QC Structured Data训练Reward Model, 针对酒店场景定制/场景区分. 比如"水面循环"在泳池是大问题(视觉焦点).
- 目标 QC Accuracy > 85%

**Plan**:
- Short-term: Gemini QC + VideoScore2 补充 (验证专业模型效果) + consistency check
- Mid-term: 积累足够数据后训练 Reward Model (定制化场景判断)
- Long-term: Align scoring dimensions with VBench 2.0 standards (Human Fidelity / Physics / Commonsense)

---

## 5. Model Training & Closed Loop

### 5.1 The Full Loop

```
Generate Video → QC Auto-Evaluation → Data Accumulation → Train Reward Model → RLHF/DPO → Improved Generation → Loop Back
```

Positive feedback cycle: more generation = more data = better model = better generation.

### 5.2 Current State

```
[IMPLEMENTED]                              [PLANNED]
生成 Seedance → 评测 Gemini QC → 数据积累 Lark  →  Reward Model → RLHF / DPO
                                                    ↑                    |
                                                    └────────────────────┘
```

### 5.3 Why Infrastructure Matters

Causal chain: No evaluation system → No Reward Signal → No RLHF/DPO → Model can't evolve.

What we're doing now (structured prompt / auto QC / data formatting) is ALL preparation for this closed loop. Surface-level it's "make prompts better", but the chain involves multiple models: MLM for scene analysis, VLM for QC, Seedance for generation. **Prompt optimization is the short-term play; long-term is using Reward signals to drive model evolution itself. 评测系统的终极价值是产出能驱动模型进化的数据。**

### 5.4 Three Concrete Directions

| Direction | Details |
|-----------|---------|
| **方向1: QC数据训练 Reward Model** | 参考 VideoScore 架构 + VBench-2.0 评估维度, 训练酒店场景专用视频评分模型 |
| **方向2: Reward信号优化生成模型** | 参考 T2V-Turbo-v2 做法, 用 RLHF/DPO 让模型学会"生成高分视频" (长期目标, 需要数据量和计算资源) |
| **方向3: Fine-tune Prompt Generator** | 参考 VPO 方法, 用 QC 高分视频对应的 Prompt 微调 Prompt 生成模块 (更轻量, 更快见效) |

### 5.5 Validation Case: T2V-Turbo-v2

[T2V-Turbo-v2](https://t2v-turbo-v2.github.io/) (2024) proves the closed-loop approach works: VBench 85.13, beat Gen-3/Kling (academic reward-based methods beating industry products).

**Method**: 3 Reward Model combination:
- Visual Quality → HPSv2.1
- Text Alignment → ViCLIP
- Aesthetic → PickScore

Multi-reward signal combination outperforms single signal. 4-step generation quality beats 50-step. Inference 12.5× faster.

**Takeaway**: Reward signal DESIGN matters more than raw training data. 10× more data might only improve 10%, but a well-designed reward signal can improve 50%.

### 5.6 T2V-Turbo-v2 vs Our Approach

| Dimension | T2V-Turbo-v2 | Our Plan |
|-----------|--------------|----------|
| **Reward Model** | Generic (CLIPScore, PickScore, HPSv2) | 自训练, 场景定制 (如"泳池水面循环"这种瑕疵通用模型识别不出) |
| **Training Data** | Public datasets (量大但不垂直) | 自己积累的 (video, prompt, qc_result) (量小但高度垂直) |
| **Optimization Target** | Generic quality (好看、清晰、流畅) | 生服特定指标 (真实感、静态稳定性、品牌调性) |
| **Evaluation Benchmark** | VBench 16 dimensions | 自定义维度; 可参考 VBench-2.0 新增 Human Fidelity / Physics 维度 |

---

## 6. Current Status & Test Results

### 6.1 API Spend

- Kie.ai: **$20.31** total (as of 2026-03-12)
  - ~20 Seedream image generations
  - ~39 Seedance video generations
  - Split across Mar 10 (~$8) and Mar 11 (~$12)
- Gemini: ~70 calls (mix of Flash and Pro), likely $0 on AI Studio free tier

### 6.2 Test Results Summary

**Round 1 — Baseline (19 samples, 10 scenes)**

High performers (reward 88+): pool, simple room, exterior
Mid performers (55-82): restaurant, spa
Low performers (<60): lobby, beach, bathroom

Most common auto-fails:
- `object_morphing` — 8 occurrences (furniture, chandeliers, architectural elements changing shape)
- `action_loop` — 5 occurrences (water, steam, vegetation repeating cycles)
- `structural_collapse` — 4 occurrences (walls, frames, ceilings warping)

**Round 2 — OPRO "Improved" (16 samples)**
Average reward dropped from 70.8 → 62.9. Constraint-based optimization made things worse.

**Key Finding**: Scene complexity is the dominant factor, not prompt wording. Simple scenes (pool, single room) score well; complex scenes (lobby with chandeliers, beach with people) consistently fail. This is likely a Seedance 1.5 Pro model-level limitation.

### 6.3 Module Status

| Component | Status | Notes |
|-----------|--------|-------|
| models.py | ✅ Done | Full Pydantic models, well-tested |
| RewardCalculator | ✅ Done | Multi-dim scoring, configurable weights |
| PromptAnalyzer | ✅ Done | Feature extraction + correlation |
| QCCalibrator | ✅ Done | Full metrics suite |
| PromptOptimizer (OPRO) | ⚠️ Functional but underperforming | Being replaced — see §4.1 |
| kie_client.py | ✅ Done | Live-tested, 1080p default, budget tracking |
| gemini_client.py | ✅ Done | QC on 2.5 Pro, LLM on 2.5 Flash |
| pipeline.py | ✅ Done | Full orchestration |
| System Prompt v3 | 🔄 In Progress | Rewritten, partial test (3/12 samples). Full comparison pending. |
| Tests | ✅ 141 passing | |

---

## 7. Leo's Status Assessment (from his doc)

| Status | Content |
|--------|---------|
| **已完成** | 结构化Prompt: 五段式模板稳定运行, 每天生成的视频都用统一结构. QC评分模块: Gemini 2.5 Pro 自动评分上线, Auto-Fail + Minor Issues 分层逻辑就绪. Pipeline集成: QC结果自动写入Lark Base, Dashboard可查询, 数据持续积累. |
| **进行中** | 数据积累: 每天随生产自然积累QC数据, 目前已有数百条记录. 人工校准: 开始人工审核一批QC结果, 建立Ground Truth. |
| **下一步** | 1. 人工标注校准: 选取100-200条样本, 人工判定Pass/Fail, 和Gemini对比验证Accuracy (目标>85%), 同时识别Gemini系统性偏差, 这批标注也是Reward Model种子数据. 2. DSPy集成: 把Prompt生成模块用DSPy重写, 定义模块签名和评估函数, 接入QC数据, 让框架自动搜索最优结构. 3. Reward Model训练 (数据量依赖). |

---

## 8. Phases & Roadmap

### Phase 1: Evaluation Infrastructure (Current)
- [x] Structured prompt template (5-section)
- [x] Kie.ai API integration (image + video)
- [x] Gemini QC auto-evaluation (multimodal, comprehensive prompt)
- [x] RewardCalculator with multi-dimensional scoring
- [x] PromptAnalyzer feature extraction
- [x] Full E2E pipeline: generate → QC → reward → analyze
- [x] 141 unit tests, pip installable
- [ ] System prompt v3 full comparison test (12 samples)
- [ ] Gemini QC consistency validation (3-run median)
- [ ] VideoScore2 HF Space integration test

### Phase 2: Data-Driven Optimization
- [ ] Replace OPRO with DSPy (primary) or TextGrad (backup)
- [ ] Implement VPO-inspired video-specific prompt optimization logic
- [ ] Expand prompt search dimensions (structure, vocabulary, temporal, few-shot)
- [ ] Accumulate 100+ scored samples for statistical significance
- [ ] Feature importance analysis → identify which prompt dimensions actually move the needle

### Phase 3: Pipeline Integration
- [ ] Connect prompt_evaluator module into Leo's production pipeline
- [ ] Wire QC results → Lark Base auto-write
- [ ] Integrate with existing infrastructure on Leo's machine (or rebuild here)
- [ ] End-to-end: production generates → auto QC → data accumulates → optimizer improves → loop

### Phase 4: Reward Model & Closed Loop
- [ ] Design multi-signal reward (visual quality + text alignment + aesthetic — T2V-Turbo-v2 pattern)
- [ ] Train Reward Model on accumulated QC data
- [ ] RLHF/DPO pipeline for generation model improvement
- [ ] Positive feedback loop: generate → score → train → improve → generate

---

## 9. Infrastructure Notes

Leo's existing infrastructure (structured prompt, Gemini QC → Lark Base pipeline integration) is on his other computer. We'll rebuild as needed here since Zane is more in-sync with current code state. The main goal is integration into the whole pipeline at the end.

### API Access
- **Kie.ai**: Seedream 4.5 (text→image) + Seedance 1.5 Pro (text/image→video). Key stored in `.credentials/kie-api.json`.
- **Gemini**: Google AI Studio key. Used for QC (2.5 Pro) and prompt generation (2.5 Flash). Key stored in `.credentials/gemini-api.json`.
- **Seedance 2.0**: NOT available via API yet. Kie.ai has a placeholder page but it's "Coming Soon". Official BytePlus API delayed indefinitely due to copyright dispute. Third-party wrappers exist (PiAPI, Atlas Cloud) but unofficial. We stay on 1.5 Pro for now — interface should be swap-compatible when 2.0 opens.

### Constraints
- `generate_audio`: always `false`
- Default resolution: 1080p
- Video duration: 8 seconds (only accepted value for Seedance 1.5 Pro)
- Aspect ratio: 9:16 (vertical, for short-form video)
- No GPU environment available — use API-based and HF Space solutions only
- Budget: monitor Kie.ai spend, stop and report if anomalous

---

## 10. Key References

- [OPRO — Large Language Models as Optimizers](https://github.com/google-deepmind/opro) (DeepMind, ICLR 2024)
- [DSPy — Programming with Foundation Models](https://github.com/stanfordnlp/dspy) (Stanford, 2024-2025)
- [TextGrad — Automatic Differentiation via Text](https://github.com/zou-group/textgrad) (Nature, 2024)
- [VPO — Video Prompt Optimization](https://openaccess.thecvf.com/content/ICCV2025/papers/Cheng_VPO_A...) (ICCV 2025)
- [T2V-Turbo-v2](https://t2v-turbo-v2.github.io/) (2024) — Reward-guided video generation SOTA
- [Anthropic Evaluation Harness](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) — Pipeline evaluation philosophy
- [VideoScore2](https://huggingface.co/spaces/TIGER-Lab/VideoScore2) (TIGER-AI-Lab, 2025) — Specialized video quality scorer
- [VBench 2.0](https://vchitect.github.io/VBench-project/) (CVPR 2025) — Video generation benchmark suite
