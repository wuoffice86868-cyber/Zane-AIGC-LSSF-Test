# Infrastructure Audit & Phased Planning

Date: 2026-03-14
Researcher: Zane Jenkins

---

## PART 1: INFRASTRUCTURE AUDIT

### Module-by-Module Assessment

Total codebase: 6,137 lines across 22 Python files (11 source + 11 test).

---

#### 1. models.py (165 lines) — ✅ KEEP, minor updates

**What it does**: Pydantic data models for the entire system (QCResult, EvalSample, RewardBreakdown, PromptFeatures, etc.)

**What's good**:
- Clean type annotations, Pydantic validation
- Good separation of concerns (input/output/prompt/QC/human label as separate models)
- Extensible structure

**What needs changing**:
- `SceneType` enum is hardcoded to hotel scenes (pool, room, bathroom...). Needs to be configurable or replaced with free-form strings for generalizability.
- `CameraMove` enum is incomplete — missing dolly, track, crane, zoom that Seedance docs mention.
- Missing: `ImageAnalysisResult` model (for the upstream Gemini image analysis output). This is a gap — we have the production prompt but no data model for its output.
- Missing: Multi-dimensional QC scores as first-class fields. Currently aesthetic/motion/adherence scores are extracted ad-hoc in RewardCalculator from raw dicts. Should be structured in QCResult.
- `PromptFeatures` has `has_sky_freeze` and `has_single_continuous_shot` — both are v1/v2 concepts we've deprecated. Dead fields.

**Verdict**: Keep, refactor. Add ImageAnalysisResult, make SceneType configurable, add structured QC scores to QCResult, remove dead fields.

---

#### 2. reward_calculator.py (256 lines) — 🔄 REWRITE core logic

**What it does**: Calculates reward scores from QC results. Two modes: QC-only (pass/fail + penalties) and multi-dimensional (weighted sum of QC + aesthetic + clip + motion).

**What's good**:
- Clean interface: `calculate()` and `batch_calculate()`
- Weight fitting via grid search (conceptually sound)
- Auto-extraction of Gemini extra scores from raw dict

**What needs rewriting**:
- The 4-dimension model (qc/aesthetic/clip/motion) is ad-hoc. From our research, the industry standard is 3 dimensions: **visual quality, text-video alignment, physical/common-sense consistency** (VideoScore2's taxonomy). Should align.
- `clip` as a dimension name is misleading — it's actually "prompt adherence" from Gemini, not a CLIP score.
- The reward formula is hand-tuned heuristics (pass_score - issue_penalty + confidence_boost). No theoretical basis. Should support pluggable reward functions (Gemini QC, VideoScore2, VisionReward, ViCLIP) that can be swapped.
- `fit_weights` grid search is O(n^3) with step=0.05. Works for 4 dimensions but won't scale. Also requires human labels we don't have.
- The confidence_boost logic (>=0.95 → +10, >=0.80 → +5) is arbitrary. No evidence this helps.

**Verdict**: Rewrite. New design:
- Pluggable scorer interface: any scorer returns `{dimension: score}` dict
- Configurable dimension weights (not hardcoded to 4 specific dimensions)
- Composite reward = weighted sum of scorer outputs
- Remove hand-tuned heuristics (pass_score/issue_penalty/confidence_boost)
- Keep `fit_weights` concept but generalize to N dimensions

---

#### 3. optimizer.py (422 lines) — ❌ TEAR DOWN

**What it does**: OPRO-style meta-prompt optimizer. Builds a meta-prompt with (prompt, score) history sorted ascending, sends to LLM, gets back improved prompt.

**Why tear down**:
- OPRO is confirmed outdated. We ran 2 rounds, it made things WORSE (-8 points).
- The meta-prompt template hardcodes hotel-specific constraints ("Single continuous shot", "Sky and clouds remain completely still") that we've proven don't work.
- The approach (sort by score, hope LLM finds patterns) has no directional gradient. DSPy critique module already proved more effective.
- 422 lines of dead code.

**Verdict**: Delete entirely. Replaced by dspy_optimizer.py (already done) and will be further evolved toward VPO-inspired architecture.

---

#### 4. dspy_optimizer.py (695 lines) — 🔄 KEEP structure, REFACTOR internals

**What it does**: DSPy-based optimizer with 3 modules (SceneToPrompt, PromptCritique, TemplateImprover). Uses DSPy Signatures and ChainOfThought.

**What's good**:
- Architecture is sound: structured input → prompt generation → critique → template improvement
- Protocol-based interfaces (VideoGenerator, VideoScorer) enable pluggability
- SceneInput dataclass maps to Leo's production image analysis output
- The 3-module decomposition (generate / critique / improve) is clean

**What needs refactoring**:
- DSPy dependency is heavy and we concluded DSPy itself isn't the right framework for T2V optimization. The MODULES and LOGIC are good; the DSPy framework wrapper adds complexity without domain-specific value.
- Should extract the core logic (critique-based improvement loop) from DSPy into a standalone implementation. Keep the Signature-like structure but don't require DSPy as a dependency.
- The optimization loop doesn't implement VPO's key insight: dual feedback (text-level + video-level). Currently only uses video-level quality scores.
- Missing: PhyPrompt's dynamic reward curriculum concept (start with semantic fidelity, shift to physics).
- `make_video_quality_metric` is tightly coupled to DSPy's metric interface.

**Verdict**: Keep the 3-module architecture, extract from DSPy wrapper into standalone Python. Add text-level feedback (prompt quality check before generation). Add dynamic reward weighting.

---

#### 5. gemini_client.py (457 lines) — 🔄 KEEP, refactor into abstract interface

**What it does**: Two classes: GeminiVideoQC (video upload + multimodal QC scoring) and GeminiLLM (text generation).

**What's good**:
- GeminiVideoQC works end-to-end (tested with real API calls)
- The QC prompt (HOTEL_VIDEO_QC_PROMPT) is comprehensive: 10 auto-fail rules + 7 minor issues + aesthetic/motion/adherence 1-10 scoring
- File upload → wait for processing → multimodal analysis flow is correct

**What needs refactoring**:
- HOTEL_VIDEO_QC_PROMPT is 300+ lines hardcoded in the module. Should be externalized to a prompt file (like Leo's production prompts in prompts/).
- The QC prompt is hotel-specific. Need a configurable prompt loader for different domains.
- GeminiVideoQC should implement the same QCClientProtocol as qc_client.py defines, but currently they're separate implementations with different interfaces.
- Missing: inter-run consistency check (score same video 3x, take median).
- Should be one of several possible QC backends, not THE QC backend. Need abstract interface that VideoScore2/VisionReward/Gemini all implement.

**Verdict**: Keep working code, wrap behind abstract interface, externalize QC prompt, add consistency checking.

---

#### 6. kie_client.py (514 lines) — ✅ KEEP, minor updates

**What it does**: Kie.ai API wrapper for Seedream image generation and Seedance video generation. Create task → poll with backoff → parse result → return URLs. Budget tracking built in.

**What's good**:
- Works end-to-end (tested with real API calls, generated 39+ videos)
- Budget tracking and rate limiting
- Proper error handling and retry logic
- Supports both 720p and 1080p (defaulted to 1080p per Leo's request)

**What needs changing**:
- Hardcoded to Kie.ai. Should be behind a GeneratorProtocol interface so we can swap backends (direct ByteDance API, other providers).
- Default scene descriptions are hotel-specific. Should be configurable.
- `generate_audio` hardcoded to False (correct, but should be configurable).
- Temp file URLs expire in ~24h. Need local download + caching for evaluation data persistence.

**Verdict**: Keep. Add GeneratorProtocol interface, add local file caching.

---

#### 7. qc_client.py (249 lines) — 🔄 MERGE with gemini_client.py

**What it does**: Defines QCClientProtocol + StubQCClient + RandomQCClient. The protocol is good but the real implementation (GeminiQCClient) is in gemini_client.py, creating confusion.

**Verdict**: Merge into a single `evaluation/` subpackage. QCClientProtocol stays as the abstract interface. GeminiVideoQC implements it. VideoScore2Client, VisionRewardClient will also implement it.

---

#### 8. prompt_analyzer.py (262 lines) — 🔄 PARTIAL REWRITE

**What it does**: Extracts prompt features (word count, camera move, shot size, motion elements, etc.) and runs correlation analysis against reward scores.

**What's good**:
- Feature extraction is thorough (camera moves, speed modifiers, shot sizes, motion patterns)
- Correlation analysis using pandas/numpy

**What needs changing**:
- Camera vocabulary is incomplete and based on v1/v2 assumptions. Needs updating from Seedance official docs.
- `has_single_continuous_shot` and `has_sky_freeze` features are deprecated (proven ineffective).
- Correlation analysis requires many samples for significance; with our current 35 samples it's noise.
- Missing: Seedance PE layer awareness. Since Seedance rewrites prompts internally, features of our prompt may not correlate with video quality — the PE layer mediates.

**Verdict**: Keep feature extraction logic, update vocabulary, remove deprecated features, add caveat about PE layer mediation.

---

#### 9. calibration.py (308 lines) — ✅ KEEP, defer usage

**What it does**: Compares automated QC against human labels. Computes accuracy/precision/recall/F1, per-rule analysis, generates recommendations.

**What's good**:
- Clean implementation of standard calibration metrics
- Per-rule analysis is valuable for identifying which auto-fail rules are too strict/lenient
- Generates actionable recommendations

**Current status**: We have no human labels, so this module hasn't been used. With Gemini-as-human-substitute approach, we'd use VideoScore2 vs Gemini cross-calibration instead.

**Verdict**: Keep for future use when we have human labels or cross-scorer calibration data. Low priority for refactoring.

---

#### 10. pipeline.py (819 lines) — 🔄 SIGNIFICANT REFACTOR

**What it does**: End-to-end orchestrator: image gen → video gen → QC → reward → analysis → optimization.

**What's good**:
- Complete orchestration of all components
- BatchResult with summary statistics
- Scene definitions and defaults
- JSON result saving for analysis

**What needs refactoring**:
- Too monolithic (819 lines, the largest file). Should be decomposed into: generation pipeline, evaluation pipeline, optimization pipeline.
- Tightly coupled to hotel domain (DEFAULT_HOTEL_SCENES, scene-specific image prompts).
- DSPy and OPRO both wired in with fallback logic. Remove OPRO, simplify.
- `_generate_prompt` uses raw string formatting when it should use the structured SceneInput → prompt template flow from production.
- Missing: Leo's actual production flow (image → Gemini image analysis → Seedream enhancement → Gemini cinematography → Seedance). Currently skips image analysis entirely.
- `EvalResult` dataclass duplicates much of what `EvalSample` model already covers.

**Verdict**: Major refactor. Decompose into 3 sub-pipelines. Remove hotel-specific defaults to config. Wire in the full production flow (image analysis → enhancement → cinematography → generation).

---

#### 11. Tests (1,932 lines across 8 test files) — 🔄 UPDATE after refactor

142 tests, all passing. Good coverage of current code. Will need updating as modules change.

---

#### 12. Scripts (11 files) — 🧹 CLEAN UP

Most are one-off experiment runners from specific test sessions. Should consolidate into 2-3 canonical scripts:
- `run_evaluation.py` — standard evaluation run
- `run_optimization.py` — optimization loop
- `run_comparison.py` — A/B comparison between prompt versions

---

### Summary Table

| Module | Lines | Verdict | Priority |
|--------|-------|---------|----------|
| models.py | 165 | Keep + update | Medium |
| reward_calculator.py | 256 | Rewrite core | High |
| optimizer.py | 422 | Delete | High |
| dspy_optimizer.py | 695 | Extract from DSPy, keep logic | High |
| gemini_client.py | 457 | Abstract interface | High |
| kie_client.py | 514 | Keep + minor | Low |
| qc_client.py | 249 | Merge into evaluation/ | Medium |
| prompt_analyzer.py | 262 | Partial rewrite | Medium |
| calibration.py | 308 | Keep, defer | Low |
| pipeline.py | 819 | Major refactor | High |
| Tests | 1,932 | Update after | After all above |
| Scripts | ~1,200 | Consolidate | Low |

---

## PART 2: PHASED PLANNING

### Phase 1: Foundation Cleanup (3-5 days)

**Goal**: Clean codebase with correct abstractions, ready for new integrations.

Tasks:
1. Delete optimizer.py (OPRO) entirely
2. Restructure into subpackages:
   ```
   prompt_evaluator/
     models/          — data models (split by domain)
     evaluation/      — QC scoring backends (abstract + implementations)
     generation/      — video/image generation backends (abstract + implementations)  
     optimization/    — prompt improvement logic (extracted from DSPy)
     analysis/        — prompt feature analysis + correlation
     pipeline/        — orchestration
   ```
3. Define abstract interfaces:
   - `ScorerProtocol`: evaluate(video_url, prompt) → ScorerResult (multi-dimensional)
   - `GeneratorProtocol`: generate_video(prompt, image_url) → GenerationResult
   - `RewardProtocol`: calculate(scorer_results) → RewardBreakdown
4. Externalize all prompts (QC prompt, system prompts) to config files
5. Remove hotel-specific hardcoding (SceneType enum, DEFAULT_HOTEL_SCENES, hotel QC prompt)
6. Update models.py: add ImageAnalysisResult, structured QC scores, remove dead fields
7. Consolidate scripts to 3 canonical runners
8. Update tests to match new structure

**Deliverable**: Refactored codebase, all tests passing, pip installable, README updated.

---

### Phase 2: Evaluation Layer Upgrade (3-5 days)

**Goal**: Move from single Gemini QC to multi-scorer evaluation with cross-validation.

Tasks:
1. Implement VideoScore2Client (via HF Space API)
   - Upload video → get 3-dimension scores + chain-of-thought
   - Compare with Gemini QC on existing 35 videos
   - Measure agreement rate and systematic differences
2. Implement VisionRewardClient (for later optimization use)
   - Evaluate feasibility: CPU inference vs API vs HF Space
3. Add ViCLIP-based alignment score as supplementary metric
   - Frame-level CLIP + ViCLIP on video
4. Design composite reward signal:
   - Configurable weighted combination of multiple scorers
   - Support for scorer agreement requirements (e.g., 2/3 must agree on pass/fail)
5. Add QC consistency checking:
   - Same video scored N times → report variance
   - Median aggregation for stable scores
6. Cross-calibrate Gemini vs VideoScore2:
   - Run both on same 30+ videos
   - Identify systematic biases
   - Generate calibration report

**Deliverable**: Multi-scorer evaluation layer, cross-calibration report, composite reward function.

---

### Phase 3: Optimization Engine Rebuild (5-7 days)

**Goal**: Replace OPRO/DSPy with VPO-inspired architecture that demonstrably improves prompts.

Tasks:
1. Extract optimization logic from DSPy wrapper into standalone Python
   - Keep: 3-module architecture (generate, critique, improve)
   - Keep: Protocol-based interfaces
   - Remove: DSPy dependency, DSPy Signatures
2. Implement VPO-inspired dual feedback:
   - Text-level feedback: Is the rewritten prompt grammatically correct, detailed, safe?
   - Video-level feedback: Did the generated video score well on evaluation?
   - Both signals feed into the improvement cycle
3. Implement PhyPrompt-inspired dynamic curriculum:
   - Round 1-2: Weight semantic fidelity (does video match intent?)
   - Round 3-4: Shift weight toward physical consistency (no morphing, stable geometry)
   - Round 5+: Balance all dimensions
4. Test VPO pre-trained model:
   - Download from github.com/thu-coai/VPO
   - Test on our prompt inputs
   - Compare output quality vs our Gemini-based approach
5. Seedance PE layer investigation:
   - Design test: 100 varied prompts → generate → analyze which aspects are preserved/modified
   - Identify: what input patterns survive the PE rewrite
   - Update optimization strategy based on findings
6. Run minimum 3 optimization rounds with new engine:
   - Target: measurable reward improvement between rounds
   - Minimum 30 samples per round for statistical significance

**Deliverable**: Working optimization engine, PE layer analysis report, demonstrated improvement across rounds.

---

### Phase 4: Production Pipeline Integration (3-5 days)

**Goal**: Wire prompt_evaluator into Leo's actual production flow.

Tasks:
1. Implement ImageAnalyzer class wrapping gemini_image_analysis.txt
   - Input: image URL → Output: structured JSON (scene, subject, camera, lighting, etc.)
2. Wire full production flow:
   - Scraped image → Image analysis → Seedream enhancement → Gemini cinematography prompt → Seedance → QC
3. Lark Base integration (coordinate with Leo on ownership):
   - Read evaluation results from Lark Base
   - Write scores + metadata back
4. Accumulate 100+ scored samples from production flow
5. Design A/B testing infrastructure:
   - Current template vs optimized template
   - Side-by-side scoring on same inputs
   - Statistical significance testing

**Deliverable**: Production-integrated evaluation pipeline, 100+ scored samples, A/B test results.

---

### Phase 5: Closed-Loop Optimization (ongoing)

**Goal**: Self-improving system that gets better with data.

Tasks:
1. Implement continuous optimization loop:
   - New videos from production → auto-scored → feed optimization → improve template → deploy → repeat
2. Domain-specific reward model (if GPU available):
   - Fine-tune VisionReward on our scored data
   - Train domain-aware scorer (hotel/travel content specifics)
3. Monitor for reward hacking:
   - Optimizer gaming the scorer (high scores but bad videos)
   - Human spot-checks on top-scored and bottom-scored videos
4. Periodic reports:
   - Weekly: score distribution, failure mode trends, optimization deltas
   - Monthly: system performance review, scorer accuracy audit

**Deliverable**: Self-improving pipeline, domain reward model, monitoring dashboard.

---

### Timeline Summary

| Phase | Duration | Dependencies | Key Output |
|-------|----------|-------------|------------|
| 1: Foundation | 3-5 days | None | Clean codebase |
| 2: Evaluation | 3-5 days | Phase 1 | Multi-scorer + calibration |
| 3: Optimization | 5-7 days | Phase 2 | Working optimizer |
| 4: Integration | 3-5 days | Phase 3 + Leo's pipeline | Production pipeline |
| 5: Closed-Loop | Ongoing | Phase 4 | Self-improving system |

Total to production-ready (Phase 1-4): ~14-22 days
