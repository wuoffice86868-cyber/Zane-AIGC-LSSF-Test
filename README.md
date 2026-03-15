# prompt_evaluator

---

## 🟢 Status (live)

**Last updated**: 2026-03-15 00:50 UTC  
**Version**: 0.2.0  
**Tests**: 146 passing

**Phase 1 Foundation — DONE** ✓  
**Phase 2 PE Layer Research — IN PROGRESS** 🔄

**Current work**: PE Layer Probe running
- 12 prompt variants × same reference image × Gemini 2.5 Pro QC
- Testing: structure order, length, adverbs, negatives, vocabulary
- Results → v4 system prompt → v4 vs v3 comparison

**Key findings so far**:
- v3 beats v1 by +9.5 avg reward (84.8 vs 75.3)
- OPRO made things worse (-8 pts) — confirmed wrong approach
- Seedance has internal Qwen2.5-14B PE layer that rewrites prompts (arXiv:2506.09113)
- VPO + PhyPrompt are correct optimization methods (not DSPy/TextGrad)
- Negative constraints confirmed ineffective — PE strips them
- VideoScore2 can't discriminate our quality levels (scores everything 4/4/4)

**API spend to date**: Kie.ai ~$26 | Gemini ~$0 (free tier)

---

## Folder Map

```
prompt_evaluator/              ← repo root
│
├── prompt_evaluator/          ← CORE MODULE (pip installable Python package)
│   ├── models.py              │  All Pydantic data types (EvalSample, QCResult, RewardBreakdown, etc.)
│   ├── reward_calculator.py   │  3-dim reward scoring (Foundational/Motion/Aesthetic)
│   ├── prompt_analyzer.py     │  Prompt feature extraction + correlation with scores
│   ├── dspy_optimizer.py      │  DSPy optimizer — critique + template improvement
│   ├── calibration.py         │  QC vs human label accuracy metrics (precision/recall/F1)
│   ├── pipeline.py            │  EvalPipeline orchestrator — wires everything together
│   ├── kie_client.py          │  Kie.ai API wrapper (Seedream 4.5 image + Seedance 1.5 Pro video)
│   ├── gemini_client.py       │  GeminiVideoQC (multimodal scoring) + GeminiLLM (text gen)
│   ├── qc_client.py           │  QC client interface
│   └── tests/                 │  142 unit tests
│       ├── conftest.py        │    Shared fixtures + sample factories
│       ├── test_models.py     │    Data model validation
│       ├── test_reward.py     │    Reward calculation
│       ├── test_analyzer.py   │    Feature extraction
│       ├── test_calibration.py│    Calibration metrics
│       ├── test_optimizer.py  │    Optimizer logic
│       ├── test_pipeline.py   │    Pipeline orchestration
│       ├── test_qc_client.py  │    QC client
│       └── test_kie_client.py │    API client
│
├── prompts/                   ← LEO'S PRODUCTION PROMPTS (from his pipeline)
│   ├── gemini_image_analysis.txt       Upstream: image → structured JSON (scene, subject, lighting, defects...)
│   ├── gemini_cinematography_prompt.txt Downstream: analysis JSON → 30-45 word Seedance prompt
│   └── integration_architecture.md     How the prompts connect in the pipeline
│
├── system_prompts/            ← SYSTEM PROMPT VERSIONS (what we're optimizing)
│   ├── hotel_v1.txt           Baseline. 5-section: shot+scene+camera+motion+stability. Avg reward ~71
│   ├── hotel_v2_diverse.txt   v1 + more camera variety. ~71 (no improvement)
│   ├── hotel_v3.txt           Rewritten per Seedance guide. 4-section, no neg constraints. Avg ~85
│   ├── v1_improved_*.txt      OPRO Round 2 output (made things worse, -8 pts)
│   └── v2_diverse_improved_*.txt  OPRO Round 2 diverse variant
│
├── eval_results/              ← RAW TEST DATA (every evaluation run, timestamped JSON)
│   ├── v3_comparison_*.json   v3 vs v1 comparison runs
│   ├── full_pipeline_*.json   End-to-end pipeline test outputs
│   ├── gemini_qc_*.json       QC scoring test results
│   └── round2_recovered.json  OPRO Round 2 data
│
├── scripts/                   ← TEST RUNNERS (how we run experiments)
│   ├── run_eval.py            Basic evaluation
│   ├── run_full_pipeline.py   Full generate → QC → score pipeline
│   ├── run_v3_comparison.py   v3 vs v1 A/B test
│   ├── run_dspy_test.py       DSPy optimizer tests
│   └── run_*.py               Various experiment scripts
│
├── reports/                   ← GENERATED REPORTS (human-readable summaries)
│   ├── v3_experiment_report.md    v3 comparison analysis
│   ├── v3_experiment_report.pdf   PDF version
│   └── v3_report_*.png            Report screenshots
│
├── research/                  ← RESEARCH DOCS
│   ├── seedance_vocabulary.md          What words/structures Seedance responds to
│   └── seedance_vocabulary_research.md Extended vocabulary research
│
├── docs/                      ← API DOCUMENTATION
│   └── kie_api_reference.md   Kie.ai API endpoint reference
│
├── logs/                      ← RUN LOGS (debug output from pipeline runs)
│
├── PROJECT.md                 ← SOURCE OF TRUTH (full project context, ~390 lines)
├── README.md                  ← THIS FILE (status dashboard + folder map)
├── setup.py                   ← Package setup (pip install -e .)
└── requirements.txt           ← Dependencies: numpy, pandas, scikit-learn, pydantic, dspy
```

### How folders relate to each other

**The flow**: Leo's production prompts (`prompts/`) define how images become videos. System prompts (`system_prompts/`) are the templates we're optimizing. The core module (`prompt_evaluator/`) evaluates and improves those templates. Scripts (`scripts/`) run experiments that produce raw data (`eval_results/`) and human-readable reports (`reports/`). Research (`research/`) feeds insights back into system prompt design.

```
prompts/ (Leo's production templates — the "what")
    ↓ feeds into
prompt_evaluator/ (core module — the "engine")
    ├── generates videos via kie_client.py
    ├── scores them via gemini_client.py
    ├── calculates reward via reward_calculator.py
    ├── analyzes prompt features via prompt_analyzer.py
    └── optimizes templates via dspy_optimizer.py
            ↓ outputs
system_prompts/ (improved versions — the "result")
            ↓ tested by
scripts/ → eval_results/ → reports/
            ↓ insights feed
research/ → back into prompt_evaluator/ design
```

---

## Current Stage

**We are in Phase 1 (Evaluation Infrastructure), nearly complete.**

Phase 1 is about building the measurement + generation loop: can we generate videos, automatically score them, and reliably compare prompt versions? Answer: yes, it works.

Phase 2 (Data-Driven Optimization) is starting — DSPy is integrated but hasn't run a real optimization loop yet. That's the immediate next step.

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Evaluation Infrastructure | ✅ Done — 3-dim reward, OPRO removed, Gemini QC working |
| Phase 2 | PE Layer Research + Prompt Optimization | 🔄 In progress — PE probe running, v4 design pending |
| Phase 3 | Pipeline Integration | 0% — connect into Leo's production pipeline + Lark Base |
| Phase 4 | Reward Model & Closed Loop | 0% — VPO/PhyPrompt training, RLHF/DPO |

See `PROJECT.md` for full phase breakdown with checkboxes.

---

## Install

```bash
pip install -e .
# or with dev dependencies
pip install -e ".[dev]"
```

## Tests

```bash
PYTHONPATH=. python3 -m pytest prompt_evaluator/tests/ -v
# 142 tests, all passing
```

## Core Classes

**RewardCalculator** — Multi-dimension reward: QC pass/fail + aesthetic + motion + adherence. Configurable weights. Grid search for weight tuning.

**PromptAnalyzer** — Extracts features (camera move, shot size, word count, stability anchors) + Pearson correlation with reward scores.

**DSPyOptimizer** — Critique module (identifies WHY prompts fail) + Template improver (rewrites based on evidence). Replaces OPRO.

**QCCalibrator** — Accuracy/precision/recall/F1 vs ground truth labels.

**KieClient** — Kie.ai API: Seedream 4.5 (text→image) + Seedance 1.5 Pro (image→video). 1080p, budget tracking.

**GeminiVideoQC** — Upload video to Gemini File API → multimodal QC (10 auto-fail rules, 4-axis 1-10 scoring).

## References

- [DSPy](https://github.com/stanfordnlp/dspy) (Stanford, 2024-2025) — Primary optimization framework
- [OPRO](https://github.com/google-deepmind/opro) (DeepMind, ICLR 2024) — Legacy optimizer
- [TextGrad](https://github.com/zou-group/textgrad) (Nature, 2024) — Backup optimization approach
- [VideoScore2](https://huggingface.co/spaces/TIGER-Lab/VideoScore2) — Specialized video quality scorer
- [VBench 2.0](https://vchitect.github.io/VBench-project/) (CVPR 2025) — Video generation benchmark
