"""Run end-to-end evaluation with StubQCClient simulating realistic QC outcomes.

This demonstrates the full pipeline including QC scoring and reward calculation.
Uses real Kie.ai API for generation, StubQCClient for QC (no Gemini key needed).

For real QC, replace StubQCClient with GeminiQCClient:
    from prompt_evaluator.qc_client import GeminiQCClient
    qc = GeminiQCClient(api_key="YOUR_GEMINI_KEY", qc_prompt=open("qc_prompt.txt").read())

Usage:
    python3 run_eval_with_qc.py
    python3 run_eval_with_qc.py --scenes 2   # Run 2 scenes only
    python3 run_eval_with_qc.py --no-real-api  # Use cached/stub everything
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

from prompt_evaluator.kie_client import KieClient
from prompt_evaluator.qc_client import StubQCClient
from prompt_evaluator.pipeline import EvalPipeline, DEFAULT_HOTEL_SCENES, SceneSpec
from prompt_evaluator.calibration import QCCalibrator
from prompt_evaluator.models import EvalSample, QCResult, HumanLabel, PromptInfo, InputInfo, OutputInfo

# ── System prompt (v1) ────────────────────────────────────────────────────────
SYSTEM_PROMPT_V1 = Path("system_prompts/hotel_v1.txt").read_text()

# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--scenes", type=int, default=4, help="Number of scenes to run")
parser.add_argument("--no-real-api", action="store_true", help="Skip real API calls")
args = parser.parse_args()

scenes = DEFAULT_HOTEL_SCENES[:args.scenes]

# ── QC client: stub with mix of pass/fail to simulate real QC ────────────────
# 75% pass rate, occasional action_loop and sky_freeze_missing
import random
random.seed(42)

class RealisticStubQC:
    """Simulates a real QC client with realistic pass/fail distribution."""
    FAIL_MODES = [
        (["action_loop"], []),
        (["sky_freeze_missing"], []),
        ([], ["slight_flicker"]),
        ([], ["edge_distortion"]),
        (["action_loop"], ["slight_flicker"]),
    ]

    def evaluate(self, video_url: str, **kwargs):
        r = random.random()
        if r < 0.70:
            return {
                "pass": True,
                "confidence": round(random.uniform(0.82, 0.97), 2),
                "human_present": False,
                "auto_fail_triggered": [],
                "minor_issues": random.choice([[], ["slight_flicker"]]),
                "summary": "Video passes QC. Clean cinematography.",
            }
        else:
            fail_mode = random.choice(self.FAIL_MODES)
            return {
                "pass": len(fail_mode[0]) == 0,
                "confidence": round(random.uniform(0.65, 0.85), 2),
                "human_present": False,
                "auto_fail_triggered": fail_mode[0],
                "minor_issues": fail_mode[1],
                "summary": f"QC issues detected: {fail_mode[0] or fail_mode[1]}",
            }

qc = RealisticStubQC()

if args.no_real_api:
    print("⚠ --no-real-api: skipping generation, generating mock report only")
    # Build mock samples for calibration demo
    samples = []
    for i in range(10):
        qc_result_data = qc.evaluate(f"https://mock/{i}.mp4")
        human_pass = qc_result_data["pass"] or random.random() < 0.15  # 15% false negative
        samples.append(EvalSample(
            sample_id=f"mock_{i:03d}",
            input=InputInfo(scene_type=random.choice(["pool", "room", "lobby", "spa"])),
            prompt=PromptInfo(generated_prompt=f"Single continuous shot. Mock prompt {i}. Camera pushes slowly. Element stays fixed."),
            output=OutputInfo(video_url=f"https://mock/{i}.mp4"),
            qc_result=QCResult(**qc_result_data),
            human_label=HumanLabel(
                labeled_by="mock",
                human_pass=human_pass,
                human_score=random.randint(5, 9) if human_pass else random.randint(2, 5),
            )
        ))

    calibrator = QCCalibrator()
    report = calibrator.calibrate(samples)
    print(f"\n{'='*50}")
    print("MOCK CALIBRATION REPORT")
    print(f"{'='*50}")
    print(f"  Accuracy:  {report.accuracy:.1%}")
    print(f"  Precision: {report.precision:.1%}")
    print(f"  Recall:    {report.recall:.1%}")
    print(f"  F1:        {report.f1:.1%}")
    print(f"  Confusion matrix: TP={report.confusion_matrix[0][0]}, FP={report.confusion_matrix[0][1]}, FN={report.confusion_matrix[1][0]}, TN={report.confusion_matrix[1][1]}")
    for rec in report.recommendations:
        print(f"  → {rec}")
    sys.exit(0)

# ── Real run ──────────────────────────────────────────────────────────────────
client = KieClient(max_requests=args.scenes * 2 + 4)

pipeline = EvalPipeline(
    system_prompt=SYSTEM_PROMPT_V1,
    kie_client=client,
    qc_client=qc,
    output_dir="eval_results",
)

print(f"\n{'='*60}")
print(f"Running evaluation: {len(scenes)} scenes | QC: RealisticStubQC")
print(f"Budget: {client.max_requests} max API calls")
print(f"{'='*60}\n")

batch = pipeline.evaluate_batch(scenes=scenes, save=True)

print(f"\n{'='*60}")
print(batch.summary())
print(f"{'='*60}\n")

for r in batch.results:
    status = "✓" if r.success else "✗"
    reward = f"{r.reward.total_score:.0f}" if r.reward else "N/A"
    qc_status = "QC-pass" if (r.qc_result and r.qc_result.qc_pass) else "QC-fail"
    cam = r.features.camera_move_type if r.features else "?"
    words = r.features.word_count if r.features else 0
    print(f"  {status} [{r.scene_type:6s}] reward={reward:>5s} | {qc_status} | cam={cam} | words={words}")
    if r.video_url:
        print(f"         video: {r.video_url}")
    if r.error:
        print(f"         error: {r.error}")

# ── Report ────────────────────────────────────────────────────────────────────
report_md = pipeline.generate_report(batch)
report_path = "eval_results/report_latest.md"
Path(report_path).write_text(report_md)
print(f"\nReport saved → {report_path}")

# ── API stats ─────────────────────────────────────────────────────────────────
print(f"\n{client.stats.summary()}")
