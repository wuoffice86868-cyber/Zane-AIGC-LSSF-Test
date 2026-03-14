"""Run a real end-to-end evaluation with the 4 default hotel scenes.

This is a one-shot script to validate the full pipeline against the live API.
Budget: 4 image + 4 video = 8 API calls.
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")

from prompt_evaluator.kie_client import KieClient
from prompt_evaluator.pipeline import EvalPipeline, DEFAULT_HOTEL_SCENES

# Conservative budget: 4 scenes = 4 images + 4 videos = 8 calls
client = KieClient(max_requests=12)  # small buffer for retries

system_prompt = """You are a cinematography director creating short hotel marketing video clips.

Rules:
- Always start with "Single continuous shot"
- Use exact Seedance camera verbs: pushes, pulls, circles around, moves left/right, pans left/right, rises, tilts up
- Keep prompts 30-45 words
- Add a speed modifier: slowly, gradually, gently
- For outdoor scenes with sky: add "Sky and clouds remain completely still"
- End with a stability anchor: "[element] stays fixed/still/anchored"
- Include one subtle motion element (water ripples, curtains sway, candle flickers)
- No equipment names, no negative phrases
"""

pipeline = EvalPipeline(
    system_prompt=system_prompt,
    kie_client=client,
    output_dir="eval_results",
)

print(f"\n{'='*60}")
print("Running evaluation on {0} default hotel scenes".format(len(DEFAULT_HOTEL_SCENES)))
print(f"Budget: {client.max_requests} max API calls")
print(f"{'='*60}\n")

batch = pipeline.evaluate_batch(save=True)

print(f"\n{'='*60}")
print(batch.summary())
print(f"{'='*60}\n")

# Print per-scene results
for r in batch.results:
    status = "✓" if r.success else "✗"
    reward = f"{r.reward.total_score:.0f}" if r.reward else "N/A"
    print(f"  {status} [{r.scene_type}] reward={reward}")
    if r.video_url:
        print(f"    video: {r.video_url}")
    if r.error:
        print(f"    error: {r.error}")

# Generate report
report = pipeline.generate_report(batch)
report_path = "eval_results/report.md"
with open(report_path, "w") as f:
    f.write(report)
print(f"\nReport saved to {report_path}")

# Print API stats
print(f"\n{client.stats.summary()}")
