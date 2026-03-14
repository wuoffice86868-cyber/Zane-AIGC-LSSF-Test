# Parallel Work Integration Points

## Interface Contract

| Your Side | This Module | Contract |
|-----------|------------|---------|
| Image URL ready | analyze_video_image() | Pass HTTPS URL, get JSON |
| Need prompt for Seedance | generate_cinematography_prompt() | Get string ≤45 words |
| Video URL ready | analyze_video_quality() | Pass HTTPS URL, get pass/fail |
| Where to store results | Lark Base write | We handle this internally |

## Integration Pseudo-code

```python
# Pseudo-code for integration
from prompt_evaluator import PromptEvaluator

evaluator = PromptEvaluator()  # Or API client

# In your video generation loop:
for image in enhanced_images:
    prompt = evaluator.generate_prompt(image.url)
    video = seedance.generate(image.url, prompt)
    qc = evaluator.evaluate(video.url)

    if qc["pass"]:
        keep_video(video)
    else:
        discard_and_log(video, qc)
```

## Field Schema (from gemini_image_analysis.txt)

### Video Prompt Generator Input Fields

| Field | Type | Example | Values |
|-------|------|---------|--------|
| scene_description | string | "luxury hotel room with king bed..." | - |
| main_subject | string | "king bed with white linens" | - |
| foreground | string | "potted plant on left side" | - |
| background | string | "floor-to-ceiling window with city view" | - |
| camera_move | enum | "push" | push / pull / circle around / move left / move right / pan left / pan right / rise / tilt up |
| camera_direction | string | "forward toward the bed" | - |
| shot_size | enum | "medium" | close-up / medium / wide |
| lighting | string | "warm afternoon sunlight" | - |
| subtle_motion | array | [] or ["water ripples softly"] | - |
| stable_element | string | "furniture" / "horizon" / "walls" | - |
| human_detected | bool | false | - |

### Final Prompt Template

```
Single continuous shot. [shot_size] of [scene_description].
The camera [camera_move] slowly [camera_direction], [passing foreground],
[subtle_motion]. [stable_element] stays perfectly still.
```
