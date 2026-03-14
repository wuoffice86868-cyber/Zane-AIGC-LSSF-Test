# Integration Architecture (from Leo's production pipeline)

## Data Flow

```
Production Pipeline → Prompt Evaluator (This Module) → Data Store (Lark Base)
```

## Input Data (what you send)
```json
{
    "image_url": "https://drive.google.com/...",    // Enhanced image
    "video_url": "https://drive.google.com/...",    // Generated video (for QC)
    "poi_name": "Four Seasons Maui"                 // For tracking
}
```

## Output Data (what you get back)
```json
{
    "analysis": {
        "scene_description": "luxury hotel room with king bed...",
        "camera_move": "push",
        "camera_direction": "forward toward the bed",
        "shot_size": "medium",
        "subtle_motion": [],
        "stable_element": "furniture"
    },
    "prompt": "Single continuous shot. Medium shot of a luxury hotel room...",
    "qc_result": {
        "pass": true,
        "auto_fail_triggered": [],
        "minor_issues": [],
        "summary": "Clean video, no artifacts detected"
    }
}
```

## Current Interface

### Option 1: Python Import (Current Implementation)
```python
from web.workers.optimization.ai_clients.gemini_video_prompter import GeminiClient

client = GeminiClient()

# Step 1: Analyze image → get structured analysis
analysis = client.analyze_video_image(image_url)

# Step 2: Generate cinematography prompt
prompt = client.generate_cinematography_prompt(analysis, image_url)

# Step 3: After video generation, QC check
qc_result = client.analyze_video_quality(video_url)
```

### Option 2: API Endpoint (If you prefer HTTP)
```
POST /api/v1/prompt/analyze      # image_url → analysis JSON
POST /api/v1/prompt/generate     # analysis + image_url → prompt string
POST /api/v1/qc/evaluate         # video_url → qc_result JSON
```

## Analysis Schema (Step 1 output)

| Field              | Type   | Example                                                    |
|--------------------|--------|------------------------------------------------------------|
| scene_description  | string | "luxury hotel room with king bed, afternoon sunlight"      |
| main_subject       | string | "king bed with white linens"                               |
| foreground         | string | "potted plant" or "none"                                   |
| background         | string | "window with city view"                                    |
| camera_move        | enum   | push/pull/circle around/move left/move right/pan left/pan right/rise/tilt up |
| camera_direction   | string | "forward toward the bed"                                   |
| shot_size          | enum   | close-up / medium / wide                                   |
| lighting           | string | "warm afternoon sunlight"                                  |
| subtle_motion      | array  | [] or ["water ripples softly"]                             |
| stable_element     | string | "furniture" / "horizon" / "walls"                          |
| human_detected     | bool   | false                                                      |

## Final Prompt Template
```
Single continuous shot. [shot_size] of [scene_description].
The camera [camera_move] slowly [camera_direction], [passing foreground].
[subtle_motion]. [stable_element] stays perfectly still.
```

## Parallel Work Integration Points

| Your Side              | This Module                        | Contract                     |
|------------------------|------------------------------------|------------------------------|
| Image URL ready        | analyze_video_image()              | Pass HTTPS URL, get JSON     |
| Need prompt for Seedance | generate_cinematography_prompt() | Get string ≤45 words         |
| Video URL ready        | analyze_video_quality()            | Pass HTTPS URL, get pass/fail|
| Where to store results | Lark Base write                    | We handle this internally    |

## Integration Pseudo-code
```python
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

## Session Notes (from Leo's environment)
- Session context was lost mid-work and had to restart the re-filter process
- Total re-filter time: 25 minutes for 45 folders (1004 images)
- Average processing: ~2.5 seconds per image (CLIP + YOLO + aesthetic scoring)
