# KIE API Reference (for prompt_evaluator)

## Base URL
`https://api.kie.ai`

## Auth
`Authorization: Bearer <API_KEY>`

## Create Task (both models)
POST `/api/v1/jobs/createTask`

### Seedream 4.5 - Text to Image
```json
{
  "model": "seedream/4.5-text-to-image",
  "callBackUrl": "https://your-domain.com/api/callback",
  "input": {
    "prompt": "...",
    "aspect_ratio": "1:1",
    "quality": "basic"
  }
}
```

### Seedance 1.5 Pro - Video Generation
```json
{
  "model": "seedance-1.5-pro",
  "callBackUrl": "https://your-domain.com/api/callback",
  "input": {
    "prompt": "...",
    "input_urls": ["https://..."],  // 0-2 input images
    "aspect_ratio": "1:1",
    "resolution": "720p",
    "duration": "8",  // NOTE: must be string, only "8" accepted for 1.5 Pro
    "fixed_lens": false,
    "generate_audio": false  // ALWAYS false per Leo's instruction
  }
}
```

Response: `{"code": 200, "msg": "success", "data": {"taskId": "task_xxx"}}`

## Query Task Status
GET `/api/v1/jobs/recordInfo?taskId=<taskId>`

States: waiting → queuing → generating → success / fail

Response (success):
```json
{
  "code": 200,
  "data": {
    "taskId": "task_xxx",
    "model": "...",
    "state": "success",
    "resultJson": "{\"resultUrls\":[\"https://...\"]}", 
    "costTime": 15000,
    "completeTime": 1698765432000,
    "createTime": 1698765400000
  }
}
```

## Constraints
- generate_audio: ALWAYS false (saves money, audio handled separately in pipeline)
- Budget is limited - monitor spending, stop and report if risk of draining
- API key has no extra quota - be conservative
- Download results immediately, URLs expire after ~24h
- Use callbacks for production; polling with exponential backoff (start 2-3s)
