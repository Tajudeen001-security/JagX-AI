# Optional NVIDIA cloud backend

JagX can call NVIDIA’s OpenAI-compatible API (`https://integrate.api.nvidia.com/v1`) when you provide API keys. This is **external inference**, not JagX-trained weights.

## GitHub / deployment secrets

Add repository secrets (Settings → Secrets and variables → Actions):

| Secret | Purpose |
|--------|---------|
| `NVIDIA_API_KEY` | Primary key (`nvapi-...` from [build.nvidia.com](https://build.nvidia.com)) |
| `NVIDIA_API_KEYS` | Optional comma-separated keys for rotation and fewer rate limits |
| `NVIDIA_API_BASE` | Optional; default `https://integrate.api.nvidia.com/v1` |

Locally:

```bash
export NVIDIA_API_KEY=nvapi-...
# or
export NVIDIA_API_KEYS=nvapi-key1,nvapi-key2,nvapi-key3
jagx serve
```

Never commit keys. Never log full keys.

## Public model names (shown in the app)

Clients only see:

| Public id | Typical use |
|-----------|-------------|
| `jagx-chat` | General chat |
| `jagx-fast` | Short / fast |
| `jagx-code` | Coding |
| `jagx-reason` | Hard reasoning |
| `jagx-vision` | Vision / images |
| `jagx-vision-fast` | Faster vision |

Upstream catalog model ids are **server-internal only** and are not returned in API JSON.

Operators may override upstream mapping with env:

```bash
export JAGX_NVIDIA_MODEL_CODE=...
export JAGX_NVIDIA_MODEL_VISION=...
```

## Vision

```json
POST /v1/generate
{
  "prompt": "What is in this image?",
  "capability": "vision",
  "image_url": "https://..."
}
```

or `image_base64` + optional `media_type`.

## Honesty

Responses include `"provider": "nvidia"` and `"external_ai_api_required": true` so the system does not pretend answers are from a local JagX checkpoint.
