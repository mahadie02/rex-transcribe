# WhisperX API Server

A FastAPI-based Speech-to-Text transcription server using [WhisperX](https://github.com/m-bain/whisperX). Accepts audio via file upload, base64, or URL, and returns transcripts in JSON (n8n-friendly) or other formats.

## Features

- **Multiple Input Methods**: Audio/video uploads and URLs, plus stream URLs (YouTube, Facebook, Instagram, TikTok, X, Threads, etc. via yt-dlp)
- **Flexible Output**: JSON (default, n8n-friendly), TXT, SRT, VTT, TSV
- **Output Detail**: Segments only, words only, or both (for JSON)
- **Segment Splitting**: Splits at natural breaks (comma, semicolon, period, etc.) into separate segments with timestamps; `max_line_width` for long phrases
- **Full WhisperX Parameters**: Model, batch size, language, alignment, diarization, ASR options, VAD options
- **Similar API Style**: Matches pocket-tts-server (.env, API_KEY, structure)
- **GPU/CPU Support**: Auto-detects CUDA; configurable device and compute type

## Installation

### Prerequisites

- Python 3.10+ (3.12 recommended)
- FFmpeg (required for audio loading)
- CUDA 12.x (optional, for GPU acceleration)

#### Install FFmpeg

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

> **Note:** The server uses the system FFmpeg binary (from PATH), not a Python package. Ensure FFmpeg is installed on your system and available in your PATH. The `torchcodec` package is not used (uninstalled) to avoid libtorchcodec/FFmpeg DLL errors on Windows.

### Python Dependencies

1. Create a virtual environment (recommended):
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

2. Install dependencies:
```bash
# Windows: run install.bat (installs deps and removes torchcodec to avoid FFmpeg DLL errors)
install.bat

# Or manually:
pip install -r requirements.txt
pip uninstall torchcodec -y   # Required on Windows to avoid libtorchcodec errors
```

## Configuration

### 1. Create `.env` file

Copy `.env.example` to `.env` and configure:

```env
API_KEY=your_secret_api_key_here
PORT=5505
WHISPER_MODEL=base
DEVICE=cuda
COMPUTE_TYPE=float16
HF_TOKEN=
```

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY` | Secret API key for authentication (required) | - |
| `PORT` | Server port | 5505 |
| `WHISPER_MODEL` | Model: tiny, base, small, medium, large-v2, large-v3 | base |
| `DEVICE` | cuda or cpu | auto (cuda if available) |
| `COMPUTE_TYPE` | float16 (GPU) or int8 (CPU/low memory) | float16 on GPU |
| `HF_TOKEN` | HuggingFace token for speaker diarization | - |

### 2. HuggingFace Token (for Diarization)

To enable speaker diarization:

1. Create a token at [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Accept terms at [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
3. Set `HF_TOKEN` in `.env`

## Running the Server

```bash
python server.py
```

The server will:
- Load the WhisperX model (first run may download models to `./models`)
- Listen on `0.0.0.0:5505` (or your configured `PORT` in `.env`)

## API Reference

### Health Check

```http
GET /health
```

```bash
curl -s http://localhost:5505/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### Transcribe Audio

```http
POST /transcript
```

**Headers:**
- `API_KEY`: Your API key from `.env`

**Content-Type:** `multipart/form-data` (when using file upload) or `application/x-www-form-urlencoded` (when using base64/URL only)

---

### Input Parameters (provide exactly one)

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio` | File | Binary audio file (mp3, wav, m4a, ogg, flac, webm) |
| `audio_base64` | string | Base64-encoded audio data |
| `audio_url` | string | Direct URL to an audio file |
| `video` | File | Binary video file (mp4, mkv, avi, mov, webm) |
| `video_base64` | string | Base64-encoded video data |
| `video_url` | string | Direct URL to a video file |
| `youtube_url` | string | YouTube video URL |
| `fb_url` | string | Facebook video / reel URL |
| `insta_url` | string | Instagram reel / post URL |
| `tiktok_url` | string | TikTok video URL |
| `x_url` | string | X (Twitter) video URL |
| `threads_url` | string | Threads post URL |

Stream URLs (`youtube_url`, `fb_url`, `insta_url`, `tiktok_url`, `x_url`, `threads_url`) use [yt-dlp](https://github.com/yt-dlp/yt-dlp) (same pipeline as YouTube). Many sites are supported; availability can change when platforms update.

---

### Output Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_format` | string | json | Output format: **json**, txt, srt, vtt, tsv |
| `output_detail` | string | segments | For JSON: **segments**, words (adds words array), or both |
| `max_line_width` | int | null | Max characters per segment; splits long phrases at word boundaries |
| `max_line_count` | int | null | Max segments per original block (srt/vtt) |

---

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | base | Whisper model: tiny, base, small, medium, large-v2, large-v3 |
| `batch_size` | int | 16 | Batch size for inference (reduce if low on GPU memory) |
| `language` | string | null | Language code (e.g. en, de, fr). null = auto-detect |
| `device` | string | null | cuda or cpu (null = use env default) |
| `compute_type` | string | null | float16 or int8 |
| `no_align` | bool | false | Skip word-level alignment (faster, less accurate timestamps) |
| `task` | string | transcribe | transcribe or translate |

---

### Diarization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `diarize` | bool | false | Enable speaker diarization (identifies who spoke when; requires HF_TOKEN) |
| `min_speakers` | int | null | Minimum number of speakers |
| `max_speakers` | int | null | Maximum number of speakers |

---

### Advanced ASR Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beam_size` | int | 5 | Beam size for decoding |
| `patience` | float | 1.0 | Patience for beam search |
| `length_penalty` | float | 1.0 | Length penalty |
| `compression_ratio_threshold` | float | 2.4 | Compression ratio threshold |
| `log_prob_threshold` | float | -1.0 | Log probability threshold |
| `no_speech_threshold` | float | 0.6 | No speech detection threshold |
| `initial_prompt` | string | null | Initial prompt for context |
| `suppress_numerals` | bool | false | Suppress numeral outputs |

---

### VAD Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vad_onset` | float | 0.5 | VAD onset threshold |
| `vad_offset` | float | 0.363 | VAD offset threshold |
| `chunk_size` | int | 30 | Chunk size in seconds |

---

## Response Formats

### JSON (default, n8n-friendly)

Returns JSON data directly (not a file). Ideal for n8n and other automation tools.

```json
{
  "language": "en",
  "text": "Full transcript as single string",
  "transcript": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello world",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.6, "end": 2.5}
      ]
    }
  ]
}
```

- `output_detail=segments`: transcript entries only (no words)
- `output_detail=words`: transcript with `words` array per entry + top-level `words` array (all words)
- `output_detail=both`: same as words
- Text is split at natural breaks (comma, semicolon, etc.); each phrase becomes a separate segment with its own `start` and `end`

### TXT

Plain text, one segment per line. Speaker labels if diarization enabled:
```
[SPEAKER_00]: Hello world
[SPEAKER_01]: How are you?
```

### SRT / VTT

Standard subtitle formats with timestamps. Each segment is split at natural breaks (comma, semicolon, period) into separate subtitle blocks. Use `max_line_width` to split long phrases.

### TSV

Tab-separated: `start_ms`, `end_ms`, `text`

---

## Usage Examples

### cURL – File Upload (JSON output)

```bash
curl -X POST http://localhost:5505/transcript \
  -H "API_KEY: your_api_key" \
  -F "audio=@recording.mp3" \
  -F "output_format=json"
```

### cURL – Audio URL (JSON output)

```bash
curl -X POST http://localhost:5505/transcript \
  -H "API_KEY: your_api_key" \
  -F "audio_url=https://example.com/audio.mp3" \
  -F "output_format=json"
```

### cURL – Video File / Video URL

```bash
# Video file
curl -X POST http://localhost:5505/transcript \
  -H "API_KEY: your_api_key" \
  -F "video=@presentation.mp4" \
  -F "output_format=json"

# Video URL
curl -X POST http://localhost:5505/transcript \
  -H "API_KEY: your_api_key" \
  -F "video_url=https://example.com/video.mp4" \
  -F "output_format=json"
```

### cURL – Stream URLs (YouTube, TikTok, Instagram, X, Facebook, Threads)

```bash
# YouTube
curl -X POST http://localhost:5505/transcript \
  -H "API_KEY: your_api_key" \
  -F "youtube_url=https://www.youtube.com/watch?v=VIDEO_ID" \
  -F "output_format=json"

# TikTok
curl -X POST http://localhost:5505/transcript \
  -H "API_KEY: your_api_key" \
  -F "tiktok_url=https://www.tiktok.com/@user/video/123" \
  -F "output_format=json"

# Instagram / Facebook / X / Threads — use the matching parameter
curl -X POST http://localhost:5505/transcript \
  -H "API_KEY: your_api_key" \
  -F "insta_url=https://www.instagram.com/reel/..." \
  -F "output_format=json"
```

### cURL – Base64 Audio

```bash
# Encode file to base64 first
AUDIO_B64=$(base64 -w 0 recording.mp3)

curl -X POST http://localhost:5505/transcript \
  -H "API_KEY: your_api_key" \
  -F "audio_base64=$AUDIO_B64" \
  -F "output_format=json"
```

### cURL – SRT with Line Limits

```bash
curl -X POST http://localhost:5505/transcript \
  -H "API_KEY: your_api_key" \
  -F "audio=@recording.mp3" \
  -F "output_format=srt" \
  -F "max_line_width=42"
```

### cURL – With Diarization

```bash
curl -X POST http://localhost:5505/transcript \
  -H "API_KEY: your_api_key" \
  -F "audio=@meeting.mp3" \
  -F "diarize=true" \
  -F "min_speakers=2" \
  -F "max_speakers=4" \
  -F "output_format=json" \
  -F "output_detail=both"
```

### Python

```python
import requests

url = "http://localhost:5505/transcript"
headers = {"API_KEY": "your_api_key"}

# File upload
with open("audio.mp3", "rb") as f:
    r = requests.post(url, headers=headers, files={"audio": f}, data={"output_format": "json"})
print(r.json())

# URL
r = requests.post(url, headers=headers, data={
    "audio_url": "https://example.com/audio.mp3",
    "output_format": "json",
})
print(r.json())
```

### n8n Integration

Use the **HTTP Request** node:

- **Method:** POST
- **URL:** `http://your-server:5505/transcript`
- **Authentication:** Header `API_KEY` = your key
- **Body Content Type:** Form-Data
- **Body Parameters:**
  - `audio` (File) – from previous node’s binary data, or
  - `audio_url` (String) – URL to audio
  - `output_format`: `json`
  - `output_detail`: `both` (optional)

The response is JSON data, suitable for use in subsequent n8n nodes.

---

## Directory Structure

```
whisperx-server/
├── server.py           # Main server
├── requirements.txt    # Dependencies
├── .env                # Configuration (create from .env.example)
├── .env.example        # Example configuration
├── models/             # Downloaded models (auto-created)
└── README.md           # This file
```

---

## Troubleshooting

### Models Not Downloading

- Ensure internet access
- For large models (large-v2, large-v3), ensure sufficient disk space
- Models are stored in `./models`

### Out of Memory (GPU)

- Reduce `batch_size` (e.g. 4 or 8)
- Use smaller model (`base` or `small`)
- Use `compute_type=int8`

### Diarization Fails

- Set `HF_TOKEN` in `.env`
- Accept terms at [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

### FFmpeg Not Found

- Install FFmpeg and add to PATH
- Restart the server after installing

### Port Already in Use

- Change `PORT` in `.env`
- Or stop the process using the port

---

## API Response Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (invalid parameters, missing/conflicting input) |
| 401 | Unauthorized (invalid API key) |
| 500 | Internal Server Error |
| 503 | Service Unavailable (model not loaded) |

---

## License

This project uses WhisperX. See [WhisperX](https://github.com/m-bain/whisperX) for license terms.
