"""
WhisperX API Server - Speech-to-Text transcription API
Supports audio file upload, base64, or URL input.
"""

# Suppress torchcodec warning from pyannote.audio (WhisperX uses system ffmpeg via subprocess)
import warnings
warnings.filterwarnings("ignore", module="pyannote.audio.core.io")

import os
import io
import gc
import re
import shutil
import subprocess
import base64
import tempfile
import urllib.request
import socket
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, File, UploadFile, Form
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

# Set models download directory
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Get public IPv4 for URL generation
def get_public_ipv4_address():
    """Get the public IPv4 address of the machine."""
    services = [
        "https://api.ipify.org",
        "https://ifconfig.me/ip",
        "https://icanhazip.com",
        "https://ident.me",
    ]
    for service in services:
        try:
            with urllib.request.urlopen(service, timeout=5) as response:
                ip = response.read().decode("utf-8").strip()
                socket.inet_aton(ip)
                return ip
        except Exception:
            continue
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

SERVER_IP = get_public_ipv4_address()
SERVER_PORT = int(os.getenv("PORT", "8001"))

# HuggingFace cache
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_DIR)

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY not found in .env file")

# Global model cache
whisperx_model = None
align_model_cache = {}  # language -> (model, metadata)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global whisperx_model
    print(f"Loading WhisperX model (models directory: {MODELS_DIR})...")
    print("This may take a few minutes on first run as models download...")
    try:
        import whisperx
        device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        # Fall back to CPU if CUDA requested but PyTorch has no CUDA support
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            print("CUDA requested but not available (PyTorch CPU-only). Using CPU.")
        compute_type = os.getenv("COMPUTE_TYPE", "float16" if device == "cuda" else "int8")
        model_name = os.getenv("WHISPER_MODEL", "base")
        whisperx_model = whisperx.load_model(
            model_name,
            device,
            compute_type=compute_type,
            download_root=str(MODELS_DIR),
        )
        print(f"WhisperX model '{model_name}' loaded successfully on {device}!")
    except Exception as e:
        print(f"Error loading WhisperX model: {e}")
        raise
    yield
    print("Shutting down WhisperX server...")
    if whisperx_model is not None:
        del whisperx_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(
    title="WhisperX API",
    description="Speech-to-Text transcription API using WhisperX",
    version="1.0.0",
    lifespan=lifespan,
)


def get_audio_path_from_content(content: bytes, suffix: str = ".wav") -> str:
    """Write audio/video content to temp file and return path."""
    if suffix not in (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4", ".mkv", ".avi", ".mov"):
        suffix = ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.close()
    return tmp.name


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")
AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm")


def extract_audio_from_video(video_path: str) -> str:
    """Extract audio from video file using ffmpeg. Returns path to temp wav file."""
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                out_path,
            ],
            capture_output=True,
            check=True,
        )
        return out_path
    except subprocess.CalledProcessError as e:
        try:
            os.unlink(out_path)
        except Exception:
            pass
        raise RuntimeError(f"FFmpeg failed to extract audio: {e.stderr.decode() if e.stderr else e}") from e


def is_video_file(path: str) -> bool:
    """Check if file is a video (needs audio extraction)."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def _ytdlp_cookie_options() -> dict:
    """
    YouTube often requires authenticated cookies (--cookies / --cookies-from-browser).
    Configure via env:
    - YTDLP_COOKIES_FILE: path to Netscape cookies.txt (export from browser; see yt-dlp wiki).
    - YTDLP_COOKIES_FROM_BROWSER: browser name, optional profile after colon. Examples: chromium (typical on
      Linux/arm64 with Chromium only), chrome, edge, firefox; or chromium:Default for a named profile. Browser may
      need to be closed while the server runs.
    File wins if both are set and the file exists.
    """
    opts: dict = {}
    cookie_file = (os.getenv("YTDLP_COOKIES_FILE") or "").strip()
    if cookie_file:
        p = Path(cookie_file).expanduser()
        try:
            p = p.resolve()
        except OSError:
            p = Path(cookie_file).expanduser()
        if p.is_file():
            opts["cookiefile"] = str(p)

    if opts:
        return opts

    browser_spec = (os.getenv("YTDLP_COOKIES_FROM_BROWSER") or "").strip()
    if not browser_spec:
        return opts
    if ":" in browser_spec:
        name, profile = browser_spec.split(":", 1)
        name, profile = name.strip().lower(), profile.strip()
        if name:
            opts["cookiesfrombrowser"] = (name, profile) if profile else (name,)
    else:
        opts["cookiesfrombrowser"] = (browser_spec.lower(),)
    return opts


def _resolve_js_runtime() -> tuple[str | None, str | None]:
    """
    Resolve which JS runtime to use and its path.
    Returns (runtime_name, runtime_path) or (None, None) if disabled.

    yt-dlp only enables 'deno' by default — it does NOT auto-detect node/bun.
    You MUST explicitly tell it to use node via --js-runtimes or the Python API.

    Configure via env:
    - YTDLP_JS_RUNTIMES=auto (default): detect node/deno/bun on PATH.
    - YTDLP_JS_RUNTIMES=none: disable all.
    - YTDLP_JS_RUNTIMES=node:/usr/bin/node: explicit runtime + path.
    See https://github.com/yt-dlp/yt-dlp/wiki/EJS
    """
    spec_raw = (os.getenv("YTDLP_JS_RUNTIMES") or "auto").strip()
    lower = spec_raw.lower()
    if lower in ("none", "off", "false", "0"):
        return (None, None)
    if lower != "auto":
        if ":" in spec_raw:
            name, _, path = spec_raw.partition(":")
            name, path = name.strip().lower(), path.strip()
            if name in ("deno", "node", "bun", "quickjs") and path:
                return (name, path)
        elif lower in ("deno", "node", "bun", "quickjs"):
            found = shutil.which(lower)
            return (lower, found) if found else (lower, None)
        return (None, None)
    # Auto-detect: try each runtime on PATH
    for name in ("node", "deno", "bun", "quickjs"):
        path = shutil.which(name)
        if path:
            return (name, path)
    return (None, None)


def _ytdlp_js_runtime_options() -> dict:
    """
    Build yt-dlp Python API options for JS runtimes.
    Uses the 'js_runtimes' key supported by newer yt-dlp versions.
    Also enables remote_components so yt-dlp can download the EJS
    challenge solver script required for YouTube JS challenges.
    """
    opts: dict = {}
    # Enable remote EJS challenge solver (required for YouTube on datacenter IPs)
    opts["remote_components"] = ["ejs:github"]
    rt_name, rt_path = _resolve_js_runtime()
    if not rt_name:
        return opts
    if rt_path:
        opts["js_runtimes"] = {rt_name: {"path": rt_path}}
    else:
        opts["js_runtimes"] = {rt_name: {}}
    return opts


def _ytdlp_js_runtime_cli_args() -> list[str]:
    """
    Build yt-dlp CLI arguments for JS runtimes.
    Returns e.g. ['--remote-components', 'ejs:github', '--js-runtimes', 'node:/usr/bin/node'] or [].
    Also includes --remote-components so the EJS challenge solver can be fetched.
    """
    args: list[str] = ["--remote-components", "ejs:github"]
    rt_name, rt_path = _resolve_js_runtime()
    if not rt_name:
        return args
    if rt_path:
        args.extend(["--js-runtimes", f"{rt_name}:{rt_path}"])
    else:
        args.extend(["--js-runtimes", rt_name])
    return args


def _ytdlp_youtube_extractor_args() -> dict:
    """
    Without a JS runtime, yt-dlp only uses android_vr by default — often too few formats for our selector.
    Request several InnerTube clients so DASH/HLS formats exist on headless ARM64/Linux VPSes.

    Override with YTDLP_YOUTUBE_PLAYER_CLIENTS=comma-separated list (e.g. android,ios,tv,mweb).
    """
    raw = (os.getenv("YTDLP_YOUTUBE_PLAYER_CLIENTS") or "").strip()
    if raw:
        clients = [c.strip() for c in raw.split(",") if c.strip()]
        if clients:
            return {"youtube": {"player_client": clients}}
    return {
        "youtube": {
            "player_client": [
                "web",
                "android",
                "ios",
                "android_vr",
                "tv",
                "mweb",
                "web_safari",
            ],
        },
    }


def _log_ytdlp_diagnostics() -> None:
    """Print yt-dlp diagnostics at first use (once)."""
    if getattr(_log_ytdlp_diagnostics, "_done", False):
        return
    _log_ytdlp_diagnostics._done = True  # type: ignore[attr-defined]
    try:
        import yt_dlp
        ver = getattr(yt_dlp, "version", getattr(yt_dlp, "__version__", "unknown"))
        if hasattr(ver, "__version__"):
            ver = ver.__version__
        print(f"[yt-dlp] version: {ver}")
    except Exception:
        print("[yt-dlp] WARNING: could not determine yt-dlp version")
    # Check JS runtimes
    for rt in ("node", "deno", "bun"):
        rt_path = shutil.which(rt)
        if rt_path:
            print(f"[yt-dlp] JS runtime '{rt}' found at: {rt_path}")
    js_cfg = _ytdlp_js_runtime_options()
    print(f"[yt-dlp] JS runtime config: {js_cfg}")
    ext_cfg = _ytdlp_youtube_extractor_args()
    print(f"[yt-dlp] YouTube extractor args: {ext_cfg}")


def _download_with_ytdlp_cli(url: str) -> str:
    """
    Last-resort fallback: call yt-dlp as a subprocess (CLI).
    The CLI handles JS runtimes natively and more reliably than the Python API,
    because the Python API's `js_runtimes` option may not be supported on all versions.
    """
    base = tempfile.NamedTemporaryFile(delete=False, suffix="").name
    out_tmpl = base + ".%(ext)s"

    # Build the yt-dlp CLI command
    # Find yt-dlp binary: prefer the one in venv, then system
    ytdlp_bin = shutil.which("yt-dlp")
    if not ytdlp_bin:
        # Try the venv's scripts directory
        venv_dir = Path(__file__).parent / "venv"
        for candidate in [
            venv_dir / "bin" / "yt-dlp",
            venv_dir / "Scripts" / "yt-dlp.exe",
            venv_dir / "Scripts" / "yt-dlp",
        ]:
            if candidate.exists():
                ytdlp_bin = str(candidate)
                break
    if not ytdlp_bin:
        raise RuntimeError("yt-dlp CLI binary not found in PATH or venv")

    cmd = [
        ytdlp_bin,
        "-f", "bestaudio/best",
        "-x",                          # extract audio
        "--audio-format", "m4a",
        "--no-playlist",
        "-o", out_tmpl,
        "--no-color",
    ]
    # Critical: yt-dlp only enables deno by default, must explicitly enable node/bun
    cmd.extend(_ytdlp_js_runtime_cli_args())
    # Add cookies if configured
    cookie_opts = _ytdlp_cookie_options()
    if "cookiefile" in cookie_opts:
        cmd.extend(["--cookies", cookie_opts["cookiefile"]])
    elif "cookiesfrombrowser" in cookie_opts:
        browser_tuple = cookie_opts["cookiesfrombrowser"]
        cmd.extend(["--cookies-from-browser", ":".join(browser_tuple)])

    cmd.append(url)

    ytdlp_verbose = (os.getenv("YTDLP_VERBOSE") or "").lower() in ("1", "true", "yes")
    if ytdlp_verbose:
        cmd.append("-v")

    print(f"[yt-dlp] CLI fallback: {' '.join(cmd)}")

    env = os.environ.copy()
    # Ensure the JS runtime binary directory is on PATH for yt-dlp's subprocess
    _, rt_path = _resolve_js_runtime()
    if rt_path and os.path.isfile(rt_path):
        rt_dir = str(Path(rt_path).parent)
        if rt_dir not in env.get("PATH", ""):
            env["PATH"] = rt_dir + os.pathsep + env.get("PATH", "")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        stdout = result.stdout.strip() if result.stdout else ""
        error_detail = stderr or stdout or "Unknown CLI error"
        raise RuntimeError(f"yt-dlp CLI failed: {error_detail}")

    # Find the output file
    for ext in (".m4a", ".mp3", ".webm", ".opus", ".wav", ".ogg", ".mp4"):
        p = base + ext
        if os.path.exists(p):
            return p

    raise RuntimeError("yt-dlp CLI did not produce an audio file")


def download_stream_audio_with_ytdlp(url: str) -> str:
    """
    Download best audio from a URL supported by yt-dlp (YouTube watch + /shorts/, Facebook, Instagram,
    TikTok, X/Twitter, Threads, and many other sites).
    Returns path to a temp audio file.

    Strategy:
    1. Try the Python API with progressively more lenient format selectors.
    2. If ALL Python API attempts fail, fall back to calling yt-dlp as a CLI subprocess,
       which handles JS runtimes natively and more reliably.
    """
    _log_ytdlp_diagnostics()

    try:
        import yt_dlp
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="yt-dlp is required for stream URLs. Install with: pip install yt-dlp",
        )

    # Progressively more lenient fallback strategies.
    # Each entry is a tuple: (format_string, use_extractor_args, use_js_runtimes)
    FALLBACK_STRATEGIES = [
        ("bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio[ext=mp3]/bestaudio", True, True),
        ("bestaudio*", True, True),
        ("best[acodec!=none]", True, True),
        ("bestaudio/best", False, True),
        ("worst[acodec!=none]/worst/best", False, False),
    ]

    ytdlp_verbose = (os.getenv("YTDLP_VERBOSE") or "").lower() in ("1", "true", "yes")
    last_error: Exception | None = None

    for strat_idx, (fmt, use_extractor_args, use_js_runtimes) in enumerate(FALLBACK_STRATEGIES):
        base = tempfile.NamedTemporaryFile(delete=False).name
        out_tmpl = base + ".%(ext)s"
        ydl_opts = {
            "format": fmt,
            "outtmpl": out_tmpl,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}],
            "quiet": not ytdlp_verbose,
            "no_warnings": not ytdlp_verbose,
            "no_color": True,
            "noplaylist": True,
        }
        if use_extractor_args:
            ydl_opts["extractor_args"] = _ytdlp_youtube_extractor_args()
        if "+" in fmt:
            ydl_opts["merge_output_format"] = "mp4"
        if use_js_runtimes:
            ydl_opts.update(_ytdlp_js_runtime_options())
        ydl_opts.update(_ytdlp_cookie_options())

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            for ext in (".m4a", ".mp3", ".webm", ".opus", ".wav", ".ogg"):
                p = base + ext
                if os.path.exists(p):
                    return p
            last_error = RuntimeError("yt-dlp did not produce an audio file")
            print(f"[yt-dlp] format '{fmt}' produced no audio file, trying next fallback...")
            continue
        except Exception as e:
            last_error = e
            msg = str(e)
            # Fatal errors — stop immediately
            if "Sign in to confirm" in msg or "not a bot" in msg:
                msg += (
                    ". Set YTDLP_COOKIES_FROM_BROWSER (e.g. chromium on Linux/arm64, edge or chrome on Windows) or "
                    "YTDLP_COOKIES_FILE to a cookies.txt exported while logged into YouTube. "
                    "See https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp"
                )
                raise HTTPException(status_code=400, detail=f"Failed to download audio from URL: {msg}") from e
            if "Video unavailable" in msg or "Private video" in msg or "removed" in msg.lower():
                raise HTTPException(status_code=400, detail=f"Failed to download audio from URL: {msg}") from e

            if strat_idx < len(FALLBACK_STRATEGIES) - 1:
                print(f"[yt-dlp] format '{fmt}' failed ({msg}), trying next fallback...")
                continue
            # All Python API strategies exhausted — fall through to CLI fallback below
            print(f"[yt-dlp] All Python API strategies failed. Trying CLI subprocess fallback...")

    # === FINAL FALLBACK: CLI subprocess ===
    # The Python API's js_runtimes option may not be supported on all yt-dlp versions.
    # The CLI handles JS runtimes natively and more reliably.
    try:
        return _download_with_ytdlp_cli(url)
    except Exception as cli_err:
        msg = str(cli_err)
        if "Sign in to confirm" in msg or "not a bot" in msg:
            msg += (
                ". Set YTDLP_COOKIES_FROM_BROWSER or YTDLP_COOKIES_FILE. "
                "See https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp"
            )
        api_msg = str(last_error) if last_error else "unknown"
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download audio from URL. Python API error: {api_msg} | CLI error: {msg}",
        ) from cli_err


async def resolve_audio_input(
    audio: Optional[UploadFile] = None,
    audio_base64: Optional[str] = None,
    audio_url: Optional[str] = None,
    video: Optional[UploadFile] = None,
    video_base64: Optional[str] = None,
    video_url: Optional[str] = None,
    youtube_url: Optional[str] = None,
    fb_url: Optional[str] = None,
    insta_url: Optional[str] = None,
    tiktok_url: Optional[str] = None,
    x_url: Optional[str] = None,
    threads_url: Optional[str] = None,
) -> str:
    """
    Resolve audio/video from file, base64, direct URL, or stream URLs (yt-dlp).
    Returns path to temp file (caller must delete). Video files need audio extraction.
    """
    stream_url = (
        youtube_url or fb_url or insta_url or tiktok_url or x_url or threads_url
    )
    sources = sum([
        audio is not None, bool(audio_base64), bool(audio_url),
        video is not None, bool(video_base64), bool(video_url),
        bool(stream_url),
    ])
    if sources == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "Provide exactly one of: audio, audio_base64, audio_url, video, video_base64, video_url, "
                "youtube_url, fb_url, insta_url, tiktok_url, x_url, or threads_url"
            ),
        )
    if sources > 1:
        raise HTTPException(
            status_code=400,
            detail="Provide only one input source",
        )

    suffix = ".wav"
    if stream_url:
        return download_stream_audio_with_ytdlp(stream_url)

    if audio:
        content = await audio.read()
        if not content:
            raise HTTPException(status_code=400, detail="Audio file is empty")
        if audio.filename:
            ext = Path(audio.filename).suffix.lower() or ".wav"
            if ext in AUDIO_EXTENSIONS:
                suffix = ext
    elif audio_base64:
        try:
            content = base64.b64decode(audio_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")
        if not content:
            raise HTTPException(status_code=400, detail="audio_base64 is empty")
    elif audio_url:
        try:
            req = urllib.request.Request(audio_url, headers={"User-Agent": "WhisperX-API/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                content = resp.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")
        if not content:
            raise HTTPException(status_code=400, detail="URL returned empty content")
        url_lower = audio_url.lower()
        if ".mp3" in url_lower:
            suffix = ".mp3"
        elif ".wav" in url_lower:
            suffix = ".wav"
        elif ".m4a" in url_lower:
            suffix = ".m4a"
        elif ".ogg" in url_lower or ".flac" in url_lower:
            suffix = ".m4a"  # fallback
    elif video:
        content = await video.read()
        if not content:
            raise HTTPException(status_code=400, detail="Video file is empty")
        if video.filename:
            ext = Path(video.filename).suffix.lower() or ".mp4"
            suffix = ext if ext in VIDEO_EXTENSIONS else ".mp4"
    elif video_base64:
        try:
            content = base64.b64decode(video_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")
        if not content:
            raise HTTPException(status_code=400, detail="video_base64 is empty")
        suffix = ".mp4"
    else:
        # video_url
        try:
            req = urllib.request.Request(video_url, headers={"User-Agent": "WhisperX-API/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                content = resp.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")
        if not content:
            raise HTTPException(status_code=400, detail="URL returned empty content")
        url_lower = video_url.lower()
        if ".mp4" in url_lower:
            suffix = ".mp4"
        elif ".mkv" in url_lower:
            suffix = ".mkv"
        elif ".webm" in url_lower:
            suffix = ".webm"
        elif ".avi" in url_lower or ".mov" in url_lower:
            suffix = ".mp4"

    return get_audio_path_from_content(content, suffix)


def transcribe_audio(audio_path: str, params: dict) -> dict:
    """Run WhisperX transcription on audio file."""
    import whisperx

    device = params.get("device") or (os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    compute_type = params.get("compute_type") or (os.getenv("COMPUTE_TYPE") or ("float16" if device == "cuda" else "int8"))
    model_name = params.get("model") or os.getenv("WHISPER_MODEL", "base")
    batch_size = params.get("batch_size", 16)
    no_align = params.get("no_align", False)
    diarize = params.get("diarize", False)
    min_speakers = params.get("min_speakers")
    max_speakers = params.get("max_speakers")
    hf_token = params.get("hf_token") or os.getenv("HF_TOKEN")

    asr_options = {
        "beam_size": params.get("beam_size", 5),
        "patience": params.get("patience", 1.0),
        "length_penalty": params.get("length_penalty", 1.0),
        "compression_ratio_threshold": params.get("compression_ratio_threshold", 2.4),
        "log_prob_threshold": params.get("log_prob_threshold", -1.0),
        "no_speech_threshold": params.get("no_speech_threshold", 0.6),
        "condition_on_previous_text": False,
        "initial_prompt": params.get("initial_prompt") or None,
        "suppress_numerals": params.get("suppress_numerals", False),
    }
    vad_options = {
        "chunk_size": params.get("chunk_size", 30),
        "vad_onset": params.get("vad_onset", 0.5),
        "vad_offset": params.get("vad_offset", 0.363),
    }

    # Use global model if it matches, otherwise load fresh
    default_model = os.getenv("WHISPER_MODEL", "base")
    if whisperx_model is not None and model_name == default_model:
        model = whisperx_model
    else:
        model = whisperx.load_model(
            model_name,
            device,
            compute_type=compute_type,
            download_root=str(MODELS_DIR),
            language=params.get("language"),
            asr_options=asr_options,
            vad_options=vad_options,
            task=params.get("task", "transcribe"),
        )

    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size, chunk_size=params.get("chunk_size", 30))

    # Align (unless no_align)
    if not no_align and len(result.get("segments", [])) > 0:
        lang = result.get("language", "en")
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=lang,
                device=device,
                model_dir=str(MODELS_DIR),
            )
            result = whisperx.align(result["segments"], model_a, metadata, audio, device)
            del model_a
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ValueError:
            # No alignment model for this language — skip silently
            print(f"[whisperx] No alignment model for language '{lang}', skipping alignment (segment-level timestamps only)")

    # Diarize (optional)
    if diarize and hf_token:
        diarize_model = whisperx.DiarizationPipeline(
            token=hf_token,
            device=device,
            cache_dir=str(MODELS_DIR),
        )
        diarize_segments = diarize_model(
            audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)
        del diarize_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Only delete if we loaded a fresh model (not the global one)
    if model is not whisperx_model:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


def _smart_split_words(words: list[str], max_width: int) -> list[str]:
    """
    Split a list of words into lines of at most max_width characters.
    Instead of greedily packing words until hitting the limit
    (which produces an ugly short leftover), this finds a more
    balanced break — roughly equal-length halves — so lines look
    natural for subtitles.

    Example with max_width=40:
      Greedy:  "Explanation by the tongue makes most" + "things clear."
      Smart:   "Explanation by the tongue" + "makes most things clear."
    """
    if not words:
        return []

    full = " ".join(words)
    if len(full) <= max_width:
        return [full]

    # How many lines do we need at minimum?
    n_lines = max(2, -(-len(full) // max_width))  # ceil division
    target_per_line = len(full) / n_lines

    lines: list[str] = []
    remaining = list(words)

    while remaining:
        # If what's left fits, emit it and stop
        rest = " ".join(remaining)
        if len(rest) <= max_width:
            lines.append(rest)
            break

        # Try to find the break closest to target_per_line
        best_idx = 0
        best_diff = float("inf")
        running = 0
        for i, w in enumerate(remaining):
            running += len(w) + (1 if i > 0 else 0)
            if running > max_width:
                break
            diff = abs(running - target_per_line)
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        # Take at least one word
        split_at = best_idx + 1
        lines.append(" ".join(remaining[:split_at]))
        remaining = remaining[split_at:]

    return lines


def _split_text_into_phrases(
    text: str,
    max_line_width: Optional[int] = None,
    max_line_count: Optional[int] = None,
    split_on_punctuation: bool = True,
    smart_split: bool = False,
) -> list[str]:
    """
    Split text into phrases for subtitle display.

    Parameters:
      split_on_punctuation: If True (default), first split at punctuation marks
        (comma, semicolon, period, etc.) and then honour max_line_width.
        If False, treat the text as one block and split only by max_line_width.
      smart_split: If True, produce more balanced line breaks instead
        of greedily packing words until hitting max_line_width.
        e.g. "Explanation by the tongue | makes most things clear."
        instead of "Explanation by the tongue makes most | things clear."
    """
    text = (text or "").strip()
    if not text:
        return []

    # --- Step 1: initial parts ---
    if split_on_punctuation:
        # Split after comma, semicolon, period, exclamation, question mark, colon
        parts = re.split(r"(?<=[,;.!?:])\s+", text)
        parts = [p.strip() for p in parts if p.strip()]
    else:
        # Treat entire text as a single block
        parts = [text]

    # --- No width limit ---
    if max_line_width is None:
        phrases = parts if split_on_punctuation else [text]
        if max_line_count is not None and max_line_count > 0 and len(phrases) > max_line_count:
            phrases = phrases[:max_line_count]
        return phrases

    # --- Step 2: split/merge parts to fit max_line_width ---
    phrases: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            phrases.append(" ".join(current))
            current = []

    for part in parts:
        part_len = len(part)

        if part_len > max_line_width:
            flush()
            # This part alone exceeds the limit — split at word boundaries
            words = part.split()
            if smart_split:
                phrases.extend(_smart_split_words(words, max_line_width))
            else:
                # Greedy: pack words until hitting the limit
                line_words: list[str] = []
                line_len = 0
                for w in words:
                    need = line_len + (1 if line_words else 0) + len(w)
                    if line_words and need > max_line_width:
                        phrases.append(" ".join(line_words))
                        line_words = []
                        line_len = 0
                    line_words.append(w)
                    line_len += len(w) + (1 if line_len else 0)
                if line_words:
                    phrases.append(" ".join(line_words))
            continue

        if split_on_punctuation:
            # Punctuation-split parts stay separate — don't merge them back
            flush()
            current.append(part)
        else:
            # No punctuation splitting — merge short parts that fit within max_line_width
            if current:
                combined = " ".join(current) + " " + part
                if len(combined) > max_line_width:
                    flush()
            current.append(part)

    flush()

    # --- Step 3: smart rebalance pass ---
    # If smart mode is on, rebalance any phrases that can be split more evenly
    if smart_split:
        rebalanced: list[str] = []
        for phrase in phrases:
            if len(phrase) > max_line_width:
                rebalanced.extend(_smart_split_words(phrase.split(), max_line_width))
            else:
                rebalanced.append(phrase)
        phrases = rebalanced

    if max_line_count is not None and max_line_count > 0 and len(phrases) > max_line_count:
        phrases = phrases[:max_line_count]

    return phrases


def _split_segment_into_subsegments(
    seg: dict,
    max_line_width: Optional[int] = None,
    max_line_count: Optional[int] = None,
    split_on_punctuation: bool = True,
    smart_split: bool = False,
) -> list[dict]:
    """
    Split a segment by natural breaks (comma, semicolon, etc.) into sub-segments,
    each with its own start, end, and text. Uses word-level timestamps when available.
    """
    txt = (seg.get("text") or "").strip()
    if not txt:
        return [seg] if seg.get("start") is not None else []

    phrases = _split_text_into_phrases(
        txt, max_line_width, max_line_count,
        split_on_punctuation=split_on_punctuation,
        smart_split=smart_split,
    )
    if len(phrases) <= 1:
        return [seg]

    seg_start = seg.get("start", 0.0)
    seg_end = seg.get("end", 0.0)
    words = seg.get("words") or []
    speaker = seg.get("speaker")

    subsegs: list[dict] = []
    word_idx = 0

    for phrase in phrases:
        phrase_words = phrase.split()
        n_words = len(phrase_words)

        if words and word_idx < len(words):
            end_idx = min(word_idx + n_words, len(words)) - 1
            if end_idx >= word_idx:
                p_start = words[word_idx].get("start", seg_start)
                p_end = words[end_idx].get("end", seg_end)
                phrase_words_data = words[word_idx : end_idx + 1] if "words" in seg else None
            else:
                p_start = seg_start
                p_end = seg_end
                phrase_words_data = None
            word_idx = end_idx + 1
        else:
            # Proportional split by character length
            total_chars = sum(len(p) for p in phrases)
            if total_chars <= 0:
                p_start, p_end = seg_start, seg_end
            else:
                prev_chars = sum(len(phrases[i]) for i in range(len(subsegs)))
                ratio_start = prev_chars / total_chars
                ratio_end = (prev_chars + len(phrase)) / total_chars
                duration = seg_end - seg_start
                p_start = seg_start + duration * ratio_start
                p_end = seg_start + duration * ratio_end
            phrase_words_data = None

        sub = {"start": p_start, "end": p_end, "text": phrase}
        if speaker is not None:
            sub["speaker"] = speaker
        if phrase_words_data is not None and phrase_words_data:
            sub["words"] = phrase_words_data
        subsegs.append(sub)

    return subsegs


def format_result(
    result: dict,
    output_format: str,
    output_detail: str,
    max_line_width: Optional[int],
    max_line_count: Optional[int],
    split_on_punctuation: bool = True,
    smart_split: bool = False,
) -> str | dict:
    """Format transcription result. Splits segments at natural breaks (comma, semicolon, etc.) into separate segments with timestamps."""

    def expand_segments(segs: list[dict]) -> list[dict]:
        """Split each segment by natural breaks; each phrase becomes its own segment with start/end."""
        out: list[dict] = []
        for s in segs:
            subsegs = _split_segment_into_subsegments(
                s, max_line_width, max_line_count,
                split_on_punctuation=split_on_punctuation,
                smart_split=smart_split,
            )
            out.extend(subsegs)
        return out

    segments = expand_segments(result.get("segments", []))

    if output_format == "json":
        out = {
            "language": result.get("language", "en"),
            "transcript": [],
            "text": "",
        }
        all_words: list[dict] = []
        full_text_parts = []
        for seg in segments:
            seg_out = {
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": seg.get("text", "").strip(),
            }
            if "speaker" in seg:
                seg_out["speaker"] = seg["speaker"]
            if output_detail in ("words", "both") and "words" in seg:
                seg_words = seg["words"]
                for w in seg_words:
                    word_entry = {"word": w.get("word", ""), "start": w.get("start"), "end": w.get("end")}
                    if "speaker" in seg:
                        word_entry["speaker"] = seg["speaker"]
                    all_words.append(word_entry)
                seg_out["words"] = [{"word": w.get("word", ""), "start": w.get("start"), "end": w.get("end")} for w in seg_words]
            out["transcript"].append(seg_out)
            full_text_parts.append(seg_out["text"])
        out["text"] = " ".join(full_text_parts).strip()
        if output_detail in ("words", "both"):
            out["words"] = all_words
        return out

    # Text-based formats
    buf = io.StringIO()

    if output_format == "txt":
        for seg in segments:
            spk = seg.get("speaker")
            txt = seg.get("text", "").strip()
            if spk:
                buf.write(f"[{spk}]: {txt}\n")
            else:
                buf.write(f"{txt}\n")
    elif output_format == "tsv":
        buf.write("start\tend\ttext\n")
        for seg in segments:
            txt = seg.get("text", "").strip().replace(chr(9), " ")
            buf.write(f"{round(1000*seg['start'])}\t{round(1000*seg['end'])}\t{txt}\n")
    elif output_format in ("srt", "vtt"):
        def fmt_ts(s, h=False):
            ms = round(s * 1000)
            hh, ms = divmod(ms, 3600000)
            mm, ms = divmod(ms, 60000)
            ss, ms = divmod(ms, 1000)
            if h or hh > 0:
                return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}" if output_format == "srt" else f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"
            return f"{mm:02d}:{ss:02d},{ms:03d}" if output_format == "srt" else f"{mm:02d}:{ss:02d}.{ms:03d}"
        if output_format == "vtt":
            buf.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, 1):
            start = fmt_ts(seg["start"], output_format == "srt")
            end = fmt_ts(seg["end"], output_format == "srt")
            txt = seg.get("text", "").strip().replace("-->", "->")
            if "speaker" in seg:
                txt = f"[{seg['speaker']}]: {txt}"
            if output_format == "srt":
                buf.write(f"{i}\n{start} --> {end}\n{txt}\n\n")
            else:
                buf.write(f"{start} --> {end}\n{txt}\n\n")
    else:
        import json as _json
        buf.write(_json.dumps(result, ensure_ascii=False, indent=2))

    return buf.getvalue()


@app.post("/transcript")
async def transcript(
    audio: UploadFile = File(None, description="Audio file (mp3, wav, m4a, ogg, flac, webm)"),
    audio_base64: Optional[str] = Form(None, description="Audio as base64-encoded string"),
    audio_url: Optional[str] = Form(None, description="Direct URL to audio file"),
    video: UploadFile = File(None, description="Video file (mp4, mkv, avi, mov, webm)"),
    video_base64: Optional[str] = Form(None, description="Video as base64-encoded string"),
    video_url: Optional[str] = Form(None, description="Direct URL to video file"),
    youtube_url: Optional[str] = Form(None, description="YouTube watch or /shorts/ URL"),
    fb_url: Optional[str] = Form(None, description="Facebook video/reel URL (yt-dlp)"),
    insta_url: Optional[str] = Form(None, description="Instagram reel/post URL (yt-dlp)"),
    tiktok_url: Optional[str] = Form(None, description="TikTok video URL (yt-dlp)"),
    x_url: Optional[str] = Form(None, description="X (Twitter) video URL (yt-dlp)"),
    threads_url: Optional[str] = Form(None, description="Threads post URL (yt-dlp)"),
    # Output
    output_format: str = Form("json", description="Output format: json, txt, srt, vtt, tsv"),
    output_detail: str = Form("segments", description="For JSON: segments (default), words (adds words array), or both"),
    max_line_width: Optional[int] = Form(None, description="Max characters per segment; splits long phrases at word boundaries"),
    max_line_count: Optional[int] = Form(None, description="Max segments per original block (srt/vtt)"),
    split_on_punctuation: bool = Form(True, description="Split at punctuation marks (comma, period, etc.) before applying max_line_width. Set False to split only by width."),
    smart_split: bool = Form(False, description="Produce balanced line breaks instead of greedy packing. E.g. 'Explanation by the tongue / makes most things clear' instead of 'Explanation by the tongue makes most / things clear'"),
    # Model
    model: str = Form("base", description="Whisper model: tiny, base, small, medium, large-v2, large-v3"),
    batch_size: int = Form(16, description="Batch size for inference"),
    language: Optional[str] = Form(None, description="Language code (e.g. en, de, fr) or None for auto"),
    device: Optional[str] = Form(None, description="Device: cuda or cpu"),
    compute_type: Optional[str] = Form(None, description="Compute type: float16, int8"),
    no_align: bool = Form(False, description="Skip word-level alignment"),
    # Diarization
    diarize: bool = Form(False, description="Enable speaker diarization"),
    min_speakers: Optional[int] = Form(None, description="Min number of speakers"),
    max_speakers: Optional[int] = Form(None, description="Max number of speakers"),
    # Advanced ASR options
    beam_size: int = Form(5, description="Beam size for decoding"),
    patience: float = Form(1.0, description="Patience for beam search"),
    length_penalty: float = Form(1.0, description="Length penalty"),
    compression_ratio_threshold: float = Form(2.4, description="Compression ratio threshold"),
    log_prob_threshold: float = Form(-1.0, description="Log probability threshold"),
    no_speech_threshold: float = Form(0.6, description="No speech detection threshold"),
    initial_prompt: Optional[str] = Form(None, description="Initial prompt for context"),
    suppress_numerals: bool = Form(False, description="Suppress numeral outputs"),
    vad_onset: float = Form(0.5, description="VAD onset threshold"),
    vad_offset: float = Form(0.363, description="VAD offset threshold"),
    chunk_size: int = Form(30, description="Chunk size in seconds"),
    task: str = Form("transcribe", description="Task: transcribe or translate"),
    # API key
    api_key: str = Header(..., alias="API_KEY", description="API key for authentication"),
):
    """
    Transcribe audio/video to text.

    Provide exactly one of:
    - **audio** / **audio_base64** / **audio_url**: Audio input
    - **video** / **video_base64** / **video_url**: Video input (audio extracted with FFmpeg)
    - **youtube_url**, **fb_url**, **insta_url**, **tiktok_url**, **x_url**, **threads_url**: Stream URLs (audio via yt-dlp; supports many sites)

    Returns transcript as JSON (default) or text formats.
    """
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if whisperx_model is None:
        raise HTTPException(status_code=503, detail="WhisperX model not loaded")

    tmp_path = None
    extracted_audio_path = None
    try:
        tmp_path = await resolve_audio_input(
            audio=audio, audio_base64=audio_base64, audio_url=audio_url,
            video=video, video_base64=video_base64, video_url=video_url,
            youtube_url=youtube_url,
            fb_url=fb_url, insta_url=insta_url, tiktok_url=tiktok_url,
            x_url=x_url, threads_url=threads_url,
        )
        # Extract audio from video if needed
        if is_video_file(tmp_path):
            extracted_audio_path = extract_audio_from_video(tmp_path)
            audio_path = extracted_audio_path
        else:
            audio_path = tmp_path
        params = {
            "model": model,
            "batch_size": batch_size,
            "language": language,
            "device": device,
            "compute_type": compute_type,
            "no_align": no_align,
            "diarize": diarize,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "hf_token": os.getenv("HF_TOKEN"),
            "beam_size": beam_size,
            "patience": patience,
            "length_penalty": length_penalty,
            "compression_ratio_threshold": compression_ratio_threshold,
            "log_prob_threshold": log_prob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "initial_prompt": initial_prompt,
            "suppress_numerals": suppress_numerals,
            "vad_onset": vad_onset,
            "vad_offset": vad_offset,
            "chunk_size": chunk_size,
            "task": task,
        }
        result = transcribe_audio(audio_path, params)

        fmt = (output_format or "json").lower()
        if fmt not in ("json", "txt", "srt", "vtt", "tsv"):
            fmt = "json"

        detail = (output_detail or "segments").lower()
        if detail not in ("segments", "words", "both"):
            detail = "segments"

        formatted = format_result(
            result, fmt, detail, max_line_width, max_line_count,
            split_on_punctuation=split_on_punctuation,
            smart_split=smart_split,
        )

        if isinstance(formatted, dict):
            return JSONResponse(formatted)
        return PlainTextResponse(formatted, media_type="text/plain; charset=utf-8")
    finally:
        if extracted_audio_path and os.path.exists(extracted_audio_path):
            try:
                os.unlink(extracted_audio_path)
            except Exception:
                pass
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": whisperx_model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "WhisperX API",
        "version": "1.0.0",
        "endpoints": {
            "/transcript": "POST - Transcribe audio (file, base64, or URL)",
            "/health": "GET - Health check",
        },
        "input_methods": [
            "audio", "audio_base64", "audio_url",
            "video", "video_base64", "video_url",
            "youtube_url", "fb_url", "insta_url", "tiktok_url", "x_url", "threads_url",
        ],
        "output_formats": ["json", "txt", "srt", "vtt", "tsv"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
