"""yt-dlp wrappers: list channel videos, download a 30s middle clip."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import yt_dlp
from yt_dlp.utils import DownloadError

from medea.config import CLIP_DURATION_SECONDS, DATA_DIR, RAW_DIR

log = logging.getLogger(__name__)


def _ensure_ffmpeg_on_path() -> str:
    """Materialize the bundled ffmpeg under its canonical filename and put it on PATH.

    imageio-ffmpeg ships ffmpeg as `ffmpeg-win-x86_64-vX.Y.exe`. yt-dlp resolves
    its postprocessors by looking for an `ffmpeg(.exe)` basename — both via its
    `ffmpeg_location` option and via PATH lookups. Copy once to a stable name,
    and prepend the directory to PATH so every code path inside yt-dlp finds it.

    Returns the directory containing the canonical binary.
    """
    src = Path(imageio_ffmpeg.get_ffmpeg_exe())
    cache = DATA_DIR / ".bin"
    cache.mkdir(parents=True, exist_ok=True)
    dst_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    dst = cache / dst_name
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    cache_str = str(cache)
    if cache_str not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = cache_str + os.pathsep + os.environ.get("PATH", "")
    return cache_str


@dataclass
class VideoMeta:
    id: str
    title: str | None
    description: str | None
    upload_date: str | None  # YYYYMMDD
    duration: int | None
    view_count: int | None
    clip_path: Path | None
    channel_handle: str | None
    yt_channel_id: str | None


def _videos_tab_url(channel_url: str) -> str:
    url = channel_url.rstrip("/")
    if url.endswith("/videos"):
        return url
    return f"{url}/videos"


def _ffmpeg_dir() -> str:
    """Return the directory containing the canonically-named ffmpeg binary."""
    return _ensure_ffmpeg_on_path()


def list_channel_videos(channel_url: str, n: int) -> list[dict]:
    """Return up to N most recent long-form video entries from a channel.

    Uses yt-dlp's flat-extract mode against the /videos tab so we don't trigger
    full info fetches for each video. Returned entries always have at least
    `id`; other fields may be missing depending on the extractor.
    """
    opts = {
        "extract_flat": "in_playlist",
        "playlistend": n,
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(_videos_tab_url(channel_url), download=False)

    entries = (info or {}).get("entries") or []
    return [e for e in entries if e and e.get("id")][:n]


def _middle_clip_range(info_dict: dict, ydl) -> list[dict]:
    """download_ranges callback: yields the [mid - 15s, mid + 15s] window."""
    duration = info_dict.get("duration")
    if not duration or duration < CLIP_DURATION_SECONDS:
        return []
    mid = duration / 2.0
    half = CLIP_DURATION_SECONDS / 2.0
    return [
        {
            "start_time": max(0.0, mid - half),
            "end_time": min(float(duration), mid + half),
        }
    ]


def download_middle_clip(video_id: str, out_dir: Path = RAW_DIR) -> VideoMeta | None:
    """Download a 30-second middle slice of the video.

    Returns VideoMeta on success, None on failure (private, geo-blocked,
    too-short, etc.). Idempotent at the file level: yt-dlp's no-overwrite
    behavior plus our caller's DB check mean re-runs are safe and cheap.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / "%(id)s.%(ext)s")

    opts = {
        "format": "bv*[height<=720]+ba/b[height<=720]/b",
        "merge_output_format": "mp4",
        "outtmpl": out_template,
        "download_ranges": _middle_clip_range,
        "force_keyframes_at_cuts": True,
        "ffmpeg_location": _ffmpeg_dir(),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "ignoreerrors": False,  # we catch DownloadError ourselves
        "overwrites": False,
    }
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except DownloadError as e:
        log.warning("skip %s: %s", video_id, e)
        return None

    if info is None:
        return None

    duration = info.get("duration")
    if duration is None or duration < CLIP_DURATION_SECONDS:
        log.info("skip %s: too short (duration=%s)", video_id, duration)
        return None

    # The merged file should be <id>.mp4, but if something exotic happened
    # (e.g. only audio, or merge fallback), fall back to whatever was produced.
    clip_path: Path | None = out_dir / f"{info['id']}.mp4"
    if not clip_path.exists():
        candidates = sorted(out_dir.glob(f"{info['id']}.*"))
        clip_path = candidates[0] if candidates else None

    return VideoMeta(
        id=info["id"],
        title=info.get("title"),
        description=info.get("description"),
        upload_date=info.get("upload_date"),
        duration=int(duration),
        view_count=info.get("view_count"),
        clip_path=clip_path,
        channel_handle=info.get("uploader_id") or info.get("channel"),
        yt_channel_id=info.get("channel_id"),
    )
