"""Handcrafted per-video and channel-level scalar features.

These are signals embedding models can't easily capture — upload cadence,
hype keywords, link spam — represented as a small numeric block that gets
concatenated with the embedding blocks in ``pipeline.py``.

Per-video features (computable from one ``videos`` row):
    title_len           chars
    title_word_count    whitespace-split tokens
    title_caps_ratio    uppercase letters / alphabetic letters (0 if no letters)
    title_excl_count    number of '!' chars
    title_quest_count   number of '?' chars
    title_emoji_count   pictographic-range chars in the title
    title_digit_count   digits in the title
    title_clickbait     1 if regex matches a known clickbait phrase
    desc_len            chars
    desc_url_count      occurrences of http:// or https://
    desc_hashtag_count  occurrences of '#word'
    view_count_log      log10(1 + view_count)

Channel-level features (replicated to each video on that channel):
    channel_video_count_observed  count of videos in our DB for this channel
    channel_age_days              span between oldest and newest observed upload
    channel_mean_iud_days         mean inter-upload delta in days; 0.0 if <2 dates

Z-scoring is deferred to the classifier-side scaler in M6 — the right scale
depends on the train split, not on the feature definition.
"""

from __future__ import annotations

import math
import re
from datetime import date

# Loose hype-words regex. Misses are fine because the title embedding also
# carries signal; this exists so the classifier sees a literal hype indicator.
CLICKBAIT_RE = re.compile(
    r"\b(shocking|you won'?t believe|insane|unbelievable|gone wrong|"
    r"reveal(?:ed)?|exposed|truth about|secret|finally|must see|"
    r"epic|crazy|hidden|nobody told you|will blow your mind)\b",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://", re.IGNORECASE)
HASHTAG_RE = re.compile(r"(?:^|\s)#\w+")
# Approximation of the emoji/pictograph BMP-and-above ranges that show up in
# YouTube titles — exhaustive Unicode-emoji parsing would need a dependency.
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F6FF"   # symbols & pictographs, transport
    "\U0001F700-\U0001F9FF"   # alchemical, geometric, supplemental
    "\U0001FA00-\U0001FAFF"   # symbols & pictographs ext-A
    "☀-➿"           # misc symbols + dingbats
    "]"
)


def _caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _parse_yyyymmdd(s: str | None) -> date | None:
    if not s or len(s) != 8:
        return None
    try:
        return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None


def video_features(*, title: str, description: str, view_count: int | None) -> dict:
    title = title or ""
    description = description or ""
    return {
        "title_len": len(title),
        "title_word_count": len(title.split()),
        "title_caps_ratio": _caps_ratio(title),
        "title_excl_count": title.count("!"),
        "title_quest_count": title.count("?"),
        "title_emoji_count": len(EMOJI_RE.findall(title)),
        "title_digit_count": sum(1 for c in title if c.isdigit()),
        "title_clickbait": 1 if CLICKBAIT_RE.search(title) else 0,
        "desc_len": len(description),
        "desc_url_count": len(URL_RE.findall(description)),
        "desc_hashtag_count": len(HASHTAG_RE.findall(description)),
        "view_count_log": math.log10(1 + max(0, view_count or 0)),
    }


def channel_features(upload_dates: list[str | None]) -> dict:
    """Summarize a channel from the upload-date strings (yyyymmdd) we observed.

    ``channel_age_days`` is the span between oldest and newest *observed*
    upload — we don't fetch true channel-creation dates from yt-dlp.
    ``channel_mean_iud_days`` is the mean gap between consecutive uploads;
    0.0 when fewer than 2 datable videos exist (the classifier will see this
    as "no cadence signal" via the count column).
    """
    parsed = sorted(d for d in (_parse_yyyymmdd(s) for s in upload_dates) if d is not None)
    n_observed = len(upload_dates)
    if not parsed:
        return {
            "channel_video_count_observed": n_observed,
            "channel_age_days": 0,
            "channel_mean_iud_days": 0.0,
        }
    span = (parsed[-1] - parsed[0]).days
    if len(parsed) < 2:
        iud = 0.0
    else:
        gaps = [(parsed[i] - parsed[i - 1]).days for i in range(1, len(parsed))]
        iud = sum(gaps) / len(gaps)
    return {
        "channel_video_count_observed": n_observed,
        "channel_age_days": span,
        "channel_mean_iud_days": iud,
    }
