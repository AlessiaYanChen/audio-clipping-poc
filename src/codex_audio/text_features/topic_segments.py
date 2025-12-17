from __future__ import annotations

import json
import os
import re
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterator, List, Sequence

from openai import AzureOpenAI

from codex_audio.boundary.candidates import BoundaryCandidate
from codex_audio.transcription import TranscriptWord

DEFAULT_LLM_MODEL = os.getenv("AZURE_OPENAI_LLM_MODEL", "gpt-4o-mini")
DEFAULT_SYSTEM_PROMPT = (
    "You segment radio news transcripts into distinct stories. "
    "Return JSON with a 'boundaries' array of objects {"
    '"time_s": number seconds since audio start, '
    '"quote": string of 5-15 exact words copied verbatim from the transcript near the boundary, '
    '"type": one of [new_story, return_to_anchor, ad_break, tease, weather, traffic, sports], '
    '"confidence": number between 0 and 1 representing boundary certainty}'
)
WINDOW_DURATION_S = 10 * 60  # 10-minute content windows keep context tight
WINDOW_OVERLAP_S = 2 * 60    # 2-minute overlap lets adjacent windows vote
MERGE_TOLERANCE_S = 5.0      # boundaries agreeing within 5s collapse into one
BASE_SCORE = 3.0
VOTE_BONUS = 0.5
BOUNDARY_TYPES = {
    "new_story",
    "return_to_anchor",
    "ad_break",
    "tease",
    "weather",
    "traffic",
    "sports",
}

ResponseProvider = Callable[[str, str, str | None, str | None, str | None, str], str]


@dataclass(frozen=True)
class ParsedBoundary:
    time_s: float
    quote: str | None
    boundary_type: str | None
    confidence: float | None


def detect_topic_boundaries(
    words: Sequence[TranscriptWord],
    *,
    model: str | None = None,
    system_prompt: str | None = None,
    key: str | None = None,
    endpoint: str | None = None,
    api_version: str | None = None,
    response_provider: ResponseProvider | None = None,
) -> List[BoundaryCandidate]:
    word_list = list(words)
    if not word_list:
        return []

    provider = response_provider or _call_chat_completion
    aggregate_entries: List[ParsedBoundary] = []

    for window_words in _iter_windows(word_list):
        prompt = _build_prompt(window_words)
        response_text = provider(
            prompt,
            model or DEFAULT_LLM_MODEL,
            key or os.getenv("AZURE_OPENAI_KEY"),
            endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            system_prompt or DEFAULT_SYSTEM_PROMPT,
        )
        aggregate_entries.extend(_parse_boundaries(response_text))

    merged_entries = _merge_boundary_votes(aggregate_entries)
    return [
        BoundaryCandidate(
            time_s=time,
            score=BASE_SCORE + VOTE_BONUS * (votes - 1),
            reason="llm_topic_change" if votes == 1 else f"llm_topic_change (votes={votes})",
            quote=quote,
            boundary_type=boundary_type,
            confidence=confidence,
        )
        for time, quote, votes, boundary_type, confidence in merged_entries
    ]


def _build_prompt(words: Sequence[TranscriptWord]) -> str:
    if not words:
        return "Analyze the transcript and return JSON boundaries."

    lines = [f"[{word.start_s:.2f}] {word.text}" for word in words]
    transcript = "\n".join(lines)
    window_start = words[0].start_s
    window_end = words[-1].start_s
    type_values = ", ".join(sorted(BOUNDARY_TYPES))
    instructions = (
        "Analyze the following radio news transcript excerpt ("  # keep request concise
        f"{window_start:.2f}s to {window_end:.2f}s). "
        "Identify timestamps (in seconds) where the topic changes. "
        'Respond with JSON: {"boundaries": [{"time_s": number, "quote": string of 5-15 exact words copied from the transcript near the boundary, '
        '"type": one of ['
        + type_values
        + '], "confidence": number between 0 and 1}, ...]}. '
        "Quotes must be verbatim phrases, not summaries."
    )
    return f"{instructions}\n\nTranscript:\n{transcript}"


def _call_chat_completion(
    prompt: str,
    model: str,
    key: str | None,
    endpoint: str | None,
    api_version: str,
    system_prompt: str,
) -> str:
    if not key or not endpoint:
        raise ValueError("Azure OpenAI key/endpoint must be configured")
    client = AzureOpenAI(api_key=key, api_version=api_version, azure_endpoint=endpoint)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    message = response.choices[0].message
    return message.content or ""


def _parse_boundaries(response_text: str) -> List[ParsedBoundary]:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        pattern = re.compile(r"\[(\d+(?:\.\d+)?)\]")
        matches = pattern.findall(response_text)
        if not matches:
            matches = re.findall(r"(\d+(?:\.\d+)?)", response_text)
        return [
            ParsedBoundary(
                time_s=float(value),
                quote=None,
                boundary_type=None,
                confidence=None,
            )
            for value in matches
            if _valid_time(value)
        ]

    entries = payload.get("boundaries", []) if isinstance(payload, dict) else []
    parsed: List[ParsedBoundary] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        time_value = item.get("time_s")
        if time_value is None:
            time_value = item.get("time") or item.get("timestamp")
        if not _valid_time(time_value):
            continue
        quote = _clean_quote(item.get("quote") or item.get("reason"))
        boundary_type = _coerce_boundary_type(item.get("type"))
        confidence = _coerce_confidence(item.get("confidence"))
        parsed.append(
            ParsedBoundary(
                time_s=float(time_value),
                quote=quote,
                boundary_type=boundary_type,
                confidence=confidence,
            )
        )
    return parsed


def _iter_windows(
    words: Sequence[TranscriptWord],
    *,
    window_duration_s: float = WINDOW_DURATION_S,
    overlap_s: float = WINDOW_OVERLAP_S,
) -> Iterator[Sequence[TranscriptWord]]:
    if window_duration_s <= 0:
        yield words
        return

    effective_overlap = min(overlap_s, max(window_duration_s - 1.0, 0.0))
    hop_s = max(window_duration_s - effective_overlap, 1.0)
    start_times = [word.start_s for word in words]
    start_idx = 0

    while start_idx < len(words):
        start_time = start_times[start_idx]
        end_time = start_time + window_duration_s
        end_idx = bisect_left(start_times, end_time, lo=start_idx)
        if end_idx == start_idx:
            end_idx = min(start_idx + 1, len(words))
        yield words[start_idx:end_idx]
        if end_idx >= len(words):
            break
        next_start_time = start_time + hop_s
        start_idx = bisect_left(start_times, next_start_time, lo=start_idx + 1)


def _merge_boundary_votes(
    entries: Sequence[ParsedBoundary], *, tolerance_s: float = MERGE_TOLERANCE_S
) -> List[tuple[float, str | None, int, str | None, float | None]]:
    if not entries:
        return []

    sorted_entries = sorted(entries, key=lambda item: item.time_s)
    merged: List[tuple[float, str | None, int, str | None, float | None]] = []
    cluster: List[ParsedBoundary] = [sorted_entries[0]]

    for current in sorted_entries[1:]:
        previous_time = cluster[-1].time_s
        if current.time_s - previous_time <= tolerance_s:
            cluster.append(current)
            continue
        merged.append(_collapse_cluster(cluster))
        cluster = [current]

    merged.append(_collapse_cluster(cluster))
    return merged


def _collapse_cluster(
    cluster: Sequence[ParsedBoundary],
) -> tuple[float, str | None, int, str | None, float | None]:
    votes = len(cluster)
    avg_time = sum(item.time_s for item in cluster) / votes
    quote = next((item.quote for item in cluster if item.quote), cluster[0].quote)
    boundary_type = _select_boundary_type(cluster)
    confidence = _average_confidence(cluster)
    return avg_time, quote, votes, boundary_type, confidence


def _select_boundary_type(cluster: Sequence[ParsedBoundary]) -> str | None:
    labels = [entry.boundary_type for entry in cluster if entry.boundary_type]
    if not labels:
        return cluster[0].boundary_type
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


def _average_confidence(cluster: Sequence[ParsedBoundary]) -> float | None:
    confidences = [entry.confidence for entry in cluster if entry.confidence is not None]
    if not confidences:
        return None
    return sum(confidences) / len(confidences)


def _valid_time(value: object) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _clean_quote(value: object) -> str | None:
    if value is None:
        return None
    quote = str(value).strip()
    if not quote:
        return None
    words = quote.split()
    if not words:
        return None
    if len(words) > 15:
        words = words[:15]
    return " ".join(words)


def _coerce_boundary_type(value: object) -> str | None:
    if value is None:
        return None
    boundary_type = str(value).strip().lower()
    if not boundary_type:
        return None
    if boundary_type not in BOUNDARY_TYPES:
        return None
    return boundary_type


def _coerce_confidence(value: object) -> float | None:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence < 0.0 or confidence > 1.0:
        return None
    return confidence


__all__ = ["detect_topic_boundaries"]
