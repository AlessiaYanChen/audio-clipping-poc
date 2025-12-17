from __future__ import annotations

import json

from codex_audio.boundary.candidates import BoundaryCandidate
from codex_audio.text_features.topic_segments import detect_topic_boundaries
from codex_audio.transcription import TranscriptWord


def _words() -> list[TranscriptWord]:
    return [
        TranscriptWord(text="Hello", start_s=0.0, end_s=0.5),
        TranscriptWord(text="world", start_s=0.5, end_s=1.0),
    ]


def test_detect_topic_boundaries_parses_json(monkeypatch) -> None:
    def fake_provider(prompt, model, key, endpoint, api_version, system_prompt):
        return json.dumps(
            {
                "boundaries": [
                    {
                        "time_s": 42.5,
                        "quote": "Hello world new story pivot now",
                        "type": "new_story",
                        "confidence": 0.87,
                    }
                ]
            }
        )

    candidates = detect_topic_boundaries(
        _words(),
        model="gpt",
        key="key",
        endpoint="https://example.com",
        response_provider=fake_provider,
    )

    assert len(candidates) == 1
    assert isinstance(candidates[0], BoundaryCandidate)
    assert candidates[0].time_s == 42.5
    assert candidates[0].reason == "llm_topic_change"
    assert candidates[0].quote == "Hello world new story pivot now"
    assert candidates[0].boundary_type == "new_story"
    assert candidates[0].confidence == 0.87


def test_detect_topic_boundaries_handles_brackets(monkeypatch) -> None:
    def fake_provider(prompt, model, key, endpoint, api_version, system_prompt):
        return "[10.0] Politics\n[55.0] Weather"

    candidates = detect_topic_boundaries(
        _words(),
        model="gpt",
        key="key",
        endpoint="https://example.com",
        response_provider=fake_provider,
    )

    assert [round(c.time_s) for c in candidates] == [10, 55]
    assert all(candidate.quote is None for candidate in candidates)
