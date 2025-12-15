from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from dotenv import load_dotenv

from codex_audio.boundary.candidates import BoundaryCandidate
from codex_audio.clipper.ffmpeg import clip_segments
from codex_audio.config.station import StationConfig, load_station_config
from codex_audio.features.embeddings import AudioEmbedding, get_audio_embeddings
from codex_audio.features.vad import VadSegment, run_vad
from codex_audio.ingest import AudioMetadata, load_and_normalize_audio
from codex_audio.segmentation import (
    ChangePoint,
    RefinementParams,
    SegmentPlan,
    build_segments,
    compute_change_points,
    from_vad,
    refine_chunk_segments,
)
from codex_audio.segmentation.selection import SegmentConstraint
from codex_audio.text_features import TextChunk, build_text_chunks, detect_topic_boundaries
from codex_audio.text_features.embeddings import DEFAULT_EMBED_MODEL, ChunkEmbedding, embed_chunks
from codex_audio.transcription import (
    TranscriptWord,
    TranscriptionOutput,
    match_quote_to_timestamps,
    refine_range_with_silence,
    transcribe_audio,
)
from codex_audio.utils import get_logger

load_dotenv()

logger = get_logger(__name__)

DEFAULT_AUDIO_WINDOW_S = 5.0
DEFAULT_AUDIO_HOP_RATIO = 0.5
DEFAULT_TEXT_CHUNK_S = 5.0
DEFAULT_TEXT_OVERLAP = 0.5
DEFAULT_MIN_STORY_S = 30.0
DEFAULT_MAX_STORY_S = 120.0
DEFAULT_CANDIDATE_MIN_SCORE = 0.8
DEFAULT_HARD_MIN_SCORE = 1.2
DEFAULT_SNAP_WINDOW_S = 1.0
DEFAULT_SMOOTHING_WINDOW = 1


@dataclass
class PipelineConfig:
    station: str
    config_path: Optional[Path] = None
    sample_rate: int = 16_000
    working_dir: Optional[Path] = None
    vad_aggressiveness: int = 2
    vad_frame_duration_ms: int = 30
    min_silence_s: float = 1.0
    min_segment_s: float = 20.0
    transcription_enabled: bool = True
    transcription_language: str = "en-US"
    transcription_key: Optional[str] = None
    transcription_region: Optional[str] = None

    def resolve_station_config(self) -> StationConfig:
        if self.config_path:
            return load_station_config(self.config_path)
        return StationConfig(name=self.station, sample_rate=self.sample_rate)


@dataclass
class PipelineResult:
    segments: List[dict[str, Any]] = field(default_factory=list)
    output_dir: Path = Path("out")
    manifest_path: Optional[Path] = None
    normalized_audio: Optional[Path] = None
    metadata: Optional[AudioMetadata] = None
    clip_paths: List[Path] = field(default_factory=list)
    transcript_path: Optional[Path] = None


class StorySegmentationPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.station_config = config.resolve_station_config()
        self._refinement_params = self._build_refinement_params()
        text_cfg = self.station_config.text or {}
        self._llm_segmentation_enabled = bool(text_cfg.get("llm_segmentation"))
        self._llm_model = text_cfg.get("llm_model")
        self._llm_prompt = text_cfg.get("llm_prompt")
        logger.debug(
            "Initialized pipeline",
            extra={"station": self.station_config.name, "sample_rate": self.station_config.sample_rate},
        )

    def run(self, audio_path: Path, output_dir: Path) -> PipelineResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        work_dir = self.config.working_dir or (output_dir / "work")
        metadata, normalized_path = load_and_normalize_audio(
            audio_path, work_dir=work_dir, target_sample_rate=self.config.sample_rate
        )

        vad_segments = run_vad(
            normalized_path,
            aggressiveness=self.config.vad_aggressiveness,
            frame_duration_ms=self.config.vad_frame_duration_ms,
        )

        boundary_candidates = from_vad(
            vad_segments,
            min_silence_s=self.config.min_silence_s,
        )
        chunk_plans = build_segments(
            boundary_candidates,
            duration_s=metadata.duration_s,
            min_segment_s=self.config.min_segment_s,
        )

        audio_embeddings = self._compute_audio_embeddings(normalized_path)

        transcription: Optional[TranscriptionOutput] = None
        if self.config.transcription_enabled:
            transcription = self._run_transcription(normalized_path)

        text_chunks = self._build_text_chunks(transcription.words) if transcription else []
        text_embeddings = self._build_text_embeddings(text_chunks)

        transcript_words = transcription.words if transcription else None
        llm_candidates: List[BoundaryCandidate] = []
        if self._llm_segmentation_enabled and transcription and transcription.words:
            llm_candidates = self._generate_llm_candidates(transcription.words, audio_path=normalized_path)

        change_kwargs = self._change_point_kwargs()
        change_points = compute_change_points(
            audio_embeddings=audio_embeddings,
            text_embeddings=text_embeddings or None,
            vad_segments=vad_segments,
            diarization_segments=None,
            transcript_words=transcript_words,
            **change_kwargs,
        )

        segment_plans = self._refine_chunks(
            chunk_plans=chunk_plans,
            change_points=change_points,
            vad_segments=vad_segments,
            transcript_words=transcript_words,
            extra_candidates=llm_candidates,
        )

        segment_ranges = [(plan.start_s, plan.end_s) for plan in segment_plans]
        clip_dir = output_dir / "clips"
        clip_paths = clip_segments(normalized_path, segment_ranges, clip_dir)

        transcription_payload: Optional[dict[str, Any]] = None
        transcript_path: Optional[Path] = None
        if transcription:
            transcription_payload = transcription.to_payload()
            transcript_path = output_dir / "transcript.json"
            transcript_path.write_text(json.dumps(transcription_payload, indent=2))

        segments_payload: List[dict[str, Any]] = []
        for idx, plan in enumerate(segment_plans):
            clip_path = clip_paths[idx] if idx < len(clip_paths) else None
            segments_payload.append(
                {
                    "start": plan.start_s,
                    "end": plan.end_s,
                    "label": plan.label,
                    "clip_path": str(clip_path) if clip_path else None,
                }
            )

        manifest_path = output_dir / "segments.json"
        manifest_payload = {
            "audio": str(audio_path),
            "normalized_audio": str(normalized_path),
            "segments": segments_payload,
            "metadata": {
                "duration_s": metadata.duration_s,
                "sample_rate": metadata.sample_rate,
                "channels": metadata.channels,
            },
            "transcript": transcription_payload,
            "transcript_path": str(transcript_path) if transcript_path else None,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2))
        logger.info(
            "Pipeline executed",
            extra={
                "audio": str(audio_path),
                "normalized": str(normalized_path),
                "segments": len(segment_plans),
                "clips": len(clip_paths),
                "out": str(manifest_path),
                "transcript": bool(transcription_payload),
            },
        )
        return PipelineResult(
            segments=segments_payload,
            output_dir=output_dir,
            manifest_path=manifest_path,
            normalized_audio=normalized_path,
            metadata=metadata,
            clip_paths=clip_paths,
            transcript_path=transcript_path,
        )

    def _run_transcription(self, audio_path: Path) -> Optional[TranscriptionOutput]:
        try:
            return transcribe_audio(
                audio_path,
                key=self.config.transcription_key,
                region=self.config.transcription_region,
                language=self.config.transcription_language,
            )
        except Exception as exc:  # pragma: no cover - logging path
            logger.warning("Transcription failed", extra={"error": str(exc)})
        return None

    def _compute_audio_embeddings(self, audio_path: Path) -> List[AudioEmbedding]:
        window_s, hop_ratio = self._audio_embedding_options()
        try:
            return get_audio_embeddings(audio_path, window_s=window_s, hop_ratio=hop_ratio)
        except Exception as exc:  # pragma: no cover - logging path
            logger.warning("Audio embeddings failed", extra={"error": str(exc)})
            return []

    def _build_text_chunks(self, words: Sequence[TranscriptWord]) -> List[TextChunk]:
        if not words:
            return []
        chunk_size, overlap = self._text_chunk_options()
        try:
            return build_text_chunks(words, chunk_size_s=chunk_size, overlap_ratio=overlap)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to build text chunks", extra={"error": str(exc)})
            return []

    def _build_text_embeddings(self, chunks: Sequence[TextChunk]) -> List[ChunkEmbedding]:
        if not chunks:
            return []
        text_cfg = self.station_config.text or {}
        model = text_cfg.get("embedding_model") or DEFAULT_EMBED_MODEL
        api_version = text_cfg.get("embedding_api_version")
        key = text_cfg.get("embedding_key")
        endpoint = text_cfg.get("embedding_endpoint")
        try:
            return embed_chunks(
                chunks,
                model=model,
                api_version=api_version,
                key=key,
                endpoint=endpoint,
            )
        except Exception as exc:  # pragma: no cover - logging path
            logger.warning("Text embeddings failed", extra={"error": str(exc)})
            return []

    def _generate_llm_candidates(
        self,
        words: Sequence[TranscriptWord],
        audio_path: Path | None = None,
    ) -> List[BoundaryCandidate]:
        if not self._llm_segmentation_enabled or not words:
            return []
        try:
            candidates = detect_topic_boundaries(
                words,
                model=self._llm_model,
                system_prompt=self._llm_prompt,
            )
        except Exception as exc:  # pragma: no cover - logging path
            logger.warning("LLM segmentation failed", extra={"error": str(exc)})
            return []
        return self._align_llm_candidates(
            candidates=candidates,
            words=words,
            audio_path=audio_path,
        )


    def _align_llm_candidates(
        self,
        *,
        candidates: Sequence[BoundaryCandidate],
        words: Sequence[TranscriptWord],
        audio_path: Path | None,
    ) -> List[BoundaryCandidate]:
        if not candidates:
            return []
        aligned: List[BoundaryCandidate] = []
        for candidate in candidates:
            aligned.append(
                self._match_llm_candidate(candidate=candidate, words=words, audio_path=audio_path)
            )
        return aligned


    def _match_llm_candidate(
        self,
        *,
        candidate: BoundaryCandidate,
        words: Sequence[TranscriptWord],
        audio_path: Path | None,
    ) -> BoundaryCandidate:
        quote = getattr(candidate, "quote", None)
        if not quote:
            return candidate
        try:
            match_range = match_quote_to_timestamps(quote, words)
        except RuntimeError as exc:  # pragma: no cover - optional dependency missing
            logger.debug("LLM quote matching unavailable", extra={"error": str(exc)})
            return candidate
        if not match_range:
            return candidate
        start_ms, end_ms = match_range
        refined_start, refined_end = start_ms, end_ms
        if audio_path:
            try:
                refined_start, refined_end = refine_range_with_silence(
                    audio_path, (start_ms, end_ms)
                )
            except Exception as exc:  # pragma: no cover - optional dependency missing
                logger.debug("LLM quote refinement skipped", extra={"error": str(exc)})
        new_time_s = refined_start / 1000.0
        return BoundaryCandidate(
            time_s=new_time_s,
            score=candidate.score,
            reason=candidate.reason,
            quote=candidate.quote,
        )



    def _audio_embedding_options(self) -> tuple[float, float]:
        heuristics = self.station_config.heuristics or {}
        window_s = _first_float(heuristics, ["audio_window_s"], DEFAULT_AUDIO_WINDOW_S)
        hop_ratio = _first_float(heuristics, ["audio_hop_ratio"], DEFAULT_AUDIO_HOP_RATIO)
        if window_s <= 0:
            window_s = DEFAULT_AUDIO_WINDOW_S
        if hop_ratio <= 0 or hop_ratio > 1:
            hop_ratio = DEFAULT_AUDIO_HOP_RATIO
        return window_s, hop_ratio

    def _text_chunk_options(self) -> tuple[float, float]:
        text_cfg = self.station_config.text or {}
        chunk_size = _first_float(text_cfg, ["chunk_s", "chunk_size_s"], DEFAULT_TEXT_CHUNK_S)
        overlap = _first_float(text_cfg, ["chunk_overlap_ratio"], DEFAULT_TEXT_OVERLAP)
        if chunk_size <= 0:
            chunk_size = DEFAULT_TEXT_CHUNK_S
        if overlap < 0 or overlap >= 1:
            overlap = DEFAULT_TEXT_OVERLAP
        return chunk_size, overlap

    def _refine_chunks(
        self,
        *,
        chunk_plans: Sequence[SegmentPlan],
        change_points: Sequence[ChangePoint],
        vad_segments: Sequence[VadSegment],
        transcript_words: Sequence[TranscriptWord] | None,
        extra_candidates: Sequence[BoundaryCandidate] | None = None,
    ) -> List[SegmentPlan]:
        refined: List[SegmentPlan] = []
        for chunk in chunk_plans:
            chunk_extras = None
            if extra_candidates:
                chunk_extras = [
                    candidate
                    for candidate in extra_candidates
                    if chunk.start_s < candidate.time_s < chunk.end_s
                ]
            chunk_segments = refine_chunk_segments(
                chunk.start_s,
                chunk.end_s,
                change_points=change_points,
                params=self._refinement_params,
                vad_segments=vad_segments,
                transcript_words=transcript_words,
                extra_candidates=chunk_extras,
            )
            refined.extend(self._label_refined_segments(chunk, chunk_segments))
        return refined or list(chunk_plans)

    def _label_refined_segments(
        self,
        chunk: SegmentPlan,
        refined: Sequence[SegmentPlan],
    ) -> List[SegmentPlan]:
        if not refined:
            return [SegmentPlan(chunk.start_s, chunk.end_s, chunk.label)]
        if (
            len(refined) == 1
            and refined[0].label.startswith("chunk_")
            and self._covers_chunk(chunk, refined[0])
        ):
            return [SegmentPlan(chunk.start_s, chunk.end_s, chunk.label)]
        prefix = chunk.label or "chunk"
        labeled: List[SegmentPlan] = []
        for idx, segment in enumerate(refined):
            start = max(chunk.start_s, segment.start_s)
            end = min(chunk.end_s, segment.end_s)
            if end <= start:
                continue
            suffix = segment.label or f"segment_{idx}"
            labeled.append(SegmentPlan(start, end, f"{prefix}|{suffix}"))
        return labeled or [SegmentPlan(chunk.start_s, chunk.end_s, chunk.label)]

    @staticmethod
    def _covers_chunk(chunk: SegmentPlan, candidate: SegmentPlan, tolerance: float = 1e-3) -> bool:
        return (
            abs(candidate.start_s - chunk.start_s) <= tolerance
            and abs(candidate.end_s - chunk.end_s) <= tolerance
        )

    def _build_refinement_params(self) -> RefinementParams:
        heuristics = self.station_config.heuristics or {}
        min_story = _first_float(
            heuristics,
            ["min_story_len", "min_story_s", "min_segment_s"],
            DEFAULT_MIN_STORY_S,
        )
        max_story = _first_optional_float(
            heuristics,
            ["max_story_len", "max_story_s"],
            DEFAULT_MAX_STORY_S,
        )
        if max_story is not None and max_story <= min_story:
            max_story = None

        candidate_min = _first_float(
            heuristics,
            ["candidate_min_score"],
            DEFAULT_CANDIDATE_MIN_SCORE,
        )
        hard_min = _first_float(
            heuristics,
            ["hard_min_cut_score"],
            DEFAULT_HARD_MIN_SCORE,
        )
        smoothing = int(
            max(
                0,
                round(
                    _first_float(
                        heuristics,
                        ["change_score_smoothing"],
                        DEFAULT_SMOOTHING_WINDOW,
                    )
                ),
            )
        )
        snap_window = _first_float(
            heuristics,
            ["snap_window_s", "boundary_snap_window"],
            DEFAULT_SNAP_WINDOW_S,
        )
        weight_overrides = heuristics.get("change_score_weights") or {}
        weights: Dict[str, float] = dict(RefinementParams().weights)
        for key, value in weight_overrides.items():
            try:
                weights[key] = float(value)
            except (TypeError, ValueError):
                continue

        constraints = SegmentConstraint(min_len=max(1.0, min_story), max_len=max_story)
        return RefinementParams(
            constraints=constraints,
            weights=weights,
            candidate_min_score=max(0.0, candidate_min),
            hard_min_cut_score=max(0.0, hard_min),
            smoothing_window=smoothing,
            snap_window_s=max(0.0, snap_window),
        )

    def _change_point_kwargs(self) -> Dict[str, object]:
        heuristics = self.station_config.heuristics or {}
        patterns_raw = heuristics.get("keyword_patterns")
        if isinstance(patterns_raw, str):
            keyword_patterns = [patterns_raw]
        elif isinstance(patterns_raw, (list, tuple)):
            keyword_patterns = [str(item) for item in patterns_raw if item is not None]
        else:
            keyword_patterns = None
        return {
            "silence_window_s": _first_float(
                heuristics,
                ["silence_window_s", "silence_min_s"],
                1.0,
            ),
            "silence_norm_s": _first_float(
                heuristics,
                ["silence_norm_s", "silence_min_s"],
                1.0,
            ),
            "anchor_tolerance_s": _first_float(
                heuristics,
                ["anchor_tolerance_s"],
                0.5,
            ),
            "audio_threshold": _clamped_threshold(
                _first_float(heuristics, ["audio_threshold", "embedding_threshold"], 0.6)
            ),
            "text_threshold": _clamped_threshold(
                _first_float(heuristics, ["text_threshold"], 0.7)
            ),
            "keyword_patterns": keyword_patterns,
            "keyword_score": _first_float(heuristics, ["keyword_score"], 5.0),
        }


def _first_float(mapping: Mapping[str, Any], keys: Sequence[str], default: float) -> float:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            try:
                return float(mapping[key])
            except (TypeError, ValueError):
                continue
    return default


def _first_optional_float(
    mapping: Mapping[str, Any], keys: Sequence[str], default: Optional[float] = None
) -> Optional[float]:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            try:
                return float(mapping[key])
            except (TypeError, ValueError):
                continue
    return default


def _clamped_threshold(value: float) -> float:
    if value <= -1.0:
        return -0.99
    if value > 1.0:
        return 1.0
    return value

