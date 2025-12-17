# News Audio Story Segmentation

Hybrid audio + text tooling for automatically slicing 24/7 radio news recordings
into story-level clips. The pipeline combines speech-to-text, diarization,
acoustic change detection, and an LLM-based topic segmenter to deliver
production-ready boundaries with precise timestamps, quotes, and labels.

---

## Key Capabilities
- **Audio feature stack** – VAD, diarization, sliding-window embeddings, silence
  and jingle detection, and acoustic change-points feed candidate generation.
- **Text feature stack** – Azure/Whisper transcripts, word-aligned timestamps,
  text embeddings, and LLM topic votes enrich the signal surface.
- **LLM boundary schema** – Every topic boundary now emits `time_s`, a verbatim
  5–15 word `quote`, a categorical `type`, and a 0–1 `confidence`, making
  transcript alignment deterministic and ready for silence snapping.
- **FFmpeg clipping** – Selected boundaries become gap-free WAV segments plus a
  JSON manifest for downstream workflows.
- **Evaluation harness** – Ground-truth manifests, precision/recall/F1 summaries,
  and hyper-parameter sweeps keep tuning measurable.

---

## Architecture Overview
1. **Ingestor** – Normalizes any input into 16 kHz mono WAV and extracts
   metadata.
2. **Dual feature extraction** – Audio track (VAD, diarization, embeddings) and
   text track (transcription, chunking, embeddings) run in parallel.
3. **Hybrid boundary detector** – Scores silence gaps, acoustic deltas, semantic
   shifts, anchors, and LLM topic votes into ranked `BoundaryCandidate` objects.
4. **Planner & refinement** – Deduplicates, enforces min spacing, and snaps to
   word boundaries or nearby silence/jingles.
5. **Clipper** – Invokes FFmpeg to cut WAVs per segment and emits
   `segments.json` + `transcript.json` for consumers.

---

## Repository Layout
```
audio-clipping-poc/
  README.md
  pyproject.toml
  requirements.txt
  requirements-dev.txt
  samples/                  # shared WAV snippets & fixtures
  src/codex_audio/          # pipeline code grouped by responsibility
  tests/                    # pytest suite mirroring src layout + fixtures
  docs/assets/              # lightweight SVGs / diagrams
```

---

## Getting Started
1. **Python & venv**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # or source .venv/bin/activate on macOS/Linux
   ```
2. **Install deps**
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```
3. **Environment** – Set Azure credentials in `.env` or your shell:
   - `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`
   - `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`
   - Optional: `AZURE_OPENAI_LLM_MODEL` override

---

## Running the CLI
### Segment an Audio File
```bash
codex-audio segment samples/snare.wav \
  --station CKNW \
  --config config/stations/CKNW.yaml \
  --out out/cknw_demo
```
Produces:
```
out/cknw_demo/
  segments/
    segment_000.wav
    segment_001.wav
  segments.json
  transcript.json
```

`segments.json` contains ranked boundaries with the new schema:
```json
{
  "segments": [
    {
      "id": "segment_001",
      "start_s": 185.92,
      "end_s": 246.07,
      "llm": {
        "time_s": 187.11,
        "quote": "the finance minister returned to the legislature",
        "type": "new_story",
        "confidence": 0.82
      }
    }
  ]
}
```
The quote is copied verbatim from the transcript so `match_quote_to_timestamps`
can deterministically snap the boundary to real word timings before the final
silence/jingle refinement.

#### Boundary schema
| field        | Type    | Notes                                                                 |
|--------------|---------|-----------------------------------------------------------------------|
| `time_s`     | float   | Initial LLM boundary guess in seconds (snapped later via quote match) |
| `quote`      | string  | 5–15 exact transcript words near the transition                       |
| `type`       | string  | One of `new_story`, `return_to_anchor`, `ad_break`, `tease`, `weather`, `traffic`, `sports` |
| `confidence` | float   | LLM certainty between 0 and 1                                         |

These attributes now flow through the refinement pipeline so downstream tools
can key on consistent labels while still benefiting from deterministic quote
alignment.

### Azure STT + Diarization
```yaml
transcription:
  provider: azure_speech
  diarization: true
  max_speakers: 6
```
Running with `diarization: true` inserts a `speaker` value for every word in
`transcript.json` while keeping LLM quote alignment, silence refinement, and the
new schema intact.

---

## Evaluation & Sweeps
1. **Manifest** (`evaluation_manifest.csv`)
   ```csv
   audio_path,annotation_path,station
   data/CKNW_2025-01-15.wav,annotations/CKNW_2025-01-15.csv,CKNW
   ```
2. **Ground-truth CSV** – one `time_s` column listing human boundaries.
3. **Run evaluation**
   ```bash
   codex-audio eval \
     --manifest evaluation_manifest.csv \
     --tolerance-s 3.0 \
     --config station_config.yaml
   ```
4. **Threshold sweeps**
   ```bash
   codex-audio sweep \
     --manifest evaluation_manifest.csv \
     --param silence_min_s 0.7 1.0 1.3 \
     --param min_boundary_score 3.0 4.0 5.0
   ```

---

## Testing & QA
- `pytest -q`
- `ruff check src tests`
- `black --check src tests`
- `mypy src`

Keep coverage above 85% with `pytest --cov=audio_clipping --cov-report=term-missing`.

---

## Extending the Pipeline
- LLM-based story titling and summaries.
- Automatic jingle classifiers per station.
- Real-time streaming segmentation.
- Station-specific auto-tuning.
- Combined audio+video segmentation experiments.

---

## Contributing
1. Follow Conventional Commits, e.g., `feat(dsp): add soft clipper`.
2. Keep PRs focused with proof (CLI output, pytest, screenshots if applicable).
3. Request review and wait for CI green lights before merging.

Need help bootstrapping configs, notebooks, or scripts? Open an issue and we can
chat through it.
