"""End-to-end pipeline wiring, with the models stubbed out.

Proves stage sequencing, output layout, timing capture, and the rename
round-trip without downloading weights or needing a Hugging Face token — so
this stays fast enough to run on every change.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scribe import pipeline as pipe
from scribe.config import Settings
from scribe.types import Diarization, Segment, Word

A, B = "SPEAKER_00", "SPEAKER_01"


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    """A real (silent) media file, so ffprobe/ffmpeg run for real."""
    path = tmp_path / "meeting.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-t", "6",
         "-i", "anullsrc=r=16000:cl=mono", "-c:a", "pcm_s16le", str(path)],
        check=True,
    )
    return path


class FakeAsr:
    name = "fake"
    model_id = "fake-asr-v1"

    def __init__(self, **_: object) -> None:
        self.calls = 0

    def transcribe(self, wav: Path) -> list[Word]:
        self.calls += 1
        texts = "we should ship it yes agreed lets go".split()
        return [
            Word(t=text, start=round(i * 0.5, 2), end=round(i * 0.5 + 0.4, 2), conf=0.9)
            for i, text in enumerate(texts)
        ]


class FakeDiarizer:
    name = "fake"
    model_id = "fake-diar-v1"

    def __init__(self, **_: object) -> None:
        pass

    def diarize(self, wav: Path) -> Diarization:
        segs = [Segment(A, 0.0, 2.05), Segment(B, 2.05, 6.0)]
        return Diarization(exclusive=segs, overlapped=segs)


@pytest.fixture
def stubbed(monkeypatch):
    asr = FakeAsr()
    monkeypatch.setattr("scribe.asr.base.load_backend", lambda *a, **k: asr)
    monkeypatch.setattr("scribe.diarize.base.load_backend", lambda *a, **k: FakeDiarizer())
    return asr


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    return s


class TestRun:
    def test_writes_every_output_plus_the_sidecar(self, wav, settings, stubbed):
        result = pipe.run(wav, settings)
        names = {p.name for p in result.written}
        assert names == {
            "transcript.json", "transcript.md", "transcript.txt",
            "transcript.srt", "transcript.vtt", "speakers.yaml",
        }

    def test_every_stage_is_timed(self, wav, settings, stubbed):
        result = pipe.run(wav, settings)
        assert set(result.durations) == set(pipe.STAGES)
        assert all(v >= 0 for v in result.durations.values())

    def test_progress_is_reported_for_all_stages_and_never_regresses(
        self, wav, settings, stubbed
    ):
        seen: list[tuple[str, float]] = []
        pipe.run(wav, settings, on_progress=lambda s, f: seen.append((s, f)))
        assert {s for s, _ in seen} == set(pipe.STAGES)
        order = [pipe.STAGES.index(s) for s, _ in seen]
        assert order == sorted(order), "stages reported out of order"

    def test_transcript_records_source_and_backends(self, wav, settings, stubbed):
        t = pipe.run(wav, settings).transcript
        assert t.source.filename == "meeting.wav"
        assert len(t.source.sha256) == 64
        assert t.source.duration_s == pytest.approx(6.0, abs=0.2)
        assert t.pipeline.asr.model == "fake-asr-v1"
        assert t.pipeline.diarization.model == "fake-diar-v1"

    def test_two_speakers_are_separated(self, wav, settings, stubbed):
        t = pipe.run(wav, settings).transcript
        assert [s.label for s in t.speakers] == ["Speaker 1", "Speaker 2"]
        assert [x.speaker for x in t.turns] == [A, B]

    def test_output_dir_is_stable_for_the_same_content(self, wav, settings, stubbed):
        first = pipe.run(wav, settings).out_dir
        second = pipe.run(wav, settings).out_dir
        assert first == second

    def test_normalized_wav_is_cleaned_up(self, wav, settings, stubbed):
        pipe.run(wav, settings)
        assert not list(settings.work.rglob("*.wav"))

    def test_keep_wav_retains_it(self, wav, settings, stubbed):
        pipe.run(wav, settings, keep_wav=True)
        assert list(settings.work.rglob("*.wav"))

    def test_video_container_is_accepted(self, tmp_path, settings, stubbed):
        # Zoom/Meet exports arrive as video; ffmpeg must drop the video stream.
        mp4 = tmp_path / "call.mp4"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y",
             "-f", "lavfi", "-t", "4", "-i", "testsrc=size=160x120:rate=10",
             "-f", "lavfi", "-t", "4", "-i", "anullsrc=r=16000:cl=mono",
             "-shortest", str(mp4)],
            check=True,
        )
        result = pipe.run(mp4, settings)
        assert result.transcript.source.filename == "call.mp4"

    def test_existing_names_survive_a_rerun(self, wav, settings, stubbed):
        first = pipe.run(wav, settings)
        sidecar = first.out_dir / "speakers.yaml"
        sidecar.write_text(sidecar.read_text().replace("name: null", "name: Darin", 1))

        second = pipe.run(wav, settings)
        assert second.transcript.speakers[0].name == "Darin"
        assert "Darin" in (second.out_dir / "transcript.md").read_text()


class TestRerender:
    def test_applies_names_without_running_inference(self, wav, settings, stubbed):
        result = pipe.run(wav, settings)
        assert stubbed.calls == 1

        sidecar = result.out_dir / "speakers.yaml"
        sidecar.write_text(sidecar.read_text().replace("name: null", "name: Darin", 1))

        again = pipe.rerender(result.out_dir)
        assert stubbed.calls == 1, "rerender must not re-transcribe"
        assert again.transcript.speakers[0].name == "Darin"
        for name in ("transcript.md", "transcript.txt", "transcript.srt"):
            assert "Darin" in (result.out_dir / name).read_text()

    def test_missing_transcript_is_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            pipe.rerender(tmp_path)
