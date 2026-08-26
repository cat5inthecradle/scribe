"""Renderer tests: format correctness and the rename round-trip."""

from __future__ import annotations

from conftest import sequence

from scribe import speakers as sidecar
from scribe.merge import merge
from scribe.render import render_all
from scribe.render.md import render_md
from scribe.render.srt import MAX_CUE_CHARS, build_cues, render_srt, render_vtt
from scribe.render.timecode import hhmmss, subtitle_time
from scribe.render.txt import render_txt
from scribe.types import Diarization, Segment, Transcript

A, B = "SPEAKER_00", "SPEAKER_01"


def two_speaker(make_transcript) -> Transcript:
    words = sequence(*"hello there how are you doing today friend".split())
    d = Diarization(
        exclusive=[Segment(A, 0.0, 1.10), Segment(B, 1.10, 10.0)],
        overlapped=[],
    )
    turns, spk = merge(words, d)
    return make_transcript(turns, spk)


class TestTimecode:
    def test_hhmmss_drops_the_hour_when_zero(self):
        assert hhmmss(75.4) == "01:15"

    def test_hhmmss_includes_hours(self):
        assert hhmmss(3725.0) == "1:02:05"

    def test_subtitle_time_srt_and_vtt_separators(self):
        assert subtitle_time(3725.5) == "01:02:05,500"
        assert subtitle_time(3725.5, sep=".") == "01:02:05.500"

    def test_millisecond_rounding_carries_correctly(self):
        # 59.9999 must not render as :59,1000
        assert subtitle_time(59.9999) == "00:01:00,000"

    def test_negative_is_clamped(self):
        assert subtitle_time(-1.0) == "00:00:00,000"


class TestMarkdown:
    def test_heading_appears_once_per_speaker_change(self, make_transcript):
        t = two_speaker(make_transcript)
        md = render_md(t)
        assert md.count("**Speaker 1**") == 1
        assert md.count("**Speaker 2**") == 1

    def test_names_override_labels(self, make_transcript):
        t = two_speaker(make_transcript)
        t.speakers[0].name = "Darin"
        md = render_md(t)
        assert "**Darin**" in md
        assert "**Speaker 1**" not in md

    def test_metadata_header_present(self, make_transcript):
        md = render_md(two_speaker(make_transcript))
        assert "meeting.m4a" in md
        assert "**Speakers:** 2" in md


class TestTxt:
    def test_one_line_per_turn_with_timestamp(self, make_transcript):
        t = two_speaker(make_transcript)
        lines = render_txt(t).strip().splitlines()
        assert len(lines) == len(t.turns)
        assert lines[0].startswith("[00:00] Speaker 1: ")


class TestSubtitles:
    def test_srt_blocks_are_sequentially_numbered(self, make_transcript):
        srt = render_srt(two_speaker(make_transcript))
        assert srt.startswith("1\n")
        assert " --> " in srt

    def test_vtt_has_header_and_voice_spans(self, make_transcript):
        vtt = render_vtt(two_speaker(make_transcript))
        assert vtt.startswith("WEBVTT")
        assert "<v Speaker 1>" in vtt

    def test_long_turn_is_rechunked_into_readable_cues(self, make_transcript):
        # A 60s monologue must not become one unreadable cue.
        words = sequence(*[f"word{i}" for i in range(120)], dur=0.4, gap=0.1)
        d = Diarization(exclusive=[Segment(A, 0.0, 999.0)], overlapped=[])
        turns, spk = merge(words, d)
        cues = build_cues(make_transcript(turns, spk))
        assert len(cues) > 5
        assert all(len(c.text) <= MAX_CUE_CHARS for c in cues)
        assert all(c.end > c.start for c in cues)

    def test_cues_are_balanced_not_greedy(self, make_transcript):
        # Greedy filling leaves a runt final cue (one full cue + "review."),
        # which flashes on screen. Cues should be roughly even instead.
        words = sequence(*("this turn is deliberately long enough that it must "
                           "be broken across more than one subtitle cue before "
                           "it can be read comfortably on screen").split())
        d = Diarization(exclusive=[Segment(A, 0.0, 999.0)], overlapped=[])
        turns, spk = merge(words, d)
        cues = build_cues(make_transcript(turns, spk))
        assert len(cues) >= 2
        shortest, longest = min(len(c.text) for c in cues), max(len(c.text) for c in cues)
        assert shortest * 2 >= longest, f"unbalanced cues: {shortest} vs {longest}"

    def test_no_single_word_orphan_cue(self, make_transcript):
        words = sequence(*("one two three four five six seven eight nine ten "
                           "eleven twelve thirteen fourteen fifteen").split())
        d = Diarization(exclusive=[Segment(A, 0.0, 999.0)], overlapped=[])
        turns, spk = merge(words, d)
        cues = build_cues(make_transcript(turns, spk))
        if len(cues) > 1:
            assert len(cues[-1].text.split()) > 1, "final cue is a lone word"

    def test_cues_are_chronological(self, make_transcript):
        cues = build_cues(two_speaker(make_transcript))
        assert cues == sorted(cues, key=lambda c: c.start)

    def test_turn_without_word_detail_still_yields_a_cue(self, make_transcript):
        t = two_speaker(make_transcript)
        for turn in t.turns:
            turn.words = []
        cues = build_cues(t)
        assert len(cues) == len(t.turns)


class TestRenderAll:
    def test_writes_every_format(self, make_transcript, tmp_path):
        written = render_all(two_speaker(make_transcript), tmp_path)
        names = {p.name for p in written}
        assert names == {
            "transcript.json", "transcript.md",
            "transcript.txt", "transcript.srt", "transcript.vtt",
        }
        assert all(p.stat().st_size > 0 for p in written)

    def test_json_round_trips_exactly(self, make_transcript, tmp_path):
        original = two_speaker(make_transcript)
        render_all(original, tmp_path)
        reloaded = Transcript.from_json((tmp_path / "transcript.json").read_text())
        assert reloaded == original

    def test_word_detail_survives_the_round_trip(self, make_transcript, tmp_path):
        original = two_speaker(make_transcript)
        render_all(original, tmp_path)
        reloaded = Transcript.from_json((tmp_path / "transcript.json").read_text())
        assert reloaded.turns[0].words[0].text == original.turns[0].words[0].text
        assert reloaded.turns[0].words[0].start == original.turns[0].words[0].start


class TestSpeakersSidecar:
    def test_write_then_read_returns_no_names_initially(self, make_transcript, tmp_path):
        sidecar.write(two_speaker(make_transcript), tmp_path)
        assert sidecar.read(tmp_path) == {}

    def test_rename_round_trip_reaches_every_format(self, make_transcript, tmp_path):
        t = two_speaker(make_transcript)
        render_all(t, tmp_path)
        sidecar.write(t, tmp_path)

        # Simulate the user editing the file.
        path = tmp_path / sidecar.FILENAME
        path.write_text(path.read_text().replace("name: null", "name: Darin", 1))

        names = sidecar.read(tmp_path)
        assert names == {A: "Darin"}

        renamed = sidecar.apply(t, names)
        render_all(renamed, tmp_path)
        assert "Darin" in (tmp_path / "transcript.md").read_text()
        assert "Darin" in (tmp_path / "transcript.srt").read_text()
        assert "Darin" in (tmp_path / "transcript.txt").read_text()

    def test_apply_does_not_mutate_the_original(self, make_transcript):
        t = two_speaker(make_transcript)
        sidecar.apply(t, {A: "Darin"})
        assert t.speakers[0].name is None

    def test_rewrite_preserves_existing_names(self, make_transcript, tmp_path):
        t = two_speaker(make_transcript)
        sidecar.write(t, tmp_path)
        path = tmp_path / sidecar.FILENAME
        path.write_text(path.read_text().replace("name: null", "name: Darin", 1))

        sidecar.write(t, tmp_path)  # e.g. a re-run of the same file
        assert sidecar.read(tmp_path) == {A: "Darin"}

    def test_missing_and_malformed_files_are_tolerated(self, tmp_path):
        assert sidecar.read(tmp_path) == {}
        (tmp_path / sidecar.FILENAME).write_text("{[not yaml")
        assert sidecar.read(tmp_path) == {}
