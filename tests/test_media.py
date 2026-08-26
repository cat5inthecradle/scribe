"""Media detection and normalization.

The intake folder is a hostile input: it collects OS junk, in-flight downloads,
and containers no allowlist anticipated. A file wrongly skipped here is
indistinguishable from a broken pipeline to whoever dropped it in.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scribe import media


def make_audio(path: Path, seconds: float = 2.0) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-t", str(seconds),
         "-i", "anullsrc=r=44100:cl=stereo", str(path)],
        check=True,
    )
    return path


class TestIsMediaFile:
    @pytest.mark.parametrize("name", ["a.mp3", "a.M4A", "a.wav", "a.mp4", "a.qta"])
    def test_known_extensions_accepted_without_probing(self, tmp_path, name):
        # Note: no file is created — a known extension must not need to exist.
        assert media.is_media_file(tmp_path / name)

    @pytest.mark.parametrize(
        "name",
        [".DS_Store", ".hidden.mp3", "Thumbs.db",
         "movie.mp4.partial", "song.mp3.crdownload", "x.tmp"],
    )
    def test_junk_and_inflight_writes_rejected(self, tmp_path, name):
        assert not media.is_media_file(tmp_path / name)

    def test_unknown_extension_with_audio_is_accepted(self, tmp_path):
        # The real-world case: Apple's screen capture writes a QuickTime
        # container named `.qta`. Write a real container, then give it an
        # extension no allowlist would carry — only probing can classify it.
        real = make_audio(tmp_path / "real.wav")
        path = tmp_path / "recording.weirdext"
        path.write_bytes(real.read_bytes())
        assert media.is_media_file(path)

    def test_unknown_extension_without_audio_is_rejected(self, tmp_path):
        path = tmp_path / "notes.xyz"
        path.write_text("just text")
        assert not media.is_media_file(path)


class TestInspect:
    def test_reports_duration_size_and_hash(self, tmp_path):
        path = make_audio(tmp_path / "clip.wav", seconds=3.0)
        info = media.inspect(path)
        assert info.duration_s == pytest.approx(3.0, abs=0.2)
        assert info.bytes == path.stat().st_size
        assert len(info.sha256) == 64

    def test_hash_is_content_based_not_name_based(self, tmp_path):
        a = make_audio(tmp_path / "a.wav")
        b = tmp_path / "b.wav"
        b.write_bytes(a.read_bytes())
        assert media.inspect(a).sha256 == media.inspect(b).sha256

    def test_missing_file_is_a_clear_error(self, tmp_path):
        with pytest.raises(media.MediaError):
            media.inspect(tmp_path / "nope.wav")


class TestNormalize:
    def test_produces_16k_mono_pcm(self, tmp_path):
        src = make_audio(tmp_path / "in.wav")  # 44.1k stereo
        out = media.normalize(src, tmp_path / "out.wav")
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=sample_rate,channels,codec_name", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert probe == "pcm_s16le,16000,1"

    def test_selects_the_first_audio_stream_of_a_video(self, tmp_path):
        # Real recordings carry video and sometimes a second exotic audio track;
        # normalization must land on the ordinary one.
        src = tmp_path / "call.mkv"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y",
             "-f", "lavfi", "-t", "2", "-i", "testsrc=size=160x120:rate=10",
             "-f", "lavfi", "-t", "2", "-i", "anullsrc=r=48000:cl=stereo",
             "-shortest", str(src)],
            check=True,
        )
        out = media.normalize(src, tmp_path / "out.wav")
        assert out.is_file() and out.stat().st_size > 0

    def test_no_partial_file_is_left_behind(self, tmp_path):
        src = make_audio(tmp_path / "in.wav")
        media.normalize(src, tmp_path / "out.wav")
        assert not list(tmp_path.glob("*.partial"))

    def test_failure_leaves_no_output_and_raises(self, tmp_path):
        bogus = tmp_path / "bogus.wav"
        bogus.write_text("not audio at all")
        with pytest.raises(media.MediaError):
            media.normalize(bogus, tmp_path / "out.wav")
        assert not (tmp_path / "out.wav").exists()
        assert not list(tmp_path.glob("*.partial"))
