"""Configuration.

Layered, lowest priority first: defaults -> ``scribe.toml`` -> environment
(``SCRIBE_*``) -> explicit CLI flags. Every merge threshold is configurable
because good values depend on the audio, and hard-coding them would make the
speaker-separation quality untunable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

AsrBackend = Literal["mlx", "onnx"]
Device = Literal["auto", "cpu", "mps", "cuda"]

CONFIG_FILE = Path("scribe.toml")


class MergeSettings(BaseModel):
    """Thresholds for word-to-speaker attribution. See `scribe.merge`."""

    nearest_window_s: float = 0.25
    """A word overlapping no segment adopts a speaker within this distance."""

    min_segment_s: float = 0.12
    """Discard diarization segments shorter than this before attributing words.

    During crosstalk, pyannote emits segments as short as 0.02s while it
    flickers between speakers. No word is 20ms long, so such a segment cannot
    meaningfully own one — but it still wins the overlap vote for whatever word
    it touches, which is a direct cause of mid-sentence speaker flips. Words
    left uncovered fall through to the nearest-segment and carry-forward rules,
    which is the better guess.
    """

    smooth_max_words: int = 5
    """Runs of at most this many words can be absorbed into their neighbours.

    Duration is the meaningful test; this is a generous safety cap. Rapid
    conversational speech fits 3-4 words into 0.4s, so a tight word limit
    silently defeats the duration rule -- observed on real audio, where a
    3-word 0.32s fragment escaped smoothing and split one question across
    three turns.
    """

    smooth_max_duration_s: float = 0.4
    """...but only if the run is also shorter than this. Both must hold."""

    turn_gap_s: float = 1.5
    """Silence longer than this splits one speaker's words into separate turns."""

    turn_max_duration_s: float = 60.0
    """Past this length, a turn may split at sentence-final punctuation."""


class AsrSettings(BaseModel):
    backend: AsrBackend = "mlx"
    """``mlx`` is Metal-accelerated and macOS-only; ``onnx`` is portable CPU."""

    mlx_model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    onnx_model_dir: Path | None = None
    """Directory of sherpa-onnx Parakeet files. Downloaded on first use if unset."""


class DiarizeSettings(BaseModel):
    model: str = "pyannote/speaker-diarization-community-1"
    device: Device = "auto"
    num_speakers: int | None = None
    """Pin the speaker count when known — materially improves the split."""

    min_speakers: int | None = None
    max_speakers: int | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SCRIBE_",
        env_nested_delimiter="__",
        toml_file=CONFIG_FILE,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    intake_dir: Path | None = None
    work_dir: Path | None = None
    out_dir: Path | None = None

    hf_token: str | None = None
    """Needed once to download the gated pyannote weights.

    Usually left unset: `hf auth login` stores a token that huggingface_hub
    finds on its own. Set it explicitly for containers, where injecting an env
    var from a Secret beats mounting a cache."""

    asr: AsrSettings = Field(default_factory=AsrSettings)
    diarize: DiarizeSettings = Field(default_factory=DiarizeSettings)
    merge: MergeSettings = Field(default_factory=MergeSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    # Subdirectories default to living under data_dir but can each be overridden,
    # which is what lets k8s point them at different volumes.
    @property
    def intake(self) -> Path:
        return self.intake_dir or self.data_dir / "intake"

    @property
    def work(self) -> Path:
        return self.work_dir or self.data_dir / "work"

    @property
    def out(self) -> Path:
        return self.out_dir or self.data_dir / "out"

    def ensure_dirs(self) -> None:
        for d in (self.intake, self.work, self.out):
            d.mkdir(parents=True, exist_ok=True)

    def resolved_hf_token(self) -> str | None:
        """Find a token from any of the places one might legitimately live.

        Checked in order: explicit config, the env vars the huggingface libs
        honour, then the token `hf auth login` writes to
        ``~/.cache/huggingface/token``. The last one matters — passing None to
        `from_pretrained` would still work via that stored token, so without
        this check `scribe doctor` would report "missing" for a setup that is
        in fact fine.
        """
        import os

        if token := (
            self.hf_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        ):
            return token

        try:
            from huggingface_hub import get_token

            return get_token()
        except Exception:  # noqa: BLE001 - absence is not an error
            return None


def load_settings(**overrides: object) -> Settings:
    """Load settings, dropping None overrides so CLI flags left unset don't win."""
    return Settings(**{k: v for k, v in overrides.items() if v is not None})
