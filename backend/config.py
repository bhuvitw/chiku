from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "development"
    database_url: str = "postgresql+psycopg://chiku:chiku@localhost:5432/chiku"
    redis_url: str = "redis://localhost:6379/0"
    cors_origins: list[str] = ["http://localhost:3000"]

    # Replaced by real authentication in Phase 2; every study is attributed to
    # this account until then so the users FK is never nullable.
    dev_user_email: str = "dev@localhost"

    # --- uploads (System Design §4) ---
    storage_dir: Path = Path("data/uploads")
    max_upload_bytes: int = 64 * 1024 * 1024
    #: Phase 2 supports PNG and JPEG. DICOM is rejected with an explicit
    #: UNSUPPORTED_MODALITY rather than parsed on a best-effort basis
    #: (implementation-plan §2) — silently mishandling it is the worse failure.
    allowed_content_types: list[str] = ["image/png", "image/jpeg"]
    #: Below this on either axis the image cannot carry a wrist fracture's
    #: detail at the model's working resolution; reject as LOW_IMAGE_QUALITY.
    min_image_pixels: int = 128

    #: Phase 2 may run analysis inline (implementation-plan §2 allows it as a
    #: placeholder). The API contract does not change either way: a job row is
    #: created and polled identically, it is simply already finished on the
    #: first poll. Flip this on once Redis and the worker are running.
    async_jobs: bool = False

    # --- inference ---
    #: Defaults to the real model. A deployment that cannot find its
    #: checkpoint should fail loudly rather than quietly serve the stub's
    #: hash-derived numbers under a `stub:v0` version nobody reads.
    inference_backend: str = "torch"
    model_checkpoint: str = "experiments/outputs/resnet18-best.pt"
    #: Drives the stub into a chosen FR-04 state; ignored by real backends.
    stub_probability: float | None = None

    # --- operating point (PRD FR-04) ---
    #: The evaluation artifact that *chose* the threshold and abstention band,
    #: on validation, from the calibration curve. It is the source of truth:
    #: `backend.inference.operating_point` reads it rather than keeping a
    #: hand-copied set of the same numbers here, because a copy cannot notice
    #: when the model behind it changes.
    operating_point_path: str = "experiments/outputs/xray-eval.json"
    #: Explicit overrides. Unset by default — `None` means "take it from the
    #: artifact". All three must be set together or not at all; a threshold
    #: moved without its band is the likeliest way to serve a wrong one.
    decision_threshold: float | None = None
    abstention_low: float | None = None
    abstention_high: float | None = None


settings = Settings()
