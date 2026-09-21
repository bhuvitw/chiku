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
    inference_backend: str = "stub"
    model_checkpoint: str = "experiments/outputs/resnet18-best.pt"
    #: Drives the stub into a chosen FR-04 state; ignored by real backends.
    stub_probability: float | None = None

    # --- operating point (PRD FR-04) ---
    # PROVISIONAL. These are placeholders that keep the abstention path
    # exercisable while the model does not exist. They are NOT a calibrated
    # operating point and must be replaced wholesale by the `selection` block
    # that `ml/evaluation/run_xray.py` writes to experiments/outputs/
    # xray-eval.json — which picks them from the validation calibration curve,
    # as PRD §3 requires and a guessed round number does not.
    decision_threshold: float = 0.5
    abstention_low: float = 0.40
    abstention_high: float = 0.60

    @property
    def operating_point_is_calibrated(self) -> bool:
        return self.inference_backend != "stub"


settings = Settings()
