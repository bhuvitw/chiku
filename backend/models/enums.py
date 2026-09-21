import enum


class UserRole(enum.StrEnum):
    STUDENT = "student"
    CLINICIAN = "clinician"
    ADMIN = "admin"


class Modality(enum.StrEnum):
    XRAY = "xray"
    MRI = "mri"


class StudyStatus(enum.StrEnum):
    """Study lifecycle, per System Design §18.

    FAILED and UNSUPPORTED are distinct terminal states on purpose: FAILED means
    a retry may help, UNSUPPORTED means it will not.
    """

    UPLOADED = "UPLOADED"
    VALIDATING = "VALIDATING"
    PREPROCESSING = "PREPROCESSING"
    ANALYZING = "ANALYZING"
    GENERATING_3D = "GENERATING_3D"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    UNSUPPORTED = "UNSUPPORTED"


class JobType(enum.StrEnum):
    ANALYZE = "analyze"
    GENERATE_3D = "generate_3d"


class JobStatus(enum.StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class PredictionLabel(enum.StrEnum):
    """PRD FR-04 outcomes. `UNABLE_TO_ASSESS` is a first-class result, not an
    error: the product abstains out loud rather than dressing up a coin flip."""

    POSSIBLE_FRACTURE = "possible_fracture"
    NO_FRACTURE = "no_fracture"
    UNABLE_TO_ASSESS = "unable_to_assess"
