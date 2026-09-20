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
