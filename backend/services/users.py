from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.config import settings
from backend.models import User, UserRole


def get_or_create_dev_user(db: Session) -> User:
    """Stand-in for authentication until Phase 2 adds it.

    Studies are always attributed to a real user row so the FK and the audit
    trail are correct from the start; only the identification is fake.
    """
    user = db.scalar(select(User).where(User.email == settings.dev_user_email))
    if user is None:
        user = User(email=settings.dev_user_email, role=UserRole.STUDENT)
        db.add(user)
        db.flush()
    return user
