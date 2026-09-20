from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User
from backend.services.users import get_or_create_dev_user

DbSession = Annotated[Session, Depends(get_db)]


def get_current_user(db: DbSession) -> User:
    """Phase 0/1 stand-in. Phase 2 replaces the body, not the signature, so
    call sites already depend on an authenticated user."""
    user = get_or_create_dev_user(db)
    db.commit()
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
