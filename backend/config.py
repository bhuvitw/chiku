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


settings = Settings()
