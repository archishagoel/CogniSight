from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env file."""

    APP_NAME: str = "CogniSight"
    DEBUG: bool = True

    # Camera Settings
    CAMERA_INDEX: int = 0
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480

    # Face Detection & Recognition Parameters
    DETECTION_THRESHOLD: float = 0.50
    FACE_RECOGNITION_THRESHOLD: float = 0.50
    RECOGNITION_INTERVAL: int = 3  # Perform recognition every N frames

    # Voice Assistant & Periodic Reminder Settings
    VOICE_ENABLED: bool = True
    SPEECH_COOLDOWN_SECONDS: float = 5.0  # Min time between different announcements
    REMINDER_INTERVAL_SECONDS: float = 30.0  # Periodic reminder interval (every 30 seconds)

    # Model Configuration
    INSIGHTFACE_MODEL_NAME: str = "buffalo_s"

    # Database
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'data' / 'cognisight.db'}"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
