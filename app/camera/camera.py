import threading
from typing import Generator, Optional, Tuple
import cv2
import numpy as np

from app.config.settings import settings
from app.utils.logging_config import logger


class Camera:
    """Manages OpenCV camera video capture stream."""

    _latest_frame: Optional[np.ndarray] = None
    _frame_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        camera_index: int = settings.CAMERA_INDEX,
        width: int = settings.CAMERA_WIDTH,
        height: int = settings.CAMERA_HEIGHT,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

    @classmethod
    def get_latest_frame(cls) -> Optional[np.ndarray]:
        """Returns a thread-safe copy of the latest captured webcam frame."""
        with cls._frame_lock:
            if cls._latest_frame is not None and cls._latest_frame.size > 0:
                return cls._latest_frame.copy()
        return None

    def start(self) -> bool:
        """Opens video capture device."""
        logger.info(f"Opening camera index {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not self.cap or not self.cap.isOpened():
            # Try fallback default backend
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            logger.error(f"Could not open camera device at index {self.camera_index}.")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        logger.info(
            f"Camera successfully started ({self.width}x{self.height})."
        )
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads a single frame from webcam stream."""
        if not self.cap or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("Failed to grab frame from camera.")
            return False, None

        with Camera._frame_lock:
            Camera._latest_frame = frame.copy()

        return True, frame

    def release(self):
        """Releases the camera device."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("Camera device released.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

