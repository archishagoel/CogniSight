import cv2
import numpy as np
from app.face.detector import FaceDetector
from app.utils.logging_config import logger


def main():
    logger.info("Initializing Face Detector Demo...")
    detector = FaceDetector()

    # Create a synthetic canvas image with a drawn face representation for offline verification
    img = np.ones((480, 640, 3), dtype=np.uint8) * 220  # Light gray background

    # Draw head outline
    cv2.ellipse(img, (320, 240), (100, 140), 0, 0, 360, (180, 150, 120), -1)
    # Draw eyes
    cv2.circle(img, (280, 200), 12, (255, 255, 255), -1)
    cv2.circle(img, (360, 200), 12, (255, 255, 255), -1)
    cv2.circle(img, (280, 200), 5, (0, 0, 0), -1)
    cv2.circle(img, (360, 200), 5, (0, 0, 0), -1)
    # Draw nose
    cv2.line(img, (320, 220), (320, 250), (100, 80, 50), 3)
    # Draw mouth
    cv2.ellipse(img, (320, 280), (35, 15), 0, 0, 180, (50, 50, 200), 3)

    logger.info("Running detection on test image frame...")
    detected_faces = detector.detect_faces(img)

    logger.info(f"Detection completed. Total faces detected: {len(detected_faces)}")
    for i, face in enumerate(detected_faces, 1):
        box = face.bbox
        logger.info(
            f"Face #{i}: BBox=[x1={box.x1}, y1={box.y1}, x2={box.x2}, y2={box.y2}], "
            f"Confidence={face.confidence * 100:.2f}%"
        )

    # Annotate frame
    annotated = detector.draw_detections(img, detected_faces)
    output_path = "data/detector_demo_output.jpg"
    cv2.imwrite(output_path, annotated)
    logger.info(f"Saved annotated demo output image to {output_path}")


if __name__ == "__main__":
    main()
