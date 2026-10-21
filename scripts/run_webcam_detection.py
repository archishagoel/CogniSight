import cv2
from app.camera.camera import Camera
from app.face.detector import FaceDetector
from app.utils.logging_config import logger


def main():
    logger.info("Starting CogniSight Face Detection (Webcam Mode)...")
    logger.info("Press 'q' or 'ESC' in the preview window to exit.")

    detector = FaceDetector()
    camera = Camera()

    if not camera.start():
        logger.error("Failed to start camera feed. Please check webcam connection.")
        return

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                logger.warning("Frame grab failed, exiting...")
                break

            # 1. Detect faces in current frame
            faces = detector.detect_faces(frame)

            # 2. Draw bounding boxes & confidence scores
            output_frame = detector.draw_detections(frame, faces)

            # 3. Display live feed window
            cv2.imshow("CogniSight - Face Detection (Step 3)", output_frame)

            # Press 'q' or ESC (27) to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                logger.info("User requested exit.")
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
