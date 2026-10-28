import cv2
from app.camera.camera import Camera
from app.database.database import SessionLocal, init_db
from app.face.recognizer import FaceRecognizer
from app.utils.logging_config import logger


def main():
    logger.info("Initializing CogniSight Real-Time Face Recognition Engine...")
    init_db()
    db = SessionLocal()

    recognizer = FaceRecognizer()
    camera = Camera()

    if not camera.start():
        logger.error("Failed to start camera feed.")
        db.close()
        return

    logger.info("Camera live feed active. Press 'q' or 'ESC' to quit.")

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                break

            # 1. Run recognition pipeline
            results = recognizer.process_frame(frame, db)

            # 2. Draw styled boxes & labels
            annotated_frame = recognizer.draw_recognition(frame, results)

            # 3. Display live stream
            cv2.imshow("CogniSight - Real-Time Recognition (Step 13-16)", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                logger.info("User requested exit.")
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        db.close()


if __name__ == "__main__":
    main()
