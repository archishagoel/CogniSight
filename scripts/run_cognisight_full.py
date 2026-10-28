import cv2
from app.camera.camera import Camera
from app.database.database import SessionLocal, init_db
from app.services.cognisight_pipeline import CogniSightPipeline
from app.utils.logging_config import logger


def main():
    logger.info("Initializing FULL CogniSight Integrated System...")
    logger.info("Pipeline: Face Recognition -> Person Memory -> Context Engine -> Voice Assistant")

    init_db()
    db = SessionLocal()

    pipeline = CogniSightPipeline()
    camera = Camera()

    if not camera.start():
        logger.error("Could not open camera stream.")
        db.close()
        return

    logger.info("CogniSight full system active. Press 'q' or 'ESC' to exit.")

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                break

            # Process frame through full pipeline
            output_frame, context = pipeline.process_frame_full_pipeline(
                frame=frame, db=db, speak_voice=True
            )

            # Display live UI window
            cv2.imshow("CogniSight - Complete System", output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                logger.info("User requested system shutdown.")
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        pipeline.shutdown()
        db.close()


if __name__ == "__main__":
    main()
