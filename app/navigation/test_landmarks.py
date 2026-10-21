
import cv2

from app.navigation.landmark_detector import IndoorLandmarkDetector

camera = cv2.VideoCapture(0)
detector = IndoorLandmarkDetector()

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit(1)

print("Show one printed ArUco marker to the webcam.")
print("Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read a camera frame.")
            break

        results = detector.detect(frame)

        for result in results:
            label = (
                f"Marker {result['marker_id']}: "
                f"{result['location']}"
            )
            cv2.putText(
                frame,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            print(label)

        cv2.imshow("AuraGlasses - Landmark Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
