
import cv2

from app.navigation.landmark_detector import IndoorLandmarkDetector
from app.navigation.indoor_navigation import find_indoor_route
from app.voice.voice_assistant import VoiceAssistant

VALID_DESTINATIONS = {
    "entrance",
    "corridor",
    "dining_room",
    "bedroom",
}

destination = input(
    "Enter destination (entrance, corridor, dining_room, bedroom): "
).strip().lower().replace(" ", "_")

if destination not in VALID_DESTINATIONS:
    raise SystemExit("Invalid destination. Please run the script again.")

detector = IndoorLandmarkDetector()
voice = VoiceAssistant()
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    voice.stop()
    raise SystemExit(
        "Could not open webcam. Close other camera applications and retry."
    )

last_marker = None

print("Show a printed location marker to the webcam.")
print("Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read a camera frame.")
            break

        results = detector.detect(frame)

        if results:
            result = results[0]
            marker_id = result["marker_id"]
            start = result["location"]

            cv2.putText(
                frame,
                f"Detected: {start}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if marker_id != last_marker:
                last_marker = marker_id

                print(f"\nDetected marker {marker_id}: {start}")

                try:
                    route = find_indoor_route(start, destination)
                    print("Route:", " -> ".join(route["route"]))

                    voice.speak(
                        f"Starting location: {route['start']}. "
                        f"Destination: {route['destination']}."
                    )

                    for instruction in route["instructions"]:
                        print(instruction)
                        voice.speak(instruction)

                    print(
                        "\nRoute announcement finished. "
                        "Show another marker to test a new route."
                    )

                except ValueError as exc:
                    print(f"Route error: {exc}")
                    voice.speak("Sorry, no route is available.")

        else:
            last_marker = None

        cv2.imshow("AuraGlasses - Live Navigation Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    voice.stop()
