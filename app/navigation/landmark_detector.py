
import cv2


# Temporary sample map: marker ID -> indoor location.
MARKER_LOCATIONS = {
    10: "entrance",
    20: "corridor",
    30: "dining_room",
    40: "bedroom",
}


class IndoorLandmarkDetector:
    """Detect printed ArUco landmarks in an OpenCV camera frame."""

    def __init__(self):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "OpenCV ArUco support is unavailable."
            )

        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        if hasattr(cv2.aruco, "DetectorParameters"):
            parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(
                self.dictionary, parameters
            )
            self._legacy = False
        else:
            self.parameters = cv2.aruco.DetectorParameters_create()
            self._legacy = True

    def detect(self, frame):
        """Return the recognized marker IDs and mapped locations."""
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._legacy:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.parameters
            )
        else:
            corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None:
            return []

        results = []

        for marker_id in ids.flatten().tolist():
            if marker_id in MARKER_LOCATIONS:
                results.append({
                    "marker_id": marker_id,
                    "location": MARKER_LOCATIONS[marker_id],
                })

        return results
