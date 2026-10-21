
from pathlib import Path
import cv2

output_dir = Path("markers")
output_dir.mkdir(exist_ok=True)

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

marker_ids = {
    10: "entrance",
    20: "corridor",
    30: "dining_room",
    40: "bedroom",
}

for marker_id, location in marker_ids.items():
    if hasattr(cv2.aruco, "generateImageMarker"):
        image = cv2.aruco.generateImageMarker(
            dictionary, marker_id, 400
        )
    else:
        image = cv2.aruco.drawMarker(
            dictionary, marker_id, 400
        )

    # Add a white border to help preserve the marker edges when printed.
    image = cv2.copyMakeBorder(
        image, 50, 50, 50, 50,
        cv2.BORDER_CONSTANT, value=255
    )

    filename = output_dir / f"{marker_id}_{location}.png"
    if not cv2.imwrite(str(filename), image):
        raise RuntimeError(f"Could not save {filename}")

    print(f"Created: {filename}")

print("All four markers generated.")
