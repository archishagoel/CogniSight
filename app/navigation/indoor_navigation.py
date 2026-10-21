
from collections import deque


# Example map for testing only.
# Replace these connections with your actual indoor layout.
INDOOR_MAP = {
    "entrance": {
        "name": "Entrance",
        "neighbors": ["corridor"],
    },
    "corridor": {
        "name": "Main corridor",
        "neighbors": ["entrance", "dining_room", "bedroom"],
    },
    "dining_room": {
        "name": "Dining room",
        "neighbors": ["corridor"],
    },
    "bedroom": {
        "name": "Bedroom",
        "neighbors": ["corridor"],
    },
}


def find_indoor_route(start: str, destination: str) -> dict:
    """Find the shortest route through the configured indoor map."""

    start = start.strip().lower()
    destination = destination.strip().lower()

    if start not in INDOOR_MAP:
        raise ValueError(f"Unknown starting location: {start}")

    if destination not in INDOOR_MAP:
        raise ValueError(f"Unknown destination: {destination}")

    queue = deque([[start]])
    visited = {start}

    while queue:
        path = queue.popleft()
        current = path[-1]

        if current == destination:
            locations = [INDOOR_MAP[node]["name"] for node in path]

            instructions = [
                f"Proceed to {name}."
                for name in locations[1:]
            ]

            if not instructions:
                instructions = ["You are already at your destination."]

            return {
                "start": INDOOR_MAP[start]["name"],
                "destination": INDOOR_MAP[destination]["name"],
                "route": locations,
                "instructions": instructions,
            }

        for neighbor in INDOOR_MAP[current]["neighbors"]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])

    raise ValueError("No walkable route connects these locations.")
