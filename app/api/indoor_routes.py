
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.navigation.indoor_navigation import (
    INDOOR_MAP,
    find_indoor_route,
)
from app.navigation.landmark_detector import MARKER_LOCATIONS

router = APIRouter()


class IndoorRouteRequest(BaseModel):
    start: str
    destination: str


class MarkerRouteRequest(BaseModel):
    marker_id: int
    destination: str


@router.get("/indoor/map")
def get_indoor_map():
    return {
        "locations": {
            key: value["name"]
            for key, value in INDOOR_MAP.items()
        },
        "marker_locations": {
            marker_id: location
            for marker_id, location in MARKER_LOCATIONS.items()
        },
        "note": "Sample development map, not a real building map.",
    }


@router.post("/indoor/route")
def calculate_indoor_route(payload: IndoorRouteRequest):
    try:
        return find_indoor_route(
            payload.start, payload.destination
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=str(exc)
        ) from exc


@router.post("/indoor/route-from-marker")
def calculate_route_from_marker(payload: MarkerRouteRequest):
    start = MARKER_LOCATIONS.get(payload.marker_id)

    if start is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown indoor marker ID: {payload.marker_id}",
        )

    try:
        route = find_indoor_route(start, payload.destination)
        route["start_marker_id"] = payload.marker_id
        route["location_source"] = "ArUco landmark mapping"
        return route
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=str(exc)
        ) from exc
