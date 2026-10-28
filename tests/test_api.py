import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.database import Base, get_db
from app.main import app

# Setup in-memory SQLite with StaticPool so all requests share table schema
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
Base.metadata.create_all(bind=engine)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def test_health_endpoint():
    """Test GET /api/v1/health returns 200 OK."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "CogniSight" in data["service"]


def test_person_crud_endpoints():
    """Test full CRUD lifecycle for Person management endpoints."""
    # 1. Create Person
    create_resp = client.post("/api/v1/persons", json={"name": "Sarah"})
    assert create_resp.status_code == 201
    person_data = create_resp.json()
    person_id = person_data["id"]
    assert person_data["name"] == "Sarah"

    # 2. List Persons
    list_resp = client.get("/api/v1/persons")
    assert list_resp.status_code == 200
    persons_list = list_resp.json()
    assert len(persons_list) >= 1

    # 3. Get Person Detail
    get_resp = client.get(f"/api/v1/persons/{person_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "Sarah"

    # 4. Delete Person
    del_resp = client.delete(f"/api/v1/persons/{person_id}")
    assert del_resp.status_code == 200

    # 5. Verify 404 on deleted person
    get_404 = client.get(f"/api/v1/persons/{person_id}")
    assert get_404.status_code == 404


def test_recognition_predict_endpoint():
    """Test POST /api/v1/recognition/predict on uploaded JPEG image."""
    # Create dummy black image frame
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = img_encoded.tobytes()

    response = client.post(
        "/api/v1/recognition/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    # Zero faces in black box image frame
    assert len(results) == 0
