from app.database.database import SessionLocal, init_db
from app.database.repository import PersonRepository
from app.face.embedding import FaceEmbedder
from app.utils.logging_config import logger
import numpy as np


def main():
    logger.info("Initializing Database for Person Enrollment Demo...")
    init_db()
    db = SessionLocal()

    try:
        person_name = "John"
        logger.info(f"Enrolling demo person '{person_name}' into database...")

        # Create Person
        person = PersonRepository.create_person(db, person_name)

        # Generate mock 512-D normalized vector for John
        john_raw_vec = np.random.randn(512).astype(np.float32)
        john_vector = FaceEmbedder.normalize_embedding(john_raw_vec)

        # Store face embedding in DB
        emb_record = PersonRepository.add_face_embedding(db, person.id, john_vector)

        logger.info(f"Enrolled Person ID: {person.id}, Name: '{person.name}'")
        logger.info(f"Stored Face Embedding ID: {emb_record.id}")

        # Retrieve all enrolled embeddings
        enrolled = PersonRepository.get_all_enrolled_embeddings(db)
        logger.info(f"Total enrolled persons in DB: {len(enrolled)}")
        for p_id, p_name, vec in enrolled:
            logger.info(f"Enrolled DB Entry -> ID: {p_id}, Name: '{p_name}', Vector shape: {vec.shape}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
