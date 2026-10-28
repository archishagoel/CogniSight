import numpy as np
from app.database.database import SessionLocal, init_db
from app.database.repository import PersonRepository
from app.face.embedding import FaceEmbedder
from app.face.matcher import FaceMatcher
from app.utils.logging_config import logger


def main():
    logger.info("Initializing Database & Face Matcher Demo...")
    init_db()
    db = SessionLocal()

    try:
        # Load all enrolled database candidates
        candidates = PersonRepository.get_all_enrolled_embeddings(db)
        logger.info(f"Loaded {len(candidates)} enrolled person candidate(s) from SQLite database.")

        matcher = FaceMatcher(threshold=0.50)

        if candidates:
            # Case 1: Test matching an enrolled person ("John") with query vector matching stored embedding
            john_id, john_name, john_vec = candidates[0]
            logger.info(f"Testing match query for enrolled person '{john_name}'...")

            # Query vector with slight variation (~0.98 similarity)
            query_known = FaceEmbedder.normalize_embedding(john_vec + np.random.randn(512) * 0.02)
            result_known = matcher.match_face(query_known, candidates)

            logger.info(
                f"Result -> Recognized: {result_known.recognized} | Name: '{result_known.name}' | "
                f"Similarity: {result_known.similarity:.4f}"
            )

        # Case 2: Test matching an unknown person (random vector)
        logger.info("Testing match query for an unknown person...")
        query_unknown = FaceEmbedder.normalize_embedding(np.random.randn(512).astype(np.float32))
        result_unknown = matcher.match_face(query_unknown, candidates)

        logger.info(
            f"Result -> Recognized: {result_unknown.recognized} | Name: '{result_unknown.name}' | "
            f"Similarity: {result_unknown.similarity:.4f}"
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
