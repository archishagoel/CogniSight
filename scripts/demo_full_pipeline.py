import cv2
import numpy as np
from app.database.database import SessionLocal, init_db
from app.database.repository import PersonRepository
from app.face.embedding import FaceEmbedder
from app.face.schemas import BoundingBox, RecognitionResult
from app.memory.memory_service import MemoryService
from app.services.cognisight_pipeline import CogniSightPipeline
from app.utils.logging_config import logger


def main():
    print("\n" + "=" * 60)
    print("      COGNISIGHT FULL SYSTEM DEMONSTRATION & WALKTHROUGH")
    print("=" * 60)

    # 1. Initialize Database
    logger.info("Step 1: Initializing SQLite Database & Schema...")
    init_db()
    db = SessionLocal()

    memory_service = MemoryService()
    pipeline = CogniSightPipeline(memory_service=memory_service)

    try:
        # 2. Enroll Person 1: John (Brother / Caregiver)
        logger.info("\nStep 2: Enrolling Person #1 ('John')...")
        john = PersonRepository.create_person(db, "John")
        memory_service.update_memory_profile(
            db=db,
            person_id=john.id,
            relationship="Brother / Caregiver",
            notes="Loves tea with 2 sugars.",
        )
        john_vector = FaceEmbedder.normalize_embedding(np.random.randn(512).astype(np.float32))
        PersonRepository.add_face_embedding(db, john.id, john_vector)

        # 3. Enroll Person 2: Sarah (Doctor / Cardiologist)
        logger.info("\nStep 3: Enrolling Person #2 ('Sarah')...")
        sarah = PersonRepository.create_person(db, "Sarah")
        memory_service.update_memory_profile(
            db=db,
            person_id=sarah.id,
            relationship="Doctor / Cardiologist",
            notes="Prescribed daily morning vitamins.",
        )
        sarah_vector = FaceEmbedder.normalize_embedding(np.random.randn(512).astype(np.float32))
        PersonRepository.add_face_embedding(db, sarah.id, sarah_vector)

        # 4. Simulate Live Perception Frame 1: Recognized Person "John"
        logger.info("\nStep 4: Simulating Camera Feed containing 'John'...")
        # Create dummy image canvas
        frame_john = np.ones((480, 640, 3), dtype=np.uint8) * 40
        query_john = FaceEmbedder.normalize_embedding(john_vector + np.random.randn(512) * 0.02)

        candidates = PersonRepository.get_all_enrolled_embeddings(db)
        match_john = pipeline.recognizer.matcher.match_face(
            query_john, candidates, bbox=BoundingBox(x1=100, y1=100, x2=280, y2=320)
        )

        context_john = pipeline.context_engine.generate_context(db, [match_john])

        print("\n" + "-" * 50)
        print("PERCEPTION & CONTEXT RESULT [SCENARIO 1: KNOWN PERSON]")
        print("-" * 50)
        print(f"Recognized:            {context_john.recognized}")
        print(f"Person Name:           {context_john.person_name}")
        print(f"Relationship:          {context_john.relationship}")
        print(f"Headline Banner:       {context_john.headline}")
        print(f"Subtext Info:          {context_john.subtext}")
        print(f"Voice Announcement:    '{context_john.spoken_announcement}'")
        print("-" * 50)

        # Draw UI overlay
        annotated_john = pipeline.recognizer.draw_recognition(frame_john, [match_john])
        final_ui_john = pipeline._render_context_ui(annotated_john, context_john)
        cv2.imwrite("data/cognisight_demo_john.jpg", final_ui_john)
        logger.info("Saved visual demo artifact: data/cognisight_demo_john.jpg")

        # 5. Simulate Live Perception Frame 2: Unknown Person
        logger.info("\nStep 5: Simulating Camera Feed containing an 'Unknown Person'...")
        frame_unknown = np.ones((480, 640, 3), dtype=np.uint8) * 40
        query_unknown = FaceEmbedder.normalize_embedding(np.random.randn(512).astype(np.float32))

        match_unknown = pipeline.recognizer.matcher.match_face(
            query_unknown, candidates, bbox=BoundingBox(x1=350, y1=100, x2=520, y2=320)
        )
        context_unknown = pipeline.context_engine.generate_context(db, [match_unknown])

        print("\n" + "-" * 50)
        print("PERCEPTION & CONTEXT RESULT [SCENARIO 2: UNKNOWN PERSON]")
        print("-" * 50)
        print(f"Recognized:            {context_unknown.recognized}")
        print(f"Person Name:           {context_unknown.person_name}")
        print(f"Headline Banner:       {context_unknown.headline}")
        print(f"Subtext Info:          {context_unknown.subtext}")
        print(f"Voice Announcement:    '{context_unknown.spoken_announcement}'")
        print("-" * 50)

        # Draw UI overlay
        annotated_unknown = pipeline.recognizer.draw_recognition(frame_unknown, [match_unknown])
        final_ui_unknown = pipeline._render_context_ui(annotated_unknown, context_unknown)
        cv2.imwrite("data/cognisight_demo_unknown.jpg", final_ui_unknown)
        logger.info("Saved visual demo artifact: data/cognisight_demo_unknown.jpg")

        # 6. Verify Interaction History Logs
        logger.info("\nStep 6: Checking CogniSight Interaction History Logs...")
        logs = memory_service.get_recent_interactions(db, limit=5)
        print("\nRECENT INTERACTION LOGS IN DATABASE:")
        for log in logs:
            print(
                f"  Log #{log.id} | Timestamp: {log.timestamp.strftime('%H:%M:%S')} | "
                f"Person: {log.person_name:<15} | Recognized: {str(log.recognized):<5} | "
                f"Similarity: {log.similarity:.4f}"
            )

        print("\n" + "=" * 60)
        print("             DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60 + "\n")
    finally:
        pipeline.shutdown()
        db.close()


if __name__ == "__main__":
    main()
