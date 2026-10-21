import numpy as np
from app.face.embedding import FaceEmbedder
from app.utils.logging_config import logger


def main():
    logger.info("Initializing Face Embedding Generator...")
    embedder = FaceEmbedder()

    # Create dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    logger.info("Testing embedding generator on empty frame...")

    emb = embedder.generate_embedding(dummy_frame)
    if emb is None:
        logger.info("Correctly returned None for frame with 0 faces.")

    # Generate synthetic 512-D vector to demonstrate L2 normalization concept
    sample_vector = np.random.randn(512).astype(np.float32)
    normalized_vector = embedder.normalize_embedding(sample_vector)

    logger.info(f"Sample Embedding Dimension: {normalized_vector.shape}")
    logger.info(f"L2 Norm (Magnitude): {np.linalg.norm(normalized_vector):.6f}")
    logger.info(f"First 5 vector values: {normalized_vector[:5].tolist()}")


if __name__ == "__main__":
    main()
