import re
from typing import Tuple


class VoiceTranscriptParser:
    """Parses spoken audio transcripts to extract Person Name and Relationship.

    Examples:
        "My name is Alex" -> ("Alex", "Known Contact")
        "I am Rahul and I am a caregiver" -> ("Rahul", "Caregiver")
        "Suresh friend" -> ("Suresh", "Friend")
        "Dr Sarah doctor" -> ("Sarah", "Doctor")
    """

    RELATIONSHIP_KEYWORDS = {
        "caregiver": "Caregiver",
        "nurse": "Nurse",
        "doctor": "Doctor",
        "brother": "Brother",
        "sister": "Sister",
        "mother": "Mother",
        "father": "Father",
        "friend": "Friend",
        "neighbor": "Neighbor",
        "physician": "Physician",
        "son": "Son",
        "daughter": "Daughter",
        "husband": "Husband",
        "wife": "Wife",
    }

    @classmethod
    def parse_name_and_relationship(cls, transcript: str) -> Tuple[str, str]:
        """Parses a transcript string and returns (name, relationship)."""
        if not transcript or not transcript.strip():
            return "New Contact", "Known Contact"

        clean = transcript.strip().lower()

        # Extract relationship keyword if present
        found_rel = "Known Contact"
        for word, rel_title in cls.RELATIONSHIP_KEYWORDS.items():
            if word in clean:
                found_rel = rel_title
                break

        # Common phrase regex patterns
        name_patterns = [
            r"my name is ([a-zA-Z]+)",
            r"i am ([a-zA-Z]+)",
            r"call me ([a-zA-Z]+)",
            r"this is ([a-zA-Z]+)",
            r"^([a-zA-Z]+)$",
        ]

        extracted_name = None
        for pattern in name_patterns:
            match = re.search(pattern, clean)
            if match:
                extracted_name = match.group(1).capitalize()
                break

        if not extracted_name:
            # Fallback: pick first word that isn't a stopword
            stopwords = {
                "my",
                "name",
                "is",
                "i",
                "am",
                "a",
                "the",
                "and",
                "hello",
                "hi",
                "hey",
            }
            words = [w.capitalize() for w in clean.split() if w not in stopwords]
            extracted_name = words[0] if words else "New Contact"

        return extracted_name, found_rel
