import json
import os
from typing import Dict, List, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TicketSimilarityEngine:
    """TF-IDF + Cosine Similarity matching engine for support ticket resolutions."""

    def __init__(self, confidence_threshold: float = 0.30):
        self.threshold = confidence_threshold
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),  # Unigrams and bigrams
            sublinear_tf=True,    # Apply 1 + log(tf) scaling to suppress high TF noise
        )
        self.historical_tickets: List[Dict[str, Any]] = []
        self.tfidf_matrix = None

    def _prepare_historical_document(self, ticket: Dict[str, Any]) -> str:
        """Combine subject, description, category, and tags into a unified searchable text block."""
        subject = ticket.get("subject", "")
        description = ticket.get("description", "")
        category = ticket.get("category", "")
        tags = " ".join(ticket.get("tags", []))
        return f"{subject} {description} {category} {tags}"

    def fit(self, historical_tickets: List[Dict[str, Any]]) -> None:
        """Fit the TF-IDF vectorizer on the historical ticket corpus."""
        self.historical_tickets = historical_tickets
        corpus = [self._prepare_historical_document(t) for t in historical_tickets]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def find_similar(self, new_ticket: Dict[str, Any], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Calculates cosine similarity between incoming query and historical vectors.
        Returns top_n matches above the confidence threshold.
        """
        query_text = f"{new_ticket.get('subject', '')} {new_ticket.get('description', '')}"
        query_vector = self.vectorizer.transform([query_text])

        # Compute cosine similarity matrix
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Score and filter by confidence threshold
        scored_tickets = []
        for idx, score in enumerate(similarities):
            if score >= self.threshold:
                scored_tickets.append((self.historical_tickets[idx], float(score)))

        # Sort descending by score
        scored_tickets.sort(key=lambda item: item[1], reverse=True)

        # Format match results
        results = []
        for ticket, score in scored_tickets[:top_n]:
            results.append({
                "ticket_id": ticket.get("ticket_id"),
                "category": ticket.get("category"),
                "subject": ticket.get("subject"),
                "resolution_notes": ticket.get("resolution_notes"),
                "confidence_score": round(score, 3),
                "match_percentage": f"{int(round(score * 100))}%"
            })
        return results


def main():
    # File paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    historical_path = os.path.join(base_dir, "starter_kit", "data", "tickets.json")
    new_tickets_path = os.path.join(base_dir, "starter_kit", "data", "new_tickets_example.json")

    # Load data
    with open(historical_path, "r", encoding="utf-8") as f:
        historical_tickets = json.load(f)["tickets"]

    with open(new_tickets_path, "r", encoding="utf-8") as f:
        new_tickets = json.load(f)["new_tickets"]

    # Initialize and train engine
    engine = TicketSimilarityEngine(confidence_threshold=0.30)
    engine.fit(historical_tickets)

    print("=" * 70)
    print("SOLARI ANALYTICS — TICKET SIMILARITY ASSISTANT (V1 EXECUTION)")
    print("=" * 70 + "\n")

    # Process test cases
    for ticket in new_tickets:
        print(f"🎫 INCOMING TICKET [{ticket['ticket_id']}]: {ticket['subject']}")
        print(f"   Description: {ticket['description']}")
        
        matches = engine.find_similar(ticket, top_n=3)

        if not matches:
            print("\n   ⚠️  NO RELEVANT MATCH FOUND (Confidence < 20%)")
            print("   👉 AGENT GUIDANCE: This appears to be a novel issue. Investigate manually.\n")
        else:
            print(f"\n   ✅ FOUND {len(matches)} RELEVANT HISTORICAL RESOLUTION(S):")
            for idx, match in enumerate(matches, 1):
                print(f"      {idx}. [{match['ticket_id']}] Match: {match['match_percentage']} (Score: {match['confidence_score']}) | Category: {match['category']}")
                print(f"         Subject: {match['subject']}")
                print(f"         Resolution: {match['resolution_notes']}")
                print()
        print("-" * 70)


if __name__ == "__main__":
    main()