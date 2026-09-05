"""
reason_matcher.py

The second real (non-authored-lookup) technique in this project: when the
diagnoser sees a reason_key it doesn't have a mapping for, instead of
immediately failing safe to escalate_human_review, it can optionally be
given a free-text description (Razorpay error responses include a real
`description` field) and find the closest KNOWN reason by text similarity.

HONESTY NOTE: this uses TF-IDF + cosine similarity, a classical (pre-neural)
NLP technique -- NOT a pretrained neural embedding model (e.g. sentence-
transformers). That's a deliberate constraint: this sandbox's network
allowlist covers package registries (PyPI/npm/GitHub) but not
huggingface.co, so a real pretrained embedding model's weights can't
actually be downloaded here. TF-IDF is a real, legitimate, well-understood
technique -- just not a neural one. Don't oversell it as "embeddings" in
the deep-learning sense; it's a genuine but simpler text-similarity method.

This is used ONLY as a fallback for genuinely unmapped codes -- known
Razorpay reason codes still go through the exact lookup in reason_mapping.json
first (diagnoser.py), which is more reliable than any similarity match.
"""

import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CONFIG_DIR = Path(__file__).parent.parent / "config"

SIMILARITY_THRESHOLD = 0.15  # below this, don't guess -- fail safe instead


class ReasonMatcher:
    def __init__(self, reason_mapping: dict = None):
        if reason_mapping is None:
            with open(CONFIG_DIR / "reason_mapping.json") as f:
                reason_mapping = json.load(f)

        # Build the reference corpus from each KNOWN reason's rationale text
        # (skip internal "_notes" keys and the simulated leak-type entries --
        # those aren't real Razorpay codes, so they shouldn't be match targets
        # for a Razorpay error description).
        self.reason_keys = []
        corpus = []
        for key, entry in reason_mapping.items():
            if key.startswith("_"):
                continue
            if entry.get("source") == "simulated":
                continue
            self.reason_keys.append(key)
            # Combine the reason key itself (underscores->spaces) with its
            # rationale for a richer text signal to match against.
            text = key.replace("_", " ") + " " + entry.get("rationale", "")
            corpus.append(text)

        self.reason_mapping = reason_mapping
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.corpus_vectors = self.vectorizer.fit_transform(corpus)

    def match(self, description: str) -> dict:
        """
        Given a free-text description of an unmapped failure, return the
        closest known reason_key and the similarity score, or None if
        nothing is close enough to trust.
        """
        query_vector = self.vectorizer.transform([description])
        similarities = cosine_similarity(query_vector, self.corpus_vectors)[0]

        best_idx = similarities.argmax()
        best_score = float(similarities[best_idx])
        best_key = self.reason_keys[best_idx]

        if best_score < SIMILARITY_THRESHOLD:
            return {
                "matched": False,
                "best_candidate": best_key,
                "similarity": round(best_score, 3),
                "reason": f"Best match similarity ({best_score:.3f}) below threshold "
                          f"({SIMILARITY_THRESHOLD}) -- not confident enough to guess.",
            }

        return {
            "matched": True,
            "matched_reason_key": best_key,
            "similarity": round(best_score, 3),
            "matched_entry": self.reason_mapping[best_key],
        }


if __name__ == "__main__":
    matcher = ReasonMatcher()

    test_descriptions = [
        "The customer's card was declined due to insufficient balance in their account",
        "Card verification failed because the CVV entered did not match bank records",
        "Our payment gateway partner experienced a temporary service outage",
        "The customer's bank flagged this transaction as potentially fraudulent activity",
        "asdkjaslkdj random unrelated text about cooking recipes",  # should fail to match confidently
    ]

    for desc in test_descriptions:
        result = matcher.match(desc)
        print(f"\nDescription: \"{desc}\"")
        if result["matched"]:
            print(f"  -> Matched: {result['matched_reason_key']} (similarity={result['similarity']})")
            print(f"  -> Root cause: {result['matched_entry']['root_cause']}")
        else:
            print(f"  -> NO CONFIDENT MATCH (closest was {result['best_candidate']}, "
                  f"similarity={result['similarity']}) -- {result['reason']}")
