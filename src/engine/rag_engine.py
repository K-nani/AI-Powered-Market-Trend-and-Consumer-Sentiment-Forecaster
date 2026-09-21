"""RAG (Retrieval-Augmented Generation) Engine with FAISS and LLM Integration.

Enables similarity search across consumer reviews and market signals,
and provides contextual market intelligence answers via Gemini LLM or offline synthesis.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer


class ReviewRAGIndex:
    """Vector database indexing customer reviews and market signals using FAISS."""

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.vectorizer = TfidfVectorizer(max_features=embedding_dim, stop_words="english")
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.documents: List[Dict[str, Any]] = []
        self.is_indexed = False

    def build_index(self, reviews: List[Dict[str, Any]]):
        """Build FAISS vector index from review dictionaries.
        Each review has: 'text', 'rating', 'helpfulness', 'sentiment', 'date', etc.
        """
        if not reviews:
            return

        self.documents = reviews
        texts = [r.get("text", "") for r in reviews]
        
        # Fit vectorizer and compute dense representations
        tfidf_matrix = self.vectorizer.fit_transform(texts).toarray()
        
        # Pad or trim to embedding_dim
        cur_dim = tfidf_matrix.shape[1]
        if cur_dim < self.embedding_dim:
            padded = np.zeros((tfidf_matrix.shape[0], self.embedding_dim), dtype=np.float32)
            padded[:, :cur_dim] = tfidf_matrix
            embeddings = padded
        else:
            embeddings = tfidf_matrix[:, :self.embedding_dim].astype(np.float32)

        # L2-normalize vectors for Cosine Similarity in IndexFlatIP
        faiss.normalize_L2(embeddings)

        # Re-initialize index and add vectors
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        self.is_indexed = True

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant customer reviews for a query using FAISS."""
        if not self.is_indexed or not self.documents:
            return []

        # Vectorize query
        q_vec = self.vectorizer.transform([query]).toarray()
        cur_dim = q_vec.shape[1]
        if cur_dim < self.embedding_dim:
            padded = np.zeros((1, self.embedding_dim), dtype=np.float32)
            padded[:, :cur_dim] = q_vec
            q_emb = padded
        else:
            q_emb = q_vec[:, :self.embedding_dim].astype(np.float32)

        faiss.normalize_L2(q_emb)

        distances, indices = self.index.search(q_emb, min(top_k, len(self.documents)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.documents):
                doc = dict(self.documents[idx])
                doc["similarity_score"] = float(round(dist, 4))
                results.append(doc)

        return results


class MarketCopilot:
    """Contextual RAG Copilot integrating FAISS vector search with LLMs."""

    def __init__(self, rag_index: ReviewRAGIndex, gemini_api_key: Optional[str] = None):
        self.rag_index = rag_index
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")

    def answer_query(self, user_question: str, top_k: int = 5) -> Dict[str, Any]:
        """Performs RAG: retrieves relevant reviews and synthesizes an actionable insight."""
        retrieved_reviews = self.rag_index.retrieve(user_question, top_k=top_k)

        # If Gemini API is available and configured
        if self.gemini_api_key:
            try:
                from google import genai
                client = genai.Client(api_key=self.gemini_api_key)

                context_blocks = []
                for i, r in enumerate(retrieved_reviews):
                    context_blocks.append(
                        f"[Review {i+1}] Rating: {r.get('rating', 'N/A')}/5, "
                        f"Helpfulness: {r.get('helpfulness', 0)} likes, "
                        f"Sentiment: {r.get('sentiment', 'N/A')}\n"
                        f"Content: {r.get('text', '')}"
                    )
                context_str = "\n\n".join(context_blocks)

                prompt = (
                    f"You are the Senior AI Market Intelligence Copilot for the SentiTSMixer forecasting platform.\n"
                    f"The user has asked: '{user_question}'\n\n"
                    f"Here are the top retrieved customer feedback records from the FAISS vector database:\n"
                    f"{context_str}\n\n"
                    f"Provide an executive-level market intelligence answer. Include:\n"
                    f"1. Direct Answer & Executive Takeaway\n"
                    f"2. Specific Customer Evidence (citing ratings, helpfulness, and sentiments)\n"
                    f"3. Forecast & Supply Chain Implications (how this sentiment impacts future demand)\n"
                    f"4. Recommended Strategic Action for Product / Marketing teams."
                )

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                return {
                    "answer": response.text,
                    "retrieved_evidence": retrieved_reviews,
                    "engine_used": "Gemini 2.5 Flash via RAG"
                }
            except Exception as e:
                # Fall back gracefully to structured contextual synthesis
                return self._offline_contextual_synthesis(user_question, retrieved_reviews, error_msg=str(e))
        else:
            return self._offline_contextual_synthesis(user_question, retrieved_reviews)

    def _offline_contextual_synthesis(
        self,
        question: str,
        reviews: List[Dict[str, Any]],
        error_msg: Optional[str] = None
    ) -> Dict[str, Any]:
        """Intelligent offline synthesis engine when LLM API key is not provided."""
        if not reviews:
            return {
                "answer": "No directly relevant customer reviews found matching your inquiry. Try querying product attributes like 'battery', 'comfort', 'pricing', or 'shipping delay'.",
                "retrieved_evidence": [],
                "engine_used": "Heuristic Knowledge Retrieval"
            }

        ratings = [r.get("rating", 3.0) for r in reviews if "rating" in r]
        avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else 3.5
        total_votes = sum(r.get("helpfulness", 0) for r in reviews)
        
        pos_reviews = [r for r in reviews if r.get("rating", 3) >= 4]
        neg_reviews = [r for r in reviews if r.get("rating", 3) <= 2]

        # Key phrases
        sentiment_summary = (
            f"Analysis of the top {len(reviews)} customer feedback items retrieved via FAISS vector similarity "
            f"reveals an average sentiment rating of **{avg_rating}/5.0** across **{total_votes} helpfulness votes**."
        )

        bullets = []
        if pos_reviews:
            top_pos = pos_reviews[0]
            bullets.append(f"**Positive Driver:** Customer praised *'{top_pos['text'][:120]}...'* (Rating: {top_pos.get('rating')}/5, {top_pos.get('helpfulness', 0)} likes).")
        if neg_reviews:
            top_neg = neg_reviews[0]
            bullets.append(f"**Friction Point:** Customer expressed concern: *'{top_neg['text'][:120]}...'* (Rating: {top_neg.get('rating')}/5, {top_neg.get('helpfulness', 0)} likes).")

        implication = (
            f"**SentiTSMixer Forecast Implication:** "
            f"Because review helpfulness heavily influences the sentiment mixture ($SC$), "
            f"the {'positive sentiment trajectory will support upward sales momentum' if avg_rating >= 3.5 else 'negative sentiment down-votes signal potential demand softening in upcoming weeks'}. "
            f"Align inventory buffers accordingly to prevent stockouts or over-forecasting bullwhip effects."
        )

        full_answer = (
            f"### Executive Market Intelligence Summary\n\n"
            f"{sentiment_summary}\n\n"
            f"#### Key Findings from Vector Retrieval:\n"
            + "\n".join([f"- {b}" for b in bullets]) + "\n\n"
            f"#### Strategic Recommendations:\n"
            f"1. **Product / Engineering:** Directly investigate high-vote feedback clusters to address recurring pain points.\n"
            f"2. **Supply Chain:** {implication}\n"
            f"3. **Marketing:** Highlight verified high-helpfulness positive features in promotional campaigns.\n\n"
            + (f"*Note: Running in offline semantic synthesis mode. Enter a Gemini API key in the sidebar for full conversational LLM generation.*" if not error_msg else f"*API Note: {error_msg}. Fallen back to local semantic synthesis.*")
        )

        return {
            "answer": full_answer,
            "retrieved_evidence": reviews,
            "engine_used": "Local Semantic RAG Engine"
        }
