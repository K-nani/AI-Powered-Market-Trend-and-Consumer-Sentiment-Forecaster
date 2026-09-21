"""Topic Modeling Engine: LDA and BERTopic-style Semantic Clustering.

Discovers significant discussion themes, emerging consumer trends,
and contextual topic shifts over time.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import KMeans


class LDATopicModeler:
    """Latent Dirichlet Allocation (LDA) Topic Modeler for consumer feedback."""

    def __init__(self, n_topics: int = 5, max_features: int = 1000, random_state: int = 42):
        self.n_topics = n_topics
        self.max_features = max_features
        self.random_state = random_state
        self.vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=1,
            stop_words="english",
            max_features=self.max_features,
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b"
        )
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=self.random_state,
            learning_method="batch"
        )
        self.feature_names = []
        self.is_fitted = False

    def fit_transform(self, documents: List[str]) -> Tuple[np.ndarray, Dict[int, List[Tuple[str, float]]]]:
        """Fit LDA on documents and return document-topic distributions and top topic words."""
        cleaned_docs = [self._preprocess(d) for d in documents if d and isinstance(d, str)]
        if len(cleaned_docs) < self.n_topics:
            cleaned_docs = cleaned_docs * (self.n_topics + 1)

        dtm = self.vectorizer.fit_transform(cleaned_docs)
        doc_topic_dist = self.lda_model.fit_transform(dtm)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.is_fitted = True

        topic_keywords = {}
        for topic_idx, topic_weights in enumerate(self.lda_model.components_):
            top_indices = topic_weights.argsort()[:-11:-1]
            top_words = [(self.feature_names[i], round(float(topic_weights[i] / max(1e-6, topic_weights.sum())), 4)) for i in top_indices if i < len(self.feature_names)]
            topic_keywords[topic_idx] = top_words

        return doc_topic_dist, topic_keywords

    def get_topic_names(self, topic_keywords: Dict[int, List[Tuple[str, float]]]) -> Dict[int, str]:
        """Generate human-readable topic labels from top words."""
        labels = {}
        for t_idx, words in topic_keywords.items():
            top_3 = [w[0] for w in words[:3]]
            labels[t_idx] = f"Topic {t_idx + 1}: {' / '.join(top_3).title()}"
        return labels

    def _preprocess(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        return " ".join(text.split())


class SemanticBERTopicModeler:
    """Transformer / Semantic Clustering Topic Modeler (BERTopic-style c-TF-IDF pipeline)."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.tfidf = TfidfVectorizer(
            max_df=0.95,
            min_df=1,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=1500
        )
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        self.topic_c_tfidf = {}
        self.topic_docs = {}

    def fit_transform(self, documents: List[str], timestamps: Optional[List[str]] = None) -> Dict[str, Any]:
        """Fits semantic topic pipeline and computes topic distributions over time."""
        clean_docs = [d if (d and isinstance(d, str)) else "product review" for d in documents]
        if len(clean_docs) < self.n_clusters:
            clean_docs = clean_docs * (self.n_clusters + 1)

        # 1. TF-IDF & Dimensionality Reduction
        X_tfidf = self.tfidf.fit_transform(clean_docs)
        n_features = X_tfidf.shape[1]
        
        # Safely reduce dimensions or use dense representation
        if n_features > 2:
            n_comp = min(n_features - 1, self.n_clusters * 2, 10)
            reducer = TruncatedSVD(n_components=max(2, n_comp), random_state=self.random_state)
            X_dense = reducer.fit_transform(X_tfidf)
        else:
            X_dense = X_tfidf.toarray()
            if X_dense.shape[1] < 2:
                # Pad with zeros so 2D coordinates always exist
                X_dense = np.pad(X_dense, ((0, 0), (0, 2 - X_dense.shape[1])), 'constant')

        # 2. Clustering
        actual_clusters = min(self.n_clusters, len(clean_docs))
        self.clusterer.n_clusters = actual_clusters
        labels = self.clusterer.fit_predict(X_dense)

        # 3. Class-based TF-IDF (c-TF-IDF)
        feature_names = self.tfidf.get_feature_names_out()
        topic_words = {}
        topic_counts = {}

        for topic_id in range(actual_clusters):
            idx_in_topic = np.where(labels == topic_id)[0]
            topic_counts[topic_id] = int(len(idx_in_topic))
            if len(idx_in_topic) == 0:
                topic_words[topic_id] = [("general", 1.0)]
                continue

            # Aggregate words in topic cluster
            cluster_matrix = X_tfidf[idx_in_topic].sum(axis=0)
            cluster_scores = np.asarray(cluster_matrix).flatten()
            top_word_indices = cluster_scores.argsort()[:-9:-1]
            topic_words[topic_id] = [
                (feature_names[i], round(float(cluster_scores[i] / max(1e-6, cluster_scores.sum())), 4))
                for i in top_word_indices if i < len(feature_names)
            ]

        # 4. Generate expressive Topic Names
        topic_names = {}
        for t_id, words in topic_words.items():
            top_3 = [w[0] for w in words[:3]]
            topic_names[t_id] = f"Cluster {t_id + 1}: {' & '.join(top_3).title()}"

        # 5. Temporal Topic Trends
        trends_df = None
        if timestamps and len(timestamps) == len(documents):
            df_temporal = pd.DataFrame({"topic": labels[:len(timestamps)], "timestamp": timestamps})
            try:
                df_temporal["timestamp"] = pd.to_datetime(df_temporal["timestamp"])
                temporal_counts = df_temporal.groupby([pd.Grouper(key="timestamp", freq="W"), "topic"]).size().unstack(fill_value=0)
                temporal_counts.columns = [topic_names.get(c, f"Topic {c}") for c in temporal_counts.columns]
                trends_df = temporal_counts.reset_index()
            except Exception:
                trends_df = None

        return {
            "cluster_labels": labels.tolist(),
            "topic_words": topic_words,
            "topic_counts": topic_counts,
            "topic_names": topic_names,
            "embeddings_2d": X_dense[:, :2].tolist(),
            "temporal_trends": trends_df
        }
