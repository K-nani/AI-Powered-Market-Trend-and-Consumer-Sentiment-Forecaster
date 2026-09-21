"""Comprehensive Automated Unit Tests for SentiTSMixer and Market Forecaster."""

import unittest
import numpy as np
import pandas as pd

from src.engine.sentitsmixer import SentiTSMixer, train_sentitsmixer, forecast_future
from src.engine.sentiment_analyzer import SentiEngine, sentiment_engine
from src.engine.feature_importance import compute_rf_feature_importance, get_benchmark_feature_importance
from src.engine.baselines import calculate_mse, calculate_msle, calculate_mape, compute_all_metrics
from src.engine.topic_modeling import LDATopicModeler, SemanticBERTopicModeler
from src.engine.rag_engine import ReviewRAGIndex, MarketCopilot
from src.data.benchmark_data import generate_weekly_series, get_domain_sample_reviews, CASE_STUDY_METADATA


class TestSentiTSMixerEngine(unittest.TestCase):

    def test_sentitsmixer_forward_shape(self):
        batch_size = 2
        seq_len = 8
        channels = 4
        horizon = 4
        model = SentiTSMixer(seq_len=seq_len, forecast_horizon=horizon, num_features=channels)
        x = np.random.randn(batch_size, seq_len, channels)
        weights = np.array([0.1, 0.1, 0.3, 0.5])
        out = model.forward(x, feature_weights=weights)
        self.assertEqual(out.shape, (batch_size, horizon, channels))

    def test_sentitsmixer_training(self):
        train_arr = np.random.uniform(0.1, 0.5, (20, 4))
        weights = np.array([0.1, 0.1, 0.3, 0.5])
        model, losses = train_sentitsmixer(train_arr, feature_weights=weights, seq_len=6, forecast_horizon=3, epochs=5)
        self.assertEqual(len(losses), 5)
        self.assertGreater(losses[0], 0.0)

    def test_sentiment_normalizations(self):
        # Equation 8: Sn = (1 + Sn)/2
        self.assertAlmostEqual(SentiEngine.map_vader_to_zero_one(-1.0), 0.0)
        self.assertAlmostEqual(SentiEngine.map_vader_to_zero_one(0.0), 0.5)
        self.assertAlmostEqual(SentiEngine.map_vader_to_zero_one(1.0), 1.0)

        # Equation 9: S in [1, 5]
        s_min = SentiEngine.normalize_sentiment_1_to_5(0.0, 0.0, 1.0)
        s_mid = SentiEngine.normalize_sentiment_1_to_5(0.5, 0.0, 1.0)
        s_max = SentiEngine.normalize_sentiment_1_to_5(1.0, 0.0, 1.0)
        self.assertAlmostEqual(s_min, 1.0)
        self.assertAlmostEqual(s_mid, 3.0)
        self.assertAlmostEqual(s_max, 5.0)

        # Equation 10: Helpfulness H in [1, 5]
        h_zero = SentiEngine.normalize_helpfulness(0, 0, 100)
        h_full = SentiEngine.normalize_helpfulness(100, 0, 100)
        self.assertAlmostEqual(h_zero, 1.0)
        self.assertAlmostEqual(h_full, 5.0)

    def test_sentiment_processing_pipeline(self):
        review_text = "Absolutely excellent magazine! Stunning articles and wonderful layout."
        res = sentiment_engine.process_review_record(review_text, rating=5.0, helpfulness_votes=25, engine_type="bert")
        self.assertIn("normalized_sentiment_S", res)
        self.assertIn("cumulative_mixture_SC", res)
        self.assertGreaterEqual(res["normalized_sentiment_S"], 3.5)

    def test_error_metrics(self):
        actual = np.array([100.0, 200.0, 300.0])
        pred = np.array([110.0, 190.0, 315.0])
        metrics = compute_all_metrics(actual, pred)
        self.assertGreater(metrics["MSE"], 0.0)
        self.assertGreater(metrics["MSLE"], 0.0)
        self.assertGreater(metrics["MAPE"], 0.0)
        self.assertLess(metrics["MAPE"], 15.0)

    def test_feature_importance_benchmark(self):
        weights_case1 = get_benchmark_feature_importance("case_1_magazine", "bert")
        self.assertIn("mixture", weights_case1)
        self.assertAlmostEqual(weights_case1["mixture"], 0.758813, places=4)

    def test_topic_modeling_lda_and_bertopic(self):
        docs = [
            "Battery lasts for hours and charging is very fast",
            "Screen protector was broken upon arrival terrible quality",
            "Phone case fits perfectly and grip is secure",
            "Fast shipping and nice responsive touchscreen display",
            "Defective charger cable stopped working after one week",
            "Comfortable earbuds with great noise cancellation and clear sound"
        ]
        lda = LDATopicModeler(n_topics=2)
        _, lda_kws = lda.fit_transform(docs)
        self.assertEqual(len(lda_kws), 2)

        bertopic = SemanticBERTopicModeler(n_clusters=2)
        res = bertopic.fit_transform(docs)
        self.assertEqual(len(res["topic_words"]), 2)

    def test_rag_faiss_indexing(self):
        reviews = get_domain_sample_reviews("case_3_cellphones")
        rag = ReviewRAGIndex(embedding_dim=32)
        rag.build_index(reviews)
        self.assertTrue(rag.is_indexed)

        results = rag.retrieve("battery and charging performance", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIn("similarity_score", results[0])

    def test_dataset_generation(self):
        for case in ["case_1_magazine", "case_2_fashion", "case_3_cellphones"]:
            df = generate_weekly_series(case)
            self.assertEqual(len(df), 54)
            self.assertIn("actual_sales", df.columns)
            self.assertIn("ts_mixer_bert_pred", df.columns)


if __name__ == "__main__":
    unittest.main()
