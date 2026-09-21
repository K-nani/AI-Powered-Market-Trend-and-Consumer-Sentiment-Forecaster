"""Aspect-Based Sentiment Analysis (ABSA) & Multi-Head Emotion Extraction Engine.

Deconstructs consumer feedback into granular product dimensions:
- Aspects: Battery, Build Quality, Pricing/Value, Performance, Support/Shipping, Software/UI
- Polarity: Positive, Neutral, Negative with confidence scores
- Granular Emotions: Joy, Trust, Anticipation, Anger, Sadness, Disgust
"""

import re
from typing import Dict, Any, List, Tuple
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class AspectSentimentAnalyzer:
    """Aspect-Based Sentiment Analysis (ABSA) and emotion profiling engine."""

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.aspect_lexicon = {
            "Battery & Power": [
                "battery", "charging", "charge", "drain", "mah", "runtime", "overheating",
                "charger", "power", "watt", "life", "standby"
            ],
            "Build Quality & Design": [
                "build", "finish", "titanium", "glass", "aluminum", "sturdy", "scratch",
                "durability", "premium", "broken", "cheap", "plastic", "panel", "gap", "flimsy"
            ],
            "Price & Value": [
                "price", "expensive", "cost", "value", "worth", "deal", "discount",
                "ripoff", "affordable", "money", "overpriced", "bargain"
            ],
            "Performance & Speed": [
                "fast", "speed", "smooth", "lag", "throttle", "processor", "chip", "fps",
                "slow", "gaming", "snappy", "benchmark", "powerful", "responsive"
            ],
            "Customer Support & Shipping": [
                "delivery", "shipping", "customer service", "support", "return", "refund",
                "warranty", "arrived", "packaging", "box", "unresponsive"
            ],
            "Software & Usability": [
                "software", "ios", "android", "app", "ui", "update", "glitch", "bug",
                "sync", "os", "firmware", "ecosystem", "intuitive"
            ]
        }

        self.emotion_lexicon = {
            "Joy": ["love", "excellent", "perfect", "stunning", "happy", "delighted", "awesome", "great", "flawless"],
            "Trust": ["reliable", "verified", "consistent", "solid", "durable", "proven", "safe", "secure", "best"],
            "Anticipation": ["looking forward", "excited", "eager", "expecting", "hype", "waiting", "hope"],
            "Anger": ["infuriating", "terrible", "worst", "hate", "scam", "predatory", "annoying", "unacceptable"],
            "Sadness": ["disappointed", "regret", "waste", "poor", "unfortunate", "ruined", "depressing"],
            "Disgust": ["awful", "horrible", "trash", "cheap", "defective", "nasty", "gross"]
        }

    def extract_aspect_sentiments(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Extracts mentions and sentiment polarity for each product aspect."""
        text_lower = text.lower()
        sentences = re.split(r"[.!?]+", text_lower)
        results = {}

        for aspect, keywords in self.aspect_lexicon.items():
            matching_sentences = [s for s in sentences if any(kw in s for kw in keywords)]
            if matching_sentences:
                combined_snippet = " ".join(matching_sentences)
                vader_score = self.vader.polarity_scores(combined_snippet)["compound"]
                
                if vader_score >= 0.15:
                    label = "Positive"
                elif vader_score <= -0.15:
                    label = "Negative"
                else:
                    label = "Neutral"

                results[aspect] = {
                    "mentioned": True,
                    "sentiment_score": round(vader_score, 3),
                    "normalized_1_to_5": round(1.0 + ((vader_score + 1.0) / 2.0) * 4.0, 2),
                    "label": label,
                    "evidence": combined_snippet.strip()
                }
            else:
                results[aspect] = {
                    "mentioned": False,
                    "sentiment_score": 0.0,
                    "normalized_1_to_5": 3.0,
                    "label": "Not Mentioned",
                    "evidence": ""
                }

        return results

    def extract_emotions(self, text: str) -> Dict[str, float]:
        """Calculates emotional intensity distribution across Joy, Trust, Anger, Sadness, etc."""
        text_lower = text.lower()
        counts = {}
        total = 0

        for emotion, terms in self.emotion_lexicon.items():
            c = sum(1 for term in terms if term in text_lower)
            counts[emotion] = c
            total += c

        if total == 0:
            # Baseline neutral distribution
            return {
                "Joy": 0.25,
                "Trust": 0.35,
                "Anticipation": 0.20,
                "Anger": 0.05,
                "Sadness": 0.10,
                "Disgust": 0.05
            }

        return {emotion: round(count / total, 3) for emotion, count in counts.items()}


# Singleton instance
aspect_analyzer = AspectSentimentAnalyzer()
