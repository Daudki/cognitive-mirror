"""Response serialization schemas."""

from marshmallow import Schema, fields


class TopEmotionSchema(Schema):
    """Top predicted emotion with probability."""
    emotion = fields.String(description="Emotion label")
    probability = fields.Float(description="Prediction probability")


class EmotionResultSchema(Schema):
    """Complete emotion prediction result."""
    emotion = fields.String(description="Primary predicted emotion")
    confidence = fields.Float(description="Confidence score for primary emotion")
    top_emotions = fields.List(
        fields.Nested(TopEmotionSchema),
        description="Top 3 emotions with probabilities",
    )


class SentimentProbabilitiesSchema(Schema):
    """Sentiment probability distribution."""
    negative = fields.Float()
    neutral = fields.Float()
    positive = fields.Float()


class SentimentResultSchema(Schema):
    """Complete sentiment prediction result."""
    sentiment = fields.String(description="Predicted sentiment label")
    confidence = fields.Float(description="Confidence score")
    probabilities = fields.Nested(
        SentimentProbabilitiesSchema,
        allow_none=True,
        description="Probability distribution (if available)",
    )


class ExplanationSchema(Schema):
    """Model explanation result."""
    top_features = fields.List(fields.Dict(), description="Top contributing features")
    method = fields.String(description="Explanation method used")


class PredictResponseSchema(Schema):
    """Complete prediction API response."""
    request_id = fields.String(description="Unique request identifier")
    text = fields.String(description="Original input text")
    emotion = fields.Nested(EmotionResultSchema, description="Emotion prediction")
    sentiment = fields.Nested(SentimentResultSchema, description="Sentiment prediction")
    mind_state = fields.String(description="Human-readable mind state summary")
    processing_time_ms = fields.Float(description="Inference time in milliseconds")
    model_version = fields.String(description="Model version used")
    explanation = fields.Nested(
        ExplanationSchema,
        allow_none=True,
        description="Model explanation (if requested)",
    )


class ErrorResponseSchema(Schema):
    """Standard error response."""
    error = fields.String(description="Error type")
    detail = fields.String(description="Error details")
    request_id = fields.String(allow_none=True, description="Request ID if available")
    timestamp = fields.String(description="ISO 8601 timestamp")
