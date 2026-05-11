"""Request validation schemas using Marshmallow."""

from marshmallow import Schema, fields, validate, ValidationError


class PredictRequestSchema(Schema):
    """Validation for single prediction requests."""
    
    text = fields.String(
        required=True,
        validate=validate.Length(min=1, max=1000),
        metadata={"description": "Text input for emotion/sentiment analysis"},
        error_messages={
            "required": "The 'text' field is required.",
            "validator_failed": "Text must be between 1 and 1000 characters.",
        },
    )
    
    include_explanation = fields.Boolean(
        missing=False,
        metadata={"description": "Whether to generate LIME/SHAP explanation"},
    )
    
    model_version = fields.String(
        missing=None,
        validate=validate.Length(max=50),
        metadata={"description": "Specific model version tag (optional)"},
    )


class BatchPredictRequestSchema(Schema):
    """Validation for batch prediction requests."""
    
    texts = fields.List(
        fields.String(validate=validate.Length(min=1, max=1000)),
        required=True,
        validate=validate.Length(min=1, max=32),
        metadata={"description": "List of texts (max 32) for batch inference"},
        error_messages={
            "required": "The 'texts' field is required.",
            "validator_failed": "Batch must contain 1-32 items.",
        },
    )
    
    include_explanation = fields.Boolean(
        missing=False,
        metadata={"description": "Generate explanations for each prediction"},
    )
