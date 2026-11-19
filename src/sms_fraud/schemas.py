"""Request and response shapes for the API (Pydantic models).

Why Pydantic?
- FastAPI checks every request against these classes automatically.
  Wrong or missing data gets a clear HTTP 422 error, and our code never
  sees bad input.
- The same classes are used to build the docs page (/docs).
"""

from pydantic import BaseModel, Field, field_validator

# An SMS is short, but a few SMS can be joined into one long message.
# A max length protects the API from very large requests.
MAX_MESSAGE_LENGTH = 5000
# Max messages in one batch request, for the same reason.
MAX_BATCH_SIZE = 100


# PATTERN: Data Transfer Object (DTO) — these classes only describe the data that goes in and out of
#   the API, with no business logic, so the API "contract" is in one clear place.
class PredictRequest(BaseModel):
    """Input for POST /predict."""

    message: str = Field(
        ...,  # "..." means: this field is required.
        max_length=MAX_MESSAGE_LENGTH,
        examples=["Congratulations! You won a FREE prize. Call now to claim."],
    )

    # A message with only spaces is also "empty". min_length=1 would not
    # catch "   ", so we check it ourselves. Raising ValueError here makes
    # FastAPI answer with HTTP 422.
    @field_validator("message")
    @classmethod
    def message_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be empty")
        return value


class PredictResponse(BaseModel):
    """Output of POST /predict (and one item of the batch output)."""

    is_fraud: bool
    # Probability that the message is spam: 0.0 = surely normal, 1.0 = surely spam.
    probability: float = Field(..., ge=0.0, le=1.0)
    # Which model gave this answer. Very useful when you debug or compare models.
    model_version: str


class BatchPredictRequest(BaseModel):
    """Input for POST /predict/batch."""

    messages: list[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)

    # Check every message in the list, with the same rule as above.
    @field_validator("messages")
    @classmethod
    def messages_must_not_be_empty(cls, values: list[str]) -> list[str]:
        for i, value in enumerate(values):
            if not value.strip():
                raise ValueError(f"message at position {i} must not be empty")
            if len(value) > MAX_MESSAGE_LENGTH:
                raise ValueError(f"message at position {i} is longer than {MAX_MESSAGE_LENGTH}")
        return values


class BatchPredictResponse(BaseModel):
    """Output of POST /predict/batch: one result per input message, same order."""

    results: list[PredictResponse]


class HealthResponse(BaseModel):
    """Output of GET /health."""

    status: str
    model_loaded: bool
