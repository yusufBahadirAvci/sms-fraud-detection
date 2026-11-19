"""Small text statistics shared by the API and the monitoring code.

This file has NO heavy imports (no pandas, scipy, matplotlib), so the small
API Docker image can use it too.
"""


def find_vectorizer(pipeline):
    """Return the TF-IDF vectorizer we use to measure unknown words.

    Since Step 5b the "tfidf" step of a model can be one of two things:
    - one TfidfVectorizer (word OR char features), or
    - a FeatureUnion (word AND char features side by side).
    For a FeatureUnion we pick the WORD part. Why? "Share of unknown words"
    is easy to understand and stays comparable with older reports.
    """
    step = pipeline.named_steps["tfidf"]
    # A FeatureUnion keeps its parts in "transformer_list": [(name, vectorizer), ...].
    parts = dict(getattr(step, "transformer_list", []))
    if parts:
        return parts.get("word", next(iter(parts.values())))
    return step


def oov_ratio(vectorizer, message: str) -> float:
    """Share of words in the message that the model does NOT know (0.0 - 1.0).

    OOV = "out of vocabulary". The TF-IDF step only knows the words it saw
    during training (its "vocabulary"). Unknown words are simply ignored by
    the model. So if most words are unknown, the model is almost blind,
    and its prediction cannot be trusted.

    Example: a model trained on English sees "Bu mesaja tiklayiniz"
    -> 3 words, 0 known -> oov_ratio = 1.0

    For a char model (analyzer "char" / "char_wb") the vocabulary holds
    letter pieces, not words. Then we count unknown letter PIECES instead.
    """
    if vectorizer.analyzer == "word":
        # Use the SAME text cleaning and word splitting as the vectorizer, so
        # we count words exactly the way the model sees them.
        preprocess = vectorizer.build_preprocessor()  # e.g. makes text lowercase
        tokenize = vectorizer.build_tokenizer()  # splits text into words
        units = tokenize(preprocess(message))
    else:
        # build_analyzer() gives the letter pieces exactly as the model sees them.
        units = vectorizer.build_analyzer()(message)

    # A message with no words at all (e.g. "!!!") gives the model nothing it
    # knows, so we treat it as fully unknown.
    if not units:
        return 1.0

    vocabulary = vectorizer.vocabulary_
    unknown = sum(1 for unit in units if unit not in vocabulary)
    return unknown / len(units)
