from transformers import pipeline

# Load sentiment-analysis pipeline once at module import
def _load_classifier():
    return pipeline("sentiment-analysis")

_classifier = _load_classifier()

def predict(text: str) -> dict:
    """
    Run sentiment analysis on the provided text.
    Returns a dict with keys: 'label' and 'score'.
    """
    result = _classifier(text)[0]
    return {"label": result["label"], "score": float(result["score"])}
