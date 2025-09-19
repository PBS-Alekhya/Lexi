import re
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename="legal_assistant.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_event(message: str):
    """Log events for debugging and monitoring."""
    logging.info(message)

def clean_text(text: str) -> str:
    """Remove unwanted characters and normalize whitespace."""
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces/newlines
    return text.strip()

def timestamp() -> str:
    """Return current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def add_disclaimer(answer: str) -> str:
    """
    Append disclaimer to generated answers.
    """
    disclaimer = "\n\n⚖️ Disclaimer: This is an AI-generated response. Verify with official legal sources."
    return answer + disclaimer