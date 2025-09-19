"""
post_processing.py
------------------
Phase 4: Post-processing for answers
- Summarization
- Actionable highlights
"""

from transformers import pipeline

# Load summarization model once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text: str) -> str:
    """Generate a clean summary without headings or bullets."""
    summary = summarizer(text, max_length=120, min_length=30, do_sample=False)
    return summary[0]['summary_text'].strip()



def extract_action_points(text: str) -> list:
    """
    Extract simple actionable highlights from answer.
    Example: Rights, obligations, remedies.
    """
    highlights = []
    for line in text.split("."):
        if any(word in line.lower() for word in ["must", "should", "entitled", "obligation", "remedy", "rights"]):
            highlights.append(line.strip())
    return highlights
