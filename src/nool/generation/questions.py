"""Identify explicit parts of a company question."""

import re

_SECOND_QUESTION = re.compile(
    r"\s+and\s+(?=(?:what|when|where|which|who|how|why)\b)", re.IGNORECASE
)
_TWO_ACTIONS = re.compile(
    r"\bhow (?:should|can|do) (?:i|we)\s+(.+?)\s+and\s+(.+?)\?\s*$",
    re.IGNORECASE,
)


def explicit_question_parts(question: str) -> tuple[str, ...]:
    """Split clear two-part requests without guessing at implied subquestions."""
    match = _SECOND_QUESTION.search(question)
    if match is not None:
        first = question[: match.start()].strip().rstrip("?, ")
        second = question[match.end() :].strip().rstrip("?")
        if first and second:
            return first, second
    match = _TWO_ACTIONS.search(question)
    if match is not None and len(match.group(2).split()) >= 2:
        return match.group(1).strip(), match.group(2).strip()
    return ()
