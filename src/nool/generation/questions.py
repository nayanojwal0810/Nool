"""Identify explicit parts of a company question."""

import re

_SECOND_QUESTION = re.compile(
    r"\s+and\s+(?=(?:what|when|where|which|who|how|why)\b)", re.IGNORECASE
)
_TWO_ACTIONS = re.compile(
    r"\bhow (?:should|can|do) (?:i|we)\s+(.+?)\s+and\s+(.+?)\?\s*$",
    re.IGNORECASE,
)
_CONTEXT_AND_QUESTION = re.compile(
    r"(?:[.!?]|,)\s+(?=(?:what|when|where|which|who|how|why|does|do|should|can)\b)",
    re.IGNORECASE,
)
_PAIRED_TOPIC_QUESTION = re.compile(r"^(?:what|which|how)\b", re.IGNORECASE)


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
    focus = focused_question(question)
    if _PAIRED_TOPIC_QUESTION.match(focus) and not focus.casefold().startswith(
        ("what is ", "what are ")
    ):
        separator = focus.casefold().rfind(" and ")
        if separator > 0:
            first = focus[:separator].strip()
            second = focus[separator + 5 :].strip().rstrip("?")
            if len(first.split()) >= 2 and len(second.split()) >= 2:
                context = question[: question.rfind(focus)]
                return (context + first).strip(), second
    return ()


def focused_question(question: str) -> str:
    """Remove an introductory situation when the final question stands alone."""
    matches = list(_CONTEXT_AND_QUESTION.finditer(question))
    return question[matches[-1].end() :].strip() if matches else question.strip()
