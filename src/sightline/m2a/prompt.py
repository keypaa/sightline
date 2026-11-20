"""The annotator contract: one prompt, one parser, one set of answer forms.

These live in the package rather than in the pilot script so the pipeline and the
pilot cannot drift apart. The pilot's prompt hash is the annotator's identity
(DECISIONS 0040); if two callers had their own copy, that hash would stop
meaning anything.
"""

from __future__ import annotations

import json
import re

NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}

PROMPT = """Look at the attached image and write a short conversation about it.

Rules:
- 2 or 3 user turns.
- Every user turn asks about one specific, checkable attribute: colour, shape,
  count, or position.
- Every assistant turn must state its answer explicitly in the first few words.
  Do not imply an answer inside a description: write "Red.", not "A red circle."
  when the question asks for the colour.
- When a question asks how many, answer with a number word: "Three."
- Answer only from the image. Do not mention instructions or the task.

Reply with ONLY a JSON object, no prose and no code fence, in exactly this shape:
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}"""


def answer_forms(answer: str) -> list[str]:
    """Acceptable spellings of an answer.

    Count answers are digits in the dataset but words in conversation, so both
    forms count as grounded. Checking only the digit scored every correct
    `'Two.'` as a miss.
    """

    forms = [answer]
    if answer.isdigit() and int(answer) in NUMBER_WORDS:
        forms.append(NUMBER_WORDS[int(answer)])
    return [f for f in forms if f]


def parse_messages(raw: str) -> list[dict] | None:
    """Extract the JSON object, tolerating code fences and surrounding prose."""

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    return [m for m in messages if isinstance(m, dict) and "content" in m]
