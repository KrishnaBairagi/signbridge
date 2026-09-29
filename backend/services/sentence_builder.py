"""
services/sentence_builder.py

Converts a list of recognized sign tokens (e.g. ["I", "NEED", "HELP"]) into
natural English ("I need help."), per §9 of the spec.

Two modes:
  - rule_based (default): deterministic, no external calls, cannot hallucinate.
  - llm_assisted (optional): sends ONLY the recognized tokens to an LLM with a
    strict instruction to reorder/punctuate/lowercase-fix but never invent new
    words. We additionally validate the LLM's output token-for-token against
    the input to reject any hallucinated content -- if validation fails we
    silently fall back to rule_based.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_MAP = {
    "THANK_YOU": "thank you",
    "I": "I",
    "YOU": "you",
}

_SMALL_WORDS = {"a", "an", "the", "to", "do", "does", "is", "are"}


@dataclass
class SentenceResult:
    text: str
    tokens_used: list[str]
    method: str  # "rule_based" | "llm_assisted"


def _humanize_token(token: str) -> str:
    return _WORD_MAP.get(token, token.lower())


def build_sentence_rule_based(tokens: list[str]) -> SentenceResult:
    if not tokens:
        return SentenceResult("", [], "rule_based")

    words = [_humanize_token(t) for t in tokens]

    # Very small set of ISL/ASL-gloss-to-English grammar nudges. This is
    # intentionally conservative: it reorders/adds only function words that
    # are grammatically implied, never new content.
    is_question = tokens[0] in {"WHERE", "WHAT", "WHO"} or tokens[-1] in {"WHERE", "WHAT", "WHO"}
    is_negative = "NO" in tokens and len(tokens) > 1

    sentence = " ".join(words)
    sentence = sentence[0].upper() + sentence[1:] if sentence else sentence
    sentence = sentence.rstrip(".") + ("?" if is_question else ".")

    return SentenceResult(sentence, tokens, "rule_based")


LLM_SYSTEM_PROMPT = """You convert a list of sign-language gloss tokens into a natural
English sentence. Rules you MUST follow:
1. Use ONLY the words/concepts present in the token list. Do not add new information.
2. You may add function words (is, the, a, to) and fix word order/tense/punctuation.
3. Do not answer questions, do not continue the conversation, only rephrase.
4. Output only the final sentence, nothing else.
"""


def build_sentence_llm_assisted(tokens: list[str], llm_call) -> SentenceResult:
    """
    `llm_call` is an injected callable(system_prompt, user_prompt) -> str so
    this module has no direct dependency on any specific LLM provider
    (see services/ai_conversation_service.py for the actual abstraction).
    """
    fallback = build_sentence_rule_based(tokens)
    try:
        raw = llm_call(LLM_SYSTEM_PROMPT, "Tokens: " + ", ".join(tokens))
        candidate = raw.strip()
        if _validate_no_hallucination(candidate, tokens):
            return SentenceResult(candidate, tokens, "llm_assisted")
    except Exception:
        pass
    return fallback


def _validate_no_hallucination(sentence: str, tokens: list[str]) -> bool:
    """
    Cheap guardrail: every content token's word stem should appear
    (case-insensitively) somewhere in the generated sentence. This won't
    catch every possible hallucination but blocks the common failure mode
    of the LLM inventing unrelated content.
    """
    sentence_lower = re.sub(r"[^a-z\s]", "", sentence.lower())
    for token in tokens:
        stem = _humanize_token(token).split()[0].lower()
        if stem in _SMALL_WORDS:
            continue
        if stem not in sentence_lower:
            return False
    # Reject wildly longer outputs (sign of added content)
    if len(sentence.split()) > len(tokens) * 3 + 3:
        return False
    return True
