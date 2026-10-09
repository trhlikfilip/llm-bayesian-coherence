"""Turn a quartet into eight (prompt, continuation) pairs.

A causal language model can only be scored on a continuation, so each relation
has two sentence templates with a subject slot {s} and an object slot {o}:

    object_final   ends with {o}, used for p(o | s)    "The capital of {s} is {o}."
    subject_final  ends with {s}, used for p(s | o)    "{o} is the capital of {s}."

For p(a | b) the prompt is the template with b filled in, cut just before a's
slot, and the continuation is a's label with a leading space:

    p(o1 | s1)  ->  "The capital of Japan is"  +  " Tokyo"

If the template puts "a" or "an" before the predicted slot ("A {s} is a {o}."),
the article moves into the continuation and is chosen for each candidate
(" an animal", " a tree"). The choice follows the first letter of the label.
Labels are never changed.
"""

from __future__ import annotations

import re

from certificate import TERMS

_ARTICLE_BEFORE_SLOT = re.compile(r"\b([Aa]n?) (\{[so]\})")
_TRAILING_ARTICLE = re.compile(r"\s+(a|an|A|An)$")


def article_for(label: str, article: str) -> str:
    """"a" or "an" for `label`, keeping the capitalisation of `article`."""
    base = "an" if label and label[0].lower() in "aeiou" else "a"
    return base.capitalize() if article[:1].isupper() else base


def split_at_slot(template: str, predict: str, s_label: str, o_label: str) -> tuple[str, str]:
    """(prompt, continuation) for predicting slot `predict` ("s" or "o") of a template ending in it."""
    slot, given_slot = "{%s}" % predict, "{o}" if predict == "s" else "{s}"
    if not re.search(re.escape(slot) + r"\s*[.!?]?\s*$", template):
        raise ValueError(f"template {template!r} does not end with {slot}")
    predicted, given = (s_label, o_label) if predict == "s" else (o_label, s_label)
    # the article before the given slot fits the given label
    template = _ARTICLE_BEFORE_SLOT.sub(
        lambda m: f"{article_for(given, m.group(1))} {m.group(2)}" if m.group(2) == given_slot else m.group(0), template)
    prefix = template[:template.rfind(slot)].rstrip()
    article = _TRAILING_ARTICLE.search(prefix)
    if article:                                   # "... is a {o}" -> " an animal" in the continuation
        prefix = prefix[:article.start()]
        continuation = f" {article_for(predicted, article.group(1))} {predicted}"
    else:
        continuation = " " + predicted
    return prefix.format(s=s_label, o=o_label).rstrip(), continuation


def term_pairs(relation: dict, quartet: dict) -> dict[str, tuple[str, str]]:
    """The eight (prompt, continuation) pairs of one quartet, keyed by term name."""
    label = {slot: quartet[f"{slot}_label"] for slot in ("s1", "s2", "o1", "o2")}
    pairs = {}
    for term in TERMS:
        candidate, given = term.split("_given_")
        if candidate.startswith("o"):             # object given subject
            pairs[term] = split_at_slot(relation["object_final"], "o", s_label=label[given], o_label=label[candidate])
        else:                                     # subject given object
            pairs[term] = split_at_slot(relation["subject_final"], "s", s_label=label[candidate], o_label=label[given])
    return pairs
