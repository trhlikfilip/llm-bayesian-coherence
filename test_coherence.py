"""Tests for the whole package. None needs a model download: the scorer is
tested with a stub tokenizer and a stub model, the pipeline with a fake scorer.

    pytest -q
"""

import csv
import hashlib
import json
import math
import os
import random
import re

import pytest

import run
import scorer as S
from certificate import DENOMINATOR, NUMERATOR, TERMS, derive, log_ratio
from prompts import term_pairs

HERE = os.path.dirname(os.path.abspath(__file__))
QUARTETS = os.path.join(HERE, "quartets.csv.gz")
TEMPLATES = os.path.join(HERE, "templates.json")
CAPITAL = {"object_final": "The capital of {s} is {o}.", "subject_final": "{o} is the capital of {s}."}
JAPAN_GERMANY = {"s1_label": "Japan", "o1_label": "Tokyo", "s2_label": "Germany", "o2_label": "Berlin"}
quiet = lambda message: None


# ============================================================================= prompts

def test_eight_prompts_for_the_capital_example():
    t = term_pairs(CAPITAL, JAPAN_GERMANY)
    assert t["o1_given_s1"] == ("The capital of Japan is", " Tokyo")
    assert t["o2_given_s1"] == ("The capital of Japan is", " Berlin")
    assert t["o1_given_s2"] == ("The capital of Germany is", " Tokyo")
    assert t["o2_given_s2"] == ("The capital of Germany is", " Berlin")
    assert t["s1_given_o1"] == ("Tokyo is the capital of", " Japan")
    assert t["s2_given_o1"] == ("Tokyo is the capital of", " Germany")
    assert t["s1_given_o2"] == ("Berlin is the capital of", " Japan")
    assert t["s2_given_o2"] == ("Berlin is the capital of", " Germany")


def test_each_prompt_and_candidate_appears_once_on_each_side():
    """Prompt normalisers and per-candidate constants cancel in LR only because of this."""
    t = term_pairs(CAPITAL, JAPAN_GERMANY)
    assert sorted(t[k][0] for k in NUMERATOR) == sorted(t[k][0] for k in DENOMINATOR)
    assert len({t[k][0] for k in NUMERATOR}) == 4
    assert sorted(t[k][1] for k in NUMERATOR) == sorted(t[k][1] for k in DENOMINATOR)


def test_article_moves_into_the_continuation_and_fits_the_candidate():
    tpl = {"object_final": "A {s} is a {o}.", "subject_final": "One example of a {o} is the {s}."}
    q = {"s1_label": "dog", "o1_label": "animal", "s2_label": "oak", "o2_label": "tree"}
    t = term_pairs(tpl, q)
    assert t["o1_given_s1"] == ("A dog is", " an animal")
    assert t["o2_given_s1"] == ("A dog is", " a tree")
    assert t["s2_given_o2"] == ("One example of a tree is the", " oak")
    assert t["s1_given_o1"] == ("One example of an animal is the", " dog")
    assert term_pairs(tpl, {**q, "s1_label": "Owl"})["o1_given_s1"][0] == "An Owl is"


def test_labels_are_never_changed():
    tpl = {"object_final": "{s} was born in {o}.", "subject_final": "{o} is the birthplace of {s}."}
    q = {"s1_label": "Anna", "o1_label": "Spittal an der Drau", "s2_label": "Bo", "o2_label": "Oslo"}
    t = term_pairs(tpl, q)
    assert t["s1_given_o1"][0] == "Spittal an der Drau is the birthplace of"
    assert t["o1_given_s1"][1] == " Spittal an der Drau"


def test_a_template_that_does_not_end_in_its_slot_is_rejected():
    with pytest.raises(ValueError, match="does not end with"):
        term_pairs({"object_final": "{o} is in {s}.", "subject_final": "{o} is the capital of {s}."}, JAPAN_GERMANY)


def test_every_relation_in_the_data_has_valid_templates():
    templates = run.load_templates(TEMPLATES)
    rows = run.load_quartets(QUARTETS)
    assert len(rows) == 140_751
    assert {(q["source"], q["relation_id"]) for q in rows} == set(templates)
    for key, relation in templates.items():
        q = next(r for r in rows if (r["source"], r["relation_id"]) == key)
        assert len(term_pairs(relation, q)) == 8


# ============================================================================= certificate maths

def conditionals_from_table(P):
    """The eight log-conditionals of a positive 2x2 table P[s][o]:
    p(o|s) = P[s][o] / row sum and p(s|o) = P[s][o] / column sum."""
    row = [P[0][0] + P[0][1], P[1][0] + P[1][1]]
    col = [P[0][0] + P[1][0], P[0][1] + P[1][1]]
    return {"o1_given_s1": math.log(P[0][0] / row[0]), "o2_given_s1": math.log(P[0][1] / row[0]),
            "o1_given_s2": math.log(P[1][0] / row[1]), "o2_given_s2": math.log(P[1][1] / row[1]),
            "s1_given_o1": math.log(P[0][0] / col[0]), "s2_given_o1": math.log(P[1][0] / col[0]),
            "s1_given_o2": math.log(P[0][1] / col[1]), "s2_given_o2": math.log(P[1][1] / col[1])}


def random_lp(rng):
    return {t: -rng.uniform(0, 40) for t in TERMS}


def test_LR_equals_2_times_B_minus_F():
    rng = random.Random(0)
    for _ in range(500):
        d = derive(random_lp(rng))
        assert math.isclose(d["log_ratio"], 2 * (d["backward_margin"] - d["forward_margin"]), abs_tol=1e-9)
        assert 0 <= d["knowledge_score"] <= 4 and 0 <= d["relative_incoherence"] <= 1
        assert d["knowledge_score"] == d["forward_correct"] + d["backward_correct"]


def test_conditionals_of_one_joint_give_LR_0():
    """Also covers reading both directions from ONE sentence template: that is a positive
    table too, which is why each direction needs its own template."""
    rng = random.Random(1)
    for _ in range(500):
        P = [[rng.uniform(1e-6, 1) for _ in range(2)] for _ in range(2)]
        assert abs(log_ratio(conditionals_from_table(P))) < 1e-9


def test_one_changed_conditional_is_caught_with_the_right_sign():
    P = [[0.4, 0.1], [0.2, 0.3]]
    lp = conditionals_from_table(P)
    lp["s1_given_o1"] += 0.7          # a numerator term
    assert math.isclose(log_ratio(lp), 0.7, abs_tol=1e-9)
    lp = conditionals_from_table(P)
    lp["o1_given_s1"] += 0.7          # a denominator term, which also raises F
    assert math.isclose(log_ratio(lp), -0.7, abs_tol=1e-9)


def test_a_constant_per_prompt_changes_nothing():
    lp = random_lp(random.Random(2))
    shifted = dict(lp, o1_given_s1=lp["o1_given_s1"] + 3.3, o2_given_s1=lp["o2_given_s1"] + 3.3)
    assert derive(shifted) == pytest.approx(derive(lp), abs=1e-9)


def test_a_constant_per_candidate_leaves_LR_F_B_but_can_change_K():
    rng = random.Random(3)
    k_changed = 0
    for _ in range(300):
        lp = random_lp(rng)
        shifted = dict(lp, o1_given_s1=lp["o1_given_s1"] - 4.0, o1_given_s2=lp["o1_given_s2"] - 4.0)
        a, b = derive(lp), derive(shifted)
        for k in ("log_ratio", "forward_margin", "backward_margin"):
            assert math.isclose(a[k], b[k], abs_tol=1e-9)
        k_changed += a["knowledge_score"] != b["knowledge_score"]
    assert k_changed > 0


def test_a_model_that_ignores_the_prompt_scores_LR_0_and_K_2():
    prior = {"o1": -3.0, "o2": -5.0, "s1": -2.0, "s2": -7.0}
    d = derive({t: prior[t.split("_given_")[0]] for t in TERMS})
    assert d["log_ratio"] == d["forward_margin"] == d["backward_margin"] == 0
    assert d["knowledge_score"] == 2


def test_relative_incoherence_edge_cases():
    def lp(F, B):
        return {"o1_given_s1": F, "o2_given_s1": 0.0, "o2_given_s2": F, "o1_given_s2": 0.0,
                "s1_given_o1": B, "s2_given_o1": 0.0, "s2_given_o2": B, "s1_given_o2": 0.0}
    assert derive(lp(3, -3))["relative_incoherence"] == pytest.approx(1.0)    # opposite signs
    assert derive(lp(3, 0))["relative_incoherence"] == pytest.approx(1.0)     # one margin zero
    assert derive(lp(0.1, -0.05))["relative_incoherence"] == 1.0              # exactly 1, even for tiny margins
    assert derive(lp(3, 1))["relative_incoherence"] == pytest.approx(0.5)     # |B - F| / (|F| + |B|) = 2 / 4
    assert derive(lp(3, 3))["relative_incoherence"] == 0                      # F = B
    assert derive(lp(0, 0))["relative_incoherence"] == 0                      # defined as 0
    assert derive(lp(0, 0))["knowledge_score"] == 0                           # ties count as wrong


def test_bad_log_probabilities_are_rejected():
    lp = {t: -1.0 for t in TERMS}
    with pytest.raises(KeyError):
        derive({k: v for k, v in lp.items() if k != "o1_given_s1"})
    with pytest.raises(ValueError):
        derive({**lp, "o1_given_s1": float("nan")})


# ============================================================================= scorer

class WordTokenizer:
    """Tokens are words with their leading space, offsets are character spans, optional BOS."""

    def __init__(self, bos=True):
        self.bos, self.vocab, self.pad_token_id = bos, {}, 0

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=True):
        ids, offsets = ([1], [(0, 0)]) if (self.bos and add_special_tokens) else ([], [])
        for m in re.finditer(r"\s*\S+", text):
            ids.append(self.vocab.setdefault(m.group(), len(self.vocab) + 2))
            offsets.append((m.start(), m.end()))
        return {"input_ids": ids, "offset_mapping": offsets}


def stub_scorer(tokenizer, model=None):
    s = S.Scorer.__new__(S.Scorer)          # skip loading a real checkpoint
    s.tokenizer, s.model, s.device = tokenizer, model, "cpu"
    return s


def test_encode_finds_the_continuation_tokens():
    s = stub_scorer(WordTokenizer(bos=True))
    ids, start = s.encode("The capital of Japan is", " Tokyo")
    assert len(ids) == 7 and start == 6                      # BOS + 5 prompt words, then " Tokyo"
    ids, start = s.encode("Paris is", " Île de France")
    assert len(ids) - start == 3


def test_encode_refuses_what_it_cannot_split():
    s = stub_scorer(WordTokenizer(bos=False))
    for prompt, continuation in [("abc", "def"), ("", " x"), ("x", "   ")]:
        with pytest.raises(ValueError):
            s.encode(prompt, continuation)


def test_score_restores_order_and_does_not_depend_on_batch_size():
    torch = pytest.importorskip("torch")
    V = 64

    class StubModel:
        """Logits depend only on the current token, so the right score is easy to compute."""

        def __call__(self, input_ids, attention_mask):
            logits = torch.nn.functional.one_hot((input_ids * 7 + 3) % V, V).float() * 4.0
            return type("Out", (), {"logits": logits})()

    s = stub_scorer(WordTokenizer(bos=True), StubModel())
    s.torch = torch
    pairs = [("The capital of Japan is", " Tokyo"), ("A", " b c d e f"), ("Tokyo is the capital of", " Japan"), ("x y", " z")]

    def expected(prompt, continuation):
        ids, start = s.encode(prompt, continuation)
        assert max(ids) < V
        total = 0.0
        for pos in range(start, len(ids)):
            logits = torch.nn.functional.one_hot(torch.tensor((ids[pos - 1] * 7 + 3) % V), V).float() * 4.0
            total += torch.log_softmax(logits, -1)[ids[pos]].item()
        return total, len(ids) - start

    for (p, c), a, b in zip(pairs, s.score(pairs, batch_size=1), s.score(pairs, batch_size=3)):
        want, n = expected(p, c)
        assert a["n_tokens"] == b["n_tokens"] == n
        assert a["logprob"] == pytest.approx(want) and b["logprob"] == pytest.approx(want)


# ============================================================================= cache and pipeline

def fake_logprob(prompt, continuation):
    h = int(hashlib.sha1(f"{prompt}\x1f{continuation}".encode()).hexdigest(), 16)
    return -(h % 3000) / 100.0


class FakeScorer:
    """Stands in for scorer.Scorer and records what it is asked to score."""
    loaded = 0
    scored = []

    def __init__(self, model_id, device, dtype):
        FakeScorer.loaded += 1

    def score(self, pairs, batch_size):
        FakeScorer.scored.extend(pairs)
        return [{"logprob": fake_logprob(p, c), "n_tokens": len(c.split())} for p, c in pairs]


@pytest.fixture
def fake(monkeypatch):
    FakeScorer.loaded, FakeScorer.scored = 0, []
    monkeypatch.setattr(S, "Scorer", FakeScorer)
    return FakeScorer


def test_cache_scores_only_missing_pairs(tmp_path, fake):
    cache = str(tmp_path / "c.jsonl")
    pairs = [("p1", " a"), ("p1", " b"), ("p2", " a")]
    first = S.score_with_cache("m", pairs, cache, device="cpu", log=quiet)
    assert fake.loaded == 1 and len(fake.scored) == 3
    assert S.score_with_cache("m", pairs, cache, device="cpu", log=quiet) == first
    assert fake.loaded == 1                                               # nothing rescored, no model loaded
    S.score_with_cache("m", pairs + [("p3", " c")], cache, device="cpu", log=quiet)
    assert fake.scored[-1] == ("p3", " c") and len(fake.scored) == 4
    S.score_with_cache("other-model", pairs, cache, device="cpu", log=quiet)
    assert len(fake.scored) == 7                                          # a different model scores everything


def test_cache_with_a_partial_last_line_is_repaired(tmp_path, fake):
    cache = tmp_path / "c.jsonl"
    pairs = [("p1", " a"), ("p2", " b")]
    S.score_with_cache("m", pairs, str(cache), device="cpu", log=quiet)
    text = cache.read_text()
    cache.write_text(text[: len(text) - 25])                              # cut the second record mid-line
    messages = []
    S.score_with_cache("m", pairs, str(cache), device="cpu", log=messages.append)
    assert any("dropped 1 unreadable" in m for m in messages)
    assert len(fake.scored) == 3                                          # only the damaged record rescored
    assert all(json.loads(line) for line in cache.read_text().splitlines())


def test_plan_scores_each_distinct_pair_once():
    rows = run.load_quartets(QUARTETS, max_rows=200, seed=0)
    pairs, plan = run.build_plan(rows, run.load_templates(TEMPLATES))
    assert len(plan) == 200 and all(set(ids) == set(TERMS) for ids in plan)
    assert len(set(pairs)) == len(pairs) < 8 * 200
    assert run.load_quartets(QUARTETS, max_rows=50, seed=4) == run.load_quartets(QUARTETS, max_rows=50, seed=4)


def test_end_to_end_with_a_fake_scorer(tmp_path, fake):
    out = str(tmp_path / "scores.csv")
    assert run.score("fake/model", QUARTETS, TEMPLATES, out, max_rows=40, seed=1, device="cpu", log=quiet) == 40
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert list(rows[0]) == list(run.load_quartets(QUARTETS, 1)[0]) + run.ADDED_FIELDS
    for r in rows:
        assert abs(float(r["log_ratio"]) - 2 * (float(r["backward_margin"]) - float(r["forward_margin"]))) < 1e-6
        for t in TERMS:
            candidate, given = t.split("_given_")
            assert abs(fake_logprob(r[f"prompt_{given}"], r[f"cont_{candidate}"]) - float(r[f"lp_{t}"])) < 1e-9
            assert r[f"cont_{candidate}"].startswith(" ") and int(r[f"ntok_{t}"]) >= 1
    summary = run.summarize(rows)
    assert summary["overall"]["n"] == 40 and sum(t["n"] for t in summary["by_relation"].values()) == 40


def test_summary_on_a_hand_made_file(tmp_path):
    path = tmp_path / "s.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(run.SCORE_COLUMNS)
        w.writerows([("m", "wikidata", "capital", -4, 3, 1, 1.0, 4, 2, 2), ("m", "wikidata", "capital", 2, 1, 2, 0.67, 4, 2, 2),
                     ("m", "wikidata", "author", 0.5, 0, 0.25, 2.0, 1, 0, 1), ("m", "wordnet", "hypernym", -1, 1, 0.5, 0.67, 3, 2, 1)])
    s = run.summarize(run.read_csv(str(path), run.SCORE_COLUMNS, "scores"))["overall"]
    assert s["n"] == 4 and s["median_abs_LR"] == pytest.approx(1.5)        # median of 4, 2, 0.5, 1
    assert s["share_within_1_nat"] == pytest.approx(0.25) and s["share_LR_negative"] == pytest.approx(0.5)
    assert s["pairwise_accuracy_forward"] == pytest.approx(0.75)


def test_bad_inputs_give_clear_errors(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        run.load_quartets(str(tmp_path / "missing.csv"))
    (tmp_path / "nocol.csv").write_text("source,relation_id\nwikidata,P36\n")
    with pytest.raises(ValueError, match="lacks columns"):
        run.load_quartets(str(tmp_path / "nocol.csv"))
    header = "source,relation_id,relation_name,s1_label,o1_label,s2_label,o2_label"
    (tmp_path / "empty.csv").write_text(header + "\n")
    with pytest.raises(ValueError, match="no rows"):
        run.load_quartets(str(tmp_path / "empty.csv"))
    (tmp_path / "unknown.csv").write_text(header + "\nwikidata,P9999,x,A,B,C,D\n")
    with pytest.raises(ValueError, match="no templates for relation"):
        run.score("m", str(tmp_path / "unknown.csv"), TEMPLATES, str(tmp_path / "o.csv"), device="cpu", log=quiet)
    (tmp_path / "clash.csv").write_text(header + ",log_ratio\nwikidata,P36,capital,A,B,C,D,1\n")
    with pytest.raises(ValueError, match="overwrite"):
        run.score("m", str(tmp_path / "clash.csv"), TEMPLATES, str(tmp_path / "o.csv"), device="cpu", log=quiet)
    with pytest.raises(ValueError, match="output folder"):
        run.score("m", QUARTETS, TEMPLATES, str(tmp_path / "nodir" / "o.csv"), max_rows=2, device="cpu", log=quiet)
