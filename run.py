"""Score a language model on the quartets, or summarise a scores file.

    python run.py score --model Qwen/Qwen3.5-2B --out scores.csv [--max-rows 2000]
    python run.py summarize scores.csv

`score` reads the quartets and templates, builds the eight prompts of every
quartet (prompts.py), scores each distinct prompt once (scorer.py), computes the
certificate statistics (certificate.py), writes one row per quartet and prints
a summary. Progress goes to stderr, results to stdout.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import statistics
import sys
from collections import defaultdict

from certificate import TERMS, derive
from prompts import term_pairs
from scorer import score_with_cache

HERE = os.path.dirname(os.path.abspath(__file__))
QUARTET_COLUMNS = ("source", "relation_id", "relation_name", "s1_label", "o1_label", "s2_label", "o2_label")
SCORE_COLUMNS = ("model", "source", "relation_name", "log_ratio", "forward_margin", "backward_margin",
                 "relative_incoherence", "knowledge_score", "forward_correct", "backward_correct")
ADDED_FIELDS = (["model", "prompt_s1", "prompt_s2", "prompt_o1", "prompt_o2", "cont_o1", "cont_o2", "cont_s1", "cont_s2"]
                + [f"lp_{t}" for t in TERMS] + [f"ntok_{t}" for t in TERMS]
                + ["log_ratio", "abs_log_ratio", "forward_margin", "backward_margin",
                   "relative_incoherence", "knowledge_score", "forward_correct", "backward_correct"])


# ----------------------------------------------------------------------------- reading inputs

def read_csv(path: str, required: tuple[str, ...], what: str) -> list[dict]:
    """All rows of a CSV (plain or .gz), after checking it exists, has the columns and has rows."""
    if not os.path.exists(path):
        raise ValueError(f"{what} file not found: {path}")
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{what} file {path} lacks columns {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{what} file {path} has no rows")
    return rows


def load_quartets(path: str, max_rows: int = 0, seed: int = 13) -> list[dict]:
    """Quartet rows, or a seeded random sample of max_rows of them (0 means all)."""
    rows = read_csv(path, QUARTET_COLUMNS, "quartets")
    if max_rows and len(rows) > max_rows:
        rows = random.Random(seed).sample(rows, max_rows)
    return rows


def load_templates(path: str) -> dict[tuple[str, str], dict]:
    """templates.json -> {(source, relation_id): {"object_final": ..., "subject_final": ...}}."""
    if not os.path.exists(path):
        raise ValueError(f"templates file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return {tuple(key.split(":", 1)): entry for key, entry in json.load(f).items()}


# ----------------------------------------------------------------------------- scoring

def build_plan(rows: list[dict], templates: dict) -> tuple[list[tuple[str, str]], list[dict[str, int]]]:
    """The distinct (prompt, continuation) pairs, and for each quartet its term -> pair index.
    Quartets that share an entity share prompts, so each pair is scored once."""
    pairs, index, plan = [], {}, []
    for q in rows:
        key = (q["source"], q["relation_id"])
        if key not in templates:
            raise ValueError(f"no templates for relation {key[0]}:{key[1]} ({q['relation_name']})")
        ids = {}
        for term, pair in term_pairs(templates[key], q).items():
            if pair not in index:
                index[pair] = len(pairs)
                pairs.append(pair)
            ids[term] = index[pair]
        plan.append(ids)
    return pairs, plan


def score(model_id: str, quartets_path: str, templates_path: str, out_path: str,
          max_rows: int = 0, seed: int = 13, cache_path: str | None = None,
          device: str | None = None, dtype: str = "bf16", batch_size: int = 32, log=print) -> int:
    """Score `model_id` on the quartets and write one row per quartet to `out_path`."""
    rows = load_quartets(quartets_path, max_rows, seed)
    pairs, plan = build_plan(rows, load_templates(templates_path))
    clash = [c for c in rows[0] if c in ADDED_FIELDS]
    if clash:
        raise ValueError(f"quartets file has columns the output would overwrite: {clash}")
    if not os.path.isdir(os.path.dirname(os.path.abspath(out_path))):
        raise ValueError(f"output folder does not exist: {os.path.dirname(os.path.abspath(out_path))}")
    log(f"[plan] {len(rows):,} quartets, {len(pairs):,} distinct (prompt, continuation) pairs")
    scores = score_with_cache(model_id, pairs, cache_path or out_path + ".cache.jsonl",
                              device=device, dtype=dtype, batch_size=batch_size, log=log)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ADDED_FIELDS)
        w.writeheader()
        for q, ids in zip(rows, plan):
            rec = dict(q, model=model_id)
            lp = {}
            for term in TERMS:
                candidate, given = term.split("_given_")
                rec[f"prompt_{given}"], rec[f"cont_{candidate}"] = pairs[ids[term]]
                lp[term] = rec[f"lp_{term}"] = scores[ids[term]]["logprob"]
                rec[f"ntok_{term}"] = scores[ids[term]]["n_tokens"]
            rec.update(derive(lp))
            w.writerow(rec)
    log(f"[write] {len(rows):,} rows -> {out_path}")
    return len(rows)


# ----------------------------------------------------------------------------- summary

def _stats(rows: list[dict]) -> dict:
    lr = [float(r["log_ratio"]) for r in rows]
    K = [int(r["knowledge_score"]) for r in rows]
    mean = statistics.fmean
    return {
        "n": len(rows),
        "mean_abs_LR": mean(abs(x) for x in lr),
        "median_abs_LR": statistics.median(abs(x) for x in lr),
        "mean_LR": mean(lr),
        "share_within_1_nat": mean(abs(x) < 1 for x in lr),
        "share_LR_negative": mean(x < 0 for x in lr),
        "mean_F": mean(float(r["forward_margin"]) for r in rows),
        "mean_B": mean(float(r["backward_margin"]) for r in rows),
        "mean_R": mean(float(r["relative_incoherence"]) for r in rows),
        "mean_K": mean(K),
        "share_fully_known": mean(k == 4 for k in K),
        "pairwise_accuracy_forward": mean(int(r["forward_correct"]) / 2 for r in rows),
        "pairwise_accuracy_backward": mean(int(r["backward_correct"]) / 2 for r in rows),
    }


def summarize(rows: list[dict]) -> dict:
    by_relation = defaultdict(list)
    for r in rows:
        by_relation[f"{r['source']}:{r['relation_name']}"].append(r)
    ordered = sorted(by_relation.items(), key=lambda kv: -len(kv[1]))
    return {"model": rows[0]["model"], "overall": _stats(rows), "by_relation": {k: _stats(v) for k, v in ordered}}


def print_summary(summary: dict, out=sys.stdout) -> None:
    s = summary["overall"]
    print(f"model {summary['model']}   quartets {s['n']:,}", file=out)
    print(f"mean |LR| {s['mean_abs_LR']:.3f} nats   median {s['median_abs_LR']:.3f}   mean LR {s['mean_LR']:+.3f}   "
          f"within 1 nat {s['share_within_1_nat']:.1%}   LR < 0 (forward stronger) {s['share_LR_negative']:.1%}", file=out)
    print(f"mean F {s['mean_F']:.3f}   mean B {s['mean_B']:.3f}   mean R {s['mean_R']:.3f}   "
          f"mean K {s['mean_K']:.2f}   fully known {s['share_fully_known']:.1%}", file=out)
    print(f"pairwise accuracy forward {s['pairwise_accuracy_forward']:.3f}   "
          f"backward {s['pairwise_accuracy_backward']:.3f}   (chance 0.5)", file=out)
    print(f"\n{'relation':34s} {'n':>7s} {'|LR|':>7s} {'LR':>7s} {'R':>6s} {'K':>5s} {'fwd':>6s} {'bwd':>6s}", file=out)
    for rel, t in summary["by_relation"].items():
        print(f"{rel:34s} {t['n']:7,d} {t['mean_abs_LR']:7.2f} {t['mean_LR']:+7.2f} {t['mean_R']:6.3f} "
              f"{t['mean_K']:5.2f} {t['pairwise_accuracy_forward']:6.3f} {t['pairwise_accuracy_backward']:6.3f}", file=out)


# ----------------------------------------------------------------------------- command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("score", help="score a model on the quartets and write a scores CSV")
    p.add_argument("--model", required=True, help="Hugging Face model id or local path")
    p.add_argument("--out", required=True, help="scores CSV to write")
    p.add_argument("--quartets", default=os.path.join(HERE, "quartets.csv.gz"))
    p.add_argument("--templates", default=os.path.join(HERE, "templates.json"))
    p.add_argument("--max-rows", type=int, default=0, help="score a seeded random sample of this size (0 = all)")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--cache", default=None, help="cache of scored pairs (default <out>.cache.jsonl)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="used on cuda only, fp32 elsewhere")
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"], help="default: the first available")

    p = sub.add_parser("summarize", help="summary statistics of a scores CSV")
    p.add_argument("scores")

    a = ap.parse_args(argv)
    try:
        if a.cmd == "score":
            score(a.model, a.quartets, a.templates, a.out, max_rows=a.max_rows, seed=a.seed, cache_path=a.cache,
                  device=a.device, dtype=a.dtype, batch_size=a.batch_size,
                  log=lambda m: print(m, file=sys.stderr, flush=True))
            a.scores = a.out
        print_summary(summarize(read_csv(a.scores, SCORE_COLUMNS, "scores")))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
