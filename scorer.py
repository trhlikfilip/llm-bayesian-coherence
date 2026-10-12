"""Log-probability of each continuation given its prompt, with a cache.

The score of (prompt, continuation) is the sum, over the continuation's tokens,
of log p(token | everything before it). Prompt and continuation are tokenised
together, so the space between them is handled the way the model sees it, and
the continuation's tokens are found from character offsets. Nothing after the
continuation is scored.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import tempfile


class Scorer:
    """A Hugging Face model wrapped for teacher-forced scoring."""

    def __init__(self, model_id: str, device: str, dtype: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.device = torch, device
        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
        except (OSError, ValueError) as e:
            raise ValueError(f"cannot load model {model_id!r}: {str(e).splitlines()[0]}") from None
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(device).eval()
        
    def encode(self, prompt: str, continuation: str) -> tuple[list[int], int]:
        """Token ids through the continuation, and its first token index."""
        if not prompt or not continuation.strip():
            raise ValueError(
                f"empty prompt or continuation: {prompt!r} + {continuation!r}"
            )

        enc = self.tokenizer(
            prompt + continuation,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            add_special_tokens=True,
        )
        ids = list(enc["input_ids"])
        offsets = list(enc["offset_mapping"])
        masks = list(enc["special_tokens_mask"])
        boundary = len(prompt)

        positions = [
            i for i, ((a, b), special) in enumerate(zip(offsets, masks))
            if not special and a >= boundary
        ]

        if (
            not positions
            or positions[0] < 1
            or any(a < boundary < b for a, b in offsets)
        ):
            raise ValueError(
                f"cannot separate the continuation tokens of {prompt!r} + {continuation!r}"
            )

        start, stop = positions[0], positions[-1] + 1
        if positions != list(range(start, stop)):
            raise ValueError(
                f"non-text token inside continuation: {prompt!r} + {continuation!r}"
            )

        return ids[:stop], start

    def score(self, pairs: list[tuple[str, str]], batch_size: int) -> list[dict]:
        """[{"logprob": float, "n_tokens": int}] for each pair, in the input order."""
        torch = self.torch
        encoded = [self.encode(p, c) for p, c in pairs]
        order = sorted(range(len(pairs)), key=lambda i: len(encoded[i][0]))   # similar lengths per batch
        out = [None] * len(pairs)
        with torch.no_grad():
            for b in range(0, len(order), batch_size):
                idx = order[b:b + batch_size]
                seqs = [encoded[i][0] for i in idx]
                L = max(len(s) for s in seqs)
                input_ids = torch.full((len(seqs), L), self.tokenizer.pad_token_id, dtype=torch.long)
                attention = torch.zeros((len(seqs), L), dtype=torch.long)
                for k, s in enumerate(seqs):                   # pad on the right
                    input_ids[k, :len(s)] = torch.tensor(s)
                    attention[k, :len(s)] = 1
                logits = self.model(input_ids=input_ids.to(self.device),
                                    attention_mask=attention.to(self.device)).logits
                logprobs = torch.log_softmax(logits.float(), dim=-1)
                for k, i in enumerate(idx):
                    ids, start = encoded[i]
                    # the output at position pos - 1 is the distribution of the token at pos
                    total = sum(logprobs[k, pos - 1, ids[pos]].item() for pos in range(start, len(ids)))
                    out[i] = {"logprob": total, "n_tokens": len(ids) - start}
        return out


def load_cache(path: str, log=print) -> dict[str, dict]:
    """Load cached scores, repairing unreadable records and a missing final newline."""
    if not os.path.exists(path):
        return {}

    cache, good, bad = {}, [], 0
    needs_rewrite = False

    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                cache[rec["key"]] = rec

                if not line.endswith("\n"):
                    needs_rewrite = True
                    line += "\n"

                good.append(line)
            except (json.JSONDecodeError, KeyError, TypeError):
                bad += 1
                needs_rewrite = True

    if bad:
        log(f"[cache] dropped {bad} unreadable line(s) from {path}")

    if needs_rewrite:
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=os.path.dirname(os.path.abspath(path)),
                prefix=".cache-repair-",
                delete=False,
            ) as f:
                temporary_path = f.name
                f.writelines(good)

            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    return cache


def score_with_cache(model_id: str, pairs: list[tuple[str, str]], cache_path: str,
                     device: str | None = None, dtype: str = "bf16", batch_size: int = 32,
                     log=print) -> list[dict]:
    """{"logprob", "n_tokens"} for every pair, scoring only what the cache does not hold."""
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device != "cuda":
        dtype = "fp32"                                          # half precision only on cuda
    cache = load_cache(cache_path, log)
    keys = [
        hashlib.sha1(
            f"label-prefix-v2\x1f{model_id}\x1f{dtype}\x1f{p}\x1f{c}".encode()
        ).hexdigest()
        for p, c in pairs
    ]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    if not todo:
        log(f"[score] all {len(pairs):,} pairs already in {cache_path}")
    else:
        log(f"[score] loading {model_id} on {device} ({dtype}), {len(todo):,} pairs to score, "
            f"{len(pairs) - len(todo):,} cached")
        scorer = Scorer(model_id, device, dtype)
        t0 = time.time()
        chunk = max(batch_size * 8, 256)
        with open(cache_path, "a", encoding="utf-8") as fh:
            for c0 in range(0, len(todo), chunk):
                sub = todo[c0:c0 + chunk]
                for i, res in zip(sub, scorer.score([pairs[i] for i in sub], batch_size)):
                    rec = {"key": keys[i], "prompt": pairs[i][0], "continuation": pairs[i][1], **res}
                    cache[keys[i]] = rec
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done = c0 + len(sub)
                log(f"[score] {done:,}/{len(todo):,} pairs ({done / max(time.time() - t0, 1e-9):.1f}/s)")
    return [{"logprob": cache[k]["logprob"], "n_tokens": cache[k]["n_tokens"]} for k in keys]
