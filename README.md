# Coherence certificates for language models

Measures whether a causal language model's probabilities about facts could come from one joint distribution. A quartet is two true facts of one relation, (s1, o1) and (s2, o2), for example (Japan, Tokyo) and (Germany, Berlin). The model is asked for each subject given each object and each object given each subject, and the natural logarithm

```
         p(s1|o1) p(o1|s2) p(s2|o2) p(o2|s1)
LR = log -----------------------------------
         p(o1|s1) p(s1|o2) p(o2|s2) p(s2|o1)
```

is 0, in nats, whenever these conditionals come from one joint distribution. The maths section below derives it and the quantities reported with it.

```
run.py             the command line: read inputs, build prompts, score, write the CSV, print a summary
prompts.py         quartet          ->  eight (prompt, continuation) pairs
scorer.py          pairs + model    ->  log-probabilities, with a cache
certificate.py     log-probs        ->  LR, F, B, K, R
test_coherence.py  23 tests, none needs a model download
templates.json     two sentence templates for each of the 25 relations
quartets.csv.gz    140,751 quartets from Wikidata, ConceptNet and WordNet
```

## Install and run

Needs Python 3.10 or newer.

```bash
pip install -r requirements.txt
pytest -q
python run.py score --model Qwen/Qwen3.5-2B --out scores.csv --max-rows 2000
python run.py summarize scores.csv
```

Any Hugging Face causal language model works. Progress goes to stderr, results to stdout. Options of `score`:

- `--max-rows N` scores a seeded random sample of N quartets (`--seed`, default 13). Without it all 140,751 are scored, which is about 603,000 distinct prompt and continuation pairs, so use `--max-rows` unless you have a GPU.
- `--device` is cuda, mps or cpu, by default the first one available. `--dtype` is bf16 (default), fp16 or fp32 and applies on cuda only, everything else runs in float32.
- `--batch-size` defaults to 32. Lower it if the GPU runs out of memory.
- `--quartets` and `--templates` default to the files shipped here and can point to your own files. A quartets file needs `source`, `relation_id`, `relation_name` and the four labels. Its other columns are copied to the output.
- Scores are cached in `<out>.cache.jsonl` (or `--cache PATH`), keyed by model, precision, prompt and continuation, so only pairs not in the cache are scored. An interrupted run resumes where it stopped. A different model or precision scores everything.
- `HF_HUB_OFFLINE=1` loads models from the local cache only, `HF_TOKEN` is needed for gated models.

## Input

`templates.json` gives each relation two sentences, one ending in the object slot and one ending in the subject slot:

```json
"wikidata:P36": {"relation": "capital",
                 "object_final": "The capital of {s} is {o}.",
                 "subject_final": "{o} is the capital of {s}."}
```

`quartets.csv.gz` has one quartet per row and 14 columns:

| column | meaning |
|---|---|
| `source`, `relation_id`, `relation_name` | wikidata, conceptnet or wordnet, and the relation (P36, capital) |
| `s1_id`, `s1_label`, `o1_id`, `o1_label` | the first fact |
| `s2_id`, `s2_label`, `o2_id`, `o2_label` | the second fact |
| `quartet_freq` | how common the rarest of the four entities is. For Wikidata, its number of Wikimedia sitelinks. For ConceptNet and WordNet, the Zipf word frequency of its label |
| `fact1_date`, `fact2_date` | when each fact became true, ISO format, empty if unknown. For ConceptNet and WordNet, the release date of the source |

## Output

One row per quartet: the input columns (14 for the shipped file), then `model`, then

| column | meaning |
|---|---|
| `prompt_s1`, `prompt_s2` | the object-final sentence with s1 or s2 filled in, cut before the object and any "a" or "an" in front of it |
| `prompt_o1`, `prompt_o2` | the subject-final sentence with o1 or o2 filled in, cut before the subject and any "a" or "an" in front of it |
| `cont_o1`, `cont_o2`, `cont_s1`, `cont_s2` | the continuations, with a leading space and with "a" or "an" when the sentence puts one before the slot |
| `lp_<a>_given_<b>` | log-probability in nats of `cont_<a>` after `prompt_<b>`, eight columns |
| `ntok_<a>_given_<b>` | number of tokens scored for that term, eight columns |
| `log_ratio`, `abs_log_ratio` | LR and \|LR\| |
| `forward_margin`, `backward_margin` | F and B, how strongly the true object is preferred given the subject, and the true subject given the object (defined in The maths) |
| `relative_incoherence` | R, from 0 to 1 |
| `knowledge_score` | K, how many of the four comparisons pick the true partner (0 to 4) |
| `forward_correct`, `backward_correct` | the forward and backward parts of K (0 to 2 each) |

The summary is printed after `score` and by `summarize`. It gives the model and the number of quartets, then mean and median |LR|, mean LR, the share within 1 nat of 0, the share with LR < 0, mean F, B, R and K, the share with K = 4 ("fully known"), and pairwise accuracy per direction (chance 0.5). Per relation (source:relation, most quartets first) it gives the count, mean |LR|, mean LR, mean R, mean K and pairwise accuracy per direction.

## The maths

Several of the claims below are checked in `test_coherence.py`: LR = 2(B − F), LR = 0 for conditionals of one joint, what cancels, and the edge cases of K and R.

### The certificate

A model gives two families of conditional probabilities on a quartet: objects given subjects p(o | s) and subjects given objects p(s | o). They are compatible if some joint distribution P(s, o) has both as its conditionals:

```
p(o | s) = P(s, o) / P(s)        p(s | o) = P(s, o) / P(o)
```

The ratio inside LR is the certificate of incompatibility of Emerson (2026).

Compatible conditionals give LR = 0 whenever the four pairs of the quartet have positive probability. Substituting the two formulas above, the numerator is P(s1,o1) P(s2,o1) P(s2,o2) P(s1,o2) divided by P(o1) P(s2) P(o2) P(s1), and the denominator is the same four joint probabilities divided by the same four marginals. So one quartet with LR ≠ 0 proves that no joint distribution produces the model's conditionals.

The converse holds too (Arnold and Press 1989). Take finitely many subjects and objects, let every conditional be positive, and let each family sum to 1 over them. If the ratio is 1 for every choice of two subjects and two objects, fix a reference pair (s0, o0) and set

```
P(s, o)  ∝  p(s | o) p(o | s0) / p(s0 | o).
```

Only p(s | o) depends on s, so the conditional of P over subjects is p(s | o). The ratio being 1 for (s, s0, o, o0) rewrites the same P as proportional to p(o | s) p(s | o0) / p(o0 | s), so its conditional over objects is p(o | s). On the 2×2 table {s1, s2} × {o1, o2} the quartet is the only certificate, up to its reciprocal, so there LR = 0 is exactly the condition for the two families to come from a joint. The model's scores need not sum to 1 over the two candidates, because renormalising them multiplies both candidates of a prompt by the same constant, which does not change the ratio.

### The certificate as a loop

Walk around the four cells of the table, changing the object, then the subject, then the object, then the subject:

```
(s1,o1) → (s1,o2) → (s2,o2) → (s2,o1) → (s1,o1)
```

Give each step the ratio of the conditional at the new cell to the one at the old cell, taken in the direction that changed: p(o2|s1)/p(o1|s1), then p(s2|o2)/p(s1|o2), then p(o1|s2)/p(o2|s2), then p(s1|o1)/p(s2|o1). Their product is the certificate ratio. Under a joint distribution each step ratio equals P(new cell) / P(old cell), for example p(o2|s1)/p(o1|s1) = P(s1,o2)/P(s1,o1), so around a closed loop the product is 1.

### Margins, and LR = 2(B − F)

Each prompt fixes one entity and offers two candidates. The eight terms pair up into four margins, each positive when the true partner wins:

```
F1 = log p(o1|s1) − log p(o2|s1)      "The capital of Japan is"   Tokyo against Berlin
F2 = log p(o2|s2) − log p(o1|s2)      "The capital of Germany is" Berlin against Tokyo
B1 = log p(s1|o1) − log p(s2|o1)      "Tokyo is the capital of"   Japan against Germany
B2 = log p(s2|o2) − log p(s1|o2)      "Berlin is the capital of"  Germany against Japan
```

Reading LR term by term gives B1 + B2 − F1 − F2, so

```
LR = 2 (B − F),      F = (F1 + F2) / 2   (forward),      B = (B1 + B2) / 2   (backward).
```

`derive()` in `certificate.py` computes LR from its definition and stops with an error if this identity fails.

The margins are log odds ratios. exp(2F) = [p(o1|s1)/p(o2|s1)] · [p(o2|s2)/p(o1|s2)] is the odds ratio of the 2×2 table implied by the forward family, exp(2B) the one implied by the backward family, and LR = 0 means they are equal.

LR < 0 means F > B, so the forward direction implies the stronger association. Putting the backward terms in the numerator is a convention. Swapping the two facts (s1 with s2 together with o1 with o2) leaves LR, F, B and K unchanged. Swapping only the subjects, or only the objects, pairs each subject with the wrong object and flips the signs of LR, F and B.

### What cancels

- A constant added to both candidates of one prompt changes nothing, because every margin compares two candidates under the same prompt.
- A constant attached to one candidate in both of its prompts (a frequency prior, or a penalty that depends only on the label) cancels in LR, F and B, because in each direction the candidate appears once on each side of its two margins. It moves those two margins in opposite directions, though, so it can change the sign of a single margin and with it K and pairwise accuracy. A model whose scores depend only on the candidate gets LR = F = B = 0 and, unless two candidates tie, K = 2.
- Evidence that depends on both the prompt and the candidate does not cancel, and it is summed over every token of the continuation. The output records the token count of every term (`ntok_*`).

### Why two sentence forms

If both families were read from one sentence template T(s, o), p(o | s) would be P(T(s, o)) normalised over the two objects and p(s | o) the same number normalised over the two subjects. Both would come from one positive 2×2 table, and LR would be 0 for every model. So p(o | s) is read from a sentence ending in the object ("The capital of Japan is" + " Tokyo") and p(s | o) from a sentence ending in the subject ("Tokyo is the capital of" + " Japan"). A non-zero LR then means the two phrasings of the relation do not come from one joint distribution. Disagreement between the two phrasings is part of what is measured.

### K, R and pairwise accuracy

- K counts the positive margins among F1, F2, B1, B2, from 0 to 4. A tie counts as wrong. It measures a choice between the two candidates of the quartet, and chance is 2.
- R = |LR| / (2|F| + 2|B|) = |B − F| / (|F| + |B|) lies between 0 and 1. LR is the difference of the two log odds ratios 2B and 2F, and for margins of these sizes |LR| is largest, at 2|F| + 2|B|, when they have opposite signs. So R is |LR| as a share of the largest value it could take. R = 1 exactly when F and B have opposite signs or exactly one of them is zero, and R = 0 exactly when F = B. It is defined as 0 when F = B = 0, and it is unstable when both margins are near zero.
- Pairwise accuracy is the share of positive margins in one direction: forward over F1 and F2, backward over B1 and B2, chance 0.5.

### Scoring conventions

- Each conditional is the sum of the log-probabilities of all tokens of the continuation (a leading space, the article if the template has one before the slot, and the label), given the prompt, in nats. The full stop after the slot is not scored, so the event is "the text continues with this label".
- Prompts are plain text, without a chat template. Prompt and continuation are tokenised together and the continuation's tokens are found from character offsets. A pair where one token covers both prompt and continuation is refused.
- "a" or "an" before a slot follows the first letter of the label, so "an university" can occur. Labels are never changed.
- Model probabilities are positive, so every conditional exists. A non-finite log-probability stops the run.
- The two candidates of a prompt are separate events except when one label begins with the other, for example Guinea and Guinea-Bissau.
- `--dtype bf16` or `fp16` (CUDA only) changes scores slightly compared with float32, so compare models at the same precision.
- In float32 the batch size and the composition of each batch change scores slightly through padding. Runs with the same batch size and the same pairs give identical results.

## References

- B. C. Arnold and S. J. Press (1989). Compatible conditional distributions. Journal of the American Statistical Association 84(405), 152–156.
- G. Emerson (2026). On the complexity of the compatibility problem for succinctly encoded conditional distributions. arXiv 2608.31120.
