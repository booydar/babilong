# Sapience (Sonnet-4.5 reader) — BABILong submission

## System

Sapience is a memory architecture, not a long-context model. The full-length BABILong
input is ingested into a structured memory store at write time; at question time a
retrieval step assembles a small evidence set (~600–1,000 reader-input tokens at every
nominal scale, 128k through 10M), and the reader LLM answers from that evidence only.
The reader never sees the raw context.

- Reader: `claude-sonnet-4-5` (Anthropic API), temperature 0, max_tokens 2000.
  We name the reader per system-name policy: retrieval-pipeline results are
  reader-dependent, so the row is "Sapience (Sonnet-4.5 reader)", not "Sapience".
- Write gate / extractor: a closed-vocabulary bAbI event extractor (regex over the
  task's entity/location/object vocabulary) that keeps task-relevant event sentences
  and discards the PG-19 background text at ingest. qa1–qa3 use the same extractor;
  qa4/qa5 use a multitask variant (new mechanism, first pass).
- Retrieval: task-adapted hybrid retrieval over the stored events ("Fix-B move-aware"
  configuration for qa1–qa3; analogous per-task retrieval for qa4/qa5).
- Prompting: per-task instruction + few-shot examples + answer-format post-prompt,
  with the retrieved evidence in the `<context>` slot (see per-cell `.json` configs).

## What this result is, and is not (please read before comparing rows)

1. **Task-adapted pipeline.** The ingest extractor is specialized to bAbI's closed
   vocabulary. These numbers demonstrate that a write-gated memory architecture keeps
   the reader's working set bounded while nominal context grows 100x–2,500x; they are
   NOT a general-purpose long-context reading result and should not be read as one.
2. **Bounded-working-set, not haystack-survival.** The flat curve comes from the write
   gate: the reader composes over ~670–1,000 tokens at every scale. It shows working-set
   invariance under growing distractor volume, not attention robustness over a raw
   10M-token window.
3. **Naive-solvable disclosure (per-task).** With a 100%-recall extractor, deterministic
   rule solvers on the extracted events already reach ~100% on qa1, qa4, qa5
   (measured on the 0k needle sets: qa1 100/100, qa4 100/100, qa5 99/100). Those cells
   therefore measure extraction + retrieval robustness over a ~40 MB haystack at 10M
   tokens, not reasoning headroom. qa2 (naive recency floor 32–34%) and qa3
   (naive floor well below the score; two-hop temporal) are the reasoning-bearing cells.
4. **qa4/qa5 = new mechanism, first pass.** Built to the 2026-08-06 pre-registration's
   arms B/C; not evidence that the qa3 mechanism generalizes beyond this family.
5. **Seeds.** Three seeds (42/123/456) were run for every 10M cell (seed = pipeline
   stochasticity; items fixed). qa1/qa2/qa4/qa5 are seed-invariant at 10M (identical
   scores; the pipeline is largely deterministic and the reader is temperature 0 —
   pseudoreplicates, disclosed as such). qa3@10M varies 86/87/90 across seeds; the
   canonical CSV is seed 42 (the LOWEST of the three, i.e. conservative), with the
   other two seeds' CSVs included under `seeds_supplementary/`.
6. **qa3 128k cell history (full disclosure).** The as-banked 2026-06-09 rows
   (idx 0–44) contained 4 infra-failed items (SQLITE_IOERR under concurrent DB
   writes; empty reader output); with those included, the pooled cell scored 82.0
   under BOTH the official metric and our internal judges. (Our earlier locked
   internal value of 86.0 came from a prior band-pooled measurement of a different
   row vintage; the 82-vs-86 delta was dominated by those 4 infra-failed rows plus
   run-to-run variation on idx 45–99 — it was never a judge-vs-metric divergence.)
   Per our pre-declared FAILED-INFRA policy, exactly those 4 items (idx 1/2/4/7)
   were re-run 2026-08-25 under the frozen config with serialized DB writes; all 4
   scored correct, and the shipped CSV scores **86.0 under both the official metric
   and the internal judges**. The numerical agreement with the old band-pooled 86.0
   is coincidental composition; only the shipped CSV's official-metric score is
   claimed. On every other cell, official metric and dual judges agree on the
   shipped rows; the leaderboard-facing number is always the official metric.

## Data provenance

- 10M cells: official HF artifacts `RMT-team/babilong` `data/qa{1,2,3,4,5}/10M.json`,
  revision-pinned `ee0d588794c7ac098062ee0d247c733d62e94fe2`, file sha256 + per-item
  fingerprints recorded in the source runs' `run_meta.json`. n=100 = the full official
  10M split per task (idx 0–99).
- Sub-10M qa3 cells: official HF configs `128k`, `256k`, `512k`, `1M` (n=100, idx 0–99).
- No cell in this package was ever used for calibration, threshold fitting, or prompt
  tuning. (A 2026-08-06 internal pre-registration reserved idx 50–99 at 128K/1M as a
  forward holdout for a hypothetical tuned pass; no tuned pass was ever run, and no
  parameter was ever fit on any item in this package. Note: publishing the item-level
  outputs in this package CONSUMES that forward holdout — idx 50–99 at 128K/1M can no
  longer serve as a held-out set for any future tuned pass.)

## Results (official metric, canonical seed-42 CSVs in this package)

| Task | 128k | 256k | 512k | 1M | 10M |
|---|---|---|---|---|---|
| qa1 | — | — | — | — | 100.0 |
| qa2 | — | — | — | — | 100.0 |
| qa3 | 86.0 | 87.0 | 87.0 | 84.0 | 86.0 |
| qa4 | — | — | — | — | 100.0 |
| qa5 | — | — | — | — | 100.0 |

avg(qa1–5) @ 10M = 97.2 (seed 42; seeds 123/456 differ only on qa3: 87/90 → avg 97.4/98.0).
Current public leaderboard best at 10M for comparison: ARMT (fine-tuned), avg 76.6.

qa3 seeds at 10M: 86.0 / 87.0 / 90.0 (42/123/456), pooled n=300 87.7% Wilson 95%
[83.5, 90.9]. Sub-1M qa3 cells are n=100 single deterministic runs.

Cost note: because the reader input is scale-invariant, the marginal cost of the five
10M cells (3 seeds, 1,500 items) was ~$10 total in reader+judge API spend.

## Harness

Method and evaluation protocol: https://doi.org/10.5281/zenodo.21894568
Harness + adapter source available on request, as are per-cell run metadata
(dataset revision pins, per-item fingerprints, spend, timing).
