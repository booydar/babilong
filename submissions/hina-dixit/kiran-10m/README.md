# Kiran-10M
## BABILong results — frozen 4B reader

**Results submission for leaderboard review; proprietary implementation is not released.**

Kiran-10M is a proprietary memory-augmented question-answering pipeline. It recorded **84.4% at 1M** and **84.0% at 10M** on BABILong QA1–QA5 with a frozen 4B reader. These are system-level results; the nominal input length is not the reader’s native attention window.

| Nominal input length | QA1 | QA2 | QA3 | QA4 | QA5 | QA1–QA5 mean |
|---|---:|---:|---:|---:|---:|---:|
| 0k | 100.0% | 98.0% | 70.0% | 80.0% | 80.0% | **85.6%** |
| 1M | 100.0% | 98.0% | 69.0% | 78.0% | 77.0% | **84.4%** |
| 10M | 100.0% | 99.0% | 69.0% | 77.0% | 75.0% | **84.0%** |

Each primary task/length cell has 100 observations. At 10M, 420 of the 500 QA1–QA5 observations are correct. The observed 1M-to-10M difference is −0.4 percentage points; exact length-invariance is not claimed.

The complete submitted grid is in [RESULTS.md](RESULTS.md) and [RESULTS.csv](RESULTS.csv). Submitted by **Hina Dixit**. Contact: the submitting account in the pull-request thread.

## Evaluated configuration

| Field | Value |
|---|---|
| Reader | Frozen 4B reader; base-model identity withheld |
| Reader revision | Withheld from the public submission |
| Dataset | `RMT-team/babilong` |
| Dataset revision | `ee0d588794c7ac098062ee0d247c733d62e94fe2` |
| Inference | Custom pipeline; task-adapted deterministic preprocessing |
| Training condition recorded | No additional gradient-based training or fine-tuning for this evaluated pipeline |
| Decoding | Greedy; at most eight new tokens; thinking disabled where supported; no few-shot prompt examples |
| Metric | Pinned, unmodified official BABILong `compare_answers` |
| Submitted files | 125 prediction CSVs; 12,500 observations |
| Packaging configuration | `cfgv1_yes` — one configuration in this model directory |

This training statement does not mean the base model was never pretrained, the task family was unknown during development, or upstream contamination was ruled out. Detailed evidence qualifications and machine-readable settings are in [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) and [evaluation.json](evaluation.json).

## Evaluation-history disclosure

Two eval-0k examples per task (QA1–QA10) were inspected for template understanding before the training split arrived; subsequent task-specific rule development used the training split. Their exact IDs and overlap with underlying problems at longer lengths are not established. An earlier unscored canary processed the first 64 QA1-at-1M cases under the primary and control paths before stopping; the completed invocation regenerated those cases. No submitted prediction was changed during packaging. See [the complete protocol note](EVALUATION_PROTOCOL.md).

## Verify saved scores — no model required

From the extracted archive root or the repository root containing these files:

```bash
python3 submissions/hina-dixit/kiran-10m/verify.py
```

Python 3.10+ and its standard library are sufficient. The verifier checks the submitted file sets and hashes, every numerator and denominator, sample mappings, and the result table. It does not generate answers or call a network service. It checks only this submission’s two directories, allowing it to run within the upstream repository without treating unrelated repository files as an error.

Prediction files are located at `babilong_evals/hina-dixit/Kiran-10M/`. Summary CSVs remain outside that prediction directory. Scoped `.gitattributes` files preserve artifact bytes when Git would otherwise normalize line endings. Repository-collector testing uses commit `7a6efee29f5cac03c3c410e6799c80fd2ffe3610`; that does not assert identical live Space code or leaderboard acceptance.

## What is and is not released

Base-model identity and exact checkpoint revision are intentionally withheld in this release. The public verifier checks saved outputs and package integrity; it does not authenticate the undisclosed reader. Eligibility under this disclosure boundary is for the benchmark maintainers to determine.

Per-sample predictions, evaluation metadata, record-index mappings, result tables, checksums, the public benchmark metric, and a score verifier are provided. Proprietary inference code, extraction rules, prompt text, memory schemas, routing logic, weights, and internal traces are not included. Rescoring verifies the saved scores, not the proprietary generation process. Any additional reproduction arrangement requires separate agreement.

No native-10M-attention, general-purpose-memory, cost, latency, hardware-efficiency, world-record, patent-status, or future-publication claim is made. This packet requests review; it does not assert that a leaderboard entry has been accepted or listed.
