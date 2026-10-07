## Summary

This PR, submitted by **Hina Dixit**, adds **Kiran-10M**, a proprietary memory-augmented question-answering pipeline with a frozen **4B reader**, for BABILong leaderboard review.

| Nominal input length | QA1 | QA2 | QA3 | QA4 | QA5 | QA1–QA5 mean |
|---|---:|---:|---:|---:|---:|---:|
| 0k | 100.0% | 98.0% | 70.0% | 80.0% | 80.0% | **85.6%** |
| 1M | 100.0% | 98.0% | 69.0% | 78.0% | 77.0% | **84.4%** |
| 10M | 100.0% | 99.0% | 69.0% | 77.0% | 75.0% | **84.0%** |

The primary score is the mean over QA1–QA5, with 100 observations per task and length. The submission includes all 125 primary cells: QA1–QA10 at 0k, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k, 256k, 512k and 1M; QA1–QA5 at 10M. The 12,500 exported observations are not asserted to be independent underlying problems across lengths.

## System and evaluation

- Reader: frozen **4B language-model reader**. Base-model identity and exact checkpoint revision are **withheld from public disclosure**.
- Dataset: `RMT-team/babilong` @ `ee0d588794c7ac098062ee0d247c733d62e94fe2`.
- Custom inference pipeline with task-adapted deterministic preprocessing; input-length labels refer to the pipeline, not native reader attention.
- Recorded training condition: no additional gradient-based training or fine-tuning for this evaluated pipeline. This is not a claim of no task-specific engineering or an upstream contamination audit.
- Greedy decoding, at most eight new tokens, thinking disabled where supported, no few-shot prompt examples.
- Unmodified official `compare_answers` metric, pinned in the accompanying metadata. Predictions are the original saved outputs, not newly generated results.

## Evaluation history

Two eval-0k examples per task (QA1–QA10) were inspected for template understanding before the train split arrived; subsequent task-specific rule development used the training split. The exact inspected IDs and their relationship to underlying problems at longer lengths are not established.

An earlier unscored canary processed the first 64 QA1-at-1M cases under the primary and page-retrieval paths before stopping. The completed invocation regenerated those cases. This history accompanies the recorded configuration freeze; we do not describe the evaluation as wholly untouched or as having no prior attempted invocation.

## Files and verification

Predictions: `babilong_evals/hina-dixit/Kiran-10M/` — one configuration, `cfgv1_yes`.

Documentation, full tables, configuration, sample mappings, hashes and offline score verifier: `submissions/hina-dixit/kiran-10m/`.

From the repository root:

```bash
python3 submissions/hina-dixit/kiran-10m/verify.py
```

The verifier requires Python 3.10+ only and makes no model or network calls. Saved-score verification is not independent reproduction of proprietary inference. The detailed protocol describes provenance limits. A separate same-reader page-retrieval control supplement is available as supporting material and is not pooled into the primary score.

## Review request and implementation boundary

Base-model identity, exact checkpoint revision, proprietary implementation details, inference code, prompts and weights are not released. Please review this as a task-adapted memory-pipeline result and advise on its appropriate classification and any additional verification requirements under these disclosed conditions and this base-model nondisclosure boundary. Any additional access arrangement would require separate agreement. The directory placement is proposed for this PR and can be adjusted to the maintainers' convention without changing prediction contents.

Submitted by Hina Dixit.
