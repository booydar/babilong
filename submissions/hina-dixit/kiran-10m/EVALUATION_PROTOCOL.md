# Evaluation protocol and provenance scope

**Kiran-10M | Submitted by Hina Dixit | September 2026 campaign | Disclosure revision assembled 2026-10-07**

## Identity and metric

Reader: frozen 4B reader, reported dtype bfloat16. Base-model identity and exact checkpoint revision are intentionally withheld from this public submission; they remain in private execution records. Dataset: `RMT-team/babilong` at snapshot `ee0d588794c7ac098062ee0d247c733d62e94fe2`, using the campaign's evaluation files. The inference pipeline is custom; only the benchmark's scoring metric is represented as unmodified official code.

Scorer: `compare_answers`, `booydar/babilong` commit `7a6efee29f5cac03c3c410e6799c80fd2ffe3610`, SHA-256 `f77c139809690588f85ed29c1c93f22394c6b48b1fdc8afeaaa5d0969d973daf`. The scorer shipped in this submission matches the metric hash recorded for the historical campaign. No corrected or more permissive scoring rule is substituted.

Recorded generation settings: greedy decoding (`do_sample=False`), `max_new_tokens=8`, chat-template generation prompt, thinking disabled where supported by the template, and no few-shot examples in the answering prompt. The maximum of eight is an output-token limit, not a claim about the number of evidence records. Exact proprietary prompt text is withheld.

The evaluated system uses task-adapted deterministic preprocessing around a frozen reader. This document identifies the evaluation class, not the proprietary construction procedure.

## Training condition

The campaign was recorded as using frozen components without additional gradient-based training or fine-tuning for the evaluated pipeline. Task-adapted deterministic preprocessing was developed with knowledge of the benchmark family; this is not a claim of no task-specific engineering.

A later source inspection found no training path or adapter-loading code in the examined frozen scripts, and no adapter files in the examined model directory. The source investigator reports that reader weight hashes matched the publisher-side acquisition records. An explicit contemporaneous adapter inventory and a comprehensive training-history audit were not recorded. Upstream pretraining contamination is unknown. These are supporting provenance observations, not an independent runtime attestation.

## Evaluation history

Before the training split was available, two evaluation examples at 0k from each of QA1–QA10 were inspected for template understanding. Subsequent task-specific rule development used the training split. The exact inspected sample IDs and their relationship to underlying problems at other lengths are not established by the available provenance. We therefore do not describe this evaluation as wholly untouched or assert that the longer-length problems were certainly unaffected by that inspection.

An earlier invocation stopped during an unscored canary after processing the first 64 QA1-at-1M cases under the primary and page-retrieval control paths. Those outputs were hash-filed, not scored, and are not the submitted predictions. The completed E1b invocation generated those cases again. The campaign records a configuration freeze before evaluation-length runs; the earlier attempted invocation is disclosed rather than described as “no reruns.” Completed-run canary files are also excluded from the scored prediction set.

No prediction is removed, corrected, or regenerated during submission packaging. No guessed sample exclusions are applied. These recorded conditions are submitted for maintainer review; no eligibility waiver or acceptance is assumed.

## Verification scope and limitations

The accompanying verifier recalculates scores from the saved predictions, checks their byte hashes, checks row mappings, and checks the complete submitted task/length grid. The preparation review also reconciled these predictions with the supplied original-run inventory. This is score and artifact verification, not verification of the undisclosed reader identity, independent regeneration of answers, or reconstruction of proprietary inference.

The campaign inventory records a dataset-file hash and zero-based sample indices for each cell. Those file hashes refer to the campaign-local evaluation JSON files; they are not asserted to be the upstream repository's raw file digests. Per-sample full-context hashes were not recorded. The question/target/output record hashes in `sample_indices.jsonl` do not hash the full input context.

The source investigator's later static read found the evidence path using input and question, with reference targets copied into output records for subsequent scoring. Targets existed in the same in-memory sample object; there is no independent runtime-isolation attestation. The launcher files identifying the reader directory were not hash-pinned by the original run receipts. Environment versions were reconstructed later rather than recorded contemporaneously, so no contemporaneous environment-version claim is made here.

Counts at multiple lengths are exported observations, not a claim that all underlying problems across lengths are independent. The nominal 10M length is the pipeline input label, not a native reader attention-window claim. No general-purpose memory, cost, latency, hardware-efficiency, or highest-score claim is made by this submission.

## Coverage and selection

This primary entry contains the complete 125-cell slice for the campaign-designated primary reader/configuration: QA1–QA10 through 1M, and QA1–QA5 at 10M, all with 100 observations per cell. Other readers and control configurations are not mixed into this entry. A separate 121-cell same-reader page-retrieval supplement has been prepared, but it does not enter this score or request a second ranked entry.

One packaging configuration identifier is used for all cells in this directory. Filenames have been made collector-compatible; their CSV contents have not been rewritten. The packaging identifier is not an internal method description. No per-task best-of-configuration selection is used for this submitted system.

## Public references

- Benchmark and submission route: https://github.com/booydar/babilong
- Dataset: https://huggingface.co/datasets/RMT-team/babilong
- Pinned metric: https://github.com/booydar/babilong/blob/7a6efee29f5cac03c3c410e6799c80fd2ffe3610/babilong/metrics.py
- Pinned collector: https://github.com/booydar/babilong/blob/7a6efee29f5cac03c3c410e6799c80fd2ffe3610/babilong/collect_results.py
