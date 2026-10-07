# Complete submitted results

All numeric cells are percentages under the pinned official BABILong metric. The last row is the arithmetic mean of QA1–QA5 at a given length, not the mean of all ten tasks. An em dash is an unmeasured cell, not a zero or an interpolated result.

| Task | 0k | 1k | 2k | 4k | 8k | 16k | 32k | 64k | 128k | 256k | 512k | 1M | 10M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qa1 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| qa2 | 98.0 | 98.0 | 98.0 | 98.0 | 98.0 | 98.0 | 98.0 | 98.0 | 98.0 | 98.0 | 98.0 | 98.0 | 99.0 |
| qa3 | 70.0 | 77.0 | 68.0 | 68.0 | 68.0 | 68.0 | 68.0 | 68.0 | 68.0 | 69.0 | 69.0 | 69.0 | 69.0 |
| qa4 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 | 78.0 | 77.0 |
| qa5 | 80.0 | 77.0 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 | 80.0 | 79.0 | 80.0 | 77.0 | 75.0 |
| qa6 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | — |
| qa7 | 54.0 | 54.0 | 54.0 | 54.0 | 54.0 | 54.0 | 54.0 | 54.0 | 54.0 | 54.0 | 54.0 | 54.0 | — |
| qa8 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | — |
| qa9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | — |
| qa10 | 99.0 | 99.0 | 99.0 | 99.0 | 99.0 | 99.0 | 99.0 | 99.0 | 99.0 | 99.0 | 99.0 | 98.0 | — |
| **QA1–QA5 mean** | **85.6** | **86.4** | **85.2** | **85.2** | **85.2** | **85.2** | **85.2** | **85.2** | **85.2** | **85.2** | **85.4** | **84.4** | **84.0** |

Every submitted primary task/length cell contains 100 observations. Coverage is QA1–QA10 at the twelve lengths 0k through 1M and QA1–QA5 at 10M: 125 cells, 12,500 observations.

`manifest.json` contains exact numerators, denominators, and prediction-file hashes. `RESULTS.csv` contains this same table in machine-readable form. No summary CSV belongs in the prediction directory.

See `EVALUATION_PROTOCOL.md` for training, development exposure, canary history, and verification scope.
