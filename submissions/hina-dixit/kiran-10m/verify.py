#!/usr/bin/env python3
"""Offline integrity and score verification. Python 3.10+; no model or network calls.

This script only verifies released artifacts and recomputes scores. It does not
reproduce the proprietary inference system or authenticate its original execution.
New verification helper: licensed under Apache-2.0; see THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys

sys.dont_write_bytecode = True
METRIC_SHA256 = 'f77c139809690588f85ed29c1c93f22394c6b48b1fdc8afeaaa5d0969d973daf'
LENGTHS = ['0k','1k','2k','4k','8k','16k','32k','64k','128k','256k','512k','1M','10M']
TASKS = [f'qa{i}' for i in range(1,11)]
DOCS_REL = 'submissions/hina-dixit/kiran-10m'
PREDS_REL = 'babilong_evals/hina-dixit/Kiran-10M'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def safe_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and '..' not in relative.parts and '\\' not in name,
            f'Unsafe relative path: {name}')
    result = root.joinpath(*relative.parts)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), f'Linked path component: {name}')
    require(result.resolve().is_relative_to(root.resolve()), f'Path escapes root: {name}')
    return result


def row_hash(row: dict[str, str]) -> str:
    raw = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def verify(root: Path) -> dict:
    root = root.resolve()
    meta = root / DOCS_REL
    checksum_path = meta / 'SHA256SUMS'
    require(checksum_path.is_file(), 'Missing SHA256SUMS')
    expected = {}
    for line in checksum_path.read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        require(name not in expected, f'Duplicate checksum path: {name}')
        require(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'Bad SHA-256')
        require(name.startswith(DOCS_REL + '/') or name.startswith(PREDS_REL + '/'), 'Checksum path outside submission')
        path = safe_path(root, name)
        require(path.is_file() and not path.is_symlink(), f'Missing or linked file: {name}')
        require(sha256(path) == digest, f'Checksum mismatch: {name}')
        expected[name] = digest
    found = set()
    for subtree in [root / DOCS_REL, root / PREDS_REL]:
        require(subtree.is_dir() and not subtree.is_symlink(), 'Submission subtree missing or linked')
        for path in subtree.rglob('*'):
            require(not path.is_symlink(), f'Symlink is not permitted: {path.name}')
            if path.is_file():
                found.add(path.relative_to(root).as_posix())
    require(found == set(expected) | {(Path(DOCS_REL) / 'SHA256SUMS').as_posix()},
            'File set differs from SHA256SUMS (remove extra files, including OS metadata, before sharing)')

    manifest = json.loads((meta/'manifest.json').read_text(encoding='utf-8'))
    require(manifest['schema_version'] == 1, 'Unsupported manifest schema')
    metric_path = meta/'third_party'/'babilong_metrics.py'
    require(sha256(metric_path) == METRIC_SHA256, 'Official metric is not the pinned version')
    require(manifest['metric_sha256'] == METRIC_SHA256, 'Manifest metric pin mismatch')
    spec = importlib.util.spec_from_file_location('pinned_babilong_metric', metric_path)
    require(spec is not None and spec.loader is not None, 'Cannot load pinned metric')
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    require(manifest['prediction_root'] == PREDS_REL, 'Prediction-root mismatch')
    prediction_root = safe_path(root, manifest['prediction_root'])
    records = {}
    results = []
    filenames = set()
    for cell in manifest['cells']:
        task, length = cell['task'], cell['length']
        require(task in TASKS and length in LENGTHS, 'Unexpected task or length')
        key = (task, length)
        require(key not in records, f'Duplicate cell: {key}')
        path = safe_path(root, cell['prediction_file'])
        require(path.parent == prediction_root, f'Wrong prediction directory: {path.name}')
        require(path.name == f"{task}_{length}_{manifest['configuration_id']}.csv", 'Bad filename/configuration')
        require(sha256(path) == cell['sha256'], f'Cell checksum mismatch: {path.name}')
        with path.open(encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            require(reader.fieldnames == ['question', 'target', 'output'], f'Unexpected CSV fields: {path.name}')
            rows = list(reader)
        require(len(rows) == cell['n'] and len(rows) > 0, f'Wrong denominator: {path.name}')
        require(all(set(r) == {'question','target','output'} and all(isinstance(v,str) for v in r.values())
                    for r in rows), f'Malformed CSV row: {path.name}')
        correct = sum(bool(metric.compare_answers(r['target'], r['output'], r['question'],
                                                 metric.TASK_LABELS[task])) for r in rows)
        accuracy = 100.0 * correct / len(rows)
        require(correct == cell['correct'] and abs(accuracy-cell['accuracy_pct']) < 1e-10,
                f'Score mismatch: {path.name}')
        records[key] = rows
        filenames.add(path.name)
        results.append({'task':task,'length':length,'n':len(rows),'correct':correct,'accuracy_pct':accuracy})
    require(filenames == {p.name for p in prediction_root.glob('*.csv')}, 'Missing or extra prediction cell')
    require(len(records) == manifest['cell_count'], 'Cell count mismatch')
    require(sum(len(v) for v in records.values()) == manifest['prediction_rows'], 'Row total mismatch')

    expected_grid = {(task, length) for task in TASKS for length in LENGTHS[:-1]}
    expected_grid |= {(task, '10M') for task in TASKS[:5]}
    require(set(records) == expected_grid, 'Submitted task/length coverage differs from the declared scope')
    for key, rows in records.items():
        wanted_n = 100
        require(len(rows) == wanted_n, f'Declared sample count violated: {key}')
    evaluation = json.loads((meta/'evaluation.json').read_text(encoding='utf-8'))
    require(evaluation['configuration_id'] == manifest['configuration_id'], 'Metadata configuration mismatch')
    require(evaluation['scope']['cell_count'] == manifest['cell_count'], 'Metadata cell-count mismatch')
    require(evaluation['scope']['prediction_rows'] == manifest['prediction_rows'], 'Metadata row-count mismatch')
    # These checks validate the public disclosure state, not the private reader identity.
    require(evaluation['reader']['model_id'] is None, 'Reader ID must be withheld')
    require(evaluation['reader']['revision'] is None, 'Reader revision must be withheld')
    require(evaluation['reader']['identity_disclosure'] == 'withheld', 'Reader disclosure mismatch')
    require(evaluation['reader']['parameter_class'] == '4B', 'Reader parameter-class mismatch')
    require(evaluation['dataset']['revision'] == 'ee0d588794c7ac098062ee0d247c733d62e94fe2', 'Dataset revision mismatch')
    require(evaluation['metric']['sha256'] == METRIC_SHA256, 'Evaluation metric mismatch')

    seen = set()
    with (meta/'sample_indices.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            item = json.loads(line)
            require(set(item) == {'task','length','export_sample_index','csv_data_row_index','record_sha256'},
                    'Unexpected sample-index fields')
            key = (item['task'], item['length'])
            idx = item['csv_data_row_index']
            require(type(idx) is int and type(item['export_sample_index']) is int, 'Non-integer index')
            require(key in records and 0 <= idx < len(records[key]), 'Sample index outside cell')
            require(item['export_sample_index'] == idx, 'Export index/order mismatch')
            require((key,idx) not in seen, 'Duplicate sample index')
            require(item['record_sha256'] == row_hash(records[key][idx]), 'Sample/CSV content mismatch')
            seen.add((key,idx))
    require(len(seen) == manifest['prediction_rows'], 'Sample mapping is incomplete')

    lookup = {(c['task'],c['length']):c for c in results}
    averages = {}
    for length in LENGTHS:
        selected = [lookup.get((t,length)) for t in TASKS[:5]]
        averages[length] = (round(sum(c['accuracy_pct'] for c in selected)/5.0, 10)
                            if all(c is not None for c in selected) else None)
    summary_rows = list(csv.DictReader((meta/'RESULTS.csv').open(encoding='utf-8',newline='')))
    require(len(summary_rows) == 11, 'Result table must have ten task rows and one mean row')
    summaries = {r['task']:r for r in summary_rows}
    require(set(summaries) == set(TASKS)|{'avg'}, 'Result table tasks mismatch')
    for task in TASKS:
        for length in LENGTHS:
            observed = summaries[task][length]
            cell = lookup.get((task,length))
            require((observed == '' if cell is None else abs(float(observed)-cell['accuracy_pct']) < 1e-10),
                    f'Result-table mismatch: {task}/{length}')
    for length, average in averages.items():
        observed = summaries['avg'][length]
        require((observed == '' if average is None else abs(float(observed)-average) < 1e-10),
                f'Average-table mismatch: {length}')
    return {'status':'PASS','system':manifest['system_label'],
            'configuration_id':manifest['configuration_id'],
            'hashed_files':len(expected),'cells':len(records),
            'prediction_rows':manifest['prediction_rows'],'metric_sha256':METRIC_SHA256,
            'mean_qa1_qa5_pct':averages,
            'reader_identity_disclosure':'withheld', 'reader_identity_verified':False,
            'scope':'Artifact integrity and score recalculation only; no reader-identity verification, inference rerun or leaderboard acceptance.'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    try:
        result = verify(args.root)
    except (ValueError, KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
