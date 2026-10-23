"""Offline check of this release (no data needed).

1. every file matches MANIFEST.sha256 and no unlisted file is present;
2. the hashes in protocol/FREEZE.json match the protocol files, the as-run scripts and the audit outputs;
3. the released and as-run scripts match provenance/code_provenance.json;
4. no data arrays, model files or archives are present;
5. every Python file compiles.
Exit status 0 means all checks passed.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {'.npz', '.npy', '.pkl', '.pickle', '.parquet', '.pt', '.pth', '.joblib', '.h5', '.zip'}
IGNORED_DIRS = {'.git', '__pycache__', '.venv', 'venv'}
IGNORED_FILES = {'MANIFEST.sha256', '.built_by_r01'}
FREEZE_MAP = {'code/': 'provenance/as_run/v43_upgrade/code/', 'results/': 'results/pvdaq9068_evaluation/', 'protocol/': 'protocol/'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def files():
    return [p for p in ROOT.rglob('*') if p.is_file() and not IGNORED_DIRS & set(p.relative_to(ROOT).parts)
            and p.name not in IGNORED_FILES]


def main():
    fails = []
    listed = {}
    for line in (ROOT / 'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
        h, name = line.split('  ', 1)
        listed[name] = h
    present = {p.relative_to(ROOT).as_posix(): p for p in files()}
    for name, h in listed.items():
        if name not in present:
            fails.append(f'missing: {name}')
        elif sha(present[name]) != h:
            fails.append(f'changed: {name}')
    fails += [f'not in manifest: {n}' for n in sorted(set(present) - set(listed))]
    print(f'manifest: {len(listed)} files checked')

    freeze = json.loads((ROOT / 'protocol/FREEZE.json').read_text(encoding='utf-8'))
    n = 0
    for key, h in freeze.items():
        if key.startswith('_'):
            continue
        prefix = next(p for p in FREEZE_MAP if key.startswith(p))
        path = ROOT / (FREEZE_MAP[prefix] + key[len(prefix):])
        n += 1
        if not path.exists() or sha(path) != h:
            fails.append(f'FREEZE.json hash differs: {key}')
    print(f'protocol freeze: {n} hashes checked (frozen {freeze["_frozen_utc"]})')

    prov = json.loads((ROOT / 'provenance/code_provenance.json').read_text(encoding='utf-8'))
    for row in prov:
        if sha(ROOT / row['released']) != row['released_sha256'] or sha(ROOT / row['as_run']) != row['as_run_sha256']:
            fails.append(f"provenance hash differs: {row['released']}")
    print(f'provenance: {len(prov)} scripts checked ({sum(r["changed_lines"] == 0 for r in prov)} unchanged, '
          f'others differ only in path and description lines; see provenance/path_changes.diff)')

    bad = [n for n, p in present.items() if p.suffix.lower() in FORBIDDEN]
    fails += [f'data or model file present: {n}' for n in bad]
    print(f'data files: {len(bad)} found')

    code = [p for n, p in present.items() if p.suffix == '.py']
    for p in code:
        try:
            compile(p.read_text(encoding='utf-8'), str(p), 'exec')
        except SyntaxError as exc:
            fails.append(f'does not compile: {p.relative_to(ROOT)} ({exc})')
    print(f'compile: {len(code)} Python files')

    print('\nRESULT:', 'all checks passed' if not fails else f'{len(fails)} problem(s)')
    for f in fails:
        print('  -', f)
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
