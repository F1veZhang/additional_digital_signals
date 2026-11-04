"""Verify this addition's bytes and Python syntax without executing analysis code."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'release_manifest.json').read_text(encoding='utf-8'))
    declared = {entry['file'] for entry in manifest['files']}
    actual = {str(path.relative_to(ROOT)).replace('\\', '/')
              for path in ROOT.rglob('*') if path.is_file()
              and path.name != 'release_manifest.json'
              and '__pycache__' not in path.parts
              and not set(path.relative_to(ROOT).parts).intersection({'Audit', 'Figures', 'local_work'})}
    if declared != actual:
        raise AssertionError({'missing': sorted(declared - actual), 'unexpected': sorted(actual - declared)})
    parsed = 0
    for entry in manifest['files']:
        path = ROOT / entry['file']
        data = path.read_bytes()
        if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise AssertionError('File changed: ' + entry['file'])
        if path.suffix == '.py':
            ast.parse(data.decode('utf-8-sig'), filename=entry['file'])
            parsed += 1
    print(json.dumps({'status': 'pass', 'files': len(declared), 'python_files_parsed': parsed,
                      'research_modules_executed': 0, 'research_data_read': False}))


if __name__ == '__main__':
    main()
