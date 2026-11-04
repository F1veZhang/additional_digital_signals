"""Compare authorized local files with published HFGI fingerprints; do not parse rows."""
import argparse
import hashlib
import json
from pathlib import Path

def fingerprints(path):
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return {'bytes': size, 'md5': md5.hexdigest(), 'sha256': sha256.hexdigest()}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'provenance/hfgi_release_files.json')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    results = []
    for record in manifest['files']:
        path = args.input_dir / record['filename']
        if not path.is_file():
            results.append({'filename': record['filename'], 'status': 'missing'})
            continue
        actual = fingerprints(path)
        ok = all(actual[key] == record[key] for key in ('bytes','md5','sha256'))
        results.append({'filename': record['filename'], 'status': 'match' if ok else 'mismatch', **actual})
    print(json.dumps({'release': manifest['release'], 'files': results}, indent=2))
    return 0 if all(row['status'] == 'match' for row in results) else 1

if __name__ == '__main__':
    raise SystemExit(main())
