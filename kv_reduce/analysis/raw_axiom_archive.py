"""Verify the archived AXIOM outputs and rebuild plotting inputs from original rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

BUNDLE = Path(__file__).resolve().parents[2] / 'results' / 'axiom-review-2026'


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def verified_files(bundle=BUNDLE):
    manifest = json.loads((bundle/'raw-manifest.json').read_text())
    archive = bundle/manifest['archive']
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['archive_sha256']:
        raise ValueError('Archive SHA-256 mismatch')
    expected = {entry['path']: entry for entry in manifest['files']}
    if len(expected) != manifest['file_count']:
        raise ValueError('Duplicate or missing manifest entry')
    result = {}
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != len(expected) or set(z.namelist()) != set(expected):
            raise ValueError('Archive file list differs from manifest')
        for name, entry in expected.items():
            data = z.read(name)
            if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise ValueError(f'File checksum mismatch: {name}')
            result[name] = data
    return result


def rebuild_metrics(destination, bundle=BUNDLE):
    """Write derived metrics to an empty directory; original archive stays unchanged."""
    files = verified_files(bundle)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError('Metrics destination must be empty')
    quality = []
    for name in sorted(files):
        if not name.startswith('axiom_quality_') or not name.endswith('/benchmark_rows.csv'):
            continue
        run = name.split('/')[0]
        config = json.loads(files[f'{run}/summary.json'])['config']
        for r in csv.DictReader(io.StringIO(files[name].decode())):
            generated = json.loads(r['generated_token_ids'])
            target = json.loads(r['target_token_ids'])
            quality.append(dict(
                run=run, context=config['prompt_len'], residual=config['key_residual_length'],
                seed=config['seed'], config=r['config'], prompt=int(r['prompt_idx']),
                generated_tokens=int(r['generated_tokens']), accepted_tokens=int(r['accepted_tokens']),
                proposed_tokens=int(r['proposed_tokens']), accept_rate=float(r['accept_rate']),
                accepted_per_round=float(r['accepted_per_round']), num_rounds=int(r['num_rounds']),
                target_calls=int(r['target_calls']), matches_target=generated == target,
                generated_sha256=token_hash(generated), target_sha256=token_hash(target),
                round_details=json.loads(r['round_details']),
            ))
    groups = {'quality': quality, 'speculative': [], 'target_only': []}
    for kind in ('speculative', 'target_only'):
        for context in (1024, 16384):
            name = f'axiom_packed_ctx{context}/{kind}.jsonl'
            for line in files[name].decode().splitlines():
                r = json.loads(line)
                tokens = r.pop('tokens')
                r.pop('top1_margins', None)
                r.update(context=context, token_count=len(tokens), tokens_sha256=token_hash(tokens))
                r.setdefault('warmup', r['prompt'] == 0)
                groups[kind].append(r)
    for kind, rows in groups.items():
        (destination/f'{kind}.jsonl').write_text(''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in rows))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    parser.add_argument('--out', type=Path, help='Optional empty directory for reconstructed metrics')
    args = parser.parse_args()
    if args.out:
        rebuild_metrics(args.out, args.bundle)
        print(f'Verified archive and rebuilt metrics in {args.out}')
    else:
        files = verified_files(args.bundle)
        print(f'PASS: verified {len(files)} original output files against the Catalyst manifest.')


if __name__ == '__main__':
    main()
