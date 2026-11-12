"""The archived measurements must reproduce the published plot inputs."""
import json
from pathlib import Path
import tempfile
import unittest

from raw_axiom_archive import BUNDLE, rebuild_metrics, verified_files


class RawArchiveTest(unittest.TestCase):
    def test_complete_archive_matches_catalyst_manifest(self):
        files = verified_files()
        self.assertEqual(len(files),24)
        self.assertEqual(len({name.split('/')[0] for name in files}),9)
        self.assertEqual(sum(name.endswith('benchmark_rows.csv') for name in files),6)
        self.assertEqual(sum(name.endswith('speculative.jsonl') for name in files),3)

    def test_raw_rows_reconstruct_all_public_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = rebuild_metrics(Path(tmp)/'metrics')
            for name in ('quality','speculative','target_only'):
                def rows(path): return [json.loads(x) for x in path.read_text().splitlines()]
                self.assertEqual(rows(output/f'{name}.jsonl'), rows(BUNDLE/f'{name}.jsonl'))

    def test_reconstruction_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'keep.txt').write_text('keep')
            with self.assertRaisesRegex(ValueError,'must be empty'):
                rebuild_metrics(Path(tmp))
            self.assertEqual((Path(tmp)/'keep.txt').read_text(),'keep')


if __name__=='__main__':
    unittest.main()
