"""Release checks for paired metrics and portable launchers; no GPU or network."""
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

from kv_reduce.analysis.analyze_axiom_review import DEFAULT_DATA, analyze, compare, read_rows, validate

ROOT = Path(__file__).resolve().parents[1]


class ReviewMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.quality = read_rows(DEFAULT_DATA/'quality.jsonl')
        cls.systems = read_rows(DEFAULT_DATA/'speculative.jsonl')
        cls.targets = read_rows(DEFAULT_DATA/'target_only.jsonl')

    def test_frozen_results_reproduce(self):
        report = analyze()
        expected = json.loads((DEFAULT_DATA/'validated-summary.json').read_text())
        compare({k: report[k] for k in expected}, expected)
        compare(report['prefix'], json.loads((DEFAULT_DATA/'prefix-summary.json').read_text())['results'])

    def test_duplicate_prompt_is_rejected(self):
        rows = copy.deepcopy(self.quality)
        rows[1]['prompt'] = rows[0]['prompt']
        with self.assertRaisesRegex(ValueError, 'Duplicate/missing'):
            validate(rows, self.systems, self.targets)

    def test_broken_pair_is_rejected(self):
        rows = copy.deepcopy(self.quality)
        rows[0]['target_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'pairing mismatch'):
            validate(rows, self.systems, self.targets)

    def test_invalid_round_total_is_rejected(self):
        rows = copy.deepcopy(self.quality)
        rows[0]['accepted_tokens'] += 1
        with self.assertRaisesRegex(ValueError, 'Accepted count mismatch'):
            validate(rows, self.systems, self.targets)

    def test_missing_audit_is_rejected(self):
        rows = copy.deepcopy(self.systems)
        audit = next(r for r in rows if r['verification']=='sequential_audit')
        rows.remove(audit)
        with self.assertRaises(ValueError):
            validate(self.quality, rows, self.targets)

    def test_false_correctness_flag_is_rejected(self):
        rows = copy.deepcopy(self.systems)
        rows[0]['matches_target'] = not rows[0]['matches_target']
        with self.assertRaisesRegex(ValueError, 'agreement flag mismatch'):
            validate(self.quality, rows, self.targets)

    def test_warmup_mislabel_is_rejected(self):
        rows = copy.deepcopy(self.targets)
        rows[0]['warmup'] = False
        with self.assertRaisesRegex(ValueError, 'warmup mismatch'):
            validate(self.quality, self.systems, rows)


class LauncherTest(unittest.TestCase):
    def invoke(self, script, task, out):
        return subprocess.run(['bash',str(ROOT/'scripts/axiom_review'/script),str(task),'--dry-run'],cwd='/',env={**os.environ,'OUT_ROOT':str(out)},text=True,capture_output=True)

    def test_quality_matrix_and_paths_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix='axiom test ') as out:
            for task,(context,residual,seed) in enumerate([(1024,0,0),(1024,128,0),(1024,0,1),(1024,128,1),(16384,0,0),(16384,128,0)]):
                result = self.invoke('quality.sh',task,out)
                self.assertEqual(result.returncode,0,result.stderr)
                args=shlex.split(result.stdout)
                for key,val in [('--prompt_len',context),('--key_residual_length',residual),('--seed',seed),('--max_new_tokens',256),('--num_prompts',24)]:
                    self.assertEqual(args[args.index(key)+1],str(val))
                self.assertEqual(args[args.index('--out_dir')+1],str(Path(out)/f'axiom_quality_ctx{context}_r{residual}_s{seed}'))
                self.assertEqual(list(Path(out).iterdir()),[])

    def test_packed_matrix(self):
        with tempfile.TemporaryDirectory() as out:
            for task,context,prompts in [(0,1024,24),(1,16384,12)]:
                result=self.invoke('packed.sh',task,out)
                self.assertEqual(result.returncode,0,result.stderr)
                args=shlex.split(result.stdout)
                self.assertEqual(args[args.index('--context')+1],str(context))
                self.assertEqual(args[args.index('--prompts')+1],str(prompts))

    def test_invalid_task_and_existing_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as out:
            self.assertNotEqual(self.invoke('quality.sh',6,out).returncode,0)
            self.assertNotEqual(self.invoke('packed.sh',2,out).returncode,0)
            (Path(out)/'axiom_quality_ctx1024_r0_s0').mkdir()
            result=self.invoke('quality.sh',0,out)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Refusing to overwrite',result.stderr)


if __name__=='__main__':
    unittest.main()
