"""Validate and reanalyze the public AXIOM review metrics without model dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_DATA = Path(__file__).resolve().parent / "results" / "axiom-review-2026"
CONFIGS = ("none", "k8v4", "k4v8", "k4v4")
RUNS = tuple(f"axiom_quality_ctx{c}_r{r}_s{s}" for c, s in ((1024, 0), (1024, 1), (16384, 0)) for r in (0, 128))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def paired_ci(values, rng):
    values = np.asarray(values, dtype=float)
    require(values.size > 0 and np.isfinite(values).all(), "Non-finite or empty metric")
    draws = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return {"mean": float(values.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist()}


def validate(quality, systems, targets):
    require(len(quality) == 576, "Expected 576 quality rows")
    require(set(r["run"] for r in quality) == set(RUNS), "Unexpected quality settings")
    for run in RUNS:
        rows = [r for r in quality if r["run"] == run]
        require(set(r["config"] for r in rows) == set(CONFIGS), "Unexpected quality configurations")
        base = {r["prompt"]: r for r in rows if r["config"] == "none"}
        for config in CONFIGS:
            group = [r for r in rows if r["config"] == config]
            require(len(group) == 24 and {r["prompt"] for r in group} == set(range(24)), "Duplicate/missing quality prompt")
            for r in group:
                require(r['run'] == f"axiom_quality_ctx{r['context']}_r{r['residual']}_s{r['seed']}", "Run metadata mismatch")
                require(r["matches_target"] and r["generated_sha256"] == r["target_sha256"] == base[r["prompt"]]["target_sha256"], "Quality output/pairing mismatch")
                require(r["generated_tokens"] == 256 and r["target_calls"] == 257, "Unexpected sequence length/call count")
                rounds = r["round_details"]
                require(len(rounds) == r["num_rounds"], "Round count mismatch")
                require(sum(x["accepted_prefix"] for x in rounds) == r["accepted_tokens"], "Accepted count mismatch")
                require(sum(x["proposed"] for x in rounds) == r["proposed_tokens"], "Proposal count mismatch")
                for x in rounds:
                    require(0 <= x["accepted_prefix"] <= x["proposed"] <= 4, "Invalid prefix")
                    expected = x["accepted_prefix"] if x["accepted_prefix"] < x["proposed"] else -1
                    require(x["rejection_position"] == expected, "Invalid rejection position")
                require(np.isclose(r["accept_rate"], r["accepted_tokens"] / r["proposed_tokens"]), "Acceptance mismatch")
                require(np.isclose(r["accepted_per_round"], r["accepted_tokens"] / r["num_rounds"]), "Prefix mean mismatch")
    for context in (1024, 16384):
        for seed in ((0, 1) if context == 1024 else (0,)):
            refs = [{r['prompt']: r['target_sha256'] for r in quality if r['context'] == context and r['seed'] == seed and r['residual'] == residual and r['config'] == 'none'} for residual in (0,128)]
            require(refs[0] == refs[1], "Residual comparisons must use paired references")
    require(len(targets) == 38 and len(systems) == 208, "Unexpected packed record count")
    require({r['verification'] for r in systems} == {'batched','sequential_audit'}, "Unexpected verifier modes")
    require({r['context'] for r in targets+systems} == {1024,16384}, "Unexpected packed contexts")
    for context, n in ((1024,24),(16384,12)):
        t = [r for r in targets if r['context'] == context]
        require(len(t) == n+1 and {r['prompt'] for r in t} == set(range(n+1)), "Missing/duplicate target reference")
        refs = {r['prompt']: r for r in t}
        rows = [r for r in systems if r['context'] == context]
        require(all((r['mode'],r['residual']) in [('bf16',128),('packed',0),('packed',128)] for r in rows), "Unexpected packed config")
        for mode,residual in [('bf16',128),('packed',0),('packed',128)]:
            g = [r for r in rows if r['verification']=='batched' and (r['mode'],r['residual'])==(mode,residual)]
            require(len(g)==n+1 and {r['prompt'] for r in g}==set(range(n+1)), "Missing/duplicate packed prompt")
        for r in t+rows:
            require(r['token_count']==256 and r['warmup']==(r['prompt']==0), "Packed length/warmup mismatch")
            if 'seconds' in r:
                require(np.isfinite(r['seconds']) and r['seconds'] > 0, "Invalid timing")
        for r in rows:
            require(r['matches_target'] == (r['tokens_sha256']==refs[r['prompt']]['tokens_sha256']), "Packed agreement flag mismatch")
        failed = {(r['prompt'],r['mode'],r['residual']) for r in rows if r['verification']=='batched' and not r['matches_target']}
        audits = [r for r in rows if r['verification']=='sequential_audit']
        require(len(audits)==len(failed) and {(r['prompt'],r['mode'],r['residual']) for r in audits}==failed, "Missing/duplicate mismatch audit")
        require(all(r['matches_target'] for r in audits), "Failed sequential audit")


def analyze(data=DEFAULT_DATA):
    quality = read_rows(data/'quality.jsonl')
    systems = read_rows(data/'speculative.jsonl')
    targets = read_rows(data/'target_only.jsonl')
    validate(quality, systems, targets)
    rng = np.random.default_rng(1910)
    report = {'quality': {}, 'packed': {}, 'paired_residual128_minus0_k4v4': {}, 'prefix': {}}
    for run in sorted(RUNS):
        groups = {c: sorted([r for r in quality if r['run']==run and r['config']==c],key=lambda r:r['prompt']) for c in CONFIGS}
        b = np.array([r['accept_rate'] for r in groups['none']])
        report['quality'][run] = {}
        report['prefix'][run] = {}
        boot = np.random.default_rng(1109).integers(24,size=(10000,24))
        for c,g in groups.items():
            a = np.array([r['accept_rate'] for r in g])
            report['quality'][run][c] = {'mean_prompt_acceptance':float(a.mean()),'delta_pp':paired_ci((a-b)*100,rng),'matches':sum(r['matches_target'] for r in g)}
            prefix = np.array([r['accepted_per_round'] for r in g])
            delta = prefix - np.array([r['accepted_per_round'] for r in groups['none']])
            histogram = [[sum(x['rejection_position']==k for x in r['round_details'])/r['num_rounds'] for k in [-1,0,1,2,3]] for r in g]
            report['prefix'][run][c] = {'mean_prefix':float(prefix.mean()),'prefix_delta':float(delta.mean()),'prefix_delta_ci95':np.quantile(delta[boot].mean(1),[.025,.975]).tolist(),'mean_prompt_rejection_distribution':np.mean(histogram,axis=0).tolist(),'mean_rounds':float(np.mean([r['num_rounds'] for r in g])),'target_calls_per_token':257/256}
    for context in (1024,16384):
        t = [r for r in targets if r['context']==context and not r['warmup']]
        rows = [r for r in systems if r['context']==context]
        groups = {k:sorted([r for r in rows if r['verification']=='batched' and not r['warmup'] and (r['mode'],r['residual'])==k],key=lambda r:r['prompt']) for k in [('bf16',128),('packed',0),('packed',128)]}
        b = groups[('bf16',128)]
        out = {'target_only_mean_s':float(np.mean([r['seconds'] for r in t])),'target_only_peak_GiB':float(np.mean([r['peak_allocated_bytes']/2**30 for r in t])),'sequential_audits_passed_including_warmup':sum(r['verification']=='sequential_audit' for r in rows)}
        for k,g in groups.items():
            times = np.array([r['seconds'] for r in g]); bt = np.array([r['seconds'] for r in b])
            mem = np.array([r['peak_allocated_bytes']/2**30 for r in g]); bm = np.array([r['peak_allocated_bytes']/2**30 for r in b])
            out[str(k)] = {'mean_s':float(times.mean()),'latency_delta_s':paired_ci(times-bt,rng),'latency_ratio':float(times.sum()/bt.sum()),'peak_GiB':float(mem.mean()),'peak_delta_GiB':paired_ci(mem-bm,rng),'matches':sum(r['matches_target'] for r in g),'draft_cache_saving_pct':paired_ci([100*(1-sum(r['cache_bytes'].values())/sum(s['cache_bytes'].values())) for r,s in zip(g,b)],rng)}
        report['packed'][str(context)] = out
    residual_rng = np.random.default_rng(1911)
    for context, seed in [(1024,0),(1024,1),(16384,0)]:
        groups = [sorted([r for r in quality if r['context']==context and r['seed']==seed and r['residual']==residual and r['config']=='k4v4'],key=lambda r:r['prompt']) for residual in (0,128)]
        ci = paired_ci([100*(b['accept_rate']-a['accept_rate']) for a,b in zip(*groups)],residual_rng)
        report['paired_residual128_minus0_k4v4'][f'{context}_seed{seed}'] = {'delta_pp':ci['mean'],'ci95':ci['ci95']}
    return report


def compare(actual, expected, path='summary'):
    if isinstance(expected,dict):
        require(actual.keys()==expected.keys(),f'{path}: key mismatch')
        for k,v in expected.items(): compare(actual[k],v,f'{path}.{k}')
    elif isinstance(expected,list):
        require(len(actual)==len(expected),f'{path}: length mismatch')
        for i,v in enumerate(expected): compare(actual[i],v,f'{path}[{i}]')
    else:
        require(np.isclose(actual,expected,rtol=1e-10,atol=1e-10),f'{path}: {actual} != {expected}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check',action='store_true',help='Compare against frozen summaries; no downloads or GPU work')
    args=parser.parse_args()
    report=analyze(args.data)
    if args.check:
        expected=json.loads((args.data/'validated-summary.json').read_text())
        compare({k:report[k] for k in expected},expected)
        compare(report['prefix'],json.loads((args.data/'prefix-summary.json').read_text())['results'],'prefix')
        print('PASS: counts, pairing, hashes, round totals, audits and frozen summaries match.')
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(f'Wrote {args.output}')
    if not args.check and not args.output:
        print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
