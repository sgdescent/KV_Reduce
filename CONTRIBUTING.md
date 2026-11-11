# Contributing

This repository contains research code and historical experiments. Start with the
AXIOM reproduction guide before changing a benchmark protocol.

## Report a problem

Open a GitHub issue with the command, environment, GPU/driver, seed, verifier mode
and a small reproducer. Do not include access tokens, private paths, model weights
or dataset passages that you cannot redistribute. Report the first divergent token
position and relevant cache state when diagnosing verifier correctness.

The packed review benchmark currently has batched output mismatches, including in
its BF16 baseline. This is documented, not resolved by the paper-facing release.

## Validate a change

CPU analysis checks need only `requirements-analysis.txt`:

```bash
python -m unittest discover -s tests -p 'test_axiom_review_release.py' -v
python analyze_axiom_review.py --check
```

With PyTorch installed, run the cache and statistics checks:

```bash
python -m unittest discover -s tests -p 'test_packed_kv_cache.py' -v
python -m unittest discover -s tests -p 'test_spec_kv_statistics.py' -v
```

GPU tests and full experimental reruns are separate. A passing CPU suite does not
validate Triton execution, model generation or a serving-speed claim. Do not
silently replace frozen result files after changing a protocol. Preserve provenance
and distinguish new experiments from reanalysis of existing outputs.
