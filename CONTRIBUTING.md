# Contributing

This repository contains research code for a published paper. Read
[docs/REPRODUCING.md](docs/REPRODUCING.md) before changing a benchmark protocol.

## Report a problem

Open a GitHub issue with the command, environment, GPU/driver, seed, verifier mode
and a small reproducer. Do not include access tokens, private paths, model weights
or dataset passages that you cannot redistribute. Report the first divergent token
position and relevant cache state when diagnosing verifier correctness.

The packed review benchmark currently has batched output mismatches, including in
its BF16 baseline. This is documented, not resolved by the paper-facing release.

## Validate a change

The released-results check needs only `requirements-analysis.txt`:

```bash
python -m kv_reduce.analysis.analyze_axiom_review --check
python -m unittest discover -s tests -p 'test_axiom_review_release.py' -v
```

The full unit-test suite runs on CPU, as in CI:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers==5.3.0 numpy matplotlib pytest
python -m pytest
```

GPU tests and full experimental reruns are separate. A passing CPU suite does not
validate Triton execution, model generation or a serving-speed claim. Do not
silently replace frozen result files after changing a protocol. Preserve provenance
and distinguish new experiments from reanalysis of existing outputs.
