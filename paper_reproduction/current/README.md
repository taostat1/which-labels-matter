# Current manuscript reproduction

From the repository root:

```bash
python -m pip install -e ".[reproduce]"
python paper_reproduction/current/reproduce.py --output build/paper
```

The command regenerates Tables 1–8, Figure 3, and audit summaries from the committed path-level records in `data/`. It validates their hashes and recalculates aggregates; numerical summaries are not copied from prewritten tables. `reference/` preserves the source modules used by the current and matched experiments. `audits/` contains independent cache and dense-design checks.

See [the reproduction guide](../../docs/REPRODUCING.md) for fresh demo commands, record provenance, and the additional inputs needed to rerun original fits. No separate note or archive is required to regenerate the reported tables.
