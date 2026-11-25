# Current manuscript reproduction

Run these commands from the repository root.

```bash
python -m pip install -e ".[reproduce]"
python paper_reproduction/current/reproduce.py --output build/paper
```

The command rebuilds Tables 1–8, the Figure 3 TikZ graphic and a standalone PDF. It writes the manuscript's numeric macros to `reported_numbers.tex`. The file `configuration_summary.csv` gives the label counts, decisions, coverage and fallback totals for each of the 320 method–configuration pairs. `audit.json` records the path checks and stage budgets.

For the November 25 manuscript, `aligned/paths.csv.gz` contains the 21,760 paths behind the aligned experiments. `aligned/population/paths.csv.gz` records purchases by stage for the ten population methods. `aligned/initialization_paths.csv.gz` gives the matched comparison of paid and zero initialization, each with fixed and cache-refined bounds (Table 7). `aligned/population/query_first_history.json.gz` records the inspections and stage charges on the 24 population paths that changed when Algorithm 2 began querying after every undecided public check.

The original fixed-plan cache, pooling and geometry records in `data/` supply Tables 3, 5 and 6 and the shares shown in Figure 3. The earlier population archives `data/population_{zero,paid,baselines}.json.gz` remain available for reference. The November 25 population tables use the ledgers in `aligned/`. `manifest.json` identifies all inputs by SHA-256. `reference/` contains the experiment implementations and `audits/` provides independent cache and dense-design checks.

The project guide lists the commands for fresh simulations and the inputs required to refit models on the original features. [Reproduction guide](../../docs/REPRODUCING.md)
