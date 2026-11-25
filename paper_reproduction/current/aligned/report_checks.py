"""Recompute the manuscript's population cost percentages and fixed-plan error check."""
from collections import Counter
import csv
import gzip
import json
from math import comb, fsum
from pathlib import Path


HERE = Path(__file__).resolve().parent
# The stage ledger and the aligned table use different names for two methods.
METHODS = {"direct_refits": "reference", "active_testing": "at", "ams": "ams",
           "completion": "completion"}


def population_savings(stats, codes, methods):
    """Percent saved relative to the smallest baseline mean within each configuration."""
    result = []
    for code in codes:
        least = min(stats["population", code, method][0] for method in methods)
        lec = stats["population", code, "lec"][0]
        result.append(100 * (1 - lec / least))
    return result


def training_share(records, methods):
    """Ratio of pooled comparison-training purchases to pooled complete purchases."""
    training = total = 0
    for row in records:
        if row["method"] not in methods:
            continue
        stages = json.loads(row["stages"])
        assert sum(stages.values()) == int(row["labels"])
        training_key = "training" if row["method"] == "completion" else "comparison_training"
        training += stages[training_key]
        total += int(row["labels"])
    return training, total


def upper_tail(n, k, p):
    return fsum(comb(n, j) * p**j * (1-p)**(n-j) for j in range(k, n+1))


def lower_cp_limit(n, k, tail_probability):
    if k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(65):
        mid = (lo+hi)/2
        if upper_tail(n, k, mid) < tail_probability:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def error_check():
    with gzip.open(HERE / "paths.csv.gz", "rt", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if (r["setting"], r["case"], r["method"]) == ("coverage", "image|0.00375", "lec")]
    cfg = json.loads((HERE / "fixed/protocol.json").read_text(encoding="utf-8"))["coverage"]
    n = len(rows)
    paths = int(cfg["paths_per_pool"])
    assert {(int(r["pool"]), int(r["path"])) for r in rows} == {
        (i, j) for i in range(20) for j in range(paths)}
    assert n == 20 * paths
    failures = []
    for row in rows:
        gap, g = float(row["target"]), float(row["g"])
        assert g == .00375 and row["choice"] in ("A", "B")
        wrong = gap > g if row["choice"] == "A" else gap < -g
        assert int(row["acceptable"]) == int(not wrong)
        if wrong:
            failures.append((row["pool"], row["path"]))
    k = len(failures)
    lo = lower_cp_limit(n, k, .025)
    hi = 1 - lower_cp_limit(n, n-k, .025)
    delta = cfg["delta"]
    p = upper_tail(n, k, delta)
    print(f"ADE20K g=.00375: {k}/{n} outside tolerance at {failures} (zero-based). "
          f"95% CP [{lo:.10f}, {hi:.10f}], upper tail at delta={delta}: {p:.10f}.")
    return {"PathErrorCount": f"{k}/{n}", "PathErrorPercent": rf"{100*k/n:.2f}\%",
            "PathErrorLower": rf"{100*lo:.2f}\%", "PathErrorUpper": rf"{100*hi:.2f}\%",
            "PathErrorP": f"{p:.4f}"}


def format_range(values):
    return rf"{min(values):.1f}\%--{max(values):.1f}\%"


def reported_tex(stats, codes):
    with gzip.open(HERE / "population/paths.csv.gz", "rt", newline="") as f:
        records = list(csv.DictReader(f))
    three = ("direct_refits", "active_testing", "ams")
    expected = set(range(6, 18))
    indexed = {(r["code"], r["method"], int(r["group"])): int(r["labels"]) for r in records}
    assert len(indexed) == len(records)
    for method in (*METHODS, "lec"):
        for code in codes:
            totals = [indexed[code, method, i] for i in expected]
            assert sum(totals)/len(totals) == stats["population", code, METHODS.get(method, method)][0]

    four_savings = population_savings(stats, codes, METHODS.values())
    three_savings = population_savings(stats, codes, [METHODS[m] for m in three])
    four_train, four_total = training_share(records, METHODS)
    three_train, three_total = training_share(records, three)
    print(f"Comparison training / distinct total: four baselines {four_train}/{four_total}, "
          f"three baselines {three_train}/{three_total}.")
    # The D.7 sensitivity check changes the betting allocation to .02 on the same paths.
    with (HERE / "interval_control.csv").open(encoding="utf-8-sig", newline="") as f:
        controls = [r for r in csv.DictReader(f) if r["arm"] in ("betting_04", "betting_02")]
    counts = Counter((r["arm"], r["code"], r["method"], int(r["group"])) for r in controls)
    assert len(counts) == 2*len(codes)*len(three)*len(expected) and set(counts.values()) == {1}
    alternative = {(r["code"], r["method"], int(r["group"])): int(r["labels"])
                   for r in controls if r["arm"] == "betting_02"}
    for r in controls:
        if r["arm"] == "betting_04":
            assert int(r["labels"]) == indexed[r["code"], r["method"], int(r["group"])]
    reduced_savings = []
    for code in codes:
        least = min(sum(alternative[code, m, i] for i in expected) for m in three)
        lec = sum(indexed[code, "lec", i] for i in expected)
        reduced_savings.append(100*(1-lec/least))
    assert format_range(three_savings) == format_range(reduced_savings)
    values = {"PopulationSavings": format_range(four_savings),
              "PopulationTrainingShare": rf"{100*four_train/four_total:.1f}\%",
              "ThreeBaselineSavings": format_range(three_savings),
              "ThreeBaselineTrainingShare": rf"{100*three_train/three_total:.1f}\%"}
    values.update(error_check())
    return [r"% Generated by evidence/aligned/render.py from the matched path records."] + [
        rf"\newcommand{{\{key}}}{{{value}}}" for key, value in values.items()]
