"""Check the matched population initialization paths and format the appendix table."""
import csv
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path


def initialization_table(directory):
    codes = ("T1", "T2", "T3", "T4", "V1", "V2", "V3", "V4")
    names = {"pilot_fixed": "Paid pilot / fixed", "pilot_cache": "Paid pilot / cache",
             "zero_fixed": "Zero / fixed", "zero_cache": r"\textbf{Zero / cache (LEC)}"}
    cells = defaultdict(dict)
    with gzip.open(Path(directory) / "initialization_paths.csv.gz", "rt", newline="") as f:
        for row in csv.DictReader(f):
            code, method, group = row["code"], row["method"], int(row["group"])
            assert code in codes and method in names and group not in cells[code, method]
            cost = int(row["labels"])
            assert sum(json.loads(row["stages"]).values()) == cost
            gap, tol, lower, upper = (float(row[k]) for k in ("target", "g", "lower", "upper"))
            assert lower <= gap <= upper
            assert ((row["choice"] == "A" and upper <= tol and gap <= tol) or
                    (row["choice"] == "B" and lower >= -tol and gap >= -tol))
            if row["fallback"] == "True":
                assert int(row["stopping_blocks"]) == 32768
            cells[code, method][group] = cost
    groups = set(range(6, 18))
    assert all(cells[c, m].keys() == groups for c in codes for m in names)
    with gzip.open(Path(directory) / "population/paths.csv.gz", "rt", newline="") as f:
        for row in csv.DictReader(f):
            method = {"lec": "zero_cache", "fixed_bound": "zero_fixed"}.get(row["method"])
            if method:
                assert cells[row["code"], method][int(row["group"])] == int(row["labels"])
    lines = [r"\begin{table}[htbp]\centering\small",
             r"\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{1.08}",
             r"\begin{tabular}{@{}lrrrrrrrr@{}}\toprule",
             r" & \multicolumn{4}{c}{GoEmotions} & \multicolumn{4}{c}{ADE20K}\\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"Initialization / bound & " + " & ".join(codes) + r"\\\midrule"]
    for method, name in names.items():
        values = [name]
        for code in codes:
            costs = cells[code, method].values()
            mean = statistics.mean(costs)
            text = f"{mean / 1000:.1f}"
            if mean == min(statistics.mean(cells[code, m].values()) for m in names):
                text = r"\mathbf{" + text + "}"
            values.append("$" + text + r"_{\color{gray}" + f"{statistics.stdev(costs)/1000:.1f}" + "}$")
        lines.append(" & ".join(values) + r"\\")
    lines += [r"\bottomrule\end{tabular}",
              r"\caption{Distinct labels for population comparison and final training (thousands). Subscripts give standard deviations over 12 matched paths. Fixed and cache variants start with the same bound within each initialization. All column minima are in bold.}",
              r"\label{tab:population_cache_ablation}", r"\end{table}"]
    return lines
