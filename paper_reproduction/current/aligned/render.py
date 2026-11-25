"""Make the article tables from per-path records using only the Python standard library."""
import csv
import gzip
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from .initialization import initialization_table
from .check_query_first import check_query_first
from .report_checks import METHODS as POPULATION_BASELINES, population_savings, reported_tex

HERE = Path(__file__).resolve().parent
METHODS = ("reference", "at", "ams", "completion", "lec", "whole", "no_early", "separate", "fixed_bound", "uniform")
NAMES = ("Active CV", "AT", "AMS", "Completion certificate", r"\textbf{LEC}", "Whole-object purchases",
         "No early stopping", "Separate bounds", "Fixed initial bound", "Uniform terms")
DOMAINS = [("image", .0025), ("image", .00375), ("image", .005), ("text", .005)]
CODES = ["T1", "T2", "T3", "T4", "V1", "V2", "V3", "V4"]
END = r"\\"
RULE = r"\midrule"
SOURCES = {"reference": "zrnic2024", "at": "kossen2021", "ams": "hara2024",
           "completion": "kuboyama2026certificates"}


def load_costs():
    groups = defaultdict(dict)
    with gzip.open(HERE / "paths.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            setting, case, method = (row[k] for k in ("setting", "case", "method"))
            key = (int(row["pool"]), int(row["path"]))
            cell = groups[setting, case, method]
            assert key not in cell and method in METHODS
            cell[key] = (int(row["labels"]), int(row["resolved"]))
    matched = {}
    stats = {}
    for (setting, case, method), cell in groups.items():
        n = {"simulation": 100, "repeated": 60, "coverage": 60, "population": 12}[setting]
        assert len(cell) == n and set(cell) == matched.setdefault((setting, case), set(cell))
        blocks = defaultdict(list)
        for (pool, path), (count, resolved) in cell.items():
            assert count >= 0 and resolved in (0, 1)
            blocks[pool].append(count)
        means = [statistics.mean(counts) for counts in blocks.values()]
        stats[setting, case, method] = (statistics.mean(means), statistics.stdev(means),
                                       sum(ok for _, ok in cell.values()), n)
    assert len(stats) == 32 * 10 and len(matched) == 32
    return stats


def case_id(alpha, lam, signal):
    return f"{alpha:g}|{lam:g}|{signal}"


def domain_id(domain, g):
    return f"{domain}|{g:g}"


def table_start(n, wide=False):
    return [r"\begin{table}[htbp]\centering" + (r"\footnotesize" if wide else r"\small"),
            r"\setlength{\tabcolsep}{" + ("1.8pt" if wide else "4pt") + r"}\renewcommand{\arraystretch}{1.05}",
            r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l" + "r"*n + r"@{}}\toprule"]


def costs(stats, columns, digits, scale, population=False):
    sets = [[stats[setting, case, m] for m in METHODS] for setting, case in columns]
    result = []
    for j, method in enumerate(METHODS):
        if method == "whole":
            result.append(r"\ablationrule")
        row = ["Direct refits" if population and j == 0 else NAMES[j]]
        if method in SOURCES:
            key = "bates2024cv" if population and method == "reference" else SOURCES[method]
            if len(columns) == 8 and method == "completion":
                row[0] = "Certificate"
            if len(columns) == 8 and method == "reference" and not population:
                row[0] = r"\begin{tabular}[t]{@{}l@{}}Active CV\\[-2pt]\tablesource{" + key + r"}\end{tabular}"
            else:
                row[0] += r" \tablesource{" + key + "}"
        for cells in sets:
            value, sd, decided, total = cells[j]
            smallest = min(v for v, _, r, n in cells if r == n)
            mean = f"{value/scale:.{digits}f}"
            if decided == total and math.isclose(value, smallest, abs_tol=1e-9, rel_tol=0):
                mean = r"\mathbf{" + mean + "}"
            sup = "" if decided == total else r"^{\dagger}"
            row.append("$" + mean + r"_{\color{gray}" + f"{sd/scale:.{digits}f}" + "}" + sup + "$")
        result.append(" & ".join(row) + END)
    return result


def end(caption, label):
    return [r"\bottomrule\end{tabular*}", r"\caption{" + caption + "}",
            r"\label{tab:" + label + "}", r"\end{table}"]


def methods_tables(stats):
    rows = table_start(6)
    for idx, alpha in enumerate((.1, .6)):
        if idx:
            rows.append(r"\addlinespace[4pt]\midrule")
        rows += [r"\multicolumn{1}{l}{Feature distance $\alpha=" + f"{alpha:g}" + r"$} & \multicolumn{2}{c}{$\lambda=.25$} & \multicolumn{2}{c}{$\lambda=.5$} & \multicolumn{2}{c}{$\lambda=2$}" + END,
                 r"Method \quad Signal $s$ & 0 & 4 & 0 & 4 & 0 & 4" + END + RULE]
        cols = [("simulation", case_id(alpha, lam, s)) for lam in (.25,.5,2) for s in (0,4)]
        rows += costs(stats, cols, 0, 1)
    rows += end("Mean distinct labels across 100 pools, including final training (SD in grey). Bold marks unrounded minima. The Active CV reference concerns allocation. LEC ablations follow the dashed line.", "simulation_labels")
    yield "simulation_label_table.tex", rows
    rows = table_start(4) + [r" & \multicolumn{2}{c}{$\alpha=.1$} & \multicolumn{2}{c}{$\alpha=.6$}" + END,
                              r"Method \quad Signal $s$ & 0 & 4 & 0 & 4" + END + RULE]
    cols = [("simulation", case_id(a,.125,s)) for a in (.1,.6) for s in (0,4)]
    rows += costs(stats, cols, 0, 1)
    rows += end(r"Weak-penalty simulations at $\lambda=.125$. Mean distinct labels over 100 pools, with the conventions of Table~\ref{tab:simulation_labels}.", "simulation_boundary")
    yield "simulation_boundary_table.tex", rows
    rows = table_start(8, wide=True) + [
        r" & \multicolumn{4}{c}{Repeated random splits} & \multicolumn{4}{c}{Full validation coverage}"+END,
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9} & \multicolumn{3}{c}{ADE20K} & Text & \multicolumn{3}{c}{ADE20K} & Text"+END,
        r"Method \quad $g$ & .0025 & .00375 & .005 & .005 & .0025 & .00375 & .005 & .005"+END+RULE]
    cols = [(family,domain_id(d,g)) for family in ("repeated","coverage") for d,g in DOMAINS]
    rows += costs(stats, cols, 2, 1000)
    rows += end(r"Distinct labels (thousands) across 20 pools with three paths each (SD in grey). Text denotes GoEmotions. Bold marks minima. The Active CV citation identifies the allocation rule used to select validation labels.", "attribution")
    yield "attribution_table.tex", rows
    rows = table_start(8, wide=True) + [
        r" & \multicolumn{4}{c}{GoEmotions} & \multicolumn{4}{c}{ADE20K}"+END,
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}Method & "+" & ".join(CODES)+END+RULE]
    cols = [("population", code) for code in CODES]
    rows += costs(stats, cols, 1, 1000, population=True)
    savings = population_savings(stats, CODES, POPULATION_BASELINES.values())
    position = rows.index(r"\ablationrule")
    rows.insert(position, r"LEC saving (\%) & " + " & ".join(f"{v:.1f}" for v in savings) + END)
    unresolved = any(stats[s,c,m][2] < stats[s,c,m][3] for s,c in cols for m in METHODS)
    note = (r" A dagger marks cells with unresolved paths. Costs include those paths. Bold compares methods resolving all 12 paths." if unresolved else " Bold marks minima.")
    rows += end("Mean distinct labels (thousands) including final training over 12 matched paths (SD in grey). LEC saving is relative to the lowest unrounded baseline mean in each column."+note, "population_comparison")
    yield "population_comparison_table.tex", rows


def full_coverage(cases):
    data={(r["domain"],r["g"]):r for r in cases}
    fig=[r"\begin{tikzpicture}[x=0.095cm,y=0.50cm,font=\rmfamily\footnotesize]"]
    for i, key in enumerate(DOMAINS):
        r=data[key]
        m=r["methods"]["selective_paired"]
        shares=[100*m["mean"]/r["full_labels"], 100*m["saving_from_missing_attributes"], 100*m["saving_from_untouched_objects"]]
        assert abs(sum(shares)-100)<1e-8
        y=3-i
        case=("Image" if key[0]=="image" else "Text") + r", $g=" +f"{key[1]:g}".lstrip("0")+"$"
        fig.append(fr"\node[anchor=east] at (-1,{y}) {{{case}}};")
        left=0
        for value, color in zip(shares,("paperblue","papersage","paperclay")):
            right=left+value
            fig += [fr"\fill[{color}] ({left:.6f},{y-.39:.2f}) rectangle ({right:.6f},{y+.39:.2f});",
                    fr"\node[text=white] at ({(left+right)/2:.6f},{y}) {{{value:.1f}\%}};"]
            left=right
    fig.append(r"\end{tikzpicture}")
    return fig


def audit_csv(stats):
    totals = defaultdict(lambda: [0, 0, 0, 0])
    targets = {}
    with gzip.open(HERE / "paths.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            setting, case, method = (row[k] for k in ("setting", "case", "method"))
            key = setting, case, row["pool"], row["path"]
            gap, lo, hi, g = (float(row[k]) for k in ("target", "lower", "upper", "g"))
            assert abs(targets.setdefault(key, gap) - gap) < 1e-10
            choice = row["choice"]
            resolved = choice in ("A", "B")
            if resolved:
                assert lo <= hi and (hi <= g + 1e-13 if choice == "A" else lo >= -g - 1e-13)
            ok = resolved and (gap <= g if choice == "A" else gap >= -g)
            audit = [int(resolved), int(ok), int(lo <= gap <= hi), int(row["fallback"])]
            assert audit[:3] == [int(row[k]) for k in ("resolved", "acceptable", "covered")]
            acc = totals[setting, case, method]
            for i, x in enumerate(audit):
                acc[i] += x
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["setting", "case", "method", "paths", "mean_distinct_labels", "sd", "resolved", "acceptable", "covered", "fallbacks"])
    for (setting, case, method), (mean, sd, _, n) in sorted(stats.items()):
        writer.writerow([setting, case, method, n, f"{mean:.8f}", f"{sd:.8f}", *totals[setting, case, method]])
    return buffer.getvalue()


def write_article_outputs(output, coverage_cases):
    check_query_first(HERE)
    stats = load_costs()
    csv_text = audit_csv(stats)
    (output / "configuration_summary.csv").write_text(csv_text, encoding="utf-8")
    tables = list(methods_tables(stats))
    tables.extend([
        ("full_coverage_bars.tikz", full_coverage(coverage_cases)),
        ("reported_numbers.tex", reported_tex(stats, CODES)),
        ("population_cache_ablation_table.tex", initialization_table(HERE)),
    ])
    for name, lines in tables:
        (output / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list(csv.DictReader(io.StringIO(csv_text)))
