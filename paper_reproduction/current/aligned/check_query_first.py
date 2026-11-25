"""Verify the population ledger against the affected query paths."""
import csv
import gzip
import json
import math


def check_query_first(directory):
    with gzip.open(directory / "population/query_first_history.json.gz", "rt") as f:
        paths = json.load(f)
    with gzip.open(directory / "population/paths.csv.gz", "rt", newline="") as f:
        rows = {(r["code"], int(r["group"])): r for r in csv.DictReader(f)
                if r["code"] in ("T4","V3") and r["method"] == "separate"}
    assert len(paths)==len(rows)==24
    assert set(rows)=={(code,i) for code in ("T4","V3") for i in range(6,18)}
    for q in paths:
        row=q["row"]
        assert all(str(v)==rows[q["code"],q["group"]][k] for k,v in row.items())
        prior_eps=q["epsilon0"]
        assert prior_eps>=row["g"] and len(q["history"])>0
        for i,h in enumerate(q["history"]):
            assert h["blocks"]==256*2**i and h["epsilon"]<=prior_eps and h["epsilon"]>=0
            prior_eps=h["epsilon"]
            l,u=h["J"]
            lo,hi=h["interval"]
            assert math.isclose(lo,l-prior_eps) and math.isclose(hi,u+prior_eps)
            expected=("unresolved" if l>u else "A" if hi<=row["g"] else "B" if lo>=-row["g"] else "unresolved")
            assert expected==h["choice"] and (expected=="unresolved" or h is q["history"][-1])
        if row["fallback"]:
            assert q["history"][-1]["blocks"]==32768 and q["history"][-1]["choice"]=="unresolved" and q["refit_blocks"]>0
        else:
            assert q["history"][-1]["choice"]==row["choice"]
        stage=json.loads(row["stages"])
        assert sum(stage.values())==row["labels"] and stage["comparison"]==q["history"][-1]["comparison_labels"]
        assert (row["lower"]<=row["target"]<=row["upper"])==row["covered"]
        assert (row["target"]<=row["g"] if row["choice"]=="A" else row["target"]>=-row["g"] if row["choice"]=="B" else False)==row["acceptable"]
    print("Verified all 24 matched paths, stopping inspections, label unions, coverage and selection audits.")
