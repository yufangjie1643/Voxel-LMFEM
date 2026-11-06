"""Verify frozen inputs and regenerate the 25 October 2026 manuscript's numbers.

Usage: python repro/scripts/reproduce_paper_tables.py [--out DIRECTORY]
CPU only. No timings are newly measured. Never changes the frozen campaigns.
"""
from pathlib import Path
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import itertools
import json
import shutil
import statistics
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "campaigns/chapter"
EVIDENCE = CHAPTER / "manuscript_evidence"
FRACTION = ROOT / "campaigns/woven_fraction/revision"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify():
    manifest = read(ROOT / "repro/file_manifest.json")
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Integrity check failed: " + entry["path"])
    return len(manifest["files"])


def relative(a, b):
    return float(np.linalg.norm(np.asarray(a) - b) / np.linalg.norm(b))


def pair_error(a, b):
    return max(relative(x["CH"], y["CH"]) for x, y in itertools.product(a["runs"], b["runs"]))


def portable(value):
    if isinstance(value, str):
        return value.replace("\\", "/")
    if isinstance(value, list):
        return [portable(x) for x in value]
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    return value


def dump(path, value):
    path.write_text(json.dumps(portable(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def csv_write(out, name, rows):
    with (out / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(portable(v)) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def regenerate(out):
    verified = verify()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(CHAPTER))
    analytical = load("paper_analytical", CHAPTER / "analytical_laminate.py")
    legacy = load("paper_legacy", CHAPTER / "reproduce_legacy_tables.py")
    # Historical analyzers write summaries. Run them in temporary copies so the
    # measured payload remains byte-identical on Windows and Linux.
    with tempfile.TemporaryDirectory(prefix="voxel-paper-") as temporary:
        stage = Path(temporary) / "evidence"
        shutil.copytree(EVIDENCE, stage)
        old = stage / "review_results/solver_study_20261022"
        for name in ("summary.json", "summary_with_stencil27.json"):
            dump(old / name, read(old / name))
        chapter = load("paper_chapter", EVIDENCE / "scripts/summarize_chapter_aligned.py")
        chapter.ROOT = stage
        chapter.OUT = stage / "review_results/chapter_aligned_20261024"
        active = load("paper_active", EVIDENCE / "scripts/summarize_active_fraction.py")
        active.ROOT = stage
        active.AUDIT = stage / "review_results/active_fraction_20261024"
        fraction_stage = Path(temporary) / "fraction"
        shutil.copytree(FRACTION / "review_results/woven_fraction_20261025", fraction_stage)
        fractions = load("paper_fractions", FRACTION / "experiments/woven_fraction_20261025/analyze.py")
        with contextlib.redirect_stdout(io.StringIO()):
            chapter.main()
            active_summary = active.summarize()
            fraction_summary = fractions.analyze(fraction_stage)
        summary = read(chapter.OUT / "summary.json")
    groups = {(r["case"], r["variant"]): r for r in summary["groups"]}
    get = lambda case, variant: groups[case, variant]
    tables = []
    for case in ("sphere", "grosmont", "woven"):
        baseline = get(case, "stencil27_jacobi")
        for variant in ("stencil27_jacobi", "execution_fp32"):
            row = get(case, variant)
            tables.append(dict(case=case, variant=variant, grid=192, time_s=row["time"],
                time_min_s=row["time_min"], time_max_s=row["time_max"],
                peak_memory_GiB=row["memory"], iterations=row["iterations"][0],
                speedup=baseline["time"]/row["time"],
                memory_reduction_percent=100*(1-row["memory"]/baseline["memory"]),
                relative_stiffness_discrepancy=pair_error(row, baseline),
                sources=[r["source"] for r in row["runs"]]))
    assert len(tables) == 6
    csv_write(out, "table_3_solver_summary.csv", tables)
    # Table 1: material values are read from the measured woven input.
    source = EVIDENCE / get("woven", "execution_fp32")["runs"][0]["source"]
    material = read(source)["input"]["material"]
    yarn, matrix = material["yarn"], material["matrix"]
    properties = [
        dict(phase="Orthotropic", E1_GPa=yarn["E11"]/1000, E2_E3_GPa=yarn["E22"]/1000,
             G12_G13_GPa=yarn["G12"]/1000, G23_GPa=yarn["G23"]/1000,
             nu12_nu13=yarn["Nu12"], nu23=yarn["Nu23"]),
        dict(phase="Isotropic", E1_GPa=matrix["E"]/1000, E2_E3_GPa=matrix["E"]/1000,
             G12_G13_GPa=matrix["E"]/(2*(1+matrix["Nu"]))/1000,
             G23_GPa=matrix["E"]/(2*(1+matrix["Nu"]))/1000,
             nu12_nu13=matrix["Nu"], nu23=matrix["Nu"])]
    csv_write(out, "table_1_constituent_properties.csv", properties)
    old = EVIDENCE / "review_results/solver_study_20261022"
    laminate = []
    laminate_sources = []
    for group in read(old / "summary_with_stencil27.json")["groups"]:
        if group["case"] not in ("fraction50", "fraction90") or group["variant"] not in ("hybrid", "latest"):
            continue
        for name in group["records"]:
            path = old / name.replace("\\", "/")
            record = read(path)
            assert record["complete"] and record["max_true_residual"] <= 1e-8
            laminate.append(relative(record["measurement"]["CH"], analytical.from_record(record)))
            laminate_sources.append(path.relative_to(EVIDENCE).as_posix())
    verification = [dict(cell="Layered pair", grid=128, reference="Analytical solution",
                         max_relative_discrepancy=max(laminate), sources=laminate_sources)]
    for case in ("sphere", "grosmont", "woven"):
        proposed = get(case, "execution_fp32")
        baseline = get(case, "stencil27_jacobi" if case == "woven" else "chfem")
        verification.append(dict(cell=case, grid=192, reference="Stencil Jacobi-PCG" if case=="woven" else "chfem",
            max_relative_discrepancy=pair_error(proposed, baseline),
            sources=[r["source"] for r in proposed["runs"]+baseline["runs"]]))
    csv_write(out, "table_2_verification.csv", verification)
    # Figure 8: compute each tensor and its discrepancy from raw records,
    # using the current 480^3 reference, not the archived 256^3 reference.
    convergence = read(EVIDENCE / "review_results/grid_extension_20261025/convergence_summary.json")
    assert convergence["complete"] and convergence["reference_grid"] == 480
    reference = read(EVIDENCE / convergence["rows"][-1]["records"][0])["measurement"]["CH"]
    rows = []
    for item in convergence["rows"]:
        records = []
        for name in item["records"]:
            path = EVIDENCE / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == convergence["source_sha256"][name]
            record = read(path)
            assert record["complete"] and record["max_true_residual"] <= 1e-8
            records.append(record)
        assert len(records) == 3
        C = np.asarray(records[0]["measurement"]["CH"])
        repeat_error = max(relative(r["measurement"]["CH"], C) for r in records)
        assert repeat_error < 1e-12
        rows.append(dict(grid=item["grid"], C11_GPa=C[0,0]/1000, C33_GPa=C[2,2]/1000,
            C44_GPa=C[3,3]/1000, reference_grid=480, relative_discrepancy=relative(C, reference),
            repeat_difference=repeat_error, sources=item["records"]))
    assert [r["grid"] for r in rows] == list(range(64,257,32)) + [320,384,448,480]
    csv_write(out, "figure_8_grid_convergence.csv", rows)
    chain = []
    for variant in ("stencil27_fft", "modal_fft", "hybrid_fft", "background_fft", "execution_fp64", "execution_fp32"):
        row = get("woven", variant)
        assert row["iterations"] == [327]
        chain.append(dict(variant=variant, time_s=row["time"], time_min_s=row["time_min"],
            time_max_s=row["time_max"], peak_memory_GiB=row["memory"], iterations=327,
            stiffness_action_ms=row["Ku"], preconditioner_ms=row["preconditioner"],
            sources=[r["source"] for r in row["runs"]]))
    csv_write(out, "figure_9_optimization_chain.csv", chain)
    cost = []
    scaling = read(EVIDENCE / "review_results/grid_refinement_20261023/scaling_summary.json")
    for item in scaling["groups"]:
        normalized = dict(item, records=[n.replace("\\", "/") for n in item["records"]])
        group = legacy.group(normalized, EVIDENCE)
        cost.append(dict(grid=item["grid"], variant=item["variant"],
            time_s=group["time_s"], peak_memory_GiB=group["nvml_GiB"],
            iterations=group["iterations"], sources=normalized["records"]))
    csv_write(out, "figure_10_grid_cost.csv", cost)
    csv_write(out, "figure_11_woven_fraction.csv", fraction_summary["rows"])
    csv_write(out, "figure_11_gains.csv", fraction_summary["gains"])
    csv_write(out, "figure_D1_active_fraction.csv", active_summary["groups"])
    components = []
    for variant in ("direct_fft", "modal_fft", "hybrid_fft", "background_fft",
                    "stencil27_fft", "stencil14_fft", "execution_fp64"):
        row = get("woven", variant)
        components.append(dict(variant=variant, time_s=row["time"], peak_memory_GiB=row["memory"],
            stiffness_action_ms=row["Ku"], preconditioner_ms=row["preconditioner"],
            sources=[r["source"] for r in row["runs"]]))
    csv_write(out, "appendix_D_component_comparisons.csv", components)
    x, y = get("woven", "stencil27_fft"), get("woven", "execution_fp32")
    gains = fraction_summary["gains"]
    headlines = dict(speedup_range=[min(r["speedup"] for r in tables[1::2]),max(r["speedup"] for r in tables[1::2])],
        memory_reduction_percent_range=[min(r["memory_reduction_percent"] for r in tables[1::2]),
                                       max(r["memory_reduction_percent"] for r in tables[1::2])],
        max_stiffness_difference_vs_jacobi=max(r["relative_stiffness_discrepancy"] for r in tables),
        controlled_time_reduction_percent=100*(1-y["time"]/x["time"]),
        controlled_memory_reduction_percent=100*(1-y["memory"]/x["memory"]),
        fp32_fp64_stiffness_difference=pair_error(y,get("woven","execution_fp64")),
        max_woven_fraction_stiffness_difference=fraction_summary["max_stiffness_difference"],
        hybrid_memory_reduction_percent=[100*min(g["hybrid_memory_reduction"] for g in gains),
                                        100*max(g["hybrid_memory_reduction"] for g in gains)],
        background_time_reduction_percent=[100*min(g["background_time_reduction"] for g in gains),
                                          100*max(g["background_time_reduction"] for g in gains)])
    dump(out / "headline_checks.json", headlines)
    # Independent checks against printed values guard against selecting stale groups.
    expected_time = [42.931,1.440,70.043,1.501,70.885,1.890]
    expected_memory = [19.242,2.055,22.037,2.055,18.104,2.385]
    assert [round(r["time_s"],3) for r in tables] == expected_time
    assert [round(r["peak_memory_GiB"],3) for r in tables] == expected_memory
    assert [format(r["max_relative_discrepancy"],".3e") for r in verification] == [
        "6.009e-13","1.532e-08","8.708e-09","1.425e-10"]
    assert fraction_summary["record_count"] == 189 and fraction_summary["identical_iterations"]
    assert len(active_summary["groups"]) == 18
    assert verify() == verified  # regeneration did not alter frozen inputs
    report = dict(status="passed", verified_files=verified, table_3_rows=len(tables),
        convergence_grids=len(rows), finest_grid=480, cost_groups=len(cost),
        woven_fraction_records=fraction_summary["record_count"],
        active_fraction_records=len(active_summary["records"]),
        verification_scope="Archived-result regeneration and analytical laminate calculations; no new GPU timings")
    dump(out / "verification.json", report)
    manifest = read(ROOT / "repro/table_manifest.json")
    lines = ["# Current manuscript tables and numerical figure data", "",
        "Generated from verified measured records. Times are three-process medians; memory is the maximum sampled process peak.", "",
        "| Manuscript item | Section | Output |", "|---|---|---|"]
    for item in manifest["items"]:
        lines.append(f'| {item["item"]} | {item["section"]} | [{item["output"]}]({item["output"]}) |')
    lines += ["", "Raw source columns are relative to campaigns/chapter/manuscript_evidence, except Figure 11 sources,",
              "which are relative to campaigns/woven_fraction/revision/review_results/woven_fraction_20261025.", "",
              "Figure 8 uses 480³ as the reference. Table 3 uses stencil Jacobi–PCG for all three cases.",
              "Archived 19-row summaries and 256³-reference summaries are historical inputs, not current manuscript outputs.", ""]
    (out / "TABLES.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if sys.flags.optimize:
        raise SystemExit("Run without -O: numerical assertions are required.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "repro/generated")
    args = parser.parse_args()
    regenerate(args.out.resolve())
