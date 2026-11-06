"""Audit the separate FI supplement; called by the main table generator."""
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/"review_results/structural_ablation_gpu_20261020"
REMOTE=OUT/"remote"
CASES=("fi128","fi192","fi_near128")
VARIANTS=("dense","energy_recovery","energy_fixed","energy_certified")
read=lambda p:json.loads(p.read_text())


def build():
    from summarize import relative
    if read(REMOTE/"fi_status.json")["status"]!="complete":raise RuntimeError("FI supplement incomplete")
    records=[read(p) for p in sorted((REMOTE/"results").glob("fi_measure*.json"))]
    if len(records)!=9 or {(r["case"],r["repeat"]) for r in records}!={(c,i) for c in CASES for i in range(3)}:raise RuntimeError("Missing FI record")
    anchor=read(ROOT/"review_results/formula_gpu_20261020/remote/results/measure_fi_analytic_192_0.json")["provenance"]
    groups=[];checks=0;comparisons=[]
    for r in records:
        if r["status"]!="passed" or {row["variant"] for row in r["runs"]}!=set(VARIANTS):raise RuntimeError("Failed FI group")
        if r["fi_source_sha256"]!=hashlib.sha256((HERE/"fi_supplement.py").read_bytes()).hexdigest():raise RuntimeError("FI source differs")
        for name,sha in r["provenance"]["ablation_sources"].items():
            if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=sha:raise RuntimeError("Shared FI source differs")
            checks+=1
        for key in ("source_sha256","pipeline_source_sha256"):
            for path,sha in r["provenance"][key].items():
                if path in anchor[key]:
                    if sha!=anchor[key][path]:raise RuntimeError("Frozen FI source differs: "+path)
                    checks+=1
        fixed=next(row for row in r["runs"] if row["variant"]=="energy_fixed")
        certified=next(row for row in r["runs"] if row["variant"]=="energy_certified")
        actual=relative(fixed["measurement"]["CH"],certified["measurement"]["CH"],True)
        if actual>fixed["certified_relative_bound"]+certified["certified_relative_bound"]+1e-12:raise RuntimeError("FI error-bound comparison failed")
        comparisons.append(dict(case=r["case"],repeat=r["repeat"],CH_error_to_energy_fixed=actual,bound=certified["certified_relative_bound"]))
    for case in CASES:
        subset=[r for r in records if r["case"]==case]
        if len({r["input_hash"] for r in subset})!=1 or len({json.dumps(r["material"],sort_keys=True) for r in subset})!=1:raise RuntimeError("FI inputs differ")
        base_times=[next(row["measurement"]["seconds"] for row in r["runs"] if row["variant"]=="dense") for r in subset]
        for variant in VARIANTS:
            rows=[next(row for row in r["runs"] if row["variant"]==variant) for r in subset]
            times=[row["measurement"]["seconds"] for row in rows]
            if any(row["CH_Mandel_relative_to_dense"]>1e-8 for row in rows):raise RuntimeError("FI CH acceptance failed")
            if variant!="energy_certified" and any(row["max_true_residual"]>1e-8 for row in rows):raise RuntimeError("FI residual failed")
            if variant=="energy_certified" and any(row["certified_relative_bound"]>1e-8 for row in rows):raise RuntimeError("FI bound exceeds target")
            phases=[]
            for row in rows:
                phase=dict(setup=0.,rhs=0.,solve=0.,post=0.)
                for s in row["measurement"]["stages"]:
                    name=s["name"];k="rhs" if name.startswith("rhs") else "solve" if name.startswith("solve") else "post" if name.startswith("post") or name=="cross_energy" else "setup"
                    phase[k]+=s["seconds"]
                phases.append(phase)
            groups.append(dict(element="FI",case=case,variant=variant,grid=subset[0]["grid"],seconds=times,
                median_seconds=statistics.median(times),min_seconds=min(times),max_seconds=max(times),
                time_saving_percent=100*(1-statistics.median(times)/statistics.median(base_times)),
                peak_allocated_GiB=max(row["measurement"]["peak_allocated_bytes"] for row in rows)/2**30,
                peak_reserved_GiB=max(row["measurement"]["peak_reserved_bytes"] for row in rows)/2**30,
                cumulative_iterations=[sum(c["iterations"] for c in row["measurement"]["calls"]) for row in rows],
                max_true_residual=max(row["max_true_residual"] for row in rows),
                max_CH_Mandel_relative_to_dense=max(row["CH_Mandel_relative_to_dense"] for row in rows),
                max_moduli_relative_to_dense=max(row["engineering_constants_relative"] for row in rows),
                max_CH_relative_to_analytical=0.,max_certified_relative_bound=max(row.get("certified_relative_bound",0) for row in rows),
                median_stage_seconds={k:statistics.median(p[k] for p in phases) for k in phases[0]},
                engineering_constants=[row["engineering_constants"] for row in rows],CH=[row["measurement"]["CH"] for row in rows]))
    lookup={(g["case"],g["variant"]):g for g in groups}
    for g in groups:
        g["memory_saving_percent"]=100*(1-g["peak_allocated_GiB"]/lookup[g["case"],"dense"]["peak_allocated_GiB"])
        g["time_saving_vs_custom_fixed_percent"]=100*(1-g["median_seconds"]/lookup[g["case"],"energy_fixed"]["median_seconds"])
    result=dict(passed=True,cases=3,variants=4,repetitions=3,formal_full_solves=36,formal_loads=216,source_checks=checks,
        groups=groups,certificate_comparisons=comparisons,max_CH_Mandel_relative_to_dense=max(g["max_CH_Mandel_relative_to_dense"] for g in groups),
        max_moduli_relative_to_dense=max(g["max_moduli_relative_to_dense"] for g in groups),
        max_fixed_true_residual=max(g["max_true_residual"] for g in groups if g["variant"]!="energy_certified"))
    lines=["", "## 9 FI 全积分补充对照", "",
        "为检验能量路线是否依赖 HG，另做 FI 128³ STRONG、192³ STRONG、128³ NEAR 三组，每组四配置、三次独立进程重复，共 36 次正式六载荷任务、216 个正式载荷。FI 基线为上一轮最快的 modal + FI 解析 FP32 参考逆；未在 FI 中加入 HG 本构压缩或 HG 投影。FI 的证书采用相同八点积分参考符号的独立 FP64 解析逆，不含沙漏项。", "",
        "| FI 算例 | 基线 s | 能量恢复 s | 自定义固定残差 s | 能量认证 s | 相对基线时间减少 | 认证峰值 GiB |", "|---|---:|---:|---:|---:|---:|---:|"]
    for case in CASES:
        b,r,f,c=(lookup[case,v] for v in VARIANTS)
        lines.append(f"| {case} | {b['median_seconds']:.6f} | {r['median_seconds']:.6f} | {f['median_seconds']:.6f} | {c['median_seconds']:.6f} | {c['time_saving_percent']:.2f}% | {c['peak_allocated_GiB']:.3f} |")
    lines += ["", "| FI 算例 | 累计步数：固定→认证 | 认证真实残差最大 | CH 相对固定能量解差异 | 认证相对上界 |", "|---|---|---:|---:|---:|"]
    for case in CASES:
        f,c=lookup[case,"energy_fixed"],lookup[case,"energy_certified"]
        actual=max(r["CH_error_to_energy_fixed"] for r in comparisons if r["case"]==case)
        lines.append(f"| {case} | {f['cumulative_iterations']} → {c['cumulative_iterations']} | {c['max_true_residual']:.3e} | {actual:.3e} | {c['max_certified_relative_bound']:.3e} |")
    lines += ["",f"FI 全部固定残差配置均通过 1e-8 真实残差标准，最大值 {result['max_fixed_true_residual']:.3e}。FI 全部配置与其基线的最大 Mandel 刚度相对差异 {result['max_CH_Mandel_relative_to_dense']:.3e}，工程模量向量最大相对差异 {result['max_moduli_relative_to_dense']:.3e}。认证接受准则及局部场精度限制与 HG 相同。", "",
        "合计 HG 七组八配置与 FI 三组四配置，共 204 次正式全六载荷均匀化、1,224 个正式载荷，另有同量完整预热。FI 补充源、结果和 GPU 监测也保存在同一证据包。"]
    return result,lines
