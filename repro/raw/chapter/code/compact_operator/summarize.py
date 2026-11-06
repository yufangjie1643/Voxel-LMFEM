"""Audit complete seven-case ablations and regenerate all result tables."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/"review_results/structural_ablation_gpu_20261020"
REMOTE=OUT/"remote"
VARIANTS=("dense","constitutive","compact","projection","compact_projection","energy_recovery","energy_fixed","energy_certified")
CASES=("woven64","woven128","woven192","near128","aligned128","contrast128","layered128")
read=lambda p:json.loads(p.read_text(encoding="utf-8"))


def relative(x,y,mandel=False):
    x,y=np.asarray(x),np.asarray(y)
    if mandel:
        w=np.array([1.,1.,1.,2**.5,2**.5,2**.5]);x=x*w[:,None]*w[None,:];y=y*w[:,None]*w[None,:]
    return float(np.linalg.norm(x-y)/max(np.linalg.norm(x),np.linalg.norm(y),1e-300))


def main():
    if read(REMOTE/"status.json")["status"]!="complete":raise RuntimeError("Experiment incomplete")
    records=[read(p) for p in sorted((REMOTE/"results").glob("measure_*.json"))]
    if len(records)!=21 or {(r["case"],r["repeat"]) for r in records}!={(c,i) for c in CASES for i in range(3)}:raise RuntimeError("Missing or duplicate case/repetition")
    old=read(ROOT/"review_results/formula_gpu_20261020/remote/results/measure_hg_baseline_192_0.json")["provenance"]
    source_checks=0
    for r in records:
        if r["status"]!="passed" or len(r["runs"])!=8 or {row["variant"] for row in r["runs"]}!=set(VARIANTS):raise RuntimeError("Incomplete group")
        for name,sha in r["provenance"]["ablation_sources"].items():
            if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=sha:raise RuntimeError("Ablation source mismatch: "+name)
            source_checks+=1
        if r["provenance"]["structural_header_sha256"]!=hashlib.sha256((HERE.parent/"structural_formula_20261020/structural_modal.cuh").read_bytes()).hexdigest():raise RuntimeError("Changed structural header")
        for key in ("source_sha256","pipeline_source_sha256"):
            for path,sha in r["provenance"][key].items():
                if path in old[key]:
                    if sha!=old[key][path]:raise RuntimeError("Frozen dependency changed: "+path)
                    source_checks+=1
    groups=[];accuracy=[]
    for case in CASES:
        subset=[r for r in records if r["case"]==case]
        if len({r["input_hash"] for r in subset})!=1:raise RuntimeError("Inputs differ within case")
        values={variant:[next(row for row in r["runs"] if row["variant"]==variant) for r in subset] for variant in VARIANTS}
        for index,record in enumerate(subset):
            reference=values["energy_fixed"][index]
            cert=values["energy_certified"][index]
            actual=relative(cert["measurement"]["CH"],reference["measurement"]["CH"],True)
            allowance=cert["certified_relative_bound"]+reference["certified_relative_bound"]+1e-12
            if actual>allowance or actual>1e-8:raise RuntimeError("Energy certificate inconsistent with high-accuracy variational reference")
            accuracy.append(dict(case=case,repeat=record["repeat"],certified_CH_error_to_energy_fixed=actual,
                certified_bound=cert["certified_relative_bound"],comparison_rounding_allowance=1e-12))
        for variant in VARIANTS:
            rows=values[variant];measurements=[row["measurement"] for row in rows]
            if any(row["CH_Mandel_relative_to_dense"]>1e-8 for row in rows):raise RuntimeError("Stiffness acceptance failed")
            if variant!="energy_certified" and any(row["max_true_residual"]>1e-8 for row in rows):raise RuntimeError("True residual failed")
            if case=="layered128" and any(row["CH_relative_to_analytical"]>1e-8 for row in rows):raise RuntimeError("Analytical comparison failed")
            times=[m["seconds"] for m in measurements];base=[m["measurement"]["seconds"] for m in values["dense"]]
            phases=[]
            for m in measurements:
                phase=dict(setup=0.,rhs=0.,solve=0.,post=0.)
                for stage in m["stages"]:
                    name=stage["name"]
                    target="rhs" if name.startswith("rhs") else "solve" if name.startswith("solve") else "post" if name.startswith("post") or name=="cross_energy" else "setup"
                    phase[target]+=stage["seconds"]
                phases.append(phase)
            groups.append(dict(case=case,variant=variant,grid=subset[0]["grid"],seconds=times,median_seconds=statistics.median(times),
                min_seconds=min(times),max_seconds=max(times),
                paired_time_saving_percent=[100*(1*t0-t)/t0 for t,t0 in zip(times,base)],
                time_saving_percent=100*(1-statistics.median(times)/statistics.median(base)),
                peak_allocated_GiB=max(m["peak_allocated_bytes"] for m in measurements)/2**30,
                peak_reserved_GiB=max(m["peak_reserved_bytes"] for m in measurements)/2**30,
                cumulative_iterations=[sum(c["iterations"] for c in m["calls"]) for m in measurements],
                restarts=[sum(c["restarts"] for c in m["calls"]) for m in measurements],
                max_true_residual=max(row["max_true_residual"] for row in rows),
                max_CH_Mandel_relative_to_dense=max(row["CH_Mandel_relative_to_dense"] for row in rows),
                max_moduli_relative_to_dense=max(row["engineering_constants_relative"] for row in rows),
                max_CH_relative_to_analytical=max((row.get("CH_relative_to_analytical",0) for row in rows)),
                max_certified_relative_bound=max(row.get("certified_relative_bound",0) for row in rows),
                median_stage_seconds={k:statistics.median(p[k] for p in phases) for k in phases[0]},
                engineering_constants=[row["engineering_constants"] for row in rows],CH=[m["CH"] for m in measurements]))
    lookup={(g["case"],g["variant"]):g for g in groups}
    for g in groups:
        base=lookup[(g["case"],"dense")]
        g["memory_saving_percent"]=100*(1-g["peak_allocated_GiB"]/base["peak_allocated_GiB"])
        fixed=lookup[(g["case"],"energy_fixed")]
        g["time_saving_vs_custom_fixed_percent"]=100*(1-g["median_seconds"]/fixed["median_seconds"])
    result=dict(passed=True,element="HG",cases=7,variants=8,repetitions=3,formal_full_solves=168,formal_loads=1008,
        source_checks=source_checks,groups=groups,certificate_comparisons=accuracy,
        global_max_CH_Mandel_relative_to_dense=max(g["max_CH_Mandel_relative_to_dense"] for g in groups),
        global_max_moduli_relative_to_dense=max(g["max_moduli_relative_to_dense"] for g in groups),
        max_analytical_CH_error=max(g["max_CH_relative_to_analytical"] for g in groups),
        max_fixed_true_residual=max(g["max_true_residual"] for g in groups if g["variant"]!="energy_certified"))
    fi_lines=[];all_groups=groups
    for g in groups:g["element"]="HG"
    if (REMOTE/"fi_status.json").exists():
        from summarize_fi import build
        fi,fi_lines=build();result["fi_supplement"]=fi
        result["combined_formal_full_solves"]=168+fi["formal_full_solves"]
        result["combined_formal_loads"]=1008+fi["formal_loads"]
        result["combined_source_checks"]=source_checks+fi["source_checks"]
        all_groups=groups+fi["groups"]
    (OUT/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    columns=["element","case","grid","variant","median_seconds","min_seconds","max_seconds","time_saving_percent","peak_allocated_GiB","peak_reserved_GiB","memory_saving_percent","max_true_residual","max_CH_Mandel_relative_to_dense","max_moduli_relative_to_dense","max_CH_relative_to_analytical","max_certified_relative_bound"]
    with (OUT/"comparison.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction="ignore");writer.writeheader();writer.writerows(all_groups)
    with (OUT/"moduli.csv").open("w",newline="",encoding="utf-8-sig") as f:
        cols=["element","case","variant","E11","E22","E33","G23","G13","G12"]
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
        for g in all_groups:writer.writerow(dict(element=g["element"],case=g["case"],variant=g["variant"],
            **{k:statistics.median(r[k] for r in g["engineering_constants"]) for k in cols[3:]}))
    lines=["# GPU 多算例全流程消融：结构本构、存储压缩、HG 投影和能量认证", "", "日期：2026-10-20。", "",
        "主矩阵为 HG / 规则周期 HEX8，保留原 α=0.1。七组配置、八种消融、每组配置三个独立进程重复，共 168 次正式六载荷均匀化、1,008 个正式载荷；FI 的三组四配置补充对照见末节。结果表由原始 JSON 生成，完整矩阵和各方向工程模量均保留。", "",
        "## 1 算例和对照协议", "",
        "woven64、woven128、woven192：同一编织几何和 STRONG 材料，64³、128³、192³；near128：原 NEAR 近各向同性纱线；aligned128：原编织几何、STRONG、纱线方向统一 x；contrast128：沿用既有 contrast0.1 输入协议，纱线 E/G 均乘 0.1、泊松比不变；layered128：新定义的 50/50 周期层合材料，界面法向 z、纱线方向 (1,2,3)，有独立解析等效刚度。前六组并非六个独立几何，分别检验网格和材料/方向变化；分层组另检验结构和纱线体积分数变化。", "",
        "同一 RTX 5090，每算例三个独立进程；每个进程以确定性打乱的次序执行八配置，每配置先完整同尺寸六载荷预热，再计完整六载荷任务。配置之间清理缓存。物理作用均为 FP64，迭代 FFT 参考均为 FP32；能量认证另算 FP64 参考逆。计入材料构造、参考准备、新表示准备、RHS、PCG、最终物理残差、应力恢复、能量存储与交叉能量；排除 CPU 输入几何、编译、预热和磁盘。原阶段同步保留。", "",
        "## 2 消融因素", "", "| 配置 | 含义 |", "|---|---|",
        "| dense | 上轮最快 HG combined：C21、本来模态、冻结 PCG、应力 CH |",
        "| constitutive | 只把算子 C:ε 改为结构张量作用，仍保留 C21 和原 RHS/后处理 |",
        "| compact | 在上一项基础上，求解阶段释放 C21 和单元 HG 尺度，RHS/应力也用结构表示 |",
        "| projection | 只换正交补 HG 投影，保留 C21 |",
        "| compact_projection | 压缩与正交补投影同时使用 |",
        "| energy_recovery | compact + 全六载荷变分能量恢复，冻结 PCG 及原残差标准 |",
        "| energy_fixed | compact + 同一能量恢复，标准自定义 CG 按真实残差 1e-8 停止 |",
        "| energy_certified | 与 energy_fixed 相同实现，改按等效刚度误差上界 1e-8 停止 |", "",
        "compact 仍先使用公共 C21 材料构造与原 Zeman 参考准备，再释放 C21；因此没有声称已消除初始化时的全 C21 构造。构造期间临时分配也计入峰值。能量路径在 GPU 保存六组 U 和六组真实残差，认证路径另持有 FP64 参考系数，这些成本全部计入。", "",
        "## 3 固定真实残差标准下的压缩收益", "",
        "时间为三个重复的中位数；峰值为三次中最大的 PyTorch 已分配显存。预留显存另见 CSV/JSON，不能等同实际数组占用。", "",
        "| 算例 | 原方案 s | compact s | 时间减少 | 原峰值 GiB | compact 峰值 GiB | 峰值减少 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for case in CASES:
        b,c=lookup[case,"dense"],lookup[case,"compact"]
        lines.append(f"| {case} | {b['median_seconds']:.6f} | {c['median_seconds']:.6f} | {c['time_saving_percent']:.2f}% | {b['peak_allocated_GiB']:.3f} | {c['peak_allocated_GiB']:.3f} | {c['memory_saving_percent']:.2f}% |")
    lines += ["", "## 4 面向等效刚度的停止准则", "",
        "下表比较 energy_fixed 与 energy_certified，二者使用同一自定义 CG、同样的四步检查频率、同样的变分恢复与存储，主要差别是接受准则。该比较能隔离停止策略；不能将其与冻结 PCG 的迭代数直接用于推断条件数变化。冻结实现报告锁存逻辑计数，自定义实现报告实际执行步数。", "",
        "| 算例 | 固定残差 s | 能量认证 s | 时间减少 | 累计步数：固定→认证 | 认证真实残差最大 | CH 相对固定能量解差异 | 认证相对上界 |", "|---|---:|---:|---:|---|---:|---:|---:|"]
    for case in CASES:
        f,c=lookup[case,"energy_fixed"],lookup[case,"energy_certified"]
        actual=max(r["certified_CH_error_to_energy_fixed"] for r in accuracy if r["case"]==case)
        lines.append(f"| {case} | {f['median_seconds']:.6f} | {c['median_seconds']:.6f} | {c['time_saving_vs_custom_fixed_percent']:.2f}% | {f['cumulative_iterations']} → {c['cumulative_iterations']} | {c['max_true_residual']:.3e} | {actual:.3e} | {c['max_certified_relative_bound']:.3e} |")
    lines += ["", "能量认证的真实残差通常高于 1e-8。它针对完整 Mandel 等效刚度的相对 Frobenius 精度，不承诺位移、局部应力达到原标准。误差上界来自逐相 C≥aC0 以及 HG 尺度比，再用最终真实残差的 FP64 参考逆二次型计算；跨载荷按 C_energy=C_stress−UᵀR/V 恢复。高精度对照使用同算例 energy_fixed 的变分刚度，另计其小误差界及 1e-12 浮点比较余量，全部通过。", "",
        "## 5 所有因素的时间消融", "",
        "以下为相对于同算例 dense 的完整时间减少百分比，负值表示减速。不能用单独 Ku 内核收益代替此表。", "",
        "| 算例 | 仅本构 | compact | 仅投影 | compact+投影 | 仅加能量恢复 | 自定义固定残差 | 能量认证 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for case in CASES:
        lines.append("| "+case+" | "+" | ".join(f"{lookup[case,v]['time_saving_percent']:.2f}%" for v in VARIANTS[1:])+" |")
    lines += ["", "## 6 精度与解析验证", "",
        f"七个固定残差配置的正式载荷均通过 1e-8 真实相对二范数残差，最大值 {result['max_fixed_true_residual']:.3e}。全部配置相对于 dense 的完整 Mandel 刚度最大相对差异 {result['global_max_CH_Mandel_relative_to_dense']:.3e}，六个工程模量向量的最大相对差异 {result['global_max_moduli_relative_to_dense']:.3e}。工程模量由完整六载荷 CH 求逆获得，不以单载荷 C 分量代替 E。", "",
        f"分层算例所有配置和重复均对照独立解析张量，最大 Mandel 刚度相对差异 {result['max_analytical_CH_error']:.3e}。解析式通过法向牵引连续、宏观平均应变约束及相内法向应变消元得到，不使用 FE 求解结果。", "",
        "dense 的应力平均仍有残差造成的一阶输出误差；变分恢复的误差是二阶。因此认证解与 dense 的差异不等于认证解相对于真实离散解的误差，认证界主要与高精度变分 energy_fixed 对照。", "",
        "## 7 分阶段和完整数据", "",
        "每组三次总时间范围、所有峰值预留/已分配显存、累计迭代/重启、真实残差、未人工对称化的 CH、完整工程模量、认证上界和分阶段时间均在 summary.json、comparison.csv 以及 remote/results 中。下面给出 192³ 的阶段中位数。", "",
        "| 配置 | 初始化 s | RHS s | 求解 s | 后处理/交叉能量 s | 峰值已分配 GiB |", "|---|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        g=lookup["woven192",variant];s=g["median_stage_seconds"]
        lines.append(f"| {variant} | {s['setup']:.6f} | {s['rhs']:.6f} | {s['solve']:.6f} | {s['post']:.6f} | {g['peak_allocated_GiB']:.3f} |")
    lines += ["", "## 8 复现与范围", "",
        f"完成 21 个独立进程组，168 次正式全六载荷任务、1,008 个正式载荷，另有同量预热。共 {source_checks} 项源哈希核对通过，冻结物理源、上一轮模态/管线数学代码和本轮三个计算源文件一致。GPU 采样与日志保留。", "",
        "代码与命令见 experiments/structural_ablation_gpu_20261020/README.md。报告由该目录 summarize.py 生成；raw JSON 保留每个重复而非只保留最好结果。适用范围为当前 HG、规则周期网格及已声明材料对称类。本轮未修改论文或生产默认，也不把刚度认证的提前停止宣称为原场残差标准下的加速。"]
    lines+=fi_lines
    (OUT/"REPORT_zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("groups","certificate_comparisons","fi_supplement")},indent=2))
    for case in CASES:
        print(case,[(v,round(lookup[case,v]["median_seconds"],4),round(lookup[case,v]["time_saving_percent"],2)) for v in ("dense","compact","projection","energy_fixed","energy_certified")])


if __name__=="__main__":main()
