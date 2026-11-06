"""Regenerate every numerical claim in the chain report from archived JSON."""
from pathlib import Path
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[1]/"review_results"/HERE.name
RAW = REPORT/"remote"/"results"


def read(name):
    return json.loads((RAW/name).read_text(encoding="utf-8"))


def relative(a, b):
    aa = [float(v) for row in a for v in row]
    bb = [float(v) for row in b for v in row]
    return math.sqrt(sum((x-y)**2 for x,y in zip(aa,bb))/max(sum(y*y for y in bb),1e-300))


def stats(rows):
    return dict(median_seconds=statistics.median(r["measurement"]["seconds"] for r in rows),
                times=[r["measurement"]["seconds"] for r in rows],
                peak_GiB=max(r["measurement"]["peak_allocated_bytes"]/2**30 for r in rows),
                logical_iterations=[sum(c["iterations"] for c in r["measurement"]["calls"]) for r in rows],
                max_true_residual=max(v for r in rows for c in r["measurement"]["calls"] for v in c["true_residuals"]),
                restarts=sum(c["restarts"] for r in rows for c in r["measurement"]["calls"]))


def main():
    for name in ("complete.json","reference_complete.json","closed_complete.json"):
        assert read(name)["complete"]
    groups, accuracy = {}, []
    for kind in ("hg","fi"):
        groups[kind] = {}
        for mode in ("prior","baseline","closed"):
            rows = [read(f"closed_formal_{kind}_{mode}_strong_192_{r}.json") for r in range(3)]
            assert len({json.dumps(x["input_hash"],sort_keys=True) for x in rows}) == 1
            groups[kind][mode] = stats(rows)
            if mode != "prior":
                error = max(relative(x["measurement"]["CH"],
                    read(f"closed_formal_{kind}_prior_strong_192_{x['repeat']}.json")["measurement"]["CH"]) for x in rows)
                assert error < 1e-8
                groups[kind][mode]["CH_relative_to_prior"] = error
        prior, core, final = (groups[kind][k] for k in ("prior","baseline","closed"))
        final["time_reduction_percent"] = 100*(1-final["median_seconds"]/prior["median_seconds"])
        final["memory_reduction_percent"] = 100*(1-final["peak_GiB"]/prior["peak_GiB"])
        final["reference_time_reduction_percent"] = 100*(1-final["median_seconds"]/core["median_seconds"])
        assert all(v["max_true_residual"] <= 1e-8 and v["restarts"] == 0 for v in groups[kind].values())
        v = read(f"closed_validate_{kind}.json")
        assert v["complete"] and v["CH_relative_to_CPU"] < 1e-8
        accuracy.append(dict(kind=kind, CH_relative_to_CPU=v["CH_relative_to_CPU"],
            independent_true_residual=max(c["independent_true_residual"] for c in v["calls"])))
    max_ch=max(groups[k]["closed"]["CH_relative_to_prior"] for k in groups)
    max_cpu=max(x["independent_true_residual"] for x in accuracy)
    max_cpu_ch=max(x["CH_relative_to_CPU"] for x in accuracy)
    profiles=read("post_profile.json")
    meta=read("closed_formal_hg_closed_strong_192_0.json")["reference_design"]
    ablations=[]
    for path in sorted(RAW.glob("tune_*.json")):
        d=json.loads(path.read_text())
        ablations.append(dict(file=path.name, kind=d["kind"], variant=d["variant"], **stats([d])))
    reference_search=[]
    for kind in ("hg","fi"):
        for grid, case in ((192,"strong"),(128,"near")):
            row=dict(kind=kind,grid=grid,case=case)
            for mode in ("baseline","optimized"):
                d=read(f"ref_tune_{kind}_{mode}_{case}_{grid}_0.json")
                row[mode]=stats([d])
            reference_search.append(row)
    smoke=[]
    for kind in ("hg","fi"):
        pair={}
        for mode in ("baseline","closed"):
            d=read(f"closed_smoke_{kind}_{mode}_near_128_0.json")
            pair[mode]=dict(**stats([d]),CH=d["measurement"]["CH"])
        error=relative(pair["closed"].pop("CH"),pair["baseline"].pop("CH"))
        assert error<1e-8
        smoke.append(dict(kind=kind, CH_relative=error, **pair))
    summary=dict(groups=groups,accuracy=accuracy,ablations=ablations,
                 reference_search=reference_search,near128_single_runs=smoke,post_profile=profiles,
                 max_final_CH_relative_to_prior=max_ch)
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    lines=[
        "# 模态有限元、FFT 与 PCG 的联合优化及闭式参考介质",
        "",
        "已完成代码、CPU 独立组装验证、RTX 5090 GPU 消融和 192³ 六载荷三次重复。全部节点、全部频率和原物理离散保留；最终以投影后的 FP64 真实残差 ‖f−Ku‖₂/‖f‖₂≤10⁻⁸ 验收。",
        "",
        "## 完整流程结果",
        "",
        "下表来自同一轮交错顺序、独立进程的三次重复，时间取中位数。旧整链已经包括模态 Ku、原生 cuFFT 和上一轮 PCG 优化。新整链增加边界融合、首次收敛冻结；HG 同时加入此前验证过的材料紧凑表示。最终方案再加入闭式参考。各阶段收益没有相乘。",
        "",
        "| 单元 | 旧整链 / s | 新整链 / s | 加闭式参考 / s | 总时间下降 | 峰值 allocated 显存 / GiB | 六载荷逻辑迭代数 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for kind in ("hg","fi"):
        q=groups[kind];p,c,z=(q[x] for x in ("prior","baseline","closed"))
        lines.append(f"| {kind.upper()} | {p['median_seconds']:.6f} | {c['median_seconds']:.6f} | {z['median_seconds']:.6f} | {z['time_reduction_percent']:.2f}% | {p['peak_GiB']:.6f} → {z['peak_GiB']:.6f} | {p['logical_iterations'][0]} → {z['logical_iterations'][0]} |")
    lines += [
        "",
        f"最终方案与旧整链的 CH 相对 Frobenius 差最大为 {max_ch:.3e}。全部正式运行真实残差≤10⁻⁸、无重启。8³ 最终组合对独立 CPU CSR 组装的最大真实残差为 {max_cpu:.3e}，CH 对 CPU 直接解的最大相对差为 {max_cpu_ch:.3e}。",
        "",
        "迭代数为六载荷报告的逻辑迭代数之和；主机每四步检查，不能把该数字直接当作实测 FFT 调用次数。峰值为 PyTorch allocated 统计，包含显式 cuFFT 工作区，不包含驱动上下文；reserved 另存于原始记录。全流程计时包含材料、算子、参考介质、RHS、六次求解、真实残差和应力恢复，排除 CPU 几何生成、JIT、磁盘 IO；每次测量前都有同尺寸完整六载荷预热。",
        "",
        "## 为什么把三个模块放在一起还能省",
        "",
        r"成本可以写成 $T=T_{\mathrm{setup}}+n_{\mathrm{it}}(T_K+T_{\mathrm{FFT}}+T_{\mathrm{vec}})$。模态法已经降低 $T_K$，原生 FFT 降低 $T_{\mathrm{FFT}}$；本轮先减少模块交界的 $T_{\mathrm{vec}}$，再通过参考介质降低 $n_{\mathrm{it}}$。",
        "",
        "### 1. 在消费者中使用 IFFT 输出，不生成 FP64 中间向量",
        "",
        r"设 IFFT 输出的 FP32 SoA 数据为 $s$。原程序生成 $z_i=\mathrm{FP64}(s_i)$，再算 $\rho=\sum_i r_i z_i$ 和 $p_i=z_i+\beta p_i^{old}$。新程序直接算：",
        "",
        r"$$\rho=\sum_i r_i\,\mathrm{FP64}(s_i),\qquad p_i=\mathrm{FP64}(s_i)+\beta p_i^{old}.$$",
        "",
        "FP32 到 FP64 的转换精确表示原值；乘法、累加、方向更新仍用 FP64。残差先在 FP64 中更新，再同时写出下一次 FFT 的 FP32 输入。归约顺序可能改变舍入，但没有新增场量量化，也没有使用近似的 Parseval 内积替代物理内积。",
        "",
    ]
    p0,p1=(profiles["rows"][k] for k in ("baseline","boundary"))
    lines += [
        f"独立 192³ 后处理测试：IFFT 输出到新方向的时间从 {p0['median_ms']:.6f} ms 降至 {p1['median_ms']:.6f} ms；增量峰值从 {p0['incremental_peak_bytes']/2**20:.3f} MiB 降至 {p1['incremental_peak_bytes']/2**20:.3f} MiB。证实编译器没有再物化整条 FP64 z。这是局部结果，完整收益以上表为准。",
        "",
        "### 2. 把首次收敛变成冻结状态，消除历史位移快照",
        "",
        r"令 $\ell=\min\{k:\|r_k\|/\|b\|<\tau\}$。对任何 $k<\ell$，都有 $\|r_k\|/\|b\|\ge\tau>\|r_\ell\|/\|b\|$，因此首次达标的位移一定是当时的最优残差位移。原实现此前写入的 best_u 不会用于成功返回。",
        "",
        "现在先完成第 ℓ 步，再锁定状态；后续主机检查前的尾步冻结 u，直接返回该向量。省掉反复的全向量快照和一条 best_u，192³ 对应 162 MiB 的向量存储。整机峰值还受其他阶段支配，不能将这 162 MiB 直接叠加到峰值收益。",
        "",
        "若到达 max_iter 仍未收敛，则重新调用冻结的原求解器，保留其返回历史最优解的接口语义；异常路径有额外时间成本。GPU 原子求和使重跑不承诺逐位复现先前轨迹。已覆盖 max_iter=1、零 RHS、非零初值和已收敛初值。当前实现限定单个 RVE、compiled 更新和延迟主机检查；事件冻结模式不接受带外部副作用的 recycle_callback。",
        "",
        "## 底层数学改进：无需搜索的最优材料谱界参考",
        "",
        "假设某个实际材料相 Cm 是各向同性，并且对所有相满足 Cm≼Ce，即基体在任意应变方向上的能量均不大于其他相。不能只比较 E11；代码检查六维广义特征值来验证这个条件。各向同性软相使该条件对纱线取向旋转不变。",
        "",
        r"任取 SPD 材料参考张量 $C_0$，若 $aC_0\preceq C_e\preceq bC_0$ 对所有相成立，则特别有",
        "",
        r"$$aC_0\preceq C_m,\quad C_e\preceq bC_0\quad\Longrightarrow\quad C_e\preceq (b/a)C_m.$$",
        "",
        r"因此 $b/a\ge L:=\max_e\lambda_{\max}(C_e,C_m)$。取 $C_0=sC_m$ 时，$a=1/s,\ b=L/s$，恰好达到下界：",
        "",
        r"$$\boxed{\min_{C_0\succ0}\frac{b(C_0)}{a(C_0)}=\max_e\lambda_{\max}(C_e,C_m),\qquad C_0=sC_m.}$$",
        "",
        "这里优化的是所有 SPD **材料参考张量**的最坏材料能量界。它不是任意全局预条件器的最优性，也不证明实际 PCG 步数最少。代码固定参考剪切模量的尺度，仅修改体积模量/剪切模量之比，以避免无用的整体缩放。",
        "",
        r"由相同单元应变映射的合同变换和装配求和，可得 $aM\preceq K\preceq bM$，从而在公共正能量子空间上有 $\kappa(M^{-1/2}KM^{-1/2})\le b/a$。",
        "",
        "HG 也可使用这一推导：本项目稳定化尺度为 scale(C)=(I₁+7I₂)/60=tr(WC)，其中 t=(1,1,1,0,0,0)ᵀ、D=diag(1,1,1,2,2,2)、W=(ttᵀ+7D)/60≻0。它是正线性泛函，所以相同材料界自动约束 HG 尺度比。几何、稳定化系数和积分规则须对应相同；本次材料远高于实现中的数值下限，未触发尺度截断。",
        "",
        f"本次 STRONG 的参考 bulk/shear 从 {meta['baseline']['ratio']:.9f} 变成 {meta['selected']['ratio']:.9f}，最坏材料谱界从 {meta['baseline']['condition_bound']:.9f} 降为 {meta['selected']['condition_bound']:.9f}。只做几个 6×6 特征值检查，无参数扫描，每个频点的应用成本保持原样。",
        "",
        "理论界针对精确参考算子；FP32 FFT 的舍入不在该谱界证明中，最终精度仍依靠独立 FP64 真实残差认证。条件不满足时保留旧参考，界打平时也保留旧参考。",
        "",
        "此前先实现了 161 点对数参数扫描并做局部精化。它在 STRONG 降低迭代，但在 NEAR 只增加设置成本；闭式推导将这一搜索消掉。NEAR 128³ 结果仅作一次运行的适用性检查，不据此宣称普遍加速。",
        "",
        "| 单元 | NEAR 原参考 / s | 闭式检查 / s | 逻辑迭代数 | CH 相对差 |",
        "|---|---:|---:|---|---:|",
    ]
    for row in smoke:
        p,z=row["baseline"],row["closed"]
        lines.append(f"| {row['kind'].upper()} | {p['median_seconds']:.6f} | {z['median_seconds']:.6f} | {p['logical_iterations'][0]} → {z['logical_iterations'][0]} | {row['CH_relative']:.3e} |")
    lines += [
        "",
        "## 单项消融（调优阶段，各一次完整任务）",
        "",
        "| 单元 | 方案 | 秒 | 峰值 GiB | 逻辑迭代数 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in ablations:
        lines.append(f"| {row['kind'].upper()} | {row['variant']} | {row['median_seconds']:.6f} | {row['peak_GiB']:.6f} | {row['logical_iterations'][0]} |")
    lines += [
        "",
        "boundary=边界表示融合；snapshot=首次收敛冻结；combined=两者；compact=此前 HG 材料紧凑表示；compact_combined=三者。材料压缩是针对已验证四方对称纱线本构的精确表示，不能直接替代任意各向异性 6×6 本构。FI 本轮没有接入该 HG 专用材料核。14 点边通量与单元模态是不同 Ku 表示，当前仍选用实测更快的单元模态路径。",
        "",
        "## 复现与证据",
        "",
        "源代码在 `experiments/chain_optimization_20261021/`；仅新增独立实验，没有修改冻结运行库或论文。`remote/results/` 保存原始 JSON 与日志；`summary.json` 和本报告由 `write_report.py` 重建。每个测量包含输入散列、材料、版本和源文件散列。",
        "",
        "新入口：`python run_rve.py --kind hg --grid 192 --out hg192.json`，FI 改为 `--kind fi`。默认在已验证 STRONG 192³ 使用闭式参考；其他尺寸或 NEAR 保留旧参考，可显式用 `--reference closed` 测试。已有输出拒绝覆盖。路径及依赖见代码目录 README。",
        "",
        "代码入口依赖已有冻结实验环境；并非独立安装的软件包。GPU 扩展和原生 FFT 计划限定创建时的 CUDA 流，不能并发复用工作区。当前性能证据覆盖这张 RTX 5090 与所列算例，材料谱界最优不等于所有几何、硬件上的时间最优。",
        "",
        "参考介质参数优化已有研究基础，本报告不声明该方向首次提出。Moulinec–Silva 的迭代格式与这里的 PCG 不同，其工作用于定位参考介质选择问题：[2014 原论文](https://doi.org/10.1002/nme.4614)。PCG 收敛还取决于谱分布，材料条件数上界只是代理：[Axelsson–Lindskog 1986](https://doi.org/10.1007/BF01389448)。本报告的软相闭式界通过上述不等式直接推导，其文献新颖性尚未专门检索确认。",
    ]
    (REPORT/"REPORT_zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    evidence=REPORT/"evidence.tar.gz"
    manifest=dict(files={str(p.relative_to(REPORT)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted((REPORT/"remote").rglob("*")) if p.is_file()},
                  local_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in HERE.glob("*") if p.is_file()},
                  evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest() if evidence.exists() else None)
    (REPORT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(groups,indent=2))


if __name__=="__main__":
    main()
