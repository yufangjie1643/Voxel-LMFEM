"""Generate the structural-formula report from CPU checks and GPU probes."""
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/"review_results/structural_formula_20261020"
read=lambda p:json.loads(p.read_text())


def main():
    cpu=read(OUT/"checks.json");assert cpu["passed"]
    gpu=[read(OUT/"remote/results"/f"probe_hg_192_interleaved_r{i}.json") for i in range(3)]
    assert len({r["input_hash"] for r in gpu})==1
    for record in gpu:
        for name in ("structural_kernels.cu","structural_modal.cuh","gpu_probe.py"):
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==record["source_sha256"][name],name
        assert all(row.get("Ku_relative",0)<1e-12 and row.get("energy_relative",0)<1e-12 for row in record["rows"])
    rows=[]
    for energy in (False,True):
        group=[]
        for variant in ("modal_baseline","C21_projector","structural_projector","structural_modal"):
            values=[next(row["median_ms"] for row in r["rows"] if row["variant"]==variant and row["energy"]==energy) for r in gpu]
            group.append(dict(variant=variant,energy=energy,process_medians_ms=values,
                median_ms=statistics.median(values),minimum_ms=min(values),maximum_ms=max(values)))
        base=group[0]["median_ms"]
        for g in group:g["time_saving_percent"]=100*(1-g["median_ms"]/base)
        rows.extend(group)
    sizes=gpu[0]["representation_bytes"]
    original=sizes["old_C21"]+sizes["old_HG_scale"]
    compact=sum(sizes[k] for k in ("new_index","new_frame","new_parameters"))
    result=dict(passed=True,gpu_scope="resident Ku / fused pKu only; not full solves or complete-workflow peak memory",
                gpu_rows=rows,material_description_GiB=dict(original=original/2**30,structural=compact/2**30),
                material_description_saving_percent=100*(1-compact/original),cpu=cpu)
    (OUT/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# 从基础公式继续降本：本构结构、HG 正交补和目标量误差", "", "日期：2026-10-20。", "",
        "结论：当前最值得继续的是材料作用的结构分解，以及面向等效刚度的能量误差控制。HG 正交补公式代数正确，但本轮 GPU 实测慢于原模态实现，暂不采用。已完成独立 CPU 全六载荷检验，以及 16³、192³ HG GPU 算子检验；未修改生产默认或论文。", "",
        "## 1 本构矩阵可以整个省掉", "",
        r"设局部纱线方向为 $n,b,c$，三者正交归一，$P=n\otimes n$，$T=b\otimes c+c\otimes b$。令 $t=\mathrm{tr}\varepsilon$、$r=\varepsilon n$、$q=n\cdot r$，当前材料可精确写成：", "",
        r"\[\sigma=(\lambda t+\alpha q)I+2\mu\varepsilon+(\alpha t+\beta q)P+2\gamma(n\otimes r+r\otimes n)+\delta(T:\varepsilon)T.\]", "",
        "该式中的 ε 是物理对称应变张量，工程剪切应变需除以 2 后进入张量；输出为物理应力。对应局部 Voigt 顺序 11、22、33、23、13、12，系数为：", "",
        r"\[\lambda=C_{23},\quad\mu=(C_{22}-C_{23})/2,\quad\alpha=C_{12}-C_{23},\quad\gamma=C_{66}-\mu,\]", "",
        r"\[\beta=C_{11}-C_{22}-2\alpha-4\gamma,\qquad\delta=C_{44}-\mu.\]", "",
        "这里的 C 下标用通常 1–6 的工程 Voigt 记号。适用条件为 C22=C33、C12=C13、C55=C66，且局部无法向剪切耦合；脚本重建局部完整 C 检查这些条件，超出该材料对称类会拒绝。一般正交各向异性或一般 21 参数材料不能直接套用这六参数特例。", "",
        "实际 STRONG 材料 E22=E33=9000、ν23=0.35、G23=3500；其横观各向同性对应的横向剪切应为 3333.333…，所以 δ=166.666… 必须保留。去掉 δ，在局部纯横向剪切试验中产生 4.7619% 应力相对误差。NEAR 参数也保留了其微小偏差，未强行改为理想各向同性。", "",
        "均匀基体只需 σ=λm tr(ε)I+2μm ε。于是基体单元不读 C21；纱线单元只存 n 的三个分量及 T 的六个分量，六个材料系数为相共享常量。HG 尺度由代码中旋转不变量给定，也可每相计算一次。这里是材料作用的精确因子化，保留原积分、边界和物理材料。", "",
        f"192³ 的材料描述数组计数：C21 加 HG 尺度 {original/2**30:.6f} GiB；全体单元 int32 索引、活动单元九个方向系数及相参数 {compact/2**30:.6f} GiB，减少 {100*(1-compact/original):.2f}%。这只比较上述命名数组。GPU 对照为保持独立基线仍同时留有原 C21，因此没有测量降低后的完整任务峰值显存。", "",
        "真正接入无 C21 初始化时，RHS、平均应力和参考统计也需改用同一表示。当前 Zeman 参考统计的剪切样本随方向变化，不能只按相参数直接替代；应保留原采样点及统计定义，仅在这些采样点重建所需小矩阵。", "",
        "## 2 HG 可用正交补，但当前 GPU 不划算", "",
        r"在规则长方体八节点单元上，令 $A=[\mathbf1,s_x,s_y,s_z]$，列为常数及三个线性符号场，则 $A^TA=8I$。当前四个标量 HG 模态张成其正交补：", "",
        r"\[H_sH_s^T=I_8-\frac18AA^T,\qquad K_eu=VB_0^TCB_0u+\eta\left[\left(I_8-\frac18AA^T\right)\otimes I_3\right]u.\]", "",
        "因此可只算常数与三个线性节点矩，不显式计算四个高阶标量 HG 模态。公式仅对当前规则长方体、归一化正交 HG 基和统一标量罚系数成立，不能无条件推广到畸变 HEX8 或不同模态罚系数。", "",
        "GPU 对照显示：保留 C21、只换正交补的版本，以及本构分解加正交补的版本，均慢于当前模态基线；保留原模态、仅替换本构作用更好。简化投影表达式并不保证编译后的运算和访存更省。", "",
        "## 3 面向等效刚度的变分公式与误差上界", "",
        r"令六个载荷组成 $F$，周期微观解满足 $KU_*=F$，$V_\Omega$ 为 RVE 体积。定义全六载荷的变分等效刚度：", "",
        r"\[C_E^H(U)=\langle C\rangle+\frac1{V_\Omega}(U^TKU-U^TF-F^TU).\]", "",
        r"将 $U=U_*+E$ 代入并使用 $KU_*=F$，线性误差项完全消去：", "",
        r"\[C_E^H(U)-C_*^H=\frac1{V_\Omega}E^TKE\succeq0.\]", "",
        "直接应力平均对未收敛位移一般含一阶误差，而此能量形式对位移误差是二阶的。仅用 <C>−FᵀU/V 替代应力平均，在任意近似解上并不具有上述二阶保证，必须保留完整的交叉项与 UᵀKU。", "",
        r"若逐相本构 $C_e\succeq aC_0$，HG 时再要求 $\eta_e\ge a\eta_0$，则零均值子空间上 $K\succeq aM$，其中 $M$ 是相同离散的均匀参考刚度。真实残差 $R=F-KU$ 给出：", "",
        r"\[0\preceq C_E^H(U)-C_*^H\preceq\frac1{aV_\Omega}R^TM^{-1}R.\]", "",
        "a 取局部材料相对于参考 C0 的最小广义特征值；HG 再与罚系数比取最小。由于 C0 各向同性，广义特征值在物理旋转下不变，可以只算各相局部的小特征值问题。六个对角上界的和控制 Mandel 等效刚度误差的谱范数及 Frobenius 范数。", "",
        "CPU 试验用真实 STRONG 相参数、20% 随机活动单元、4×5×6 周期网格和明确指定的各向同性参考；参考逆在小模型上由 FP64 显式矩阵形成。误差目标为完整 Mandel 等效刚度相对 Frobenius 误差 ≤1e-8。该数学上界不要求把位移真实残差也降到 1e-8，因此它是另一种输出精度协议。", "",
        "| 离散 | 原真实残差停止：六载荷累计迭代 | 能量认证累计迭代 | 迭代减少 | 刚度实际相对误差 | 刚度误差上界 |", "|---|---:|---:|---:|---:|---:|"]
    for r in cpu["periodic"]:
        before=sum(x["iterations"] for x in r["conventional_true_residual_solves"])
        after=sum(x["iterations"] for x in r["energy_certified_solves"])
        lines.append(f"| {r['kind'].upper()} | {before} | {after} | {100*(1-after/before):.2f}% | {r['certified_CH_Mandel_relative_error']:.3e} | {r['certified_CH_relative_error_upper_bound']:.3e} |")
    lines += ["", "这些是 CPU 合成小模型的迭代证据，不是 192³ 的完整任务加速。能量认证停止时，真实残差约为 4e-6 至 2e-5；若还要求局部应力、位移场达到原残差标准，仍需继续求解。GPU 实现还需处理完整交叉能量的存储/归约，以及认证阶段 FP64 参考逆与浮点误差，不能直接把现有 FP32 预条件内积当成精确误差证书。", "",
        "## 4 独立验证与 GPU 算子测量", "",
        f"本构试验：STRONG、NEAR 各 2048 个随机旋转，最大逐样本应力相对差 {max(r['max_rotation_relative'] for r in cpu['constitutive']):.3e}。HG 投影：随机完整 SPD 材料、不同边长与罚系数，单元作用最大相对差 {max(r['force_relative'] for r in cpu['hourglass_projector']['rows']):.3e}。组合公式与独立 CPU 常规装配的全局算子、RHS、位移和六载荷完整等效刚度均核对通过。", "",
        "GPU：同一 RTX 5090、同一 192³ STRONG 周期编织 RVE，物理计算 FP64，无 fast_math；三个独立进程，每个先预热一秒，再交错测七批、每批十次。表中为三个进程内中位数的中位数。", "",
        "| 算子 | 版本 | 中位数 ms | 三进程中位数范围 ms | 相对基线时间减少 |", "|---|---|---:|---|---:|"]
    for r in rows:lines.append(f"| {'Ku 与能量' if r['energy'] else 'Ku'} | {r['variant']} | {r['median_ms']:.6f} | {r['minimum_ms']:.6f}–{r['maximum_ms']:.6f} | {r['time_saving_percent']:.2f}% |")
    lines += ["", "测量范围仅为驻留物理算子及融合 pᵀKp，包含输出清零、节点原子累加和能量归约；不包含 FFT、PCG、初始化、RHS、最终真实残差和应力恢复。这些数据不能替代完整均匀化总耗时。融合能量的原 HG 模态实现已较高效，因此本构分解在融合路径上的算子收益明显小于单独 Ku 路径。", "",
        "## 5 后续优先级", "",
        "1. 优先把材料结构表示贯穿初始化、RHS、算子、平均应力与参考采样，实测完整任务峰值和时间；保留原 HG 模态计算。", "2. 若目标仅为等效刚度，继续实现并验证全六载荷能量误差控制；若目标还包括高精度场解，则保留原真实残差要求。", "3. HG 正交补保留为等价公式与负结果，当前不替换模态基线。", "",
        "## 6 文献背景与本轮推导的边界", "",
        "结构张量形式的五参数横观各向同性弹性属于已有理论，可见 [DYNAmore 2007 论文的弹性应力关系](https://www.dynamore.de/de/download/papers/dynamore/de/download/papers/forum07/forum07/material02/transversely-isotropic-plasticity-with-application)。本轮针对实际六参数材料保留额外横向剪切秩一项，并独立对照原旋转本构。", "",
        "正交沙漏控制的背景见 [Flanagan–Belytschko, 1981](https://doi.org/10.1002/nme.1620170504)；这里的正交补恒等式直接由项目现有归一化基和规则几何推导。", "",
        "均匀化与离散最小势能的一致性背景见 [Liu–Reina, Computational homogenization of heterogeneous media under dynamic loading](https://arxiv.org/abs/1510.02310)。本报告中的二阶误差恒等式和参考算子上界由上面的代数步骤给出，数值结论来源于本地 checks.json，不将一般变分原理或已有材料对称性声称为新物理理论。", "",
        "## 7 复现与源文件", "",
        "CPU：experiments/structural_formula_20261020/check_structural.py；GPU：gpu_probe.py、structural_kernels.cu、structural_modal.cuh。命令与范围见同目录 README.md。checks.json 保存全部 CPU 对照，remote/results 保存初步及交错重复 GPU 记录，remote/pilot_sources 保存初步测量时的原始源码。summary.json 汇总本表；manifest.json 提供文件 SHA256。论文、原求解器默认和旧实验均未更改。"]
    (OUT/"REPORT_zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="cpu"},indent=2))


if __name__=="__main__":main()
