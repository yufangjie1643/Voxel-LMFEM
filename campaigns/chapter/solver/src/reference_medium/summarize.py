"""Summarize complete downloaded formula experiments and all negative results."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"review_results/formula_gpu_20261020"
REMOTE=OUT/"remote"
read=lambda p:json.loads(p.read_text(encoding="utf-8"))


def relative(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.linalg.norm(b),1e-300))


def main():
    if read(REMOTE/"status.json")["status"]!="complete":raise RuntimeError("Incomplete experiment")
    formal=[read(p) for p in sorted((REMOTE/"results").glob("measure_*.json"))]
    pilots=[read(p) for p in sorted((REMOTE/"pilot").glob("measure_*.json"))]
    validation=[read(p) for p in sorted((REMOTE/"results").glob("validate_*.json"))]
    if any(r.get("status")!="passed" for r in formal+pilots) or not all(r["passed"] for r in validation):raise RuntimeError("Failed record")
    if len(formal)!=15 or len(pilots)!=5 or len(validation)!=5:raise RuntimeError("Missing result")
    hashes={r["input_hash"] for r in formal+pilots}
    if len(hashes)!=1:raise RuntimeError("Different inputs")
    earlier=read(ROOT/"review_results/pipeline_gpu_20261020/remote/results/measure_hg_combined_192_0.json")["provenance"]
    audits={"checks":0,"mismatches":[]}
    for r in formal:
        for key in ("source_sha256","pipeline_source_sha256"):
            for path,sha in r["provenance"][key].items():
                if path in earlier.get(key,{}):
                    audits["checks"]+=1
                    if earlier[key][path]!=sha:audits["mismatches"].append(path)
    if audits["mismatches"]:raise RuntimeError("Frozen source changed: "+str(set(audits["mismatches"])))
    formula_checks=0
    for row in formal+pilots+validation:
        for path,sha in row["provenance"]["formula_source_sha256"].items():
            name=Path(path).name
            # Validation harness gained even-grid/orchestrator controls later;
            # its actual numerical adapter and CUDA sources are unchanged.
            if "measurement" not in row and name not in ("formula_adapter.py","formula_kernels.cu"):continue
            snapshot=REMOTE/"measured_sources"/name
            local=snapshot if snapshot.exists() else Path(__file__).parent/name
            if hashlib.sha256(local.read_bytes()).hexdigest()!=sha:raise RuntimeError("Formula source does not match measurement: "+path)
            formula_checks+=1
    audits["recorded_formula_source_checks"]=formula_checks
    groups={}
    for kind,variants in (("hg",("baseline","contrast64","single")),("fi",("baseline","analytic"))):
        baseline=next(r for r in formal if r["kind"]==kind and r["variant"]=="baseline")["measurement"]["CH"]
        for variant in variants:
            rows=[r for r in formal if r["kind"]==kind and r["variant"]==variant]
            if len(rows)!=3 or {r["repeat"] for r in rows}!={0,1,2}:raise RuntimeError("Missing repetition")
            seconds=[r["measurement"]["seconds"] for r in rows]
            stages=[]
            for row in rows:
                sums=dict(setup=0.,rhs=0.,solve=0.,post=0.)
                for stage in row["measurement"]["stages"]:
                    name=stage["name"]
                    key="rhs" if name.startswith("rhs") else "solve" if name.startswith("solve") else "post" if name.startswith("post") else "setup"
                    sums[key]+=stage["seconds"]
                stages.append(sums)
            groups[kind+"_"+variant]=dict(kind=kind,variant=variant,seconds=seconds,
                median_seconds=statistics.median(seconds),min_seconds=min(seconds),max_seconds=max(seconds),
                peak_allocated_GiB=max(r["measurement"]["peak_allocated_bytes"] for r in rows)/2**30,
                peak_reserved_GiB=max(r["measurement"]["peak_reserved_bytes"] for r in rows)/2**30,
                cumulative_iterations=[sum(c["iterations"] for c in r["measurement"]["calls"]) for r in rows],
                restarts=[sum(c["restarts"] for c in r["measurement"]["calls"]) for r in rows],
                max_true_residual=max(v for r in rows for c in r["measurement"]["calls"] for v in c["true_residuals"]),
                max_CH_relative_to_baseline=max(relative(r["measurement"]["CH"],baseline) for r in rows),
                median_stage_seconds={k:statistics.median(s[k] for s in stages) for k in stages[0]},
                CH_raw=[r["measurement"]["CH"] for r in rows])
        base=groups[kind+"_baseline"]
        for variant in variants:
            g=groups[kind+"_"+variant]
            g["speedup_vs_current"]=base["median_seconds"]/g["median_seconds"]
            g["time_saving_percent"]=100*(1-g["median_seconds"]/base["median_seconds"])
            g["memory_saving_percent"]=100*(1-g["peak_allocated_GiB"]/base["peak_allocated_GiB"])
    if any(g["max_true_residual"]>1e-8 or g["max_CH_relative_to_baseline"]>1e-8 for g in groups.values()):raise RuntimeError("Correctness failed")
    selected={kind:min((g for g in groups.values() if g["kind"]==kind),key=lambda r:r["median_seconds"])["variant"] for kind in ("hg","fi")}
    result=dict(passed=True,grid=192,complete_formal_measurements=len(formal),converged_formal_loads=6*len(formal),
        input_hash=next(iter(hashes)),source_audit=audits,selected_by_full_time=selected,
        groups=groups,validation=[{k:v for k,v in r.items() if k!="provenance"} for r in validation],
        pilots=[dict(kind=r["kind"],variant=r["variant"],seconds=r["measurement"]["seconds"]) for r in pilots])
    (OUT/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    (Path(__file__).parent/"recommended.json").write_text(json.dumps(dict(variants=selected,grid_of_performance_validation=192,tolerance=1e-8),indent=2)+"\n")
    columns=["kind","variant","median_seconds","min_seconds","max_seconds","peak_allocated_GiB","speedup_vs_current","time_saving_percent","memory_saving_percent","max_true_residual","max_CH_relative_to_baseline"]
    with (OUT/"comparison.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction="ignore");writer.writeheader();writer.writerows(groups.values())
    lines=["# 192³ RVE 公式重推的 GPU 验证", "", "日期：2026 年 10 月 20 日。", "",
        "本次独立检验 FI 对角加秩一解析参考逆、HG 频域背景加实空间材料差 CG，以及 HG 单次归约 PCG。基线为上一轮完整任务最快的 HG combined 和 FI modal。默认是否采用由完整任务耗时决定，同时报告显存与未采用方案。", "",
        "# 计算与精度协议", "",
        "同一 RTX 5090、192³ 解析周期编织 RVE、STRONG 材料和方向场、HG α=0.1、FI 八点积分。全部物理算子与最终认证保持 FP64，真实相对二范数残差不超过 1e-8。FI 解析参考采用 FP32/complex64，与当前基线相同；HG 背景差分采用 FP64/complex128，保持重推公式的固定 SPD 精度条件。HG 单次归约使用与基线相同的 FP32/complex64 参考。", "",
        "每配置三个独立进程，每进程完整六载荷同尺寸预热后测一次正式完整均匀化，配置顺序交错。计入材料构造、算子与参考准备、RHS、全部求解、投影后的真实残差与平均应力；排除 CPU 几何、磁盘、首次编译和预热。原阶段同步保留。五个单次试算不作为正式重复数据。", "",
        "新 HG 求解器每四次实际迭代检查一次递推物理残差，报告实际执行次数；基线保留原锁存的逻辑迭代计数。两者停止轮询细节不同，均需通过投影后的物理 FP64 真实残差，失败则按原机制继续求解。不能把迭代数差异全部归因于条件数变化。", "",
        "# 完整任务结果", "", "| 离散 | 配置 | 中位数 s | 最小至最大 s | 峰值已分配 GiB | 相对当前加速 | 时间减少 |", "|---|---|---:|---|---:|---:|---:|"]
    for g in groups.values():
        lines.append(f"| {g['kind'].upper()} | {g['variant']} | {g['median_seconds']:.6f} | {g['min_seconds']:.6f}–{g['max_seconds']:.6f} | {g['peak_allocated_GiB']:.6f} | {g['speedup_vs_current']:.3f}× | {g['time_saving_percent']:.2f}% |")
    lines += ["", "# 正确性与收敛", "", "| 离散 | 配置 | 六工况累计迭代 三次 | 重启 三次 | 最大真实残差 | 完整等效刚度相对当前基线差异 |", "|---|---|---|---|---:|---:|"]
    for g in groups.values():
        lines.append(f"| {g['kind'].upper()} | {g['variant']} | {g['cumulative_iterations']} | {g['restarts']} | {g['max_true_residual']:.3e} | {g['max_CH_relative_to_baseline']:.3e} |")
    lines += ["", "五组小模型验证包括奇数网格 5×6×7 与偶数网格 4×6×8，使用不等边长和完整正定各向异性材料，独立常规 B 矩阵装配 CPU CSR 并直接求解。HG 材料差测试只激活 20% 单元，其余严格等于背景；检查物理作用、换元后作用、能量、半谱 Parseval 物理范数、RHS、全部位移与局部应力以及完整未对称化的等效刚度。FI 另对原 FP64 参考与 FP32 参考分别核对。", "",
        f"共 {len(formal)} 次正式均匀化、{6*len(formal)} 个正式载荷全部通过残差标准。冻结及上一轮源文件交叉校验无差异；环境、输入和逐源文件哈希保存在原始记录。", "",
        "# 各阶段中位数", "", "| 离散 | 配置 | 初始化 s | RHS s | 求解 s | 应力 s |", "|---|---|---:|---:|---:|---:|"]
    for g in groups.values():
        s=g["median_stage_seconds"];lines.append(f"| {g['kind'].upper()} | {g['variant']} | {s['setup']:.6f} | {s['rhs']:.6f} | {s['solve']:.6f} | {s['post']:.6f} |")
    lines += ["", "# 结果解释", "",
        "FI 总时间小幅下降主要来自参考初始化；求解阶段没有加速。此次解析逆减少了符号构造和矩阵求逆准备成本，不能据此宣称 FFT 或 PCG 每轮更快。原 upper6 缓存为每频率六个 complex64，新缓存为四个 float32 加三个一维方向表；按数组结构计算，频率缓存主体减少三分之二，但完整任务峰值显存只减少约 1%，峰值由其他工作数组共同决定。", "",
        "HG 材料差方案在本例确实减少了材料存储以及 RHS、应力恢复工作量，完整任务峰值已分配显存下降约 31.8%；FP64/complex128 频域求解的总耗时仍高于现有混合精度最快方案。因此它提供了显存与速度的取舍，不能作为本例的时间优化替代。单次归约递推本例也没有时间收益。阶段计时可以定位差异，但没有进行逐内核性能剖析，不能把减速唯一归因于某个硬件瓶颈。"]
    lines += ["", "# 实现范围", "",
        "FI 解析逆缓存四个实数系数，方向因子按三个一维表预计算，不在每次谱作用时重复做双精度除法。HG 材料差版本以真实均匀基体为背景，保留原 Zeman 参考，仅保存纱线单元的 ΔC21、位置与沙漏尺度差；RHS 及平均应力也按同一差分公式计算。求解期间释放完整 C21。最终真实残差通过独立的背景傅里叶作用加材料差物理作用计算，并已由小模型 CSR 和完整等效刚度对照验证。", "",
        "背景差分版本每轮仍有一对 FFT。当前验证没有用 FP32 Krylov 状态或降低容差换取收益，也没有将 CPU 固定精度等价证明直接当作 GPU 加速证据。单次归约新增方向向量和递推，不保证单 GPU 速度收益。", "",
        "# 按完整时间选择", ""]
    for kind in ("hg","fi"):lines.append(f"- {kind.upper()}：本次完整时间优先选择 {selected[kind]}。")
    lines += ["", "该选择限定当前算例与硬件；上一轮已验证默认入口保持原样，新候选可通过本实验入口显式运行。论文和冻结生产源未修改。正式测试结束后，只修复了外来 GPU 任务停止分支缺少 os 导入的问题，计算代码未变；测量时原脚本保存在 remote/measured_sources/formula_study.py，其哈希与原始记录逐一核对。", "",
        "# 复现", "", "代码与命令见 experiments/formula_gpu_20261020/README.md。summary.json、comparison.csv 对应表格；remote/results 保存全部正式输出与小模型验证，remote/pilot 保存全部试算，remote/logs 保存日志和 GPU 采样。"]
    (OUT/"REPORT_zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(dict(selected=selected,groups={k:{a:b for a,b in g.items() if a!="CH_raw"} for k,g in groups.items()}),indent=2))


if __name__=="__main__":main()
