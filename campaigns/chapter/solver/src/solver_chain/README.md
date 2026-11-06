# 模态 Ku、原生 FFT 与 PCG 联合优化

全部节点/频率保留，FP64 物理场，FP32 预条件 FFT，外层真实残差标准 1e-8。

新增代码：

- `chain_adapter.py`：残差更新与 FFT 打包联合编译；IFFT FP32 输出直接进入 FP64 内积和方向更新；首次收敛冻结位移，消除 best_u 快照；可叠加 HG 紧凑材料表示。
- `reference_design.py`：材料谱界的一参数数值优化及 CPU 自检。
- `reference_closed_form.py`：检查实际各向同性软相及 Loewner 序，直接选择比例软相参考。界相同或条件不满足时保留原参考。
- `chain_study.py`、`reference_study.py`、`closed_study.py`：分阶段完整六载荷消融。
- `closed_validate.py`：最终组合对独立 CPU CSR 组装验证。
- `profile_steps.py`：独立测量 z 消除后的内存与时间。
- `run_rve.py`：最终 opt-in 入口；`write_report.py`：由下载证据重建报告。

已验证远程环境为 RTX 5090、Python 3.12、PyTorch 2.12.0+cu130。在 `/home/yufan/code/solver_chain`：

```bash
/home/yufan/uv-torch-py312/bin/python run_rve.py --kind hg --grid 192 --out hg192.json
/home/yufan/uv-torch-py312/bin/python run_rve.py --kind fi --grid 192 --out fi192.json
```

默认对 STRONG 192³ 启用闭式参考；其他网格/NEAR 维持原参考。可显式 `--reference baseline` 或 `--reference closed`。`--variant baseline` 指上一轮模态+原生FFT+原PCG整链；最终 HG 核心方案为 compact_combined，FI 为 combined。

每次先跑完整六载荷预热，再测完整流程，包括设置与最终认证。输出存在时拒绝覆盖。GPU 被其他进程使用时调度器停止自己的子进程。

冻结依赖目录可通过环境变量 `FEAD_FFT`、`FEAD_FORMULA`、`FEAD_PIPELINE`、`FEAD_MODAL`、`FEAD_RUNTIME`、`FEAD_ABLATION`、`FEAD_STRUCTURAL` 指定。默认分别为前序实验使用的 `/home/yufan/code/fead-{fft-contest,formula,pipeline,modal-hex8,recompute,structural-ablation,structural}-...` 目录，详见适配器中的精确路径。这是依赖现有冻结库的实验入口，并非独立分发包。

完整复现：

```bash
python chain_study.py --phase orchestrate
python profile_steps.py
python reference_study.py --phase orchestrate
python reference_design.py
python reference_closed_form.py
python closed_study.py --phase orchestrate
python closed_validate.py --kind hg
python closed_validate.py --kind fi
```

调度输出目录已存在时不要覆盖，使用新的实验副本保存重复实验。下载到 `review_results/chain_optimization_20261021/remote/` 后，在本地运行 `python write_report.py`。

当前限定 B=1、连续 FP64 节点向量、compiled PCG、延迟主机收敛检查；原生计划限定创建流。HG 材料压缩仅适用于当前已验证的四方对称材料类。达到 max_iter 未收敛时会回退原求解器，异常路径可能更慢。完整推导、计时口径和适用性见生成报告。
