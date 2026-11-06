# HEX8 模态求解流程优化

该实现沿用第一轮精确 HG/FI 模态单元算子，增加能量内积融合、HG 按需初始化及 HG 解析 FFT 参考逆。`recommended.json` 由正式完整任务的三次重复测量生成，`run_rve.py` 默认按该文件选择 HG/FI 配置。

## 运行完整六工况计算

需要 CUDA、PyTorch，以及论文冻结运行包和第一轮模态实现。当前已验证环境为 RTX 5090、Python 3.12、PyTorch 2.12.0+cu130、CUDA 13.0。物理算子、PCG 向量及真实残差保持 FP64；FFT 参考仍采用 FP32/complex64。

在当前实验服务器上：

```bash
cd /home/yufan/code/assembly_pipeline
/home/yufan/uv-torch-py312/bin/python run_rve.py --kind hg --grid 192 --out new_hg192.json
/home/yufan/uv-torch-py312/bin/python run_rve.py --kind fi --grid 192 --out new_fi192.json
```

每次先完成同尺寸六工况预热，再执行正式计算。输出包含未对称化的完整 6×6 等效刚度、逐载荷真实残差、迭代数、阶段时间、显存和源文件哈希。输出文件已存在时拒绝覆盖。

其他目录可通过环境变量指定：

```bash
export FEAD_RUNTIME=/path/to/runtime
export FEAD_MODAL=/path/to/modal_hex8_gpu_20261020
python run_rve.py --kind hg --grid 192 --out new_hg192.json
```

该入口运行冻结包的解析编织几何和材料设置。自定义材料或 RVE 的集成接口是 `pipeline_adapter.configure`；其前提是兼容冻结的构造与求解接口。当前 CUDA 扩展限定 B=1、每单元常量 C21、规则周期长方体网格，各方向至少两个单元。原冻结源文件保持原样。

## 配置含义

| 配置 | 含义 |
|---|---|
| paper | 论文 HG 直接两阶段或 FI 14 点缓存配置 |
| modal | 第一轮模态原子累加 thread128 |
| lean | HG 跳过未使用的 Jacobi 对角，融合沙漏尺度构造 |
| energy | lean 加单元模态能量与 Kp 同时计算，替代单独扫描 p 与 Kp 的内积 |
| analytic | lean 加 HG 各向同性参考的标量加秩一解析逆 |
| combined | energy 加 HG 解析参考；同时取消返回后被丢弃的内部真实残差重算 |

所有配置保留投影后最终 FP64 真实相对二范数残差 1e-8 的验收和未通过时的继续迭代。`combined` 的内部验证清理不删除最终认证。HG 的 `diag()` 保留按需构造；测量的 FFT 路线不调用它。

FI 的中心模态平均应力和积分后 RHS 已由原生代码实现。本实验保留这些路径。六载荷顺序求解，材料和参考共享；没有引入块 Krylov 或批量向量工作集。

## 验证与完整对照

`generate_modal.py` 从已验证 CPU 标量公式生成 `modal_functions.cuh`；`pipeline_kernels.cu` 实现力与能量、沙漏尺度和解析参考。修改生成公式后需重新生成并验证。

```bash
python generate_modal.py
python pipeline_study.py --phase validate --kind hg
python pipeline_study.py --phase validate --kind fi
```

小网格采用独立常规 B 矩阵装配 CPU CSR 并直接分解，检查算子、能量、延迟对角、全部六载荷位移、局部应力和完整等效刚度。FI 局部应力检查全部八个积分点。HG 解析预条件器在 FP32 下允许舍入顺序差异，最终物理解采用相同 FP64 残差判据。

完整对照使用新的实验目录，避免覆盖既有记录：

```bash
python pipeline_study.py --phase orchestrate --grid 192
```

流程先验证，再对预定候选进行一次完整试算选型，最后交错测量论文配置、第一轮模态配置和选中配置，各三个独立进程；每进程完整预热后计量。计时包含材料、算子、参考、六个 RHS、求解、真实残差和平均应力，排除 CPU 几何、磁盘、首次编译及预热。遇到其他 GPU 计算进程时，停止本实验子进程并保留记录。

`finish_checks.py` 在正式任务结束后独立验证沙漏尺度，并检查公开入口在 strong/near 两类材料上的重复调用。`summarize.py` 从下载记录生成汇总与默认配置；`pack_evidence.py` 保存完整原始证据和逐文件 SHA256。

结果与适用范围见 `../../review_results/pipeline_gpu_20261020/REPORT_zh.md`。
