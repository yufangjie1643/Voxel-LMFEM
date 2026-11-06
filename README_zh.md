# Voxel-LMFEM

低存储 GPU 体素有限元均匀化求解器。

仓库提供统一的 Python 包、命令行、算例、输入配置、测试、文档和论文复现数据。主代码位于 src/voxel_lmfem/；历史实验快照保留在 campaigns/，用于追溯原始测量。

## 安装与第一个算例

    python -m pip install -e .
    voxel-lmfem example --out runs/example
    voxel-lmfem solve --input runs/example/homogeneous.npz --materials runs/example/materials.json --backend cpu --output runs/example/result.json

基础安装只需要 NumPy 和 SciPy。CPU 参考求解器用于不超过 4096 个体素的小规模验证。GPU 主求解器使用完整积分 HEX8、FP64 刚度与迭代向量、FP32 FFT 预条件。

## 论文复现

    python run.py reproduce

脚本从归档记录再生当前稿件的 3 张表和结果图数据，并校验来源、残差、重复次数和打印值。包括六行性能表、480³ 收敛参考、21 个纱线体积分数与九个活性体积分数。原始 GPU 计时不会由此命令重跑。

统一版本为 v0.1.0；早期 v1/v2 复现包保留。生成索引位于 repro/generated/TABLES.md。

## 验证和开发

    python -m pip install -e ".[dev]"
    python -m pytest -q
    python run.py prepare-inputs

完整说明见 [安装](docs/installation.md)、[输入与使用](docs/usage.md)、[仓库结构](docs/architecture.md)、[验证](docs/validation.md) 和 [论文复现](repro/REPRODUCING.md)。Linux/Windows 自动测试覆盖 CPU 接口、安装包及论文数据再生；GPU 集成测试需要配置好的 CUDA 环境。

作者代码的许可状态沿用 NOTICE.md，第三方许可证保留。
