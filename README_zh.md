# Voxel-LMFEM

低存储 GPU 体素有限元均匀化求解器。

仓库提供统一的 Python 包、命令行、算例、输入配置、测试、文档和论文复现数据。主代码位于 src/voxel_lmfem/；历史实验快照保留在 campaigns/，用于追溯原始测量。

## 安装与第一个算例

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，再执行：

    git -c core.longpaths=true clone https://github.com/yufangjie1643/Voxel-LMFEM.git
    cd Voxel-LMFEM
    uv venv --python 3.12
    uv pip install -e .
    uv run --no-sync voxel-lmfem example --out runs/example
    uv run --no-sync voxel-lmfem solve --input runs/example/homogeneous.npz --materials runs/example/materials.json --backend cpu --output runs/example/result.json

在仓库根目录执行上述命令。uv 统一管理本地 .venv；uv run --no-sync 使用已经安装的环境，保留选定的 CPU/CUDA 依赖。

基础安装只需要 NumPy 和 SciPy。CPU 参考求解器用于不超过 4096 个体素的小规模验证。GPU 主求解器使用完整积分 HEX8、FP64 刚度与迭代向量、FP32 FFT 预条件。

## GPU 环境

在 Linux 上使用同一个 uv 环境，安装 GPU 依赖并检查 CUDA：

    uv pip install --torch-backend=auto --reinstall-package torch -e ".[gpu]"
    uv run --no-sync voxel-lmfem doctor --require-cuda
    uv run --no-sync voxel-lmfem solve --input model.npz --materials materials.json --backend cuda --warmup --output runs/model/result.json

后端会根据本机驱动选择；CUDA 13.0 环境也可显式使用 --torch-backend=cu130。工具链与版本要求见[安装说明](docs/installation.md#gpu-installation)。

## 论文复现

    uv run --no-sync python run.py reproduce

脚本从归档记录再生当前稿件的 3 张表和结果图数据，并校验来源、残差、重复次数和打印值。包括六行性能表、480³ 收敛参考、21 个纱线体积分数与九个活性体积分数。原始 GPU 计时不会由此命令重跑。

统一版本为 v0.1.0；早期 v1/v2 复现包保留。生成索引位于 repro/generated/TABLES.md。

## 验证和开发

    uv pip install -e ".[dev]"
    uv run --no-sync python -m pytest -q
    uv build
    uv run --no-sync python run.py prepare-inputs

完整说明见 [安装](docs/installation.md)、[输入与使用](docs/usage.md)、[仓库结构](docs/architecture.md)、[验证](docs/validation.md) 和 [论文复现](repro/REPRODUCING.md)。Linux/Windows 自动测试覆盖 CPU 接口、安装包及论文数据再生；GPU 集成测试需要配置好的 CUDA 环境。

作者代码的许可状态沿用 NOTICE.md，第三方许可证保留。
