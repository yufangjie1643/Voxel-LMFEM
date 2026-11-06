# Voxel-LMFEM

面向空间变化各向异性微结构的 GPU 体素有限元均匀化求解器，采用完整积分 HEX8 单元、紧凑材料存储和 FFT 预条件共轭梯度法。

[安装](installation.md) · [使用](usage.md) · [论文复现](../repro/README.md) · [English](../README.md)

## 快速开始

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后执行：

```bash
git clone https://github.com/yufangjie1643/Voxel-LMFEM.git
cd Voxel-LMFEM
uv venv --python 3.12
uv pip install -e .
uv run --no-sync voxel-lmfem example --out runs/example
uv run --no-sync voxel-lmfem solve --input runs/example/homogeneous.npz --materials runs/example/materials.json --backend cpu --output runs/example/result.json
```

示例计算均匀材料单胞的等效刚度。Python 接口、材料参数和输入格式见[使用说明](usage.md)。

## GPU 计算

在配置 NVIDIA GPU 和 CUDA 工具链的 Linux 环境中执行：

```bash
uv pip install --torch-backend=auto --reinstall-package torch -e ".[gpu]"
uv run --no-sync voxel-lmfem doctor --require-cuda
uv run --no-sync voxel-lmfem solve --input model.npz --materials materials.json --backend cuda --output runs/model/result.json
```

## 论文复现

```bash
uv run --no-sync voxel-lmfem reproduce
```

该命令根据仓库中的实测记录生成论文表格和结果图数据。数据来源、实验命令和测量规范见[复现指南](../repro/README.md)。

## 开发

```bash
uv pip install -e ".[dev]"
uv run --no-sync python -m pytest -q
```

[项目结构](architecture.md) · [开发指南](development.md) · [更新记录](changelog.md)

代码采用 [MIT 协议](../LICENSE)。第三方署名见 [NOTICE](NOTICE.md)，引用信息见 [CITATION.cff](../CITATION.cff)。
