# 投稿版复现包

对应论文：Memory-efficient GPU finite element homogenization of microstructures with spatially varying anisotropy，2026 年 10 月 25 日投稿稿件。

固定版本：fead-2026-10-25-v1。

在仓库根目录运行：

    python -m pip install -r repro/requirements-cpu.txt
    python repro/scripts/reproduce_paper_tables.py

不需要 GPU。脚本先校验文件，再从原始计算记录重新生成当前稿件的 3 张表、数值结果图的数据以及正文关键数字。输出索引位于 repro/generated/TABLES.md，表图与章节对应关系位于 repro/table_manifest.json。

本版已对齐当前六行性能表、以 480³ 为参考的收敛结果、第 5.4 节的 21 个纱线体积分数和附录 D 的九个活性体积分数。历史目录中的旧表名和旧汇总保留用于追溯；当前投稿结果以顶层索引为准。

主实验和纱线体积分数实验保留各自源码快照。原始测量、计时日志、显存采样及校验清单同时提供。新的入口不依赖原实验机器的 Python 绝对路径。

具体 CPU 检查与 GPU 重算命令见 repro/REPRODUCING.md。CPU 再生核对的是已保存的实验结果；原 GPU 性能测试没有在本次整理中重跑。
