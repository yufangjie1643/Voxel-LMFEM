# 固定网格 FFT 预条件优化

所有节点、全部频率、物理算子、参考介质、PCG递推及真实残差标准均保持原定义。物理计算FP64，预条件FFT维持FP32/complex64。新代码精确重排参考逆公式，浮点计算顺序的变化以同一物理残差验收。

实现包括合并精度转换和AoS/SoA重排、及时释放临时量、就地频谱作用、静态系数预计算、将FFT归一化折入系数、可销毁频谱的原生cuFFT逆变换及正逆计划共用工作区。`recommended.json`由三次正式完整任务比较生成。

在已验证服务器上运行完整六载荷任务：

```bash
cd /home/yufan/code/spectral
/home/yufan/uv-torch-py312/bin/python run_rve.py --kind hg --grid 192 --out new_hg192.json
/home/yufan/uv-torch-py312/bin/python run_rve.py --kind fi --grid 192 --out new_fi192.json
```

输出文件存在时拒绝覆盖。先完成同尺寸六载荷预热，再测完整流程，包含材料/参考准备、RHS、PCG、FP64真实残差认证和平均应力。`--case near`可运行另一组已归档材料。`--variant`可显式指定baseline/fused/inplace/cached/dense/native_cached/native_dense；auto采用测量生成的配置。

依赖归档运行环境及前序公式实现，可用`FEAD_FORMULA`、`FEAD_PIPELINE`、`FEAD_MODAL`、`FEAD_RUNTIME`环境变量修改相应路径。验证环境：RTX5090、PyTorch2.12.0+cu130、CUDA13.0、Python3.12。

第一阶段完整消融与重复：`python study.py --phase orchestrate`。原生cuFFT后续对照：`python native_study.py --phase orchestrate`。输出目录为results；正式单项记录存在时拒绝覆盖。调度器发现其他GPU进程时只停止自己的子进程。

下载证据到`../../review_results/fft_contest_20261020/remote/`后，运行`write_report.py`生成报告与推荐配置。

当前扩展限定单个RVE、连续FP64节点向量、FP32预条件FFT、各维至少2且索引不超32位安全范围；节点按三分量交错存储。原生计划和共用工作区归创建它们的CUDA流所有，跨流调用会明确报错，避免并发复用覆盖。返回的物理向量每次独立分配，未复用PCG持有的状态。全部频率保留，无粗网格、截频、FP16或更宽松容差。
