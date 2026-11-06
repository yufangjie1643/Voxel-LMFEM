CMC 随机仿真模块

本模块只做仿真，不读取拉伸实验，不标定曲线，不生成实验对比图或预测区间图。

入口（在项目根目录运行）：
  ./run verify-cmc-sim --gpu --out runs/cmc-sim-verify
  ./run cmc-sim --config experiments/cmc_sim/tow_elastic.json --out runs/cmc-tow-elastic
  ./run cmc-sim --config experiments/cmc_sim/tow_damage.json --out runs/cmc-tow-damage
  ./run cmc-sim --config experiments/cmc_sim/woven_elastic.json --out runs/cmc-woven-elastic
  ./run cmc-sim --config experiments/cmc_sim/fe2_woven_elastic.json --out runs/cmc-fe2-woven
  ./run cmc-sim --config experiments/cmc_sim/fe2_tow_damage.json --out runs/cmc-fe2-tow-damage

所有输出写入新的 runs/<名称>/。已有输出拒绝覆盖；恢复使用同一命令加 --resume。
--n 覆盖网格分辨率，--samples 覆盖样本数，--seed 覆盖主种子。
--max-samples 1 可测试批次中断；--max-steps 3 可测试加载路径中断。
恢复时去掉限制参数，其余参数和配置必须相同。配置和源码哈希不同会拒绝恢复。

复用原求解链：八点全积分 FI、混合材料池、均匀背景与差值算子、FP32 FFT 参考、
FP64 物理平衡。PCG 验收仍为真实相对残差不超过 1e-8，能量误差不超过 1e-7。
使用原 recover.cu 恢复全部八个高斯点，并复用原 recover_mmf.cu 中的成对 RHS；
不改动旧 T300/epoxy 算例、MMF 参数、CUDA 算子或既有 FE² 默认行为。
新增 recover_damage.cu 在 GPU 上归约八点最大主有效应力，每体素只传回一个标量；
生产路径不展开和传回 N×8×12 的全场数组。完整场仅用于小网格独立校验。

随机输入：
  每个样本使用 SeedSequence([seed, sample_index])，独立生成材料和几何，随后冻结。
  参数分布目前支持固定值、显式均匀区间和 Weibull。未知分布不得当作已识别统计。
  随机矩阵支持正标量、一般 SPD、正交各向异性分块 SPD 三种方式。
  零离散参数严格恢复平均张量。分块离散参数与完整矩阵的 Frobenius 离散不同。
  现有示例的 delta=0.03 只是敏感性设置，不来自强度变异系数或实验识别。
  当前参数采样彼此独立，尚未包含空间相关随机场或跨组分相关性。

几何：
  hexagonal_tow：周期六角排列直纤维束，显式 SiC 纤维/PyC 层/SiC 基体三相。
  legacy_woven：复用现有编织体素几何、材料方向和 256³ 固定输入读取逻辑。
  npz：自定义 phase_id[z,y,x]、domain_size[3]，可选 orientation[...,3,3]。
  长度用 mm，应力和模量用 MPa，能量密度用 MPa，G 用 J/m² 并转换为 N/mm。
  PyC 层不足两个横向体素会记录警告；完全消失则报错，不能把粗网格当收敛结果。
  旧编织几何并非真实 CMC 预制体。示例中的纱线剪切模量是注明的闭合假设。

加载和损伤：
  prescribed_strain 接受相同的六分量工程应变路径。
  uniaxial_stress 指定加载方向应变，其余宏观应力通过当前等效柔度自由释放。
  各样本采用完全相同的绝对应变路径，不按各自损伤起点重新归一化。
  支持 elastic、rankine_brittle、linear_crack_band。
  损伤由八个高斯点的最大正主有效应力驱动，历史为最大等效应变，不会卸载愈合。
  线性裂带软化使用立方根体素体积为特征长度；均匀单轴连续模型的总耗能为指定 G。
  实际离散采用显式 damage_levels 个状态、正残余刚度，二者均是必须检查的数值参数。
  此原型没有界面牵引分离、纤维拔出、裂纹闭合或经过验证的多轴 CMC 断裂定律。
  单元状态改变后在两次线性求解之间重建共享材料池；不会在一个 PCG 中修改刚度。
  这是保守正确的接入方式，暂未优化增量更新性能；模型尚不是经过验证的完整 CMC 断裂定律。

在线 FE²：
  simulation=fe2 接入原 WovenFE2，不以离线等效本构替换在线微观求解。
  每个宏观高斯点独立保存 int64 编码损伤历史，复用一个微观工作区。
  使用原宏观平衡、平面应力约束、增量回退、能量检查和微观精度收紧逻辑。
  宏观网格可选矩形小验证网格，或原 main_mesh/refined_mesh 带孔板。
  宏观沿 x2 加载。loading.history 在 FE² 模式下是完整板名义应变路径；
  实际矩形和带孔板的自由边界由原宏观求解器施加，不是每个材料点均匀单轴应力。
  自定义 NPZ 几何的方向、输入数组和源码必须保持一致，恢复前由指纹核对。

结果和恢复：
  manifest.json：配置、全局源码指纹和语义。
  sample_XXXXX/sample.json、inputs.npz、geometry.json：冻结样本、体素、相分数和警告。
  checkpoint.npz：唯一权威检查点，原子替换，保存已提交状态、历史、加载游标和记录。
  records.json：已平衡仿真增量的应力、应变、能量、损伤和误差。
  summary.json、status.json：运行状态与数值验收。它们不是实验验证结果。
  每个样本有独立状态；失败增量不提交，损伤不稳定时回退并细分步长。
  检测到其他 GPU 作业会停止，不终止其他进程。输出目录加锁防止重复运行。

建议的仿真验证顺序：
  1. verify-cmc-sim 的均匀材料张量、自由横向应力、损伤不可逆和样本隔离检查。
  2. 小网格三相束弹性、编织弹性、损伤加载卸载和中断恢复。
  3. 逐步加密薄界面，检查体积分数、六载荷等效刚度和损伤状态数敏感性。
  4. 在材料本构和几何确定后，再扩展真实试样 FE² 及不确定性后处理。
