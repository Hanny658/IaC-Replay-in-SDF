# MLP-Cortex：一个尽可能贴近人脑工作方式的神经网络（v1–v12 实验全记录）

*2026-08-27 至 2026-08-31 · 独立于课程作业；数据轴自第 10 期起为 MNIST + split-CIFAR-10（表格数据退役）*

## 1. 定位与评价框架

**目标**：不是打榜，而是把现有仿生神经网络（PC、Forward-Forward、SNN、SOM、BurstCCN……）
的可用部件与前沿神经科学发现（BTSP、CLS、局部睡眠、突触稳态）拼装成一个"各方面都贴近人脑"
的网络与训练范式，用阶梯式消融量出**每一条生物约束的真实价格**。

**评价三轴**：准确率 × 能耗代理（event-driven synops / 脉冲能量 0.9 pJ/AC vs 4.6 pJ/MAC）×
脑相似度（定性清单）。**两条数据轴**：静态 i.i.d.（MNIST / 作业三个表格数据集）+ 序列轴
（split-MNIST，5 任务 × 2 类，共享 10 路读出，class-IL）。三条护栏（7 期起）：
静态轴不得退化；BP + 等容量经验回放必须进每张表；不换基准来赢（拒绝 streaming 评测）。

**代码**：`src/models/cortex.py`（CortexNet，全部开关）、`src/models/spiking.py`（率→脉冲推理）、
`src/run_bio.py`（表格数据）、`src/run_mnist.py`（静态轴）、`src/run_seq.py`（序列轴 + 海马 +
睡眠 + DG）。结果：`results/bio/{summary.csv, mnist/summary.csv, seq/summary.csv}`。
等价锚：`act="tanh"` + 全开关关闭 == 报告中的 MLP-PC，逐位一致（每次改动后复检）。

## 2. 模型积木（最终形态 v12）

| 机制 | 实现 | 生物对应 |
|---|---|---|
| 无权重转置 | Kolen-Pollack 反馈突触 + `dale_fb` 符号一致（细胞类型定符号） | 反馈通路独立、无转置 |
| 有界误差通道 | `κ·tanh(ε/κ)` burst 饱和 + burst 门控（无事件不 burst）+ burst 基线 | 顶树突 burst 复用 |
| 单相学习 | `sweep`：一次前向 + 一次自顶向下 burst 扫描（T=0），无迭代松弛 | 单次事件内学习 |
| 稀疏编码 | k-WTA（宽层 2–10%），`adam_eps=1e-3` 防跑飞 | 侧抑制 |
| Dale 律 | 细胞类型固定、投影时投影回符号锥 | 兴奋/抑制分离 |
| 自组织连接 | 距离依赖稀疏连接（30%）+ 结构可塑性（剪弱长新） | 皮层局部连接 |
| 脉冲推理 | Bernoulli 输入脉冲 + soft-reset IF + k-WTA 逐步预算，T_s=8–16 | 事件驱动推理 |
| 海马 | K 样本 reservoir 缓冲（真实样本一次写入） | 情景记忆 |
| 夜间 NREM | ×3 可塑性离线重放（7A 默认；v9 起可被局部睡眠替代） | 慢波睡眠重放 |
| 内部状态切换 | surprise 压力 θ + 重放误差停机 + 门控 REM | 双过程模型 |
| **局部睡眠** | **refractory：刚放电单元强制轮休；重放只写入"睡着"突触（掩码内 Adam，精确隔离）** | 使用依赖的局部睡眠 |
| **稳态触发** | **单元级 Process S（使用累积、巩固放电），触发 8–16 批重放脉冲串** | 睡眠压力 / SWR |

## 3. 阶段年表与关键数字

### v1–v3（约束的价格，MNIST 30 epochs×3 seeds / 表格数据报告协议）
- **硬三件套免费**：无转置 + 有界误差 + Dale ≈ 0 代价（SUPPORT2 AUC 0.895 = 最好的 MLP；
  MNIST 宽核 97.1 vs 对照 97.8 vs BP 97.7）。`dale_fb` 是让 KP 反馈在 Dale 律下对齐的关键。
- k-WTA 是**宽层的属性**：256-128 上 25% 97.3 / 10% 97.0 / 5% 95.9 / 2% 94.2；64-32 上 10% 即塌。
- Adam 在宽 k-WTA 上会跑飞（v 极小 + 赢家翻转）→ `adam_eps=1e-3`（82.6→97.2）。
- 稳态调节只能做护栏：逐步阈值/缩放随更新数退化，均衡器塌 2/3 种子，`sleep_guard` 死区无害。
- KP 衰减的本质是**权重正则**而非对齐（6C 证实）：表格 1e-2 / MNIST 1e-3。

### 第 4 期（脉冲、松弛深度、连接）
- 脉冲推理只在窄窗口划算：T_s=8 97.4% @ 206 nJ（密集率网 1080 nJ）；T_s=4 崩，T_s>16 反而更贵。
- 松弛 T=20→5 无损（训练 4× 便宜）；**第 6 期进一步 T=0**（单相扫描，1/3 成本，各轴等好或更好）。
- 距离依赖稀疏连接在等成本下胜随机 1–2 pt；10% 密度 + 再生长恢复大半损失。

### 第 5–6 期（v5/v6 基座、负相、镜像）
- v5 基座 96.5±0.2，脉冲 T_s=8 95.9% @ **59 nJ**（≈密集率网 1/18 能耗），比 BP 低 1.2 pt。
- burst 门控/基线免费；乘性 burst 编码失败（78.8）；朴素做梦负相净亏（-1.2，强度大即毁灭）；
  spike-count 训练 -11~-16 pt；SGD 比 Adam 稳定但 -1.3 pt。

### 第 7 期（海马 + 睡眠系统，split-MNIST）
- **7A**：夜间 NREM ×3 是小缓冲的赢家——K=200 86.9 vs 交错 ER 70.5（BP）；K=1000 90.7±0.4；
  BP 在同样夜间重放下不稳定。surprise 门控写入比随机差（最难样本≠最该巩固的样本）。静态轴无损。
- **7B**：B0 解码器 + 原型种子的梦可辨认（judge ≈1.0），但生成重放在真实缓冲之上零增益，
  反向学习（Crick-Mitchison）净亏 → NREM 真实重放仍是巩固机制。
- **7C**：surprise 压力 + 误差停机 = 内部控制器在 74–80% 重放量达到固定日程精度
  （K=1000：91.0 @ 387 vs 90.7 @ 480），等重放量下胜固定时刻表 12–14 pt；2/3 的睡眠自动落在
  任务切换后；门控 REM 首次不有害。
- **8A**：局部睡眠（重放写入当前输入"睡着"的单元，隔离精确到 0.00 输出变化）可行但通道窄
  （20–30%），39×批量成本达不到夜间水平。
- **8B**：**refractory 规则把隔离通道翻倍到 ~53%**，无夜间 90.1–90.4 ≈ 最佳夜间 90.7；
  silent+夜间组合 91.4。宽稀基底提高隔离收益但夜间自身在宽稀上退化。

### 第 9 期（文献定向轮 + 通宵三方向，2026-08-31）
文献轮结论（详见记忆 `cortex-lit-review-2026-08`）：无 ANN 工作做过使用依赖门控或局部睡眠；
Bazhenov 线全是全局离线睡眠；CLS-ER 的 EMA 慢系统提升单次重放价值但不减重放量；
Cayco-Gajic 清单给出 DG 施工图；Driessen/Tononi/Cirelli (Nat Neurosci 2026) 在清醒小鼠因果证明
ON/OFF 交替（而非持续压制）缓解局部睡眠压力并救回记忆——与 8B silent<refractory 完全对应。

**9C · DG 前端 → 证伪**。冻结稀疏扩张（2025 单元 × 8 局部输入、3% k-WTA、除法归一）在输入端
完全兑现理论（任务重叠 0.79→0.18，PR 维度 62→139），但下游全线更差：refr 79.4/79.0、
DG@10% 87.2、夜间 82–84、静态 91.4（对照 95.0）；隔离通道不变宽、隐层维度塌缩（4–6 vs 11–15）。
随机采样、密集对照、密集读出全部更差。**输入端正交性穿不过习得的 k-WTA 层**。

**9A · 成本 → "39×"是记账错觉**。对 refractory 掩码重放，重放 batch 缩到 8 都免费
（br16 90.8±0.1、br8 90.5±0.3；br1 才 82.9），而同样的缩小杀死所有对照：无掩码 ER
86.9→83.9、silent 87.8、**夜间 batch 16 掉到 81.0±1.6**。→ **隔离掩码是微批重放的稳定器**。
样本记账：br8 = 15 万重放样本 = 夜间（12.3 万）的 1.2×。**cad2_br64 = 91.3±0.3，全项目最好**，
无任何夜间；br16+夜间 90.9 —— 夜间在便费连续局部睡眠之上**不再有增益**。
重放必须稀疏时：8–16 批脉冲串每 64–128 清醒批（86.8–87.6 @ 12% 事件量），涓流与"小睡"都失败；
**触发必须是稳态压力**（pburst16_t64 89.9±0.1 @ 15%），surprise 触发在任何量级全灭（19%）——
新奇性只在任务切换后开火，把后期饿死。K=200 仍逊夜间（84.4 vs 86.9）。

**9B · 静态代价 → 真实但有限**。强制轮休在 i.i.d. 上花 −1.3（w256s10）~ −3.0（w512s5+br16）,
与 Driessen 2026"局部睡眠出现在错误时刻损害表现"一致。

### 第 10 期（三个遗留问题，2026-08-31 白天；数据轴改为 MNIST + split-CIFAR-10 灰度）

**10A · 静态代价 → 相对新奇门控解决（可学习流上）**。单元级压力版（rest 当 Process-S≥θ_r）只给出
纯权衡曲线：θ_r 0.5→4 时静态 91.6→94.0、序列 88.8→80.7，无支配点；绝对 surprise 阈值不可跨数据集
（静态 MNIST 收敛误差高于任何序列内阈值：ach_frac=1.0；CIFAR 0.95）。**ACh 式相对新奇**（快 EMA ≥
β × 慢 EMA 才开轮换）一套参数双轴通吃：β=1.5 → 静态 95.0±0.4（=none 95.0±0.2，轮换只开 1.4–17%，代价归零）+
序列 90.3±0.4；β=1.1 → 序列 90.9±0.6（最高）/静态 93.8±0.5（3 seeds）。配额版 refr_frac 饱和为"带记忆的 refractory"（序列 91.0±0.6，网格最高）。
**注意**：新奇门在永远欠拟合的流上会错关（CIFAR nov13 序列 19.1）——默认仍是常开轮换，门控是
可学习流上的附件；按"学习进度"而非误差水平定义新奇是开放项。

**10B · K=200 缺口 → 主因是重放多样性**。重放加噪 σ=0.2 + 压力脉冲串：81.8±2.9 → **84.3±1.0**
（遗忘 0.08，2,763 次重放）；增益 ×1 单独 83.4。夜间 86.9 仍保有 ~2.6 优势：小缓冲下离线集中巩固
对每个独立样本的利用率仍更高。K=1000 下噪声无害（90.4±0.2）。

**10C · CIFAR 迁移成立且放大 + 脉冲能耗**。split-CIFAR-10（灰度 1024 维）序列：refr 27.1±1.7 >
BP+ER 25.1±1.5 > 夜间 23.8±1.2 > 无掩码 20.9±1.8 > 无缓冲 16.0–16.5——v9 排序完整迁移，幅度更大；
**静态轴反转**：rp4 29.3±1.8 > refr 25.3±1.0 > none 22.8±1.3——欠拟合数据上轮换是 +2.5~+6.5 的
正则化收益，"静态代价"只是易数据体制的属性。CIFAR 任务重叠 ~0.9–0.98（编码几乎不分离）而隔离照赢。
脉冲重测（静态 MNIST）：v9 网 T_s=8 = **91.6% @ 118.8 nJ**（密集率网 2461 nJ 的 1/20.7；对照网
93.7% @ 118.7 nJ）；refractory 训练的网在 T_s>8 时精度反降（阈值校准与轮换权重的交互，开放项）。

### 第 11 期（进度门 / 特征前端 / K=200 机理 / 下选择，2026-08-31 晚）

**11A** 进度门（新奇 ∨ 未掌握，式：f≥β·s 或 f≥γ·e0）一套参数统一四个体制：CIFAR 序列 27.2±0.7 =
常开轮换（纯新奇门在此只有 19.1）、CIFAR 静态 25.3、MNIST 序列 90.4、MNIST 静态 94.3±0.2（γ=0.5，
距纯新奇门 95.0 余 0.7——未掌握子句的代价）。**11B** V1 式 patch 特征前端（线性探针 49%）：局部
方案排序保持且放大（refr 37.0±0.4 > 夜间/无掩码 ~31，静态轮换 +3.9），但 BP+ER 42.6±0.5 反超——
局部睡眠对 BP 的优势限于原始像素+局部规则体制，对其他局部日程的优势跨全部三个输入体制稳健。
**11C** K=200 的"缺口"主要是基底错配：同基底夜间 84.0±3.4 vs 本地+噪声 84.3–84.8（打平且方差
小 3 倍）；窄网夜间 86.9 仍是小缓冲全场最好；噪声伤夜间（81.8）、无掩码坍塌（71.0）结论不变。
**11D** 睡眠锚定下选择（剪弱+距离偏置新生，KP 衰减为弱化力）：连续局部睡眠上免费（90.9±0.2）、
纯夜间上 +3.1（87.2 vs 84.1，离线下选择救回宽稀网的夜间）、情景脉冲串上有害（84.6 vs 89.9——
重放稀疏时突触未被保护就被剪）。

### 第 12 期（外部 review 处理 + 三个对照，2026-08-31 夜）

外部新颖性 review（`report/critic.md`）核心判断被接受：主张从"refractory rotation 是新的"重定位为
"并发的精确隔离重放构造 + rotation 作为通道扩宽器"；11 条核实过的引文入稿（Kaski–Kohonen 1994、
Maeda–Miyajima 1999、Shen AAAI 2024、Abbasi 2022、GPM、Adam-NSCL、OGD、CLNP、PackNet、Ross 2012、
Klasson 2023），review 中一条幻觉引用（arXiv:2608.26720）被识别并排除。三个 review 启发的对照：
**E1 镜像隔离**（保护过去=null-space 方向，同一套机制实现）崩至 81.3±2.9——低于无掩码 ER，
"保护方向反转"成为 9.5 pt 的实验事实并写入稿件；**E2 无状态 SGD × 隔离 = 92.0±0.7 全项目新高**
（静态 93.4±0.2；无掩码 SGD 对照 84.1）——掩码内优化器状态只是 Adam 的必要条件，无状态更简更强，
但增益需要每批节奏（SGD×cad2 只有 91.3）；**E3 软轮换**双轴皆负（88.4/86.0）——全或无的 OFF
优于减半增益，与 Driessen 2026 tonic-vs-alternation 的第三次对应。v12 默认：refr br16 cad1 + SGD。

### 第 12b 期（系统消融表补全，2026-09-01）

为稿件新增 Ablation Study 章（两张表：机制组件 + 底座阶梯），补跑 22 个格子（w512s5，K=1000，
br16 cad1，3 种子），三个新发现：**(1) Adam 动量泄漏反而更好**——矩限制在掩码内 90.8±0.1，
矩全局推进（权重仍掩码）91.9±0.4、静态 92.8 vs 92.0：精确性命题需要的"状态限制"在准确率上是
Adam 的枷锁而非保护；序列 92.0（无状态）≈ 91.9（泄漏）> 90.8（限制），无状态是唯一同时保住
保证与准确率的选择——稿件三处"矩必须限制"的表述已改写。**(2) 完整系统（SGD+轮休+prog 门）
89.1±0.3 / 静态 95.0±0.6**：门在 SGD 下静态全额退款（≥95.1 无缓冲天花板），但序列付 2.9 pt
（Adam 下同一门只付 0.4）——SGD 增益像需要每批 cadence 一样需要每批轮休，"v12 默认 = SGD+门"
不成立，SGD 下是真实的双工作点选择（92.0/93.4 常开 vs 89.1/95.0 门控）。**(3) SGD×silent
（去轮休）89.0±0.5 / 静态 96.0±0.1**——表内最高静态格：轮休是唯一有静态代价的组件，SGD 把
silent 通道从 Adam 的 87.8 抬到 89.0（轮休在 SGD 下值 +3.0）。配置：`g12_sgd_prog05/…_silent/
…_adam_leak_w512s5`（含 static\_ 前缀）、`static_g12_sgd_none_w512s5`、`ctx_none_w512s5`；
泄漏开关 `leak_moments`（cortex.py `_adam`）。

**12c 修正（诚实化）**：进一步审视发现 SGD 门控点（89.1/95.0）被 SGD×silent（89.0/96.0）
**弱支配**——时序噪声内打平、静态整低 1 分：SGD 下门没有制造出第三个有用工作点，真实的
两工作点是常开（92.0/93.4）与 silent（89.0/96.0）。补上缺失的 Adam-silent 静态格
（`static_loc16_silent_br16_w512s5` = 95.0±0.3）后确认：Adam 下门是**未被支配的折中**
（门控 90.4/94.3 vs silent 87.8/95.0，+2.6 时序换 −0.7 静态）——门的自适应价值是优化器
特有的，不是门本身的。稿件已按此改写（表格加粗改为按列最优；摘要中门控消除静态代价的
主张限定为 masked Adam；Future Work 改为最短开启时长的块状门）。

### 第 13 期（发作承诺门 / 块状最短开启时长，2026-09-01 晚）

实现 `refr_block`：触发条件沿用 prog 门（β=1.3, γ=0.5），但每次点火把"发作计数器"重置为 M 批，
计数器非零期间轮休连续运行——Saper 睡眠-觉醒翻转开关的整段发作结构（抖动即病理）。核心发现：
**按批门失败的原因是触发稀疏 + 抖动**（时序流上门只开 12%），发作承诺把稀疏触发放大成体制不对称
的长连续发作（M=2048：时序占空 94% / 静态 63%）。结果（SGD，3 种子）：

| M | 时序 | 静态 | 占空(时序/静态) |
|---|---|---|---|
| 512 | 91.7±0.4 | 94.0±0.5 | 0.61/0.38 |
| **2048** | **92.1±0.6** | **94.2±0.4** | 0.94/0.63 |
| 4096 | 92.0±0.7 | 93.4±0.1 | 1.00/0.94 |

**M=2048 弱支配常开轮休**（92.0/93.4）：时序打平、静态 +0.8——第 12b 期"SGD 下门造不出第三点"
修正为"**按批**门造不出，**发作承诺**门可以且成为新的时序默认工作点"。M=4096 静态占空回到 0.94、
退款消失；silent 的静态天花板 96.0 仍未触及（轮休即使成发作也有残余代价）——门买到的是体制
自适应，不是免费午餐。稿件：Table 1 加行（加粗移至 92.1）、解读段重写、摘要与 limits 更新、
新增 Saper 2005 引文；Future Work (ii) 更新为"补 OFF 最短时长（完整翻转开关滞回）/ 稳态发作
时长自适应"。配置：`g13_sgd_block{512,2048,4096}_w512s5`（含 static\_），开关 `block`。

**13b OFF 滞回（干净的阴性结果）**：补全翻转开关的 OFF 侧——发作过期后触发被忽略 `block_off` 批。
OFF∈{512,1024}: 静态轨迹与基线**逐位相同**（发作过期即意味着触发已静默 ≥2048 批，静态重触发
本来就不是抖动，短不应期永不生效）；OFF=4096: 静态占空 0.63→0.53 但静态精度不动（94.0±0.7），
时序单调受损（92.1→91.9→91.7→91.0，长 OFF 会盖住任务切换，seed1 掉到 89.8）。结合发作门
M=512 格（占空 0.38→94.0）：**残余静态代价源于"轮休扰动已收敛联盟"本身，与门开火频率无关**
——本系统里翻转开关的功能半边是 ON 侧。已入稿（发作段追加 + negative results + Future Work (ii)
改为"掌握度退火的轮休深度 / 效用加权的选择性轮休"）。配置：`g13_sgd_flip{512,1024,4096}_w512s5`
（含 static\_），开关 `block_off`。

### 第 14 期（η 敏感性，2026-09-02 凌晨，54 运行）

四组结论全部加固论文：**(1) SGD 非刀尖**——时序平台 0.01–0.02（91.9/92.0）、静态平台
0.02–0.05（93.4/93.7），0.02 为联合最优，平台外优雅退化；**(2) Adam 的比较本来就公平**——
扫 10× lr，Adam 时序最优就在默认 1e-3（90.8），3e-3 换静态 93.6 但时序塌到 88.9，无任何
Adam 格触及 SGD 联合点；**(3) 无掩码 SGD 跨 η 都又差又不稳**（84–88，σ 1.8–4.1 = 掩码版
3–8 倍）——隔离稳定优化是跨 η 结论；**(4) 发作门继承平台**（0.02 联合最优）。η 随体制左移
（CIFAR 峰在 0.01）。配置：`g14_*`。

### 第 15 期（通宵探索，2026-09-02 凌晨，~130 运行）

**A. SGD×CIFAR 转移**：特征体制 SGD η=0.01 = **40.6±1.0**（Adam 37.0 → +3.6；6 种子），
静态 38.6±1.4（Adam 35.6 → +3.0）；BP+ER 公平扫描后最优 43.3±1.0（lr 3e-4）——**phase 11
的 5.6 分体制边界大半是本地端优化器伪影，双方调齐后诚实差距 2.7 分，边界仍在但收窄**。
Raw CIFAR：SGD 27.2±1.4 ≈ Adam 27.1——欠拟合体制无状态增益不转移（持平）。

**B. 双时标锚突触**（Benna–Fusi 最小形式：快权向慢锚衰减 λ、锚吸收 μ，仅清醒步 tick，
不触碰重放隔离保证）：强假设"溶解静态价"**证伪**——任何耦合静态都没超过 93.4；但弱对称锚
(λ=μ=3e-4) 时序 **92.4±0.2**（基线 92.0±0.6，方差减半）+ 静态打平 93.3——**弱支配常开**，
定性为"时序稳定器"。强耦合（λ≥1e-3 且 μ 小）双轴受损（锚变滞后拖拽）。

**C. 效用豁免轮休**（top-q 长使用迹免轮休）：q=0.10 → 90.9/94.2、q=0.25 → 89.1/95.4——
与端点连成光滑权衡线，但 **发作门在等静态下时序高 1.1，支配 util10**；13b 对代价来源的定位
正确，但结构豁免付成比例时序代价，**时间承诺严格优于结构豁免**（phase 10 教训在效用层复现）。

**D. 锚×发作门组合**：92.3±0.3/93.8±0.5——落在两者之间，不支配任何一方（锚与门不叠加）。

**6 种子加固后的 Pareto 前沿**：锚 92.4/93.3 → 组合 92.3/93.8 → **发作门 92.0±0.4/94.2±0.4
（时序与常开完全打平、静态 +0.9 → 干净弱支配，常开作为工作点已过时）** → silent 89.0/96.0。
配置：`cif/cfeat_sgd_refr_e*`、`cfeat_bp_er_lr*`、`g15_sgd_anchor_*`、`g15_sgd_util*`、
`g15_sgd_anchorblock_*`；旋钮 `anchor=(lam,mu)`、`util_q`、`bp_lr`。

**15F 边界分解（特征体制的 2.7 分归因）**：把本地网的部件逐一移植给 BP+ER（每格取两档 lr
较优者）——宽度 512-256 反而帮 BP（44.5±0.4）；30% 距离接线（同掩码生成器同种子）只花 0.6
（43.9±0.7）；**5% k-WTA 花 4.4 分（39.5±0.6），把 BP+ER 压到本地学习者（40.6±1.0）之下**。
**边界 = 密集激活，不是 BP 的信用分配**——同底座下本地规则不输 BP。实现：`bp_conn`（从
CortexNet 偷掩码、每步后重置零）、`bp_kwta`（KWTA 模块）；配置 `cfeat_bp_er_w512{,_d30,_d30k5}_lr*`。
FW (i) 更新为"k-WTA 的特征体制代价能否买回（更温和/退火稀疏、学习阈值）而不失去它供能的
隔离通道"。

**15E 能耗重测**（`scripts/p15_spike.py`，新默认栈，静态 MNIST，seed 0，协议同 v10）：
发作门系统 T_s=8 → **93.5% @ 118 nJ**（密集率网 2461 nJ 的 1/20.8、事件驱动率网 516 nJ 的
1/4.4），且随 T_s 单调升（16→94.0）。**旧"校准交互"主张在 SGD 下反号**：轮休网现在是校准
稳健方（T_s 单调），rotation-free 反而 T_s>8 劣化（94.5→91.2@32）——轮休塑造更稀疏的码
（隐层维度 12.7 vs 28.8、每样本 60 vs 91 脉冲 @T_s=8）。Energy 小节与 FW (vi) 已改写。

### 第 16 期（k-WTA 税的买回 / 稀疏拨盘，2026-09-02，~110 运行）

15F 把边界定为 5% k-WTA 后的自然追问：这个税我们自己也在交吗？**在交，而且买回几乎免费**。
**cfeat 税曲线**（本地 SGD，5→25%）：时序 40.6→42.8±0.3、静态 38.6→45.6±0.5，遗忘还略降；
BP 对照同形状曲线（39.7→41.2→43.6），**本地在每个匹配档领先 ~1.1 分**（规则优势与稀疏档正交）。
**MNIST 拨盘**（全 6 种子）：时序峰在 **af=10%（92.7±0.3/94.7±0.3）——支配 5% 前沿的所有点**
（发作门、锚在内）；15% 换挡到 92.5±0.3/95.1±0.3；25% 平台（92.5/95.1）。三个对照定调：
① 无缓冲天花板随 af 升（95.1→95.8→95.9）→ 同稀疏轮休静态价从 1.8 缩到 0.7–1.0，
**门的存在意义随价缩水**（15% 时常开已不输门控）；② silent 不受益（89.0→89.3→88.7）——
拨盘只帮轮休家族（5% 时被表征空间饿着）；③ **不对称稀疏阴性**（浅松深紧：时序持平、静态
−3~4，税每层都收）。能耗侧免费：synops +3–5%，脉冲推理输入主导（15% 下 94.0% @ 120 nJ，
T_s 单调性保持）。**决策：论文保持 5% 为记录底座（机制比较内部匹配），新增 "The sparsity
dial" 小节 + 前沿图橙色系列；拨盘（而非门）是轮休静态价最便宜的解**。10% 重定基线列为
future work (i)。配置：`g16_*`、`cfeat_sgd_refr_s*/a*`；15F 系列已全部加固至 6 种子
（阶梯微调 44.8→43.6→39.7）。

### 手稿重组（2026-09-02）

主线从旁线中解放：正文压缩至 ~12.5 页（曾 ~16），六条旁线移入附录并在正文一句话引用——
A 底座阶梯（Table 2）、B Adam 时代按批门研究（含 fig_rotation）、C K=200 三角定位（含
fig_k200）、D 下选择、E η 扫描、F 残余价探针细节（发作旋钮/OFF-null/效用/锚）。摘要删旁线尾
段、补边界分解主张；"reading the table" 收紧。**重跑范围含义**：future work (i) 的 10% 重定
基线只需覆盖正文主张（Table 1、r1/微批/方向、CIFAR、能耗、拨盘已在 10%），附录内容可如实
标注"历史 5% 底座研究"无需重跑。

### 第 17 期（10% 记录底座全面重定基线，2026-09-02/03，~160 运行）

W1 机制网格 + W2 六种子加固 + W3 叙事数字，全部主文主张移至 s10+SGD 记录底座；附录保留 5%
历史网格。**排序全部保持，故事净增强**：headline 92.7±0.3（6 种子）vs 最佳夜间 90.7±0.4
（差距 1.3→2.0）；轮休 +3.9、隔离 +5.7（方差 4×）、镜像 −7.7、泄漏>限制（92.6 vs 91.8）。
**关键量变**：① 门/锚在记录底座全部**中性化**（发作门 92.3/94.9 与常开 92.7/94.7 互不支配、
锚 92.7/94.6 = 常开）——拨盘吸收了它们的存在理由，**默认系统简化为常开轮休+SGD**；
② 无状态对 Adam 优势收窄至 +0.9；③ **CIFAR 静态反转是 5% 特有现象**（s10 下 27.6±0.4 vs
none 29.5±2.8 中性）；④ surprise 触发在 s10 完全不点火（0 事件 → 19.0）；⑤ raw CIFAR 排序
拉大（refr 29.5 > night 26.2 > BP+ER 25.1）且**夜间反超 BP**；⑥ 特征体制边界经拨盘+公平双修
收至 **1.9**（41.7±1.2 vs 43.6±1.1），匹配底座下本地 41.7 > BP 40.8（k10 移植格）；
⑦ 同底座夜间在 s10 双峰不稳定（90.9±3.2，新开放项）；⑧ br 曲线平坦至 8、br4 崩（63±46）；
⑨ cad2 不再免费（91.4，−1.3）。手稿全文（摘要/Setup/r1/微批/脉冲串/轮休价/CIFAR/能耗/
Table 1/解读/dial/limits/FW）+ 三张主图（fig_batch/fig_timing/前沿图——s10 蓝色主系列 +
5% 灰色历史点）全部重建。配置：`g17_*`、`cif_s10_*`、`cfeat_s10_*`、`cfeat_bp_er_w512_d30k10_*`。

### 第 18 期（CIFAR-100 / 100 类自举失败的诊断与部分修复，2026-09-03）

**基建**：CIFAR-100 加载器（gray + V1 特征前端）、10×10 任务表、全管线类数泛化（`n_classes`/
`split_tasks`/`NC` 贯穿 run/run_local/BP/镜像/读出掩码）、旋钮 `kp_decay`/`tgt_scale` 每配置化。

**发现（探针链完整定位）**：局部 burst 学习者在 100 类输出下**完全无法自举**（静态 2.4%，
10 类子集同数据 20%+）。机制链：① 每类监督信号频率随 C 降 10 倍 → ② W2 的局部梯度跑不赢
MNIST 调定的 KP 衰减 1e-3（|W2| 按纯衰减轨迹归零，|W1| 不衰减因 l=1 无 KP）→ ③ a2 活动塌缩
→ ④ 读出层梯度 = 0（|W_L| 也按纯衰减走）→ 全链饿死。与既有记录"KP 衰减是逐数据集参数
（表格 1e-2 / MNIST 1e-3）"同一现象的更极端形态。排除项：读出层 k-WTA（预测用 x_L，误差不
被门控）、kp=0 单独（反馈失准，误差上升）、η 单独（任何档全灭）、读出 lr 放大（隐层已死）。
**部分修复组合**：kp=1e-4 + 目标 ×3 + η=0.02 → a2 恢复增长，raw 4%、特征 5.2%@4k 批仍爬升
——但整个底座在 split CIFAR-100 就是地板区（**BP+ER 也只有 6.1%**，机会 1%）。结论：100 类
需要更强前端/更宽网才是有意义的 benchmark（future work）；当前网格（20 配置 × 3 种子，含
K=5000 每类匹配缓冲）如实记录地板区的相对排序。

**网格结果（60 运行）——排序全面反转，边界陈述干净**：BP+ER 6.2–9.9（特征 K=1000 最高
9.9±0.5）> 夜间 2.6–4.8 > 无缓冲 2.3–3.1 > **本地日程全体 ~1.0–1.4**（refr/none、任一 K、
静态亦然——组合修复在探针里有效但被重放交织 ×3 增益与轮休在脆弱自举期的开销压垮）。
**结论：局部睡眠机制预设一个可自举的基任务**；100 类 + 此底座上无人自举，机制开销主导，
排序反转。CIFAR-100 成为可打 benchmark 的前置条件：更强前端或更宽网 + 类频率归一的 KP。
（绝对数字全在地板区，仅作排序与边界记录，不入正文主张。）

### 第 19 期（自适应 KP 衰减控制器，2026-09-03）

**规则**：λ_l,t = ρ·η·EMA(|g_l|)/(|W_l|+ε)——衰减恒为学习驱动的固定比例 ρ（唯一无量纲旋钮，
ρ=0.25 全数据集统一），EMA 仅清醒步更新，W/B 共享 λ_l（对齐机制不变），实测 λ 逐层入档
（`kp_eff`）。设计原理 = 项目"自参照信号"传统（新奇门/进度门）应用于巩固-遗忘平衡。

**跨数据集判定（一条 ρ，无任何逐数据集调参）**：MNIST 92.4±0.5/95.4±0.2（时序平、
**静态 +0.7 反超**固定 λ）；CIFAR 28.2±1.3/**31.1±1.1（静态 +3.5！）**；cfeat 41.0±1.6/
**43.8±0.4（静态 +1.9）**；**c100 纯控制器（无目标放大）3.5±0.4 = 本地方案在 c100 的历史
最好**（手工救援 kp=1e-4+tgt×3 只有 1.4）。实测 λ 自选值:MNIST 时序 ~8.6e-5、静态 ~9e-4、
CIFAR 静态达 3.3e-3、c100 ~1.2-1.7e-3——**λ 随体制/轴/层自动分化，per-dataset 调参史终结**。
静态轴的系统性反超说明固定 λ 一直低估了厚信号体制的最优正则强度。

**交互伪影（重要）**：tgt_scale=3 会 3× 抬高梯度估计 → λ 冲到剪裁上限 1e-2 → 重新饿死
（c100+tgt3 = 1.1%）。**控制器与目标放大不可叠加；控制器直接取代 tgt_scale**。
配置：`g19_kad_*`；旋钮 `kp_adapt=ρ`（cortex.py KP 块内实现）。

**19 收官（加固 + v2 + 交互 + 表格边界）**：v1 六种子定稿——时序 −0.5~−1.4 vs 各自调优的
固定 λ、静态 +0.5~+3.6、c100 全线 3–5×。v2（驱动比 ρ·|ΔW|/|W|，优化器无关，代码规范版）
与 v1 噪声内等价（MNIST 静态 96.1±0.2 = 全项目静态最高，含 silent）。**λ 双角色定论**：
自举/对齐角色 → 控制器全面接管；容量正则角色（SUPPORT2 的 1e-2）→ 不在驱动比信息集内
（v1 AUC 0.867 / v2 0.852 vs 固定 1e-2 的 0.895，预测先于数据）。**组合戒律**：控制器只能
看未放大的清醒流——夜间 ×3 增益与其组合灾难性（60.2±2.8，λ 冲顶抹掉白天学习），与
tgt_scale 伪影同族；锚/发作门组合正常（同签名：时序 −0.5~−0.8、静态 +1.1~+1.3，
bout+ctrl 静态 96.0±0.3）。

### 第 20 期（深度阶梯：叠层下的存活性，2026-09-03）

**问题**：叠更深（3/4/5 隐层，512-256-128-128-128 递减）后本地系统还能不能用？控制器是否
如预测在深层更关键？对照 = 同宽度 BP+ER。45 运行（d×{kad,fix}×{seq,static}×3 种子 + bp）。

**判决表（MNIST s10，seq/static）**：
- d3：fix 91.4±0.4 / 93.6±0.4；kad 90.6±1.0 / 95.6±0.1；bp_er 88.9±0.3
- d4：fix 89.3±2.5 / **9.7±0.7（死）**；kad 90.0±0.6 / 95.3±0.2；bp_er 88.8±0.3
- d5：fix **55.5±39.3（双峰死亡）** / **10.1±0.3（死）**；kad 87.6±1.7 / 91.8±0.8；bp_er 88.1±0.6

**结论**：(1) **深度 = 第 18 期同一饥饿边界的另一条轴**——类数摊薄按类稀释误差，深度按层
衰减误差；固定 λ 在信号变薄的格子上不是渐退而是悬崖（static 轴 10 类同时监督 + 深层衰减
→ d4 起纯随机；seq 轴每任务 2 类信号厚，撑到 d5 才双峰）。(2) **控制器逐格救活并优雅退化**
（每层约损 1–1.5 分；饿层实测 λ 自动降到 3e-6）——"控制器是深栈地基"的预测被 d5 的
+32 pt（seq）/+81 pt（static）差距实证。(3) **带控制器的本地系统在每个深度都不输 BP+ER**
（d3/d4 反超、d5 打平），且遗忘只有 BP 的 1/2–1/3（F 5–8 vs 13–14）。BP 深度不敏感但
不巩固；本地系统靠控制器换来同样的深度容忍。配置：`g20_d{3,4,5}_{kad,fix}[_static]`、
`g20_d*_bp_er`；`make_bp` 已通用化到任意深度。

**20B（ResNet 式跳连,同日）**：S^l 从 a^{l-2} 直入第 l 层基底驱动（越层旁路,稠密——旁路轴突
长程）,反馈孪生 Bs^l 用同一局部乘积 + 共享 λ_l 学习（KP 配对）,Dale 按 l−2 层类型投影,
重放隔离用同一"前睡或后睡"掩码（未提供掩码时重放中冻结,保精确性）。判决（3 种子）：
- **d4+skip 91.7±0.4 / 96.3±0.2（项目静态新高,超 d2 的 96.1）；d5+skip 91.5±0.4 /
  96.2±0.3——五隐层两轴全额回到 d2 记录,深度学费清零**。
- 机理探针 fixskip（固定 λ+skip,无控制器）：d5 91.3±0.8/93.6±0.2——**旁路单独救活固定 λ
  两轴** → 深度崩溃定位为**误差路径衰减**（捷径恢复深层误差厚度）,不是逐层衰减调参本身；
  控制器在旁路之上的余量集中在静态轴（seq 噪声内打平,static +2.6）。
- 修订归因：深度轴有两条救援路径——架构（旁路补信号）与自适应（控制器停掉衰减竞赛）,
  组合最佳；类数摊薄轴（c100）无架构可加,仍是控制器领地（skip×c100 未试,FW）。
配置：`g20b_d{4,5}_skip[_static]`、`g20b_d5_fixskip[_static]`；实现在 cortex.py
（S/Bs/mS/vS,forward 旁路项,sweep_errors 跳连反馈,local_update 共享 kp_l + skip_mask）。

**20C（带 skip 重返 CIFAR-100,同日,42 运行）**：
- 第一波 深度×skip（d4,控制器,raw/feat × seq/static,3 种子）：d4 无 skip 近随机
  （raw 1.4/1.3,feat 1.9/1.6——比 d2 还差,双重饥饿）；**skip 每格约翻倍**（raw 2.4/2.1,
  feat 3.8/3.0）但**全格低于 d2**（raw 3.7/4.3,feat 5.7/6.0）。→ 类摊薄轴上深度是净亏损,
  skip 只退回部分深度税,不是薄信号的解药；skip 修路径衰减的机制在 c100 上同样成立。
- 第二波 宽度/缓冲（c100f,d2+控制器）：**wide (1024,512) 6.7±0.3 / 7.5±0.2 = c100 本地
  新记录（两轴）**,方差收紧；K=5000 无效（5.6±1.1/6.4±0.4,重放量不是此处约束）；
  wide+K5000 不叠加（5.4±0.6/7.2±1.2）。参照：BP+ER 9.9±0.5（F 40 vs 本地 20）,night 4.3–4.8。
- 结论：**薄信号体制要的是短链长上的容量,不是深度**；与 BP+ER 差距 4.2→3.2,仍 floor
  体制（仅论排序）。真正突破 c100 仍需更强前端（静态轴各配置都撞 ~6–7.5 墙,前端表征封顶）。
配置：`g20c_{c100,c100f}_d4[_skip][_static]`、`g20c_c100f_{wide,k5000,wide_k5000}[_static]`。

### 代码审查轮（外部 reviewer 只读代码提出 9 项，2026-09-04，~240 运行）

逐项核实后：4 项高优先中 3 项属实、1 项部分属实；5 项中优先全部属实。修复与结果：
- **H1 CIFAR-10 特征前端阈值泄漏（属实）**：`feats()` 对 train/test 各自用前 500 张估阈值
  （测试阈值来自测试集,相差 7.2%,含 ~2–3% 采样噪声）。修复：阈值只在训练集估一次
  （缓存改名 `feat256_trthr`,训练特征逐位不变,测试特征平均变动 1.6%）。**全部 49 个 cifarf
  配置 183 运行重跑**：平均 Δ −0.20、平均 |Δ| 0.23、最大 |Δ| 1.1（bp_er_1000;BP 格比本地格
  更敏感）,**排序全部不变**；特征体制边界 1.9→1.7,拨盘曲线 40.5→42.7 / 39.4→45.6。
  旧 pickle 归档在 `parts_review_archive/cfeat_leak/`。CIFAR-100 前端用每样本空间均值,无此问题。
- **H2 spiking 阈值用测试集校准（属实）**：p10/p15/p16 改为 `Xtr[:2000]`,重测后全部数字变动
  ≤0.3（默认格 93.8% @ 119 nJ,T_s=16 94.4 @ 241;无轮换对照 94.6→90.1）。
- **H3 在测试集上反复选型（属实,不可追溯修复）**：新增 `--val`（训练集分层留出 10% 作评估集,
  `val_` 前缀 checkpoint）,12 个头条配置 × 3 种子复现：val 平均比 test 低 0.9,序列轴
  Spearman ρ=0.92,所有主张的排序成立（refr/bout > night > BP+ER > 无掩码 ER;skip d5 两轴保持;
  控制器静态签名保持）。手稿 Limits 明确声明 + 新附录 G。
- **H4 "exact isolation" 与默认实现不一致（部分属实）**：命题本身限定正确（Prop.2 带 margin,
  readout 可塑已声明）,但 Remark "0.0 drift" 与摘要/Discussion 的 "provably output-neutral"
  过度。新增 `diag_drift` 全程测量（18,760 重放步）：默认配置每步清醒批预测改变 0.27%、顶层
  隐层码改变 0.36%、睡眠单元被唤醒 ~1e-4；readout 隔离时预测改变 0.001%；Adam-5% 时代隐层码
  改变 1.0%（旧 "0.0" 是单批检查的典型结果,非总体率）。手稿三处措辞收窄为实测率。
- **M5 `syn_mask` 存在即视为 replay（属实,且是 mirror 消融的真实混淆）**：`local_update`
  新增显式 `replay=` 参数,掩码（谁动）与簿记（burst baseline / 控制器 EMA / 锚 / goodness）
  解耦；mirror 清醒更新传 `replay=False`。**mirror 重跑后更差**：s10 85.0→77.9±1.6,
  5% 81.3→70.3±10.1,static 不变——旧代码下 mirror 的 burst baseline 从未更新（等于该消融
  一直在 baseline 关闭下跑）。"反向保护代价"由 8 分改为 15 分,手稿披露旧值。
- **M1** run_bio checkpoint 含 seed（seed 0 保留旧名）；**M2** 运行参数写入 checkpoint,
  `--resume` 不符则警告重跑；**M3** runs.csv 重生成；`requirements.txt` 新增。
- **M4**（课程 evaluate.py heavy-tail 列在内层 CV 前选定）：属实但无标签统计、最终测试评估
  干净；按约定不动课程代码。
- **顺带发现**：phase 19 的 v1（梯度参照）公式曾被 v2 就地替换,`g19_kad_*` 结果不可复现；
  恢复为 `kp_adapt_mode="grad"`（λ = ρ·η_当前·EMA|g|/|W|）并标记全部 g19_kad 配置,
  cfeat 的 v1 格用该模式重跑（40.5±0.6 / 43.8±0.2,旧 40.7±1.2 / 43.5±0.6）。**复现校验**：
  `g19_kad_mnist` s0 在 OMP=4 下与原始 pickle 逐位一致（acc_matrix 最大差 0.0,kp_eff 全位相同）。
- **复现性边界（新发现）**：同一配置在 OMP_NUM_THREADS=2 vs 4 下结果不同（未改代码的
  `g16_sgd_refr_s10_w512` s0：92.53 vs 91.97）——矩阵归约顺序的浮点差经 k-WTA 的不连续性
  在 18k 步内放大。**逐位复现只在固定线程数下成立**（项目主体运行用 OMP=4）；checkpoint
  现记录 `omp_threads`。本轮 mirror/diag 格在 OMP=2、v1 cfeat 格在 OMP=2–3 下产生,统计上等价
  但不与 OMP=4 逐位对应。

### 手稿评审轮 + workshop 版（2026-09-04）

外部手稿评审（`report/critic.md` 第二部分）逐条核实后几乎全部接受并落地：式 (4) 转置
（B^ℓ∈ℝ^{n_{ℓ+1}×n_ℓ}）、命题 2 加 Δb、能耗常数改引 Horowitz 2014（0.9/3.7 pJ 45 nm FP32）
并改称"理想化算术能耗代理"、补 Rueckauer 2017 / Whittington 2017 / Vitter 1985、
four-way、"inconclusive"、3.7 倍标准差、控制器 2.6–5×、"no dataset-specific decay coefficient"、
结论限定、验证集范围限定、R 定义统一为样本数、生物类比软化（tonic ↔ soft rotation 一致化）、
实验日志腔清理；fig_timing 刻度/标注修复。**10% 底座 η 扫描（g23,36 运行）**证实 5% 结论：
SGD η=0.02 联合最优（0.01: 92.2/93.3；0.05: 91.5/94.7；0.1 崩 79.5±17）,Adam 默认 1e-3
即最优（3e-4: 89.9/89.9；3e-3: 90.9/94.3）。**Workshop 版** `report/workshop/cl4fmagents.tex`：
CL4FMAgents @ NeurIPS 2026（8 页正文、双盲、非存档、截止 9/7 AoE）,NeurIPS 2026 样式,
面向智能体的叙事,主线保留、旁线与负结果全部入附录,新增控制器×深度图（`make_fig_depth.py`）。

### GPT 手稿评审轮（对 8 页版,2026-09-04）

两个 P0 均属实并已修：(1) **重放成本**——头条配置 batch_replay=16,18,760×16 = 3.0e5 =
夜间的 2.4×,"1.2×" 是 batch-8 格（91.8）的旧数字;摘要/结果/讨论/附录改为"2.4×;匹配 1.2×
预算时 batch-8 达 91.8"（另跑 cadence-2×batch-16 探针,恰 1.2×）。(2) **命题 1**改为只声明隐层
活动不变 + "readout 满足同条件或固定则输出不变"的推论;摘要/引言"present computation
invariant"改为"hidden computation … exact channel, near-exact end to end"。P1 全部落地：
定理 batch 化（X={x_b}、每样本阈值 τ_{k,b}、严格不等式）、增益 g^ℓ 声明为固定 1 并从公式移除、
**masked heavy-ball 更新式**按实现写出（速度掩码外冻结、衰减限于掩码、偏置按单元掩码、μ=0.9）、
控制器驱动定义为"优化器输出、衰减前"的清醒增量 u（EMA 0.02、ε=1e-12）、"one constant"→
"one dataset-independent ratio target"、第 5 节改名 "Scaling the mechanism: self-referenced
decay and depth" 并写明统一原则（novelty/homeostasis/decay 均参照学习者自身状态）、
"cannot pause"→"may not afford downtime"、Driessen 推理跳跃标注、"must/never"→"tested trigger"、
深度归因→"consistent with … principal contributor"、能耗"per FP32 accumulate"、Sorrenti 换
TNNLS 36(7) 正式版、Tononi 补全标题、burst 学习加 Payeur 2021;**SESLR 未引**（arXiv 已撤稿）。
新增附录：数据流图（tikz）、底座分解瀑布图、能耗 Pareto 图（`make_fig_appendix.py`）、
基线调参范围说明 + 配对 bootstrap CI（本地−窄夜间 +2.1 [+1.4,+2.5] p=0.004;−同底座夜间
+1.7 [+0.1,+4.6] p=0.25;−ER +5.8;−BP+ER +3.5）。held-out 划分补到 6 种子：Spearman 0.97、均差 0.90；轮换 91.6±0.3 vs 同底座夜间 89.0±3.5（配对 +2.6 [+0.2,+5.2]）、BP+ER 88.8（+2.8）、ER 85.4（+6.3）；skip d5 91.0/95.8；**控制器的静态优势在该划分未复现**（93.5±4.3 vs 94.1±0.2，一个种子塌缩）——两版均如实标为 unconfirmed。匹配成本探针 cad2×br16（1.2×）：91.7±1.0/93.5±0.4。

### 第 22 期（"stateless" 是否字面成立：momentum=0 的纯 SGD,2026-09-04,66 运行）

**动机**：手稿 18 处 "stateless SGD",但 heavy-ball 的速度缓冲是每突触状态（只是无二阶矩）。
**结果**：(1) momentum=0 + 默认重放步（3η）：η∈{0.02…0.4} **全部随机水平**——隐层维度 [·,0,0],
活动完全塌缩。探针定位：清醒路径下纯 SGD 与基线同样学习（η=0.2 匹配有效步长）,塌缩来自
**重放路径**：动量对 16 样本重放微批的噪声有 ~10× 方差平均,去掉后 η=0.2 的重放步发散（NaN）,
η=0.02 则第 2 层活动逐任务衰竭。(2) 把重放增益压回动量平滑尺度后（η=0.2、nrem_gain 0.3）
纯 SGD 可行但更差：**89.9±0.3 / 90.1±0.2**（−2.8/−4.6）；加控制器 91.7±0.2 / 91.4±0.0
（−1.0/−3.3）；地形脆弱（η=0.1、gain 1.0 序列轴 10.3 而静态 90.7）。
**决定**：润色用词。全文 "stateless" → "heavy-ball SGD"（一个每突触速度、无二阶矩、速度只在
掩码内推进）；核心论证改为"速度是同时保住保证与精度的最少状态",消融表新增 momentum 0 行。
速度缓冲的真实角色 = 隔离重放微批的噪声平均。配置：`g22_sgd0_e*`、`g22b_sgd0_e{1,2}_g{01,03,1}`、
`g22b_sgd0_e2_g03_kad`；旋钮 `momentum`（run_local 已接线）。

### 方法节评审轮（外部读者只读 Method 节，2026-09-05，2 诊断运行）

**评审要点全部可吸收，两处是实质修正**：(1) "moments leak 会 surrender the guarantee" 说法过强——
`leak_moments` 里权重改变仍受掩码约束（当批不变性 Prop. 1 仍成立），丢失的是**优化器状态随时间的
约束**（泄漏的矩在下一次未掩码清醒步重新进入清醒突触）；两版 + `cortex.py` 注释均改为此表述。
(2) 训练时序此前未明说：实现顺序 = 清醒前向（W_t，当前抑制）→ **未掩码清醒更新** → 从更新前活动
读 awake 集 → 在更新后权重上做掩码重放；命题对"在被更新权重上算的掩码"逐字成立，主文漂移率是在
实现顺序下测的。**诊断**（`g25_diag_postwake_{free,isolated}`，旋钮 `mask_post_wake`：清醒更新后
在同一抑制下重推该批再取 awake 集）：隐层码改变率 0.363%→0.295%（isolated 0.213%→0.172%），
预测改变率不变（0.274→0.303%；isolated 0.001→0.000%），边际违反率不变（~3e-5 / 1.8e-4），
精度不变（92.2 vs 92.0，单种子）→ **残余漂移来自未强制的边际，不来自时序**；默认实现不改。
**文本补充（两版）**：隐层/线性读出定义（z^L 为输出，one-hot 平方误差）；κ 的逐分量界
|ε_i| ≤ κ Σ_j |B_ji|；G = 局部更新方向、λ = 每步衰减系数；重放前向去掉抑制（refractory 单元可在重放中
发放并学习）；一次 burst 复用同一掩码；ρ_ℓ 公式 |Ã∖A|/|Ã|；压力 = 该批发放样本比例、触发 = 所有隐层
睡眠单元池化均值 ≥ θ、放电 = 复位（δ=1）、e_t = 批均平方输出误差；Prop. 2 证明补"原胜者为何不变"与
τ_k=0 情形；Eq. (5) 改述为"衰减均幅 λ|W| 保持为数据驱动更新均幅的 ρ 倍"；Fig. 1 标题注明默认配置
= 常开轮换 + 每批重放，压力触发/新奇门是稀疏预算下的调度器。**新附录**（两版）"One training step,
the three channels and the skip path"：Algorithm 1（带状态快照的伪码，学习率 η_w=η·m/256、η_r=3η）、
三通道对照表、数值例子、skip 路径的前向/反馈/掩码公式。Workshop 主文守住 8 页（References 第 9 页），
代价是六轮压缩（Fig. 1 scale 0.83、Fig. 2 宽 .44、Fig. 3 宽 .88、若干句子精简）。附带修正：momentum
附录"3× the waking rate"→"3η 对比按批缩放的清醒步 η·m/256"。

### 第二轮手稿评审（Weak Reject → 处理，2026-09-05，~108 运行）

**评审六项与处理**：(1) 测试集选参：不切协议（重跑 300 run 不现实），改为把 held-out 六种子数字
提到 Results 第一段与官方测试集并排，写明"配置在复验前冻结；held-out 样本开发阶段只是训练数据、从未参与
选择、复验时不训练"，其余数字统一标注 development-phase/official test；**CIFAR held-out 复验**
（raw 7 配置 + 特征 7 配置 × 3 种子，`--val`）：排序全部复现——raw 28.9 > 25.7 > 24.8 > 15.4 ≈ 16.3
（refr > night > BP+ER > unmasked ≈ none），特征 42.3 > 34.0 ≫ 20.5，BP+ER 42.8（领先缩到 0.5），分解链
43.7/43.5/40.4。(2) "Inference as consolidation" 过强：主张限定为"监督稀疏 CL 系统中受限步间重放移除
独立离线阶段"，"while the agent acts" → "between the steps of the stream"；补更新次数 39× 与墙钟
（6.6 min vs night 0.6 / 无重放 0.4 / unmasked 6.0）；补**单遍流**（1 epoch/task，`g26_stream_*`，旋钮
`epochs`）：refr 91.8±0.4 vs BP+ER 90.1、night 75.4（每任务只一夜）、unmasked 68.3±19.1、无缓冲 18.8。
(3) exact 范围：Prop. 2 陈述统一 τ_k=0；诊断加种子（隐层码 0.29–0.36%，均 0.32%；预测 0.33%；
max|Δlogit| 0.02）+ CIFAR（隐层 0.46%，预测 3.6%——低边际预测经可塑读出易翻）。(4) 预算与 48× 学习率比：
"matched 1.2×" → "within 1.2× (22% more samples, 39× more updates)"；**重放增益扫描** {1/16,1/4,1,3,10}：
masked 71.9/88.2/91.8/92.7/90.8，unmasked 63.8/83.3/86.2/87.0/85.5——两者都在默认 3 处峰值，unmasked
任何增益下追不上 → 混杂解释不成立。(5) 因果识别：**四格表补齐**（`g26_ctrl_rot_noiso`，旋钮 `no_iso`）：
两者都无 87.0 / 仅隔离 88.8 / **仅轮换 92.1±0.5** / 都有 92.7；轮换是更大的单因子（+5.1 单独、+3.9 叠加），
隔离在轮换开启时的边际精度 +0.6（batch 16 与 8 都是 0.6；batch 4 时 masked 反而双峰 63.4±46.4，仅轮换
89.5）；静态无差别（94.7）。→ 主文改写："隔离买的是保证，轮换买的是精度"。**readout-only 重放**
（`replay_readout_only`）= 随机水平 27.3±15.1、静态还亏 12 点 → 效应全部来自隐层巩固。**随机轮换**
（`mask="refr_random"`，匹配数量）90.8±1.3、静态 92.8、F 9.4：恢复大部分收益，使用依赖再多 1.9/1.9。
Table 1 "−rotation" 行是我们标签错误（该行用的就是完整式 (3) 掩码）→ 改标签；mirror 表述软化；BP+ER
定位为外部参照、明写未含 DER++。(6) 表述与证据：五层"回到两层纪录" → "两层水平（vs 两层控制器格
91.7/96.1，序列轴低于两层默认 1.2）"；"every fixed decay collapses" 加"无 skips"；触发器标为稀疏预算调度器；
"未按深度重调固定衰减"明写。写作：评估时抑制关闭（代码 `suppress=None` 后 predict）、F 定义、附录 K 加
逐任务矩阵（六种子均值，F 5.3±0.4）与 CIFAR held-out 表；Fig. 2 左加"仅轮换"虚线系列；Section 5 压缩、
Table 2 进附录；hyperref 彩色无边框；Fig. 3/附录图字号上调。Workshop 主文守住 8 页（七轮压缩：Fig. 1
scale 0.72、Fig. 2 宽 .43、Fig. 3 宽 .72）。**教训**：5 worker × OMP=4 在 8 核笔记本上每进程只拿到 0.8 核
（第一次启动 32 min 零完成）；改为 OMP=1 × 9 worker 后每进程 1 核、MNIST 本地 run ≈ 14 min。

### 第三轮手稿评审（Weak Accept 方向的收尾，2026-09-05 晚，12 运行）

**四项收尾**：(1) 表 7（CIFAR held-out）两行 static 数据曾显示 "28.9±1.0STA / 15.4±4.2STA"——是
`fill_numbers.py` 占位符前缀冲突（VCIFREFR 被替换进 VCIFREFRSTA）造成的转录错误；回查原始 pickle 后填入
真实 static 值 28.0±1.2 / 30.0±1.9，表题改为"sequential; static for the two rows so marked"。**教训**：占位符
不能互为前缀。(2) 四组合归因统一：isolation 单独 +1.8（88.8 vs 87.0）、有 rotation 时 +0.6；Discussion/
Conclusion 里 "−5.7" 改为 1.8/0.6；Fig. 5 frontier 点 "−isolation" 改标 "−isolation, −rotation" 并加
"−isolation"（仅轮换）点；replay-gain 讨论注明只约束无轮换对照、不估计 isolation 贡献；摘要写明
"rotation carries most of the gain"。(3) 措辞收紧：摘要与 Discussion 的 "near-exactly end to end" → "当前输入、
当前抑制掩码下的隐层计算不变：证明通道上精确，其余 99.7% 的清醒隐层码不变"；"no replay update touches the
units the present input uses" → "replay leaves the hidden activities unchanged (exactly on the proven
channels, within the margin elsewhere)"（式 (3) 允许沉默前突触→活跃后突触的更新）；理论只覆盖训练时带抑制
的计算、评估时抑制关闭；readout-only "at chance / entirely" → "falls to 27.3±15.1 (ten classes)；readout-only
replay cannot account for the full system's accuracy, hidden-layer replay updates are essential"；
"affordable" → "at a higher total compute (11× the night's wall-clock, unoptimised)"；预算计数改为 "within
1.2× (22% more) …, 39× the night's updates (batch 8) or 19.5× (batch 16 every second batch)"；单遍流
"ordering survives" → "keeps its lead over every tested control, night falls below BP+ER"；preprint 5.3 标题
"state-free isolation wins" → "the least optimiser state wins"，Section 8 "Why isolation stabilises
micro-batches" 按四格表重写，附录 D/F 的 unmasked 对照注明 rotation-free。(4) **held-out rotation-only**
（`val_g26_ctrl_rot_noiso[_static]`，6 种子）：91.5±0.3 / 93.7±0.3 vs 默认 91.6±0.3 / 94.1±0.2——次可加性
在 held-out 上复现；进入 Table 5 与 Results held-out 段。**通读**：两版 PDF 全文过了一遍（pdftotext），修正
Fig. 2 标题 "0.6–0.7"→"0.6"、双括号、preprint 控制器 "at most 1.4" → "0.5–3（gradient/drive 两种形式）"、
"provably cannot serve" 软化等。Workshop 主文守住 8 页，Fig. 1 scale 0.85、Fig. 3 宽 .74（图字号优先于文字，
用 Section 5/Discussion 的压缩换回）。

### 协议切换：全部主文数字改为 held-out 划分（2026-09-05 夜 – 09-06 晨，~400 运行）

**动机**：审稿人对"官方测试集选参"的保留意见靠附录澄清始终扎眼；干净的解法是换协议——官方测试集
定位为 development set，论文报告的每个数字都在训练集的 10% held-out 划分（种子 1234，从不训练、从不
参与任何决定）上测得。**重跑**：主文引用的 104 个配置 × 3 种子（headline 行已有 6 种子）= 312 run，
8 个单线程 worker（OMP=1）约 9.5 小时；补种子 24 run（readout-only、night-on-top、gain 10、batch 4、
trickle/burst/pressure 各到 6 种子）；**第二个独立划分**（`--val-seed 4321`，pickle 前缀 `val4321_`，
20 个 headline 配置 × 3 种子）回应"单一划分"；尖峰能耗用 `scripts/val_spike.py` 在 held-out 上重测。
未重跑：附录 E–H 的 5% 旧网格、η 扫描、K=200、下选择、梯度形式控制器、c100 深度——统一标注
development-phase（官方测试集）并只作为机制排序证据。

**held-out 关键数字（vs 开发阶段）**：默认 91.6±0.3 / 94.1±0.2（92.7/94.7）；夜间窄底座 89.2±1.7、
同底座 89.0±3.5（第二划分 91.5±0.5！夜间的双峰性是划分依赖的，领先幅度从 +2.6 缩到 +0.3）；BP+ER
88.8±0.3；unmasked SGD 85.4±3.5（第二划分 80.0±14.1）；silent 88.0±0.7；**仅轮换 91.5±0.3 / 93.7**
→ isolation 的边际精度 +0.1 [−0.3, +0.5]（batch 16）、+0.2（8）、+0.9（4）、静态 +0.4；单独 +2.6；
readout-only 41.7±10.9；随机轮换 91.0±0.8 / 92.1；**masked Adam 与 SGD 打平**（91.5/94.1 vs 91.6/94.1，
leak 91.6）——开发阶段的 0.9 优势没有复现，措辞改为"least optimiser state suffices"；momentum 0
88.4/89.4；mirror 78.0±3.2；soft 89.9/84.3。**batch 4 不再双峰**（89.7±0.7 vs 开发 63.4±46.4），仅轮换
88.8；**clocked burst 失效**（80.5±6.1 ≈ trickle 78.5），只有压力触发有效（88.6±1.6）；增益 10 在两种系统
上都不稳定。控制器：CIFAR 三个体制静态 +1.2 至 +5.3，MNIST 静态第一划分 93.5±4.3（一个种子崩溃）、
第二划分 95.2±0.4 → 不主张 MNIST 静态收益；深度：d5+skips 91.0/95.8 vs 两层默认 91.6/94.1。CIFAR
排序两个划分全部复现，特征前端 BP+ER 领先缩到 0.5。能耗：默认 92.7% @ 117 nJ（21× 低于 dense），
rotation-free 对照在 T_s=32 只降 0.4——"校准交互反号"主张撤回。87 个序列配置 test-vs-held-out
Spearman 0.973、均差 −1.1。

**文稿**：两版 Section 4/5、摘要、Intro、Discussion、附录 C/D/I/J 全部数字重写；协议段一句中性表述
（official test sets served as the development set; every reported number is on a held-out tenth …）；
附录 K/H 改为"development-phase vs reported vs second split"对照表 + held-out 逐任务矩阵（F 6.4±0.6）；
图 2/3/5/6 用 `MLPC_VAL=1` 重生成（frontier 去掉 5% 灰点，Pareto 改为 held-out 的默认 vs rotation-free）。
Workshop 主文守住 8 页。**教训**：占位符互为前缀会串填（VCIFREFR/VCIFREFRSTA）；Windows 写的清单文件
带 \r 会让 shell 监视器计数为零；8 核笔记本上 OMP=1 × 8 worker 最有效。

### 补充实验：isolation 在 replay batch 2 上的边际（2026-09-06 晨，15 run）

**动机**：协议切换后 isolation 在 rotation 之上的精度边际只有 +0.1/+0.2/+0.9（batch 16/8/4，后者
仅 3 个 noiso 种子），想看曲线最小的一端是否延续"批越小价值越大"。**运行**：`g17_sgd_br2_s10`
与新配置 `g26_ctrl_rot_noiso_br2`（`run_seq.py` 的 noiso 列表补 b=2、g17 字典补 `sgd_br2`）各 6 种子，
加 `g26_ctrl_rot_noiso_br4` 种子 3–5，全部 `--val`；8 个 OMP=1 worker 约 22 分钟。

**结果**：（1）趋势不延续：batch 4 补到 6 对后配对边际 +0.3 [−0.5, +1.2]（noiso 89.3±0.7，原 3 种子
88.8±0.4 偏低），于是 16/8/4 的边际是 0.1/0.3/0.3，全部在种子噪声内，"growing as the micro-batch
shrinks"撤回，各处范围 0.1–0.9 改为 0.1–0.3。（2）batch 2 两者都大幅退化，但方式不同：isolated
79.0±3.2（F 21.5±4.2），六个种子全在 75.5–84.3；unmasked 分叉——三个种子 82.5–84.8（略高于同种子的
isolated：+0.3/+1.8/+8.1），一个 58.2，两个塌到 9.9（chance）。配对均值 +24 [−0.2, +48.5] 只描述失效
模式，不当作边际引用。**结论措辞**：isolation 的精度价值不是边际而是去掉一个失效模式；每个 batch
尺寸下它买到的是保证（live computation 不变），在 batch 2 处 rotation 的容忍度耗尽时体现为优雅退化。
泄漏到 live coalition 的 replay 不是纯损失（存活种子更高），但打开了 mask 关闭的失效通道。

**文稿**：两版摘要、引言贡献 1、Results "What rotation and isolation each buy"、图 2 说明、Discussion
/Honest limits、preprint §8 "Why micro-batches work" 更新；新附录 "Isolation across replay batch
sizes"（workshop K / preprint H）+ 图 `fig_isomargin.pdf`（`report/make_fig_isomargin.py`，`MLPC_VAL=1`）。
图 2 左面板保持 4–256（加 batch 2 会压扁 88–92 区间），batch 2 由附录图承担。Workshop 主文为守住
第 8 页缩了 Discussion/Limitations 的措辞（无主张改动）。

### 单遍流结果提到正文（2026-09-06 晨，纯文字）

用户问"主张被削弱后还能投吗"，结论：工作坊层面可投（核心论点、机制对照、形式保证、held-out 协议都在），
最贴题的证据是单遍流（一遍过、无离线相位、领先所有对照），此前只在 Limitations 括号里引到附录。改动：两版
Results 首段加一句（91.8±0.2 vs BP+ER 88.7±0.3、夜间 75.1±3.5，one night per task is too few），Table 1
加第四列 "Seq., single pass"（full system / unmasked ER 76.0±20.1 / no buffer 18.1 / night / BP+ER 五格，
其余留空；preprint 表改 footnotesize + tabcolsep 4pt 才不溢出），摘要加半句，Limitations 改指向表 1；附录的
Single-pass 段保留。Workshop 为守第 8 页再缩了 Component prices / 控制器 / 深度 / CIFAR 迁移几句的措辞，
图 2 宽 .38、图 3 宽 .64。不加 DER++：截稿前不动新代码。

### 收尾修稿：数据来源声明、成本与 exactness 按新协议重测、图 5、措辞（2026-09-06 上午）

第四轮审稿（`report/critic.md`）七点，用户要求做其中四项，且**不再向审稿人叙述测试集修复过程**：
1. **数据来源声明统一**：删掉"every reported number ..."的全称断言（与附录 E–H、表 3、附录 I/J 的
   development-phase 结果直接矛盾）。统一表述：配置在官方测试划分（development set）上选定并冻结；
   主文比较把训练集分层 10% 从流中移除后重训并在其上评估，该 10% 未参与任何选择；第二个 10%
   （不同种子，与第一个共享 9%＝563/5999 个样本，不是不相交测试集）复验 headline 行；未按此协议重跑的
   探索性结果标注 development-phase。附录 L/I 引言与表 5 表头（development set / held-out split 1 /
   split 2）同步；Spearman 0.97 注明"稳健性检查，不证明选择偏差为零"。不写"曾经在测试集上选参然后修复"
   的叙事。
2. **成本按 held-out 运行重算**（流是九成训练集）：更新数 18,760 → **16,885**（cadence 2：8,442；
   pressure 5,779≈1/3；clocked/cad8 2,096/2,110≈12.5%）；replay 样本 2.7e5，**2.2×** 夜间（原 2.4×）；
   8-sample 与 cadence-2 方案 1.35e5 ＝ **1.1×**（原 1.22×）；更新次数 **35×**（原 39×）、17.6×（原 19.5×）。
   Wall-clock 用 seed 100、OMP=1 的单线程干净计时（`val_*_s100.pkl`，只作计时不入表）：默认 8.8 min、
   unmasked 8.6、夜间 1.4、无 replay 1.0、BP+ER 0.4 → Discussion 的"11×"改 **6×**。
3. **Exactness 按新协议重测**：`g21_diag_free` ×3 + `g21_diag_isolated` ×1 `--val`（16,885 次更新）：
   hidden-code 0.33%（0.31–0.35）、prediction 0.32%、readout isolated 0.21%/0.001%、margin ~1e-4、
   平均最大 logit 变化 0.021——与开发运行几乎相同；正文/Remark 改用新数并注明协议，开发运行
   （post-wake 顺序检查、CIFAR 0.46%/3.6%、5% Adam 1.0%）保留并标注。
4. **图 5 Pareto 改为程序计算支配关系**（`make_figs.py`，按种子均值）：非支配＝bout gate 91.8/94.4、
   silent 88.0/95.3、unmasked 无 rotation 85.4/95.8；默认、adaptive λ、anchor、Adam 均被支配；图注同步
   （原"adaptive decay extends the static side"错误）。标签加引线。旧结论残留：附录 C "SGD 在
   consolidation loop 两轴胜过 Adam" → 打平；"两条 gain 曲线都在默认值达峰" → unmasked 在 gain 1–3
   是平台（85.7/85.4）。
5. **措辞收紧**："removes a collapse mode" → 在所测种子与配置下没有出现 unmasked 对照在半数种子上的
   严重失效（操作定义：最终精度 <60%：两个 chance、一个 58.2），六个种子只能给出频率上界；masked 系统
   在 gain 10 也会失稳。"beats the strongest night" → 两个划分上与所测离线方案有竞争力（第一划分 +2.5、
   第二 +0.3），对 BP+ER 与 unmasked local ER 两划分都稳定领先；1.1× 预算只在第一划分验证过。
   **未做**：审稿第 6 点（把"mask 必须与被更新权重状态匹配"的前提提前到方法正文）——工作坊第 8 页已满。

**Preprint 换 arXiv 模板**：`report/arxiv.sty`（kourgeorge/arxiv-style），`\documentclass{article}` +
`\usepackage{arxiv}`，去掉自带 geometry，`\shorttitle`/`\undertitle`/`\headeright` 设为
"Preprint"，摘要后加 `\keywords`；23 页（原 11pt 30 页）。作者块仍是匿名占位，上传 arXiv 前需恢复。

### 作者润色 + 摘要措辞软化（2026-09-06 晚）

用户通读两版并润色：去掉破折号、拆长句；术语统一（"record" 配置/基底 → "default"，preprint 小节
"The sparsity dial" → "The activity fraction"，标签 sec:dial 保留；"Falsified" → "tested and rejected"）；
两版摘要只留主线数字（91.6、单遍 91.8/88.7/75.1、"about twice"），+2.5/+0.3、1.1×/91.1、rotation-unmasked
91.5、isolation 0.1–0.3、协议句、preprint 的 5% 门控段与 14 点 mirror 句移出摘要（正文均保留）；
preprint 摘要部件顺序改为 isolation → rotation → 信号，与 workshop (i)–(iii) 一致；删去三处修复叙事
（39× 计数产物、cell 冻结 baseline、leak 开发期 +0.8）。
随后把两版摘要的 "removes a severe failure mode" 软化为 "avoids the severe failures that unmasked replay
shows in half the seeds"，与正文/引言/Limitations 的 "none of / absence of ... in the seeds tested" 一致。
重编译：workshop 19 页、正文止于第 8 页；preprint 22 页；零错误零 overfull。

### "What this offers agents" 落到具体架构：top-k MoE 例子（2026-09-06 晚）

针对 CL4FMAgents 的 relevance 风险（实验全是 MLP，FM/agents 只在框架里呼应），在 workshop Discussion
的 "What this offers agents" 段加两句：两个前提可在给定架构上核对——top-k 路由的 MoE 层按 token 天然暴露
"本 batch 未用的专家"，稀疏支撑条件由构造成立；局部误差信号则是这类系统仍需自行提供的前提。
同段删掉 "The invariance closes the channel..." 一句并压缩 isolation 数字句，正文仍止于第 8 页（19 页，
零 overfull）。preprint 无此段，在 Discussion 的 "Why micro-batches work" 与 "Correspondence with biology"
之间新增 "Prerequisites beyond this substrate." 段，内容同上。

### 引用 Shen et al. ICLR 2026（RTK-WTA）（2026-09-06 晚）

用户指出的最近邻工作："Robust Selective Activation with Randomized Temporal K-WTA in SNNs for Continual
Learning"（OpenReview uAkexWJ7dW，poster）。全文被 OpenReview 人机验证挡住（PDF/forum/API/代理均 403），
只读到摘要；前作 AAAI 2024 SA-SNN（已引 shen2024）全文读到：class-IL、无任务标签、无 replay、按 spike trace
逐时间步 top-k 掩码、阈值随使用不可逆上升、+EWC，splitMNIST class-IL 约 77–82%（h=1000，K=10）。ICLR 版把
确定性 top-k 换成概率 top-k 以减少任务间表示重叠，摘要称比确定性 K-WTA 高 3.07–5.0 点（绝对值未核实）。
定位：他们的 k-WTA/随机化是唤醒学习中的资源分配（protect-the-past，regularisation 族，无 replay，SNN 基底）；
我们的 k-WTA 用于暴露不可见突触供推理期 isolated replay，并有精确性保证。需承认的重叠："扰动确定性赢家集合
有利于 CL"——我们的随机交替对照恢复 rotation 大部分收益，使用依赖只多 0.6/2.0。novelty（isolation、推理期
巩固、触发、控制器）不受影响。两版 Related work 的 gating 段各加一句引用 shen2026 并加 bibitem。

### 上传前小修：九条明确问题（2026-09-06 夜）

按用户给的清单核实后修改（两版同步）：(2) preprint §4.1 优化器改为默认 heavy-ball SGD，ε-floored masked
Adam 标为早期 5% 阶段与对照；(3) Driessen 2026 区分单侧 ON/OFF 诱导放掉局部睡眠压力与双侧诱导（睡眠剥夺下）
恢复记忆巩固，tonic 对照只说不放压力；(4) MoE 句改为逐 token 路由只暴露该 token 未用的专家，整个 batch 路由并集
是否留下足够空闲专家需测量（load balancing 反向作用）；(5) preprint 控制器 "sits on the static side" 改为被默认
系统支配（90.6/93.5 vs 91.6/94.1）；(6) substrate decomposition 图注 "six seeds except the local learner, three"
是开发期残留——held-out 下四格全是三种子（val_ pickles 核实），改为 three seeds；(7) preprint 附录 C–G 的
development 范围声明加 "except where marked held-out（G 的 single-pass）"，H 的 gain-10 引用改指 §sec:ablation；
(8) 命题旁加一句：命题取 mask 与被保护权重于同一状态；实现中 awake set 先于 waking update，故对更新前权重
精确成立，一步滞后计入测得漂移（0.32 中的 0.02）；两版摘要的 "exact" 加条件 "when the mask matches the weights
being updated"；(9) 两版方法段定义 "during inference" ＝ 数据流步骤之间，三步顺序执行，不主张异步或服务延迟。
(1) 作者块保持匿名（仓库有匿名镜像），arXiv 上传时再填。Workshop 等量删减：引言末句改指 Discussion 的前提段、
Related work 删去与方法重复的 optimiser-state 句、headline 删 local+night +0.8 句（表中有）、component prices
的 hedge 缩短、depth 括号缩短、sleep 段删一句、per-unit gain 句缩短、exactness 段删 logit 变化句、rotation/isolation
段删保证复述。正文仍止于第 8 页（19 页）；preprint 23 页；零错误零 overfull。

### 去掉 Type 3 字体（arXiv 要求轮廓字体）（2026-09-06 夜）

`pdffonts` 显示 preprint 18 个、workshop 20 个 Type 3 字体，全部是 matplotlib 图里的 DejaVuSans（9 张图）；
workshop 另有 1 个 `ectt1000.pk` 位图字体（T1 编码下 `\url` 用的 EC 打字机字体，MiKTeX 没装 cm-super）。
四个作图脚本（`make_figs.py`、`make_fig_isomargin.py`、`make_fig_appendix.py`、`make_fig_depth.py`）加
`plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})`，用 `MLPC_VAL=1` 全部重生成；对比新旧图的
数值 token 完全一致（depth/isomargin 只是 Type 3 抽取时的分词差异），仅字体嵌入方式改变。workshop 前言加
`\renewcommand{\ttdefault}{lmtt}`（Latin Modern 打字机，Type 1）。重编译后两版 Type 3 = 0，页码不变
（workshop 19 页正文止于第 8 页，preprint 23 页）。

## 4. 正面主张（按新颖性排序，检索基准 2026-08）

1. **使用依赖的单元级局部睡眠可以完全替代睡眠夜**：refractory 轮休 + 精确隔离 + 连续微批重放，
   split-MNIST class-IL 91.3±0.3（无夜间）> 最佳夜间 90.7±0.4。文献零命中（门控类工作全部按
   输入身份选子网；睡眠类工作全部全局离线）。
2. **隔离是微批重放的稳定器**：掩码重放对 batch 大小不敏感到 8；无掩码 ER 与离线夜间在同样
   batch 下退化 4–10 pt。"推理即训练"的真实成本 ≈ 1.2× 夜间重放样本量。
3. **清醒重放的触发信号必须是稳态压力，不能是新奇性**（89.9 vs 19%），而夜间时机恰恰相反
  （7C surprise 定时最好）——一个干净的、可与生物学对话的解离。
4. 内部双信号控制器（压力 + 误差停机）在无任务边界的流上以 74–80% 重放量达到固定日程（7C）。
5. 皮层约束几乎免费、能耗 18–20× 改善（v5/v6 与 v9 复测：91.6% @ 119 nJ，密集率网 1/20.7）。
6. **ACh 式相对新奇门控**（快/慢误差比开关轮换）：一套参数同时拿到静态 95.1（代价归零）与序列 90.5，
   且绝对阈值被证明不可跨数据集——门控信号必须是相对于自身基线的抬升（第 10 期）。
7. 小缓冲的重放该"带噪"：K=200 时重放加噪 + 压力脉冲串收回大半缺口（81.8→84.3），呼应 7B 的
   多样性结论与生物学上重放的变异性。

**已证伪**（同样有价值）：输入端 DG 模式分离（9C）、生成重放替代情景缓冲（7B）、反向学习（7B）、
surprise 选样写入（7A）、乘性 burst 编码（6A）、朴素负相做梦（5C）、深度松弛的必要性（4B/6A）、
均衡器式稳态（v3）、spike-count 训练（6B）。

## 5. 局限与开放问题

- CIFAR 只做了灰度 MLP（欠拟合体制，静态 10 类 ~23–29%）；带特征前端（卷积/patch）的验证未做。
- K=200 残余 ~2.6 缺口：小缓冲下夜间对每个独立样本的巩固效率仍更高。
- 新奇门控在永不收敛的流上会错关轮换（CIFAR 序列 19.1）；"按学习进度定义新奇"未实现。
- refractory 训练的网在长脉冲积分窗（T_s>8）下精度反降，阈值校准交互未解释。
- 单元级压力做轮换选择被证伪（纯权衡）；配额版在连续压力下饱和为 refractory 等价物。
- 新颖性声明基于我们自己的检索（2026-08），发表前需系统查重。

## 6. 复现

```
# 序列轴全部（含 9 期配置；~40 配置 × 3 种子，通宵量级）
python src/run_seq.py --configs all --seeds 0 1 2 --resume
python src/run_seq.py --summary          # -> results/bio/seq/summary.csv
python tmp/bio/p9_table.py               # 9 期聚合表（含通道/重叠/维度诊断）
# 静态轴 / 表格数据
python src/run_mnist.py --configs all --seeds 0 1 2 ; python src/run_bio.py --summary
```

关键配置名：`g9a_w512s5_cad2_br64`（v9 最优）、`g9a_w512s5_br8`（最省样本）、`g10_nov15_w512s5` /
`static_g10_nov15_w512s5`（相对新奇门控）、`g10_k200_pb16_n02`（K=200 方案）、`cif_refr_br16`（CIFAR）、
`g9a_w512s5_pburst16_t64`（压力触发脉冲串）、`g8_w512s5_loc_refr`（8B 基线）、
`ctx_nrem_rand_1000`（夜间参照）。全部逐配置 checkpoint 在 `results/bio/seq/parts/`。
