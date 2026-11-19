# 系统动力学–蒙特卡洛–博弈论耦合的土木工程施工安全风险量化与决策优化研究

**——成本、进度、安全三重约束下的仿真、鲁棒决策与多方博弈**

> 系统动力学建模 · 五维拉丁超立方抽样 · 共同随机数 · 分层/重要性采样 · Sobol 全局敏感性 · CVaR 与鲁棒优化 · Stackelberg 博弈 · 公共品博弈 · RBF 代理与主动学习
>
> 本文为论文 Markdown 版，全部数值由本仓库 `src/` 下脚本复现，随机种子固定。配套报告见 [`docs/improvement_report.md`](improvement_report.md)（v2→v3 二十项机制增强）与 [`docs/game_or_report.md`](game_or_report.md)（v3.1 博弈/运筹扩展）。

---

## 摘要

施工安全管理的决策难点不在"是否重视安全"，而在"给定一笔投入和一套巡检安排，一个施工周期内事故概率、成本与工期损失呈何种分布；在多方利益主体下，激励如何设计才能让安全投入真正到位"。本文构建**系统动力学（SD）–蒙特卡洛（MCS）–博弈论/运筹学**三层耦合的风险量化与决策优化框架。

**第一层（仿真内核）**：以"未整改隐患存量"为核心存量，刻画隐患生成的正反馈放大、巡检整改的负反馈调节、安全投入对培训抑制率的指数衰减、隐患向事故的泊松转化；事故只计数、不扣减存量，检出率指数封顶以保证物理合理性。在 v2 基础上完成 **20 项机制增强**——I/f 解耦、事故后风险反馈、两层不确定性分解、参数秩相关抽样、损失–停工相关 copula、巡检质量衰减、巡检成本入账、三阶段与马尔可夫天气、CVaR、独立随机流、分级移交、RBF 代理模型等——并保留 `FEATURES` 开关字典使全部改进**可逐位回退为 v2、可逐项归因**。

**第二层（敏感性、校准与不确定性）**：用五维拉丁超立方 + 共同随机数（CRN）抽样，Sobol 全局敏感性揭示隐患生成率 `g0`（总效应 0.385）为第一主导因子，纠正了单因素分析对巡检参数的过度归因；两层不确定性分解显示偶然不确定性占 96.9%，说明事故成本本质由"运气"主导；PIT 尾部覆盖率 80/90/95% 档为 80.2/90.1/95.0%，校准近乎完美。

**第三层（决策优化与博弈）**：(1) 运筹决策层将 (投入 I, 巡检频次 f) 作为决策变量，求解期望最优、CVaR 最优与鲁棒 min-max 三类随机规划，量化"风险厌恶定价"；(2) 监管–承包商 Stackelberg 博弈给出**威慑门槛**（事故处罚/赔付基数须达约 200 万元量级，承包商才自愿诚实投入）与均衡检查强度 e\*=0（重罚到位后额外监查多余）；(3) 多承包商公共品博弈揭示**公地悲剧**——N>1 时 Nash 投入恒低于社会最优达 50%，需总包统一管理或联保连坐内部化外部性；(4) 蒙特卡洛引擎本身经分层抽样、重要性采样与主动学习优化，在稀有事故尾部估计上以约 1/5 的样本达到同等精度（并诚实记录按输入参数分层反而失效的教训）。

**主要结论**：期望最优与尾部最优分离（v3 会更尖锐）；安全治理应先立重罚、再谈检查；公共安全投入不足是结构性而非个体道德问题；MC 方法选择须匹配"要估什么、按什么分层"。三套方案（投入 8/20/40 万元）在 v3 全特性下事故概率 48.6%/26.4%/14.0%、期望总成本 63.0/73.8/111.7 万元、CVaR95 204.6/190.0/202.4 万元。

**关键词：** 施工安全管理；系统动力学；蒙特卡洛模拟；风险量化；尾部风险；Sobol 全局敏感性；CVaR；鲁棒优化；Stackelberg 博弈；公共品博弈；主动学习

## Abstract

Construction safety decisions hinge less on *whether* to invest than on *how much* investment, under *what* inspection frequency, yields what distribution of accident probability, total cost and schedule loss within a construction cycle — and, in a multi-agent setting, on *how incentives should be designed* so that safety investment is actually delivered. This paper builds a three-layer coupled framework: system dynamics (SD) for the hazard-stock dynamics, Monte Carlo simulation (MCS) for distributional risk quantification, and game theory / operations research for decision optimization.

Layer 1 enhances a baseline SD–MCS model with **twenty mechanistic upgrades** — decoupling of investment and inspection frequency, accident-triggered risk feedback, two-level uncertainty decomposition, rank-correlated parameter sampling, loss–downtime copula, inspection-quality decay, inspection-cost accounting, tri-phase and Markov weather, CVaR, independent random streams, graded hand-over and an RBF surrogate — all switchable via a `FEATURES` dictionary so each is attributable and the whole degrades exactly to the baseline. Layer 2 applies Latin hypercube sampling with common random numbers; Sobol global sensitivity identifies the hazard-generation rate `g0` (total effect 0.385) as the dominant factor, correcting single-factor over-attribution; aleatory uncertainty accounts for 96.9% of cost variance; PIT tail coverage is near-perfect (80.2/90.1/95.0%). Layer 3 adds (i) operations-research optimization of (I, f) under expected-cost, CVaR and robust min-max criteria; (ii) a regulator–contractor Stackelberg game yielding a deterrence threshold (penalty base ≳ 2.0M to make honesty rational) and an equilibrium inspection intensity e\*=0; (iii) a multi-contractor public-good game exposing a 50% commons under-provision; and (iv) variance reduction and active learning that reach equal tail accuracy with ~1/5 the samples. The mean-optimal and tail-optimal schemes diverge; governance should price penalties first and inspect later; public-safety under-investment is structural; and MC method choice must match what is estimated and how strata are formed.

**Key words:** construction safety management; system dynamics; Monte Carlo simulation; tail risk; Sobol sensitivity; CVaR; robust optimization; Stackelberg game; public-good game; active learning

**四项核心结论**

- **动态反馈决定风险水平，源头优于巡检**：三方案稳态隐患量为 7.2/3.1/1.5 个；Sobol 总效应显示 `g0`（0.385）> `θ`（0.313）> `k_d`（0.282），管理应转向"降低隐患生成"而非一味加密巡检。
- **均值与尾部不一致且更尖锐**：巡检成本入账后，期望成本最低仍非高投入，但 P95/CVaR 上高投入才压尾部——买确定性而非期望收益。
- **威慑有门槛、公地有悲剧**：处罚基数跨过约 200 万元门槛承包商才自愿诚实；重罚到位后检查多余（e\*=0）；5 家共用公共安全时 Nash 投入仅为社会最优的 50%。
- **MC 引擎可优化但需对症**：重要性采样以 2000 样本逼近 5 万样本稀有概率（ESS=576），主动学习 40 点达均匀 200 点精度；而按输入参数机械分层反而变差——分层变量须对准目标量。

---

## 1 引言

### 1.1 施工安全风险需要在项目前期给出可决策的量化答案

土木工程施工是典型的高风险作业系统：高处坠落、物体打击、基坑失稳、机械伤害等事故类型高度集中，诱因分散在人的行为、物的状态与环境扰动之中。行业现行安全检查依据《建筑施工安全检查标准》（JGJ 59-2011）以现场评分评价安全水平，风险评估多沿用作业条件危险性评价法（LEC）给出单点分值。这类方法能回答"这个作业危险不危险"，却难以回答项目管理真正需要的三个问题：给定一笔投入和一套巡检安排，一个周期内至少发生一次事故的概率是多少？总成本最坏情况有多差？事故停工对进度冲击多大？

在多方参与的建设工程中，还有第四个问题：总包、分包、监管者利益并不一致，安全投入是"公共品"还是"私人物品"？谁在为谁的懒惰买单？**本文把这三个量化问题与一个博弈问题统一到一个可标定的框架中。**

### 1.2 既有研究的三条技术路线与断层

（继承 v2 论文）既有研究形成三条技术路线：以事故统计与贝叶斯网络为主的风险评估、以系统动力学为主的系统行为建模、以蒙特卡洛为主的概率推演。三者的共同断层是：**彼此少有耦合，且几乎不进入优化与激励设计层面**。本文的工作是把三者串成"仿真—敏感性/校准—决策/博弈"的完整链条。

### 1.3 本文贡献

1. **机制层**：20 项可逐位回退的机制增强（详见 §3），并给出逐项数值归因。
2. **方法层**：Sobol 全局敏感性纠偏、两层不确定性分解、PIT 校准、分岔分析、RBF 代理加速（~9000×）与主动学习采样。
3. **决策层**：将成本模型嵌入期望/CVaR/鲁棒三类随机规划，量化风险厌恶定价。
4. **博弈层**：监管–承包商 Stackelberg 威慑门槛、多承包商公地悲剧、MC 引擎方差缩减。

---

## 2 问题设定与研究设计

（继承 v2 论文）以某 180 天主体结构施工算例为对象，标定隐患生成率、事故触发、损失/停工/进度等参数（示例标定值，见 `data/params.csv`）。三套管理方案 S1/S2/S3 对应投入 8/20/40 万元、巡检频次 0.32/0.58/1.00 次/天。研究设计分四步：建模与仿真内核 → 敏感性与校准诊断 → 决策优化 → 博弈分析。

---

## 3 仿真内核：系统动力学–蒙特卡洛耦合模型（v3，20 项增强）

### 3.1 基础存量流量结构

```
H(t+1) = H(t) + G(t) + W(t) - R(t)
G(t)   = g0 * (1 + β*H(t)) * (1 - η)            # 隐患生成（含正反馈放大）
η      = 1 - exp(-k_t * I)                       # 安全投入的培训抑制率
R(t)   = λ * H(t),  λ = 1 - exp(-(λ0 + k_d*f))   # 巡检整改，检出率封顶
A(t)   ~ Poisson(θ * H(t))                       # 隐患→事故；只计数
C      = I + c_f*f*T + Σ L_i + c_d * Σ D_i       # 总成本（v3 额外计入巡检成本 c_f·f·T）
```

两处建模约定：事故不消除隐患（否则系统性低估存量、美化高风险方案）；检出率指数封顶（保证不越物理上限）。

### 3.2 v3 的 20 项机制增强

所有增强由 `FEATURES` 开关字典控制，`V2_FEATURES` 全关时精确退化为 v2。

1. **I 与 f 解耦**：投入与巡检频次成为独立决策变量（v2 中锁死成一条线）。
2. **事故后风险反馈**：事故触发 14 天 λ×1.8 的整改强化窗口（v2 只计数零反馈）。
3. **两层不确定性分解**：偶然（样本内）与认知（参数间）方差分离。
4. **参数秩相关抽样**：Iman–Conover 高斯 copula，g0–k_d ρ=0.55、θ–loss ρ=0.40、loss–delay ρ=0.25。
5. **损失–停工时滞相关**：共享潜变量 copula（ρ=0.6）。
6. **巡检质量衰减**：长期巡检边际效用递减（k_d decay）。
7. **巡检成本入账**：`c_f=0.35` 万元/次，堵住 v2"巡检免费→角点解"的漏洞。
8. **三阶段乘子**：基础/主体/装修阶段风险不同。
9. **马尔可夫天气**：三态转移，均值校准到 v2 水平。
10. **投入时滞**：η 幂饱和 + 一阶时滞。
11. **CVaR95**：尾部风险度量。
12. **独立随机流**：weather/accident/loss/delay/state 各自 SeedSequence 派生（保护 CRN）。
13. **分级移交**：隐患按等级通道处理。
14. **Wilson 区间**：事故概率的可靠性区间（Clopper–Pearson 近似）。
15. **BCa bootstrap**：偏校正加速置信区间。
16. **校准诊断**：可靠性曲线 + Brier + PIT。
17. **分岔分析**：隐患存量随投入的分岔与安全单稳态临界点。
18. **CPM 进度冲击**：关键路径总时差 TF=6 天，关键占比 0.45。
19. **全成本口径**：赶工费、处罚、保费上浮、信誉损失。
20. **RBF 代理模型**：纯 numpy 高斯核，训练 30 点/密集 165 点，加速约 9000 倍。

（逐项 before/after 对照见 [`docs/improvement_report.md`](improvement_report.md)。）

---

## 4 敏感性、校准与不确定性诊断

### 4.1 Sobol 全局敏感性纠偏

单因素（OAT）曾显示巡检边际效应 `k_d` 最敏感。Sobol 总效应（SALib，512 基样本）揭示真实排序：

| 参数 | g0 | θ | k_d | β | λ0 | w_rate |
|---|---|---|---|---|---|---|
| 总效应 ST | **0.385** | 0.313 | 0.282 | 0.039 | 0.028 | 0.011 |

**隐患生成率 `g0` 才是第一主导因子。** 管理启示从"多巡检"转向"源头降低隐患生成"（如规范作业、源头防护）。OAT 与 Sobol 的差异见对比图 `cmp2_sobol.svg`。

### 4.2 两层不确定性分解

偶然方差 1219 vs 认知方差 39，**偶然份额 96.9%**。含义：事故成本方差绝大部分由施工过程中的"运气"（偶然事件）决定，参数标定再准也只能压掉约 3% 方差。这为"风险准备金"而非"精确点估计"的决策方式提供了依据。

### 4.3 校准与分岔

- **PIT 尾部覆盖率**：80% 档 80.2%，90% 档 90.1%，95% 档 95.0%，近乎完美校准（`fig8_calibration.svg`）。
- **Brier**：v2=0.188、v3=0.196（同量级）。
- **分岔**：标定下安全单稳态临界投入 I\*≈1.0 万元，与论文一致（`fig9_bifurcation.svg`）。

### 4.4 代理模型加速

RBF 代理训练 30 点密集 165 点：期望成本 R²=0.992（MAPE 1.3%）、事故概率 R²=0.987、P95 R²=0.68；耗时 2.06 s → 0.0014 s，**加速约 9000 倍**（`fig10_surrogate.svg`）。

---

## 5 运筹决策层：从"评价方案"到"最优化方案"

### 5.1 三类随机规划

把 (I, f) 作为决策变量，预算 180 万元：

1. **期望最优（风险中性）**：$\min_{I,f}\ \mathbb{E}[C]$
2. **CVaR 最优（风险厌恶）**：$\min_{I,f}\ \mathrm{CVaR}_{95\%}[C]$
3. **鲁棒 min-max（深度不确定）**：$\min_{I,f}\max_{\theta\in\Theta}\mathrm{CVaR}$，盒式不确定集（g0/θ ±25%）

### 5.2 结果

| 决策准则 | 最优 I（万元） | 最优 f（次/天） | E[C] | CVaR95 | 超预算概率 |
|---|---|---|---|---|---|
| 期望最优 | 6 | 0.40 | 61.1 | 202.2 | 3.17% |
| CVaR 最优 | 6 | 0.80 | 72.8 | **183.2** | 2.00% |
| 鲁棒 min-max | 8 | 0.60 | — | worst 184.7 | — |

**要点**：

- **风险厌恶定价可量化**：多花 11.7 万元期望，把 CVaR 压 19.0 万元、超预算概率从 3.17% 降到 2.00%。
- **两最优解 I 都停在 6 万下界**：因 `p_acc(I)` 弹性弱，加大投入对均值/尾部边际改善追不上成本；风险厌恶者改为**提高巡检频次 f（0.40→0.80）**压尾部。这是关键替代动作。
- **鲁棒解保守**：参数看走眼时最坏 CVaR 仍有 184.7 万，是纯期望模型给不出的保证。
- 期望–尾部有效前沿与两类最优分离，见 `fig11_orfrontier.svg`。

---

## 6 监管–承包商博弈：威慑门槛与 Stackelberg 均衡

### 6.1 序贯博弈建模

监管者定处罚与检查强度，承包商观测后最优选择投入与"偷工比例" m。N=30000 大样本保证 `p_acc(I)` 单调。

- 承包商成本 = 投入成本 + `p_acc(I)×L_acc`（事故期望损失）+ 被查处罚 − 偷工节省；检出概率 `detect = min(0.9, 0.3e)×(0.5+0.5m)`。
- 监管者成本 = 检查成本 + 社会事故损失（外部性放大 1.6）。

### 6.2 威慑门槛（Part A）

扫描事故处罚/赔付基数 `L_acc`，模型给出**威慑门槛 L_acc\* ≈ 200 万元**：低于此承包商理性维持偷工；跨过门槛稳定自愿加码、放弃偷工。门槛偏高源于低投入区 `p_acc(I)` 弹性弱——多投入的安全收益不够显眼，必须重罚才够。

### 6.3 Stackelberg 均衡（Part B，L_acc=300）

均衡 **e\*=0**：处罚已远超门槛（300 > 200），承包商已因高额赔付而诚实（m=0、I=30），监管再花钱检查纯属浪费。自由市场自发达到社会最优，**效率损失 0 万元**。

**治理启示**：处罚与检查是可替代的两种威慑工具；**先立重罚（价格信号）、再谈检查（信息信号）**，过度监查在重罚到位后低效（`fig12_game.svg`）。

---

## 7 多承包商公地悲剧：公共安全的外部性

### 7.1 公共品博弈建模

N=5 家共用同一现场公共安全环境。每家投入既护己也护人（正外部性），私人只"感知"自己承担 `(N-1)/N × PHI` 份额外部性（PHI=外部性感知比例），社会计入完整 `L_public`。

### 7.2 结果

| | 单家最优投入 | 总投入 Σ | 缺口 |
|---|---|---|---|
| Nash 均衡 | I\*=6 | 30 万 | — |
| 社会最优 | I^S=12 | 60 万 | **30 万（50%）** |

PHI 从 0→1 时 Nash 投入 0→12、缺口 100%→0%，外部性完全感知时悲剧消失。

**要点**：公地悲剧是**结构性**的——N>1 且外部性未完全内化时，Nash 投入严格低于社会最优，稀释效应随 N 增大加剧。弥合须靠提高外部性感知（联保连坐、共享安全信用）或内部化（统一安全基金、总包代建），这解释了大型工地为何普遍总包统一安全管理（`fig13_commons.svg`）。

---

## 8 蒙特卡洛引擎升级：方差缩减与主动学习

施工事故是稀有事件（p≈1%–10%），尾部与稀有概率的普通 MC 估计方差大。三项技术（方案 S2，n=2000，rep=20）：

| 指标 | 普通 MC | 改进 | 说明 |
|---|---|---|---|
| 事故概率标准差 | 0.0096 | 分层 0.0100 | 反升 10.1% |
| CVaR95 标准差 | 6.146 | 分层 8.235 | 反升 79.5% |
| 稀有概率 | 0.2677（5 万样本） | IS **0.2585** | ESS=576 |
| 达 RMSE≈1.0 | ~200 点 | 主动学习 **40 点** | 效率 ~5× |

**诚实记录**：按输入参数 `g0` 分层对概率与 CVaR 都变差——分层维度与"是否出事/成本是否进尾部"弱相关，机械分层打乱尾部覆盖。**分层须对准目标量（后分层/按成本分层），或改用重要性采样/对偶变量。** 真正奏效的是重要性采样（2000 样本逼近 5 万样本稀有概率）与主动学习（省样本 ~5×）（`fig14_variance.svg`）。

---

## 9 综合结论与管理启示

1. **动态反馈决定风险水平，源头优于巡检**：Sobol 总效应 `g0`>θ>`k_d`；管理应从"加密巡检"转向"降低隐患生成"。
2. **均值与尾部分离（巡检成本入账后更尖锐）**：买确定性而非期望收益；风险厌恶者靠提高巡检频次压尾部。
3. **威慢有门槛、公地有悲剧**：先立重罚（≥200 万元量级）再谈检查；公共安全投入不足是结构性，需制度内部化外部性。
4. **运气主导方差，决策用区间而非点**：偶然不确定性占 96.9%，应以"目标投入区间 + 尾部风险准备金"替代单一投入水平。
5. **MC 方法须对症**：IS 与主动学习真正省样本，机械分层失效——方法匹配"估什么、怎么分层"。

---

## 10 局限与展望

- 参数为示例标定值（台账反演 + 文献区间），未经现场实测校准；用于方法演示与方案比选，不可直接作为具体工程决策依据。
- 聚合型模型（无空间分辨率、无个体异质性）。
- 待深化：贝叶斯在线更新参数、与 BIM 分区清单耦合、现场实测再标定、多阶段动态博弈（监管者学习更新）、主动学习与高保真仿真闭环。

---

## 附录：复现

```bash
pip install -r requirements.txt
python src/experiments.py   # 三方案 + 两层不确定性 + Sobol + 2D 网格 + 进度 + 成本扩展
python src/validate.py      # 校准 + 分岔
python src/surrogate.py     # RBF 代理模型
python src/or_opt.py        # 运筹决策层：期望/CVaR/鲁棒最优 (I,f)
python src/game.py          # 监管–承包商 Stackelberg 博弈 + 威慑门槛
python src/game_multi.py    # 多承包商公地悲剧
python src/mc_advanced.py   # MC 方差缩减 + 主动学习
python src/compare_figs.py  # v2-vs-v3 对比图
python src/render_svg.py    # 生成 docs/figures/*.svg
pytest -q                   # 14 项单元测试
```

数值输出：`output/results.json`、`comparison.json`、`validate.json`、`surrogate.json`、`or_opt.json`、`game.json`、`game_multi.json`、`mc_advanced.json`。图：`docs/figures/fig1`–`fig14` 及三张对比图 `cmp1`–`cmp3`。
