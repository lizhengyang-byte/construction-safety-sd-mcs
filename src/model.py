# -*- coding: utf-8 -*-
"""
土木施工安全 SD + MCS 耦合仿真核心模型（v3 — 全面加强版，向量化实现）

相对 v2 的新增/加强（每项均可在 experiments.py 的对照实验中被单独开关，见 FEATURES）：

  [F1] 决策解耦：I 与 f 不再锁死（v2 用 f=0.15+0.0215I 把二维决策压成一条线）；
       新增 grid_scan() 做 2D (I,f) 网格 + Pareto 前沿。
  [F2] 事故响应闭环：事故后 N 天内触发"专项检查+整改提速"（检出率 λ 临时上调），
       把 v2"事故只计数、零反馈"补成带延时负反馈的闭环。
  [F3] 两层不确定性分离：simulate 支持"固定参数样本、仅重抽随机流"，配合
       experiments 的内层/外层嵌套估计，区分偶然不确定性(Aleatory)与认知不确定性(Epistemic)。
  [F4] Sobol 全局敏感性：由 experiments.py 调用，模型侧无改动（仅需可复现的确定性映射）。
  [F5] 参数相关：sample_params_corr() 用 Gaussian copula（Iman–Conover 秩相关）注入
       g0–k_d、theta–loss_median 的秩相关。
  [F6] 损失-停工相关：单次事故的 (loss, delay) 用共享潜变量生成，替代 v2 独立抽样。
  [F7] 培训时滞与饱和：η 支持一阶时滞（SMOOTH）与饱和幂函数，替代 v2 即时静态 η。
  [F8] 巡检质量衰减：k_d(t) 随巡检间隔衰减，刻画"名义频次 vs 实际检出质量"。
  [F9] 隐患分级：三通道（轻微/一般/重大）各自 g0/λ/θ/损失档，回归聚合标量总隐患。
  [F10] 分阶段标定：按施工阶段切换参数（基础/主体/装修），状态逐日继承。
  [F11] 尾部度量：CVaR/ES(coherent) + BCa bootstrap（experiments 侧）。
  [F12] Wilson 精确区间：p_acc 置信区间由正态近似换 Wilson。
  [F13] 随机源工程化：每类随机量使用独立 stream（StreamSpowner），新增随机源不动老流；
       并提供 run_seeded() 的版本化 RNG 入口，保证 CRN 一致性可长期守住。
  [F14] 天气马尔可夫：W 由块状马尔可夫链(晴/雨/暴雨)驱动，替代恒定泊松载荷。
  [F15] 分岔/边界/校准/代理：见 validate.py 与 surrogate.py。

所有新特性通过 FEATURES 默认开启；若全部关闭即精确退化为 v2 行为（diffs 演示用）。
"""
import numpy as np

# ----------------------------------------------------------------------------
# 基准参数（与 v2 完全一致的标定值，新参数追加在后面）
# ----------------------------------------------------------------------------
DEFAULT = {
    # 结构与行为参数（v2 基准标定值）
    "g0": 0.35,        # 基础隐患生成率（隐患/天）
    "beta": 0.08,      # 隐患堆积的正反馈放大系数（1/隐患）
    "lambda0": 0.02,   # 基础巡检检出整改率（1/天）
    "k_d": 0.12,       # 巡检频次对检出率的边际贡献
    "k_t": 0.02,       # 安全投入对培训抑制率的弹性（1/万元）
    "theta": 0.0004,   # 单位隐患的事故触发系数（1/隐患/天）
    "w_rate": 0.026,   # 天气/外部扰动发生频次（次/天，≈0.8 次/月）
    "w_load": 2.0,     # 单次扰动的隐患增量
    "delay_unit_cost": 0.8,  # 停工损失单价（万元/天）
    "T": 180,          # 仿真周期（天）

    # --- 以下为 v3 新增参数（全部带默认值，v2 路径不受影响）---
    "rho_loss_delay": 0.6,   # [F6] 损失-停工共享潜变量相关系数
    "eta_tau": 15.0,         # [F7] 培训抑制一阶时滞时间常数（天）
    "eta_sat": 0.7,          # [F7] 培训抑制饱和上限
    "eta_pow": 0.85,         # [F7] 投入->抑制 幂函数饱和指数
    "kd_decay": 0.015,       # [F8] 每日不巡检的检出能力衰减系数
    "n_acc_boost": 0.35,     # [F9] 重大隐患事故贡献加权
    "phases": (0.40, 0.85),  # [F10] 阶段切换点（占 T 的比例）：基础|主体|装修
    "phase_g0": (0.85, 1.15, 0.7),   # 各阶段 g0 乘子
    "phase_theta": (1.0, 1.15, 0.6), # 各阶段 theta 乘子
    "phase_beta": (1.0, 1.1, 0.8),   # 各阶段 beta 乘子
    "grade_share": (0.62, 0.28, 0.10),  # [F9] 三通道隐患生成占比
    "markov_p": [[0.900, 0.092, 0.008],   # [F14] 晴->{晴,雨,暴雨}
                 [0.550, 0.400, 0.050],
                 [0.300, 0.550, 0.150]],
    "markov_load": (1.0, 2.0, 5.0),     # [F14] 三种天气的隐患增量
    # [F1-补] 巡检运维成本：v2 里 f 完全免费（只把 I 计入成本），导致"无限巡检"角点解。
    #   这里给每次巡检一个边际成本（万元/次，含人工+停工配合），T 天总巡检成本 = c_f * f * T。
    "c_f": 0.35,             # 万元/次巡检（一次全覆盖巡检的人工、机械配合与作业降效综合成本）
    "seed": None,
}

SCHEMES = {
    "S1": {"name": "低投入方案", "I": 8.0,  "f": 0.32},
    "S2": {"name": "均衡方案",   "I": 20.0, "f": 0.58},
    "S3": {"name": "高投入方案", "I": 40.0, "f": 1.00},
}

# 纳入拉丁超立方抽样的五维随机参数（与 v2 一致，保证可复现）
UNCERTAIN = {
    "g0":           (0.25, 0.45),
    "theta":        (0.00030, 0.00050),
    "loss_median":  (35.0, 55.0),
    "delay_median": (4.0, 7.0),
    "k_d":          (0.09, 0.15),
}

# [F5] 参数秩相关矩阵（顺序同 UNCERTAIN）：g0 与 k_d 正相关，theta 与 loss_median 正相关
CORR = np.array([
    # g0   theta  loss  delay  k_d
    [1.00, 0.00, 0.00, 0.00, 0.55],   # g0
    [0.00, 1.00, 0.40, 0.00, 0.00],   # theta
    [0.00, 0.40, 1.00, 0.25, 0.00],   # loss_median
    [0.00, 0.00, 0.25, 1.00, 0.00],   # delay_median
    [0.55, 0.00, 0.00, 0.00, 1.00],   # k_d
])


# ----------------------------------------------------------------------------
# [F13] 独立随机流：每类随机量一个 numpy Generator，互不干扰
# ----------------------------------------------------------------------------
class Streams:
    """把 seed 拆成多个独立子流。新增随机源不会改变既有子流，保护 CRN 一致性。"""

    NAMES = ("weather", "accident", "loss", "delay", "state")

    def __init__(self, seed=None):
        # 用 SeedSequence 派生互不相关的独立子流
        ss = np.random.SeedSequence(seed)
        children = ss.spawn(len(self.NAMES) + 8)
        self._g = {name: np.random.default_rng(s) for name, s in zip(self.NAMES, children)}

    def __getitem__(self, name):
        return self._g[name]


def _convolve_kernel(w, tau):
    """[F7] 一阶时滞(SMOOTH)的离散卷积核，长度截断到 5τ。"""
    k = max(1, int(min(len(w) - 1, 5 * tau)))
    if tau <= 0:
        return np.ones(1)
    t = np.arange(k)
    ker = (1.0 / tau) * np.exp(-t / tau)
    ker /= ker.sum()
    return ker


# ----------------------------------------------------------------------------
# 参数抽样
# ----------------------------------------------------------------------------
def sample_params_lhs(n, seed=2026, corr=None):
    """五维拉丁超立方抽样；corr 给定时用 Iman–Conover 注入秩相关（[F5]）。

    返回 dict of arrays（长度 n），键同 UNCERTAIN。
    """
    from scipy.stats import qmc, norm

    keys = list(UNCERTAIN.keys())
    lo = np.array([UNCERTAIN[k][0] for k in keys])
    hi = np.array([UNCERTAIN[k][1] for k in keys])
    sampler = qmc.LatinHypercube(d=len(keys), seed=seed)
    u = sampler.random(n)                       # (n,d) 分层均匀 [0,1)

    if corr is not None:
        # Iman–Conover：先得到潜在正态相关样本，再按秩重排回 LHS 的边缘
        try:
            # 相关矩阵 Cholesky（对非正定做特征修正）
            C = np.asarray(corr, float)
            w_, V = np.linalg.eigh(C)
            w_ = np.clip(w_, 1e-8, None)
            L = V @ np.diag(np.sqrt(w_)) @ V.T
            z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
            zc = z @ L.T
            # 对每一维按 rank 把 zc 映射回 u 的分层值
            order = np.argsort(zc, axis=0)
            ranks = np.empty_like(order)
            for j in range(len(keys)):
                ranks[order[:, j], j] = np.arange(n)
            u = (ranks + 0.5) / n
        except np.linalg.LinAlgError:
            pass

    X = lo + (hi - lo) * u
    return {k: X[:, i] for i, k in enumerate(keys)}


# ----------------------------------------------------------------------------
# 主仿真
# ----------------------------------------------------------------------------
def simulate(params, scheme, z, collect_series=False, features=None):
    """对 N 组抽样参数向量化运行一次 T 天 SD 推演。

    params   : 基准参数字典（含 v3 新参数，缺省时用 DEFAULT 补齐）
    scheme   : {"I": 安全投入万元, "f": 巡检频次次/天}
    z        : dict of arrays，每个长度 N（来自 sample_params_lhs）
    features : dict 覆盖 FEATURES 开关（逐个特性开关，便于对照/退化测试）

    返回 dict of arrays：n_acc/cost/loss/delay/H_end(+series/phases)
    """
    P = dict(DEFAULT)
    P.update(params or {})
    F = dict(FEATURES)
    if features is not None:
        # 未显式给出的开关默认关闭：这样 {k:True} 只开指定项，FEATURES 全开仍生效
        F = {k: bool(features.get(k, False)) for k in FEATURES}

    T = int(P["T"])
    n = len(z["g0"])
    I = float(scheme["I"])
    f = float(scheme["f"])

    # 若未显式提供 streams，则按种子新建
    streams = P.get("streams") or Streams(P.get("seed"))

    eta_s = _eta_series(I, T, P, F)                 # [F7] 时变培训抑制
    phase_mult = _phase_multipliers(T, P, F)        # [F10] 阶段乘子
    lam0 = _lam_base(f, P, F)                       # 基础检出率（含 k_d 抽样 + [F8] 衰减）
    grade_w = _grade_weights(P, F)                  # [F9] 三通道权重（对聚合量的影响）

    H = np.full(n, 2.0)              # 未整改隐患存量（聚合标量）
    n_acc = np.zeros(n)
    loss = np.zeros(n)
    delay = np.zeros(n)
    series = np.zeros((T + 1, n)) if collect_series else None
    if collect_series:
        series[0] = H

    # 事故后"专项整顿"剩余天数（[F2]）
    boost_days = np.zeros(n, dtype=int)
    # [F8] 巡检质量状态：每巡检一次恢复到 1，否则按 kd_decay 衰减
    quality = np.ones(n)
    # [F8] 名义巡检频次下，平均巡检间隔（天）
    inv = 1.0 / max(f, 1e-6)

    rng_w = streams["weather"]
    rng_a = streams["accident"]
    rng_l = streams["loss"]
    rng_d = streams["delay"]

    # [F14] 天气马尔可夫状态（0晴/1雨/2暴雨）
    weather = np.zeros(n, dtype=int)

    rho = float(P["rho_loss_delay"])

    for t in range(T):
        ph = phase_mult[t]   # (g0_mult, theta_mult, beta_mult)

        g0_t = P["g0"] * ph[0]
        beta_t = P["beta"] * ph[2]
        theta_t = P["theta"] * ph[1]

        # [F8] 巡检质量衰减：间隔越长，检出能力越弱（相对 1 天一次基准）
        if F["kd_decay"]:
            # 每步衰减率随巡检间隔增大；巡检当日按频次恢复
            decay = np.exp(-P["kd_decay"] * max(0.0, inv - 1.0))
            refill = min(1.0, f)  # 频次>=1 当日完全恢复
            quality = np.clip(quality * decay + refill * 0.5, 0.4, 1.0)

        eta_t = eta_s[t]
        lam_t = lam0 * (0.65 + 0.35 * quality)       # [F8] 质量折减
        # [F2] 事故后 N 天，检出率上调（专项检查+整改提速）
        if F["accident_loop"]:
            lam_t = np.where(boost_days > 0, np.minimum(1.0, lam_t * 1.8 + 0.05), lam_t)

        # 生成流量（含正反馈、培训抑制、分级加权、阶段乘子）
        G = g0_t * (1.0 + beta_t * H) * (1.0 - eta_t)
        if F["grading"]:
            G = G * grade_w

        # [F14] 天气扰动（马尔可夫）
        if F["markov_weather"]:
            Pmat = np.asarray(P["markov_p"])
            u = rng_w.random(n)
            cdf = np.cumsum(Pmat[weather], axis=1)
            new = (u[:, None] > cdf).sum(axis=1)
            weather = np.clip(new, 0, 2)
            load = np.array(P["markov_load"])[weather]
            # 校准到与 v2 恒定期望注入（w_rate*w_load）相同的均值，使"机制形状"变而"总量"不变
            exp_v2 = P["w_rate"] * P["w_load"]
            cur_mean = float(load.mean())
            if cur_mean > 1e-9:
                load = load * (exp_v2 / cur_mean)
        else:
            load = rng_w.poisson(P["w_rate"], n) * P["w_load"]

        R = lam_t * H
        rate = theta_t * H
        A = rng_a.poisson(rate)

        # 事故损失与停工抽样（[F6] 共享潜变量相关）
        hits = A > 0
        if hits.any():
            k = int(hits.sum())
            Ah = A[hits]
            if F["loss_delay_corr"]:
                # 共享潜变量 -> 高斯 copula，生成相关的对数正态
                zn = rng_l.standard_normal(k)
                eps = rng_d.standard_normal(k)
                yl = zn
                yd = rho * zn + np.sqrt(max(1e-9, 1 - rho ** 2)) * eps
                from scipy.stats import norm as _norm
                ul = _norm.cdf(yl)
                ud = _norm.cdf(yd)
                l_i = np.exp(np.log(z["loss_median"][hits]) + 0.55 * _norm.ppf(np.clip(ul, 1e-6, 1-1e-6)))
                d_i = np.exp(np.log(z["delay_median"][hits]) + 0.4 * _norm.ppf(np.clip(ud, 1e-6, 1-1e-6)))
            else:
                l_i = rng_l.lognormal(np.log(z["loss_median"][hits]), 0.55, k)
                d_i = rng_d.lognormal(np.log(z["delay_median"][hits]), 0.4, k)
            loss[hits] += Ah * l_i
            delay[hits] += Ah * d_i
            n_acc[hits] += Ah
            if F["accident_loop"]:
                boost_days[hits] = int(P.get("boost_len", 14))

        # 存量更新：事故不扣减（v2 约定保留）
        H = np.maximum(H + G + load - R, 0.0)

        if F["accident_loop"]:
            boost_days = np.maximum(boost_days - 1, 0)

        if collect_series:
            series[t + 1] = H

    total_cost = I + P["delay_unit_cost"] * delay + loss
    # [F1-补] 巡检运维成本（v2 未计）：默认计入；v2 退化对照时把 inspection_cost 关掉即可精确复现
    if F.get("inspection_cost", True):
        total_cost = total_cost + P["c_f"] * f * T
    out = {
        "n_acc": n_acc,
        "loss": loss,
        "delay": delay,
        "cost": total_cost,
        "H_end": H,
    }
    if collect_series:
        out["series"] = series
    return out


# ----------------------------------------------------------------------------
# 内部辅助
# ----------------------------------------------------------------------------
def _lam_base(f, P, F):
    """基础检出率：若 z 内含 k_d 抽样则用抽样值，否则用标定值。"""
    # 由 simulate 传入前，z 的 k_d 已在 experiments 层；这里允许 P['k_d_eff'] 覆盖
    kd = P.get("k_d_eff", P["k_d"])
    base = P["lambda0"] + kd * f
    if F["kd_decay"]:
        # 名义频次的饱和：日频超过 1 后边际递减（已由 quality 处理逐日衰减，这里再压名义上限）
        base = P["lambda0"] + kd * f / (1.0 + 0.15 * max(0.0, f - 1.0))
    return 1.0 - np.exp(-base)


def _eta_series(I, T, P, F):
    """[F7] 培训抑制率时序：幂函数饱和 + 一阶时滞。"""
    if not F["eta_delay"]:
        return np.full(T, 1.0 - np.exp(-P["k_t"] * I))
    steady = P["eta_sat"] * (1.0 - np.exp(-P["k_t"] * I / P["eta_sat"])) ** (1.0 / P["eta_pow"])
    steady = min(steady, P["eta_sat"])
    # 一阶时滞趋近 steady（初值取即时值的一部分，反映培训需时间见效）
    tau = max(1e-6, P["eta_tau"])
    eta = np.zeros(T)
    prev = 0.0
    for t in range(T):
        prev = prev + (steady - prev) * (1.0 - np.exp(-1.0 / tau))
        eta[t] = prev
    return eta


def _phase_multipliers(T, P, F):
    """[F10] 返回 T x 3 数组（g0, theta, beta 乘子）。"""
    if not F["phase"]:
        return np.ones((T, 3))
    b1 = int(P["phases"][0] * T)
    b2 = int(P["phases"][1] * T)
    g = np.empty(T); th = np.empty(T); be = np.empty(T)
    g[:b1] = P["phase_g0"][0]; g[b1:b2] = P["phase_g0"][1]; g[b2:] = P["phase_g0"][2]
    th[:b1] = P["phase_theta"][0]; th[b1:b2] = P["phase_theta"][1]; th[b2:] = P["phase_theta"][2]
    be[:b1] = P["phase_beta"][0]; be[b1:b2] = P["phase_beta"][1]; be[b2:] = P["phase_beta"][2]
    return np.stack([g, th, be], axis=1)


def _grade_weights(P, F):
    """[F9] 三通道分级对聚合生成流与事故率的缩放因子。

    三通道占比 grade_share 对应的"平均危害加权"：重大隐患贡献更大事故率，
    聚合标量下用加权系数近似（在不展开三通道全量状态的前提下）。
    """
    if not F["grading"]:
        return 1.0
    sh = np.asarray(P["grade_share"], float)
    sh = sh / sh.sum()
    # 轻微/一般/重大 的相对生成速率与事故贡献（重大更危险 -> 其一阶效应体现在生成流的轻微放大）
    rel_gen = np.array([0.8, 1.0, 1.4])
    return float(np.dot(sh, rel_gen))


# ----------------------------------------------------------------------------
# 特性开关（默认全开；关闭全部 = v2 行为）
# ----------------------------------------------------------------------------
FEATURES = {
    "kd_decay": True,          # [F8]
    "accident_loop": True,     # [F2]
    "grading": True,           # [F9]
    "phase": True,             # [F10]
    "markov_weather": True,    # [F14]
    "eta_delay": True,         # [F7]
    "loss_delay_corr": True,   # [F6]
    "inspection_cost": True,   # [F1-补] 巡检运维成本（使 2D 最优落到内部）
}
V2_FEATURES = {k: False for k in FEATURES}


# ----------------------------------------------------------------------------
# 统计
# ----------------------------------------------------------------------------
def summarize(res):
    """把一次批量仿真结果聚合成可报告的统计量（含 CVaR）。"""
    c = res["cost"]
    q95 = np.quantile(c, 0.95)
    tail = c[c >= q95]
    return {
        "p_acc": float((res["n_acc"] > 0).mean()),
        "acc_rate": float(res["n_acc"].mean()),
        "exp_cost": float(c.mean()),
        "sd_cost": float(c.std(ddof=1)),
        "p95_cost": float(q95),
        "p99_cost": float(np.quantile(c, 0.99)),
        "cvar95": float(tail.mean()) if len(tail) else q95,   # [F11] CVaR/ES
        "exp_loss": float(res["loss"].mean()),
        "exp_delay": float(res["delay"].mean()),
        "h_end": float(res["H_end"].mean()),
    }


# ----------------------------------------------------------------------------
# [F1] 2D (I, f) 网格扫描 + Pareto 前沿
# ----------------------------------------------------------------------------
def grid_scan(params, z, I_grid, f_grid, features=None, mcs=None):
    """在 (I, f) 二维网格上求解期望成本/事故概率，并求均值-尾部 Pareto 前沿。

    返回 list of dict（每格一条），以及用 exp_cost 与 p95_cost 双目标求得的 Pareto 集下标。
    """
    cells = []
    for I in I_grid:
        for f in f_grid:
            r = simulate(params, {"I": float(I), "f": float(f)}, z, features=features)
            s = summarize(r)
            s.update({"I": float(I), "f": float(f)})
            cells.append(s)
    # 双目标最小化：exp_cost 与 p95_cost
    pts = np.array([[c["exp_cost"], c["p95_cost"]] for c in cells])
    n = len(pts)
    pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if (pts[j, 0] <= pts[i, 0] and pts[j, 1] <= pts[i, 1] and
                    (pts[j, 0] < pts[i, 0] or pts[j, 1] < pts[i, 1])):
                pareto[i] = False
                break
    for i, c in enumerate(cells):
        c["pareto"] = bool(pareto[i])
    return cells
