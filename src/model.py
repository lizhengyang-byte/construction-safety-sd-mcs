# -*- coding: utf-8 -*-
"""
土木施工安全 SD + MCS 耦合仿真核心模型（向量化实现）

存量-流量结构（单位：天）
    H_{t+1} = H_t + G_t + W_t - R_t
    G_t = g0 * (1 + beta * H_t) * (1 - eta)                     隐患生成（含正反馈放大）
    W_t ~ Poisson(w_rate) * w_load                              天气/外部扰动注入
    R_t = lambda * H_t,  lambda = 1 - exp(-(lambda0 + k_d * f))  巡检整改（检出率封顶为 1）
    A_t ~ Poisson(theta * H_t)                                  隐患转化为事故（只计数，不扣减存量）
    eta = 1 - exp(-k_t * I)                                     安全培训抑制率

成本
    C = I + sum(损失_i) + c_d * sum(停工天数_i)                 （万元）

两处建模约定（v2）：
1) 事故发生后现场隐患通常仍然存在，故 A_t 不参与存量更新，只做计数；
2) 检出率以指数式封顶，避免巡检频次过高时线性式给出大于 1 的无效解。
"""
import numpy as np

DEFAULT = {
    # 结构与行为参数（基准标定值）
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
}

SCHEMES = {
    "S1": {"name": "低投入方案", "I": 8.0,  "f": 0.32},
    "S2": {"name": "均衡方案",   "I": 20.0, "f": 0.58},
    "S3": {"name": "高投入方案", "I": 40.0, "f": 1.00},
}

# 蒙特卡洛随机参数（拉丁超立方抽样的上下界）
UNCERTAIN = {
    "g0":                (0.25, 0.45),
    "theta":             (0.00030, 0.00050),
    "loss_median":       (35.0, 55.0),    # 单次事故损失中位数（万元）
    "delay_median":      (4.0, 7.0),      # 单次事故停工中位数（天）
    "k_d":               (0.09, 0.15),    # 巡检边际效应（敏感性最高，故纳入抽样）
}


def simulate(params, scheme, z, collect_series=False):
    """对 N 组抽样参数向量化运行一次 180 天 SD 推演。

    params : 基准参数字典
    scheme : {"I": 安全投入万元, "f": 巡检频次次/天}
    z      : dict of arrays, 每个长度为 N
    返回   : dict of arrays（事故次数、总成本、总损失、总停工、期末隐患存量、可选时序）
    """
    T = params["T"]
    n = len(z["g0"])

    I = scheme["I"]
    f = scheme["f"]
    eta = 1.0 - np.exp(-params["k_t"] * I)
    # 检出率封顶为 1：避免巡检频次过高时线性式超出物理上限
    lam = 1.0 - np.exp(-(params["lambda0"] + params["k_d"] * f))

    H = np.full(n, 2.0)          # 初始隐患存量
    n_acc = np.zeros(n)
    loss = np.zeros(n)
    delay = np.zeros(n)
    series = np.zeros((T + 1, n)) if collect_series else None
    if collect_series:
        series[0] = H

    rng = np.random.default_rng(params.get("seed", None))

    for t in range(T):
        G = params["g0"] * (1.0 + params["beta"] * H) * (1.0 - eta)
        W = rng.poisson(params["w_rate"], n) * params["w_load"]
        R = lam * H
        rate = params["theta"] * H
        A = rng.poisson(rate)

        # 事故损失与停工抽样（仅对发生事故的个体抽样）
        hits = A > 0
        if hits.any():
            k = int(hits.sum())
            l_i = rng.lognormal(np.log(z["loss_median"][hits]), 0.55, k)
            d_i = rng.lognormal(np.log(z["delay_median"][hits]), 0.4, k)
            loss[hits] += A[hits] * l_i
            delay[hits] += A[hits] * d_i
            n_acc[hits] += A[hits]

        # 事故不再从存量中扣减：事故发生后隐患通常仍然存在，只做计数
        H = H + G + W - R
        H = np.maximum(H, 0.0)
        if collect_series:
            series[t + 1] = H

    total_cost = I + loss + params["delay_unit_cost"] * delay
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


def summarize(res):
    """把一次批量仿真结果聚合成可报告的统计量。"""
    c = res["cost"]
    return {
        "p_acc": float((res["n_acc"] > 0).mean()),
        "acc_rate": float(res["n_acc"].mean()),
        "exp_cost": float(c.mean()),
        "sd_cost": float(c.std(ddof=1)),
        "p95_cost": float(np.quantile(c, 0.95)),
        "p99_cost": float(np.quantile(c, 0.99)),
        "exp_loss": float(res["loss"].mean()),
        "exp_delay": float(res["delay"].mean()),
        "h_end": float(res["H_end"].mean()),
    }
