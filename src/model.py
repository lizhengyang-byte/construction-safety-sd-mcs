# -*- coding: utf-8 -*-
"""土木施工安全 SD + MCS 耦合仿真核心模型（向量化实现）。

存量-流量结构（单位：天）
    H_{t+1} = H_t + G_t + W_t - R_t - A_t
    G_t = g0 * (1 + beta * H_t) * (1 - eta)      隐患生成（含正反馈放大与天气冲击）
    R_t = (lambda0 + k_d * f) * H_t              巡检整改（负反馈调节）
    A_t ~ Poisson(theta * H_t)                   隐患转化为事故的通量
    eta = 1 - exp(-k_t * I)                      安全培训抑制率

成本
    C = I + sum(损失_i) + 0.8 * sum(停工天数_i)   （万元）
"""
import numpy as np

DEFAULT = {
    "g0": 0.35,
    "beta": 0.08,
    "lambda0": 0.02,
    "k_d": 0.12,
    "k_t": 0.02,
    "theta": 0.0004,
    "w_rate": 0.026,
    "w_load": 2.0,
    "delay_unit_cost": 0.8,
    "T": 180,
}

SCHEMES = {
    "S1": {"name": "低投入方案", "I": 8.0, "f": 0.32},
    "S2": {"name": "均衡方案", "I": 20.0, "f": 0.58},
    "S3": {"name": "高投入方案", "I": 40.0, "f": 1.00},
}

UNCERTAIN = {
    "g0": (0.25, 0.45),
    "theta": (0.00030, 0.00050),
    "loss_median": (35.0, 55.0),
    "delay_median": (4.0, 7.0),
}


def simulate(params, scheme, z, collect_series=False):
    """对 N 组抽样参数向量化运行一次 180 天 SD 推演。"""
    T = params["T"]
    n = len(z["g0"])
    I = scheme["I"]
    f = scheme["f"]
    eta = 1.0 - np.exp(-params["k_t"] * I)
    lam = params["lambda0"] + params["k_d"] * f

    H = np.full(n, 2.0)
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
        A = rng.poisson(params["theta"] * H)

        hits = A > 0
        if hits.any():
            k = int(hits.sum())
            l_i = rng.lognormal(np.log(z["loss_median"][hits]), 0.55, k)
            d_i = rng.lognormal(np.log(z["delay_median"][hits]), 0.4, k)
            loss[hits] += A[hits] * l_i
            delay[hits] += A[hits] * d_i
            n_acc[hits] += A[hits]

        H = np.maximum(H + G + W - R - A, 0.0)
        if collect_series:
            series[t + 1] = H

    out = {
        "n_acc": n_acc,
        "loss": loss,
        "delay": delay,
        "cost": I + loss + params["delay_unit_cost"] * delay,
        "H_end": H,
    }
    if collect_series:
        out["series"] = series
    return out


def summarize(res):
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
