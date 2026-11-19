# -*- coding: utf-8 -*-
"""实验脚本（v3 — 全面加强版）。

相对 v2，本脚本提供全部 20 项改进的数值实现，并在 output/comparison.json 里
同时记录 v2（FEATURES 全关）与 v3（FEATURES 全开）两套结果，供对照文档使用。

改进清单（编号对应 model.py 注释）：
  F1  2D (I,f) 网格 + Pareto 前沿           -> run_grid()
  F2  事故响应闭环                            -> 体现在 simulate（对照组）
  F3  两层不确定性(偶然/认知)分离             -> run_two_level_uncertainty()
  F4  Sobol 全局敏感性(一阶/总效应)           -> run_sobol()
  F5  参数相关(Gaussian copula / Iman-Conover)-> 主实验用 CORR
  F6  损失-停工相关                           -> 体现在 simulate
  F7  培训时滞与饱和                           -> 体现在 simulate
  F8  巡检质量衰减                             -> 体现在 simulate
  F9  隐患分级                                 -> 体现在 simulate
  F10 分阶段标定                               -> 体现在 simulate
  F11 CVaR/ES + BCa bootstrap                 -> boot_ci(BCa)
  F12 Wilson 精确区间                         -> describe()
  F13 独立随机流(Streams)                      -> model.Streams
  F14 天气马尔可夫                             -> 体现在 simulate
  F15 分岔/校准/代理                            -> validate.py, surrogate.py
  F16 进度耦合(CPM 关键路径)                   -> run_schedule_impact()
  F17 成本口径扩展                             -> run_cost_extended()
  F18 决策支持(export for app.py)              -> main() 末尾
  F19 代理模型(PCE/Kriging)                     -> surrogate.py
  F20 概率校准检验                             -> validate.py

运行：python src/experiments.py
输出：output/results.json, output/comparison.json, output/*.csv
"""
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (DEFAULT, SCHEMES, UNCERTAIN, CORR, FEATURES, V2_FEATURES,
                   sample_params_lhs, simulate, summarize, grid_scan)

N = 6000
REPS = 1000          # bootstrap 次数（v3 提高精度）
Z_SEED = 2026
R_SEED = 7
INNER_REPS = 30      # [F3] 两层不确定性：内层重抽随机流次数
OUT = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# 抽样 & 统计工具
# ---------------------------------------------------------------------------
def lhs(n, seed=Z_SEED, corr=None):
    return sample_params_lhs(n, seed=seed, corr=corr)


def wilson_ci(p, n, z=1.96):
    """[F12] Wilson 得分区间，比正态近似更稳（p 接近 0/1 时不越界）。"""
    denom = 1.0 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def boot_ci(x, reps=REPS, seed=11):
    """[F11] 均值/P95/P99 的 BCa bootstrap 区间 + CVaR。"""
    x = np.asarray(x)
    n = len(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(reps, n))
    s = x[idx]

    def bca(stat, point):
        bs = stat(s, axis=1)
        # 偏差校正 z0
        prop = np.mean(bs < point)
        prop = min(max(prop, 1e-6), 1 - 1e-6)
        from scipy.stats import norm
        z0 = norm.ppf(prop)
        # 加速度 a（jackknife）
        jack = np.array([stat(np.delete(x, i)) for i in range(min(n, 200))])
        jm = jack.mean()
        num = np.sum((jm - jack) ** 3)
        den = 6.0 * (np.sum((jm - jack) ** 2)) ** 1.5
        a = num / den if den > 0 else 0.0
        lo_q = norm.cdf(z0 + (z0 - 1.96) / (1 - a * (z0 - 1.96)))
        hi_q = norm.cdf(z0 + (z0 + 1.96) / (1 - a * (z0 + 1.96)))
        lo_q = min(max(lo_q, 0.001), 0.999)
        hi_q = min(max(hi_q, 0.001), 0.999)
        return [float(np.quantile(bs, lo_q)), float(np.quantile(bs, hi_q))]

    def cvar(cc):
        q = np.quantile(cc, 0.95)
        t = cc[cc >= q]
        return t.mean() if len(t) else q

    return {
        "mean": float(x.mean()),
        "mean_ci": bca(lambda a, axis=None: np.mean(a, axis=axis), x.mean()),
        "p95": float(np.quantile(x, 0.95)),
        "p95_ci": bca(lambda a, axis=None: np.quantile(a, 0.95, axis=axis), np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "p99_ci": bca(lambda a, axis=None: np.quantile(a, 0.99, axis=axis), np.quantile(x, 0.99)),
        "cvar95": float(cvar(x)),
    }


def run(params, scheme, z, features=None):
    n = len(z["g0"])
    return simulate(dict(params, seed=R_SEED), scheme, z, features=features)


def describe(res, name=None, features=None, params=None):
    cost = boot_ci(res["cost"])
    n = len(res["cost"])
    pa = float((res["n_acc"] > 0).mean())
    lo, hi = wilson_ci(pa, n)
    s = {
        "p_acc": pa, "p_acc_ci": [lo, hi],
        "acc_rate": float(res["n_acc"].mean()),
        "exp_cost": cost["mean"], "exp_cost_ci": cost["mean_ci"],
        "p95_cost": cost["p95"], "p95_ci": cost["p95_ci"],
        "p99_cost": cost["p99"], "p99_ci": cost["p99_ci"],
        "cvar95": cost["cvar95"],
        "exp_loss": float(res["loss"].mean()),
        "exp_delay": float(res["delay"].mean()),
        "h_end": float(res["H_end"].mean()),
        "n": n,
    }
    if name:
        s.update({"scheme": name, "scheme_name": SCHEMES[name]["name"],
                  "I": SCHEMES[name]["I"], "f": SCHEMES[name]["f"]})
    return s


# ---------------------------------------------------------------------------
# 主实验：v2 vs v3 方案对照
# ---------------------------------------------------------------------------
def run_schemes(z, features):
    rows, dists = [], {}
    for name in SCHEMES:
        r = run(DEFAULT, SCHEMES[name], z, features=features)
        rows.append(describe(r, name))
        dists[name] = r["cost"]
    return rows, dists


# ---------------------------------------------------------------------------
# [F3] 两层不确定性分离
# ---------------------------------------------------------------------------
def run_two_level_uncertainty(z, features):
    """内层(偶然)：固定参数样本、仅重抽随机流；外层(认知)：跨参数样本。

    对均衡方案 S2：
      - aleatory_sd：每个固定参数样本内，重抽 INNER_REPS 次随机流的均值方差
      - epistemic_sd：各参数样本均值的方差（内层均值再跨参数）
      - 总方差 = 内层方差均值 + 外层方差
    """
    sc = SCHEMES["S2"]
    n = len(z["g0"])
    # 取前 K 个参数样本，每个重抽 INNER_REPS 次随机流
    K = min(120, n)
    inner_means = np.zeros((K, INNER_REPS))
    rng = np.random.default_rng(99)
    for k in range(K):
        # 固定该参数样本，仅改变随机流种子（偶然不确定性）
        for r in range(INNER_REPS):
            seed = int(rng.integers(0, 2 ** 31 - 1))
            rr = simulate(dict(DEFAULT, seed=seed), sc,
                          {kk: v[k:k + 1] for kk, v in z.items()}, features=features)
            inner_means[k, r] = rr["cost"][0]
    aleatory_var = float(np.mean(np.var(inner_means, axis=1, ddof=1)))
    outer_var = float(np.var(inner_means.mean(axis=1), ddof=1))
    total = aleatory_var + outer_var
    return {
        "K_params": K, "inner_reps": INNER_REPS,
        "aleatory_var": aleatory_var, "epistemic_var": outer_var,
        "total_var": total,
        "aleatory_share": aleatory_var / total if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# [F4] Sobol 全局敏感性
# ---------------------------------------------------------------------------
def run_sobol(z, features, n_base=1024):
    """对 6 个连续参数做 Sobol 一阶/总效应（以均衡方案期望成本为输出）。"""
    try:
        from SALib.sample import sobol as sobol_sample
        from SALib.analyze import sobol as sobol_an
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}

    keys = ["g0", "beta", "lambda0", "k_d", "theta", "w_rate"]
    bounds = [[DEFAULT[k] * 0.7, DEFAULT[k] * 1.3] for k in keys]
    problem = {"num_vars": len(keys), "names": keys, "bounds": bounds}
    X = sobol_sample.sample(problem, n_base, calc_second_order=False)

    # 固定一组参数样本（缩小到 800 以控制成本），每次仿真返回期望成本
    zs = {kk: v[:800] for kk, v in z.items()}
    Y = np.zeros(len(X))
    for i, row in enumerate(X):
        p = dict(DEFAULT)
        for j, k in enumerate(keys):
            p[k] = float(row[j])
        r = simulate(dict(p, seed=R_SEED), SCHEMES["S2"], zs, features=features)
        Y[i] = r["cost"].mean()

    Si = sobol_an.analyze(problem, Y, calc_second_order=False, print_to_console=False)
    return {
        "names": keys,
        "S1": [round(float(v), 4) for v in Si["S1"]],
        "ST": [round(float(v), 4) for v in Si["ST"]],
        "S1_conf": [round(float(v), 4) for v in Si["S1_conf"]],
        "ST_conf": [round(float(v), 4) for v in Si["ST_conf"]],
    }


# ---------------------------------------------------------------------------
# [F16] 进度耦合：CPM 关键路径下的工期冲击
# ---------------------------------------------------------------------------
def run_schedule_impact(z, features):
    """把停工天数落到关键路径/非关键工作上，估算对工期基线的冲击。

    简化 CPM：关键路径剩余若干天，事故停工若落在关键工作直接顺延工期；
    非关键工作有浮时(total float)，停工超过浮时才顺延。
    """
    # 关键路径总浮时 = 0；非关键工作平均浮时 TF=6 天
    TF = 6.0
    res = {}
    for name in SCHEMES:
        r = run(DEFAULT, SCHEMES[name], z, features=features)
        # 期望冲击：停工超过 TF 的部分计入工期顺延
        delay_days = r["delay"]
        critical_share = 0.45   # 落在关键工作上的事件占比
        crit_delay = delay_days * critical_share
        noncrit_delay = delay_days * (1 - critical_share)
        schedule_impact = crit_delay + np.maximum(noncrit_delay - TF, 0.0)
        res[name] = {
            "exp_schedule_slip": float(schedule_impact.mean()),
            "p95_schedule_slip": float(np.quantile(schedule_impact, 0.95)),
        }
    return res


# ---------------------------------------------------------------------------
# [F17] 成本口径扩展
# ---------------------------------------------------------------------------
def run_cost_extended(z, features):
    """在 v2 (投入+损失+停工) 基础上叠加赶工费、处罚、保费上浮、信誉损失。"""
    EXTRA = {
        "crash_rate": 4000.0,     # 赶工费：元/天（关键路径延误赶工）
        "penalty_base": 20000.0,  # 监管处罚基数（元/次事故，按严重度放大）
        "premium_uplift": 0.08,   # 保费上浮（占投入比例，随事故率上升）
        "reput": 15000.0,         # 信誉/资质损失（元/次事故，尾部放大）
    }
    out = {}
    for name in SCHEMES:
        r = run(DEFAULT, SCHEMES[name], z, features=features)
        cost = r["cost"].copy()               # 万元
        n_acc = r["n_acc"]
        delay = r["delay"]
        # 赶工费（万元）：关键比例 0.45，超过 1 天的部分
        crash = 0.45 * np.maximum(delay - 1.0, 0.0) * EXTRA["crash_rate"] / 10000.0
        # 处罚 + 信誉（万元），随事故次数（含严重度按次数平方根放大）
        pen = n_acc * (EXTRA["penalty_base"] + EXTRA["reput"]) / 10000.0
        pen = pen * (1.0 + 0.5 * np.sqrt(np.maximum(n_acc - 1, 0)))
        # 保费上浮（万元）
        prem = SCHEMES[name]["I"] * EXTRA["premium_uplift"] * (0.5 + (n_acc > 0).astype(float))
        ext = cost + crash + pen + prem
        out[name] = {
            "exp_cost_v2": float(cost.mean()),
            "exp_cost_ext": float(ext.mean()),
            "p95_v2": float(np.quantile(cost, 0.95)),
            "p95_ext": float(np.quantile(ext, 0.95)),
        }
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("== 抽样（v3：含参数相关）==")
    z = lhs(N, corr=CORR)
    z_nocorr = lhs(N, corr=None)

    results = {"meta": {
        "N": N, "T": DEFAULT["T"], "REPS": REPS, "z_seed": Z_SEED, "r_seed": R_SEED,
        "version": "v3",
        "uncertain": {k: list(v) for k, v in UNCERTAIN.items()},
        "corr": CORR.tolist(),
        "features": FEATURES,
    }}

    # ---- 0) 三方案 v2 / v3 / 无相关 对照 ----
    print("== 三方案对照（v2 退化 vs v3）==")
    rows_v2, _ = run_schemes(z, V2_FEATURES)
    rows_v3, dists_v3 = run_schemes(z, FEATURES)
    rows_v3_nocorr, _ = run_schemes(z_nocorr, FEATURES)
    results["schemes_v2"] = rows_v2
    results["schemes"] = rows_v3
    results["schemes_v3_nocorr"] = rows_v3_nocorr

    pd.DataFrame([{k: v for k, v in row.items() if not k.endswith("_ci")}
                  for row in rows_v3]).to_csv(os.path.join(OUT, "scheme_summary.csv"),
                                               index=False, encoding="utf-8-sig")
    # v2 退化对照表
    pd.DataFrame([{"scheme": a["scheme"],
                   "v2_p_acc": a["p_acc"], "v3_p_acc": b["p_acc"],
                   "v2_exp": a["exp_cost"], "v3_exp": b["exp_cost"],
                   "v2_p95": a["p95_cost"], "v3_p95": b["p95_cost"],
                   "v2_cvar95": a["cvar95"], "v3_cvar95": b["cvar95"]}
                  for a, b in zip(rows_v2, rows_v3)]).to_csv(
        os.path.join(OUT, "v2_vs_v3_schemes.csv"), index=False, encoding="utf-8-sig")

    qs = np.linspace(0.01, 0.99, 41)
    results["cost_dist"] = {name: {"quantile": qs.round(3).tolist(),
                                   "value": [round(float(np.quantile(d, q)), 2) for q in qs]}
                            for name, d in dists_v3.items()}

    # ---- 1) 时序 ----
    print("== 时序 ==")
    zs = lhs(500, corr=CORR)
    ser = {}
    for name in SCHEMES:
        S = simulate(dict(DEFAULT, seed=R_SEED), SCHEMES[name], zs, collect_series=True)["series"]
        ser[name] = {"t": list(range(DEFAULT["T"] + 1)),
                     "mean": [round(float(v), 3) for v in S.mean(axis=1)],
                     "p5": [round(float(v), 3) for v in np.quantile(S, 0.05, axis=1)],
                     "p95": [round(float(v), 3) for v in np.quantile(S, 0.95, axis=1)],
                     "d30": round(float(S[30].mean()), 2), "d120": round(float(S[120].mean()), 2)}
    results["series"] = ser

    # ---- 2) 收敛（含分位数收敛，F11 相关）----
    print("== 收敛 ==")
    r2 = run(DEFAULT, SCHEMES["S2"], z)
    pacc = np.cumsum(r2["n_acc"] > 0) / np.arange(1, N + 1)
    idx = list(range(50, N + 1, 50))
    results["convergence"] = {
        "n": idx,
        "p_acc": [round(float(pacc[i - 1]), 4) for i in idx],
        "exp_cost": [round(float(np.mean(r2["cost"][:i])), 2) for i in idx],
        "p95_cost": [round(float(np.quantile(r2["cost"][:i], 0.95)), 2) for i in idx],
    }

    # ---- 3) 投入扫描（1D，保留以对照）----
    print("== 1D 投入扫描 ==")
    sweep = []
    for I in [5, 10, 15, 20, 25, 30, 35, 40, 45]:
        f = round(0.15 + 0.0215 * I, 3)
        s = describe(run(DEFAULT, {"I": float(I), "f": f}, z))
        s.update({"I": I, "f": f})
        sweep.append(s)
    results["sweep"] = sweep
    pd.DataFrame([{k: v for k, v in s.items() if not k.endswith("_ci")} for s in sweep]).to_csv(
        os.path.join(OUT, "investment_sweep.csv"), index=False, encoding="utf-8-sig")

    # ---- 4) [F1] 2D 网格 + Pareto ----
    print("== 2D (I,f) 网格 + Pareto ==")
    I_grid = [8, 12, 16, 20, 25, 30, 35, 40, 50]
    f_grid = [0.3, 0.5, 0.7, 0.9, 1.1, 1.4, 1.8]
    zz = {k: v[:3000] for k, v in z.items()}   # 网格用 3000 样本控制成本
    cells = grid_scan(DEFAULT, zz, I_grid, f_grid)
    # 网格只用了前 3000 样本，1D 最优需在同一子样本上比，才公平
    sweep_same = []
    for I in I_grid:
        f = round(0.15 + 0.0215 * I, 3)
        s = summarize(simulate(dict(DEFAULT, seed=R_SEED), {"I": float(I), "f": f}, zz))
        s.update({"I": I, "f": f})
        sweep_same.append(s)
    results["grid"] = {"I_grid": I_grid, "f_grid": f_grid,
                       "cells": [{k: c[k] for k in ("I", "f", "p_acc", "exp_cost", "p95_cost",
                                                    "cvar95", "pareto")} for c in cells]}
    pareto_cells = [c for c in cells if c["pareto"]]
    results["grid"]["pareto"] = [{"I": c["I"], "f": c["f"], "exp_cost": c["exp_cost"],
                                  "p95_cost": c["p95_cost"]} for c in pareto_cells]
    # 1D 线上最优 vs 2D 最优（同一子样本口径）
    best_1d = min(sweep_same, key=lambda s: s["exp_cost"])
    best_2d = min(cells, key=lambda c: c["exp_cost"])
    results["grid"]["best_1d"] = {"I": best_1d["I"], "f": best_1d["f"],
                                  "exp_cost": best_1d["exp_cost"], "p_acc": best_1d["p_acc"]}
    results["grid"]["best_2d"] = {"I": best_2d["I"], "f": best_2d["f"],
                                  "exp_cost": best_2d["exp_cost"], "p_acc": best_2d["p_acc"]}
    results["grid"]["sweep_same"] = [{"I": s["I"], "f": s["f"], "exp_cost": s["exp_cost"],
                                      "p_acc": s["p_acc"], "p95_cost": s["p95_cost"]}
                                     for s in sweep_same]
    print("  1D 最优:", results["grid"]["best_1d"])
    print("  2D 最优:", results["grid"]["best_2d"])

    # ---- 5) 敏感性 OAT（v3）----
    print("== OAT 敏感性 ==")
    factors = [("g0", "隐患生成率 g0"), ("beta", "正反馈系数 β"), ("lambda0", "基础检出率 λ0"),
               ("k_d", "巡检边际效应 k_d"), ("theta", "事故触发系数 θ"), ("w_rate", "扰动频次 w")]
    base = float(run(DEFAULT, SCHEMES["S2"], z)["cost"].mean())
    tor = []
    for key, label in factors:
        for pct in (-0.3, 0.3):
            p = dict(DEFAULT); p[key] = DEFAULT[key] * (1 + pct)
            r = run(p, SCHEMES["S2"], z)
            tor.append({"factor": key, "label": label, "pct": pct,
                        "exp_cost": round(float(r["cost"].mean()), 2),
                        "p_acc": round(float((r["n_acc"] > 0).mean()), 4),
                        "d_cost": round(float(r["cost"].mean()) - base, 2)})
    results["sensitivity"] = {"base_cost": round(base, 2),
                              "base_p_acc": round(float((run(DEFAULT, SCHEMES["S2"], z)["n_acc"] > 0).mean()), 4),
                              "runs": tor}

    # ---- 6) [F3] 两层不确定性 ----
    print("== 两层不确定性 ==")
    tlu = run_two_level_uncertainty(z, FEATURES)
    results["two_level"] = tlu
    print("  偶然份额:", round(tlu["aleatory_share"], 3))

    # ---- 7) [F4] Sobol ----
    print("== Sobol（可能较慢）==")
    results["sobol"] = run_sobol(z, FEATURES, n_base=512)
    print("  Sobol ST:", results["sobol"].get("ST"))

    # ---- 8) [F16] 进度耦合 ----
    results["schedule"] = run_schedule_impact(z, FEATURES)

    # ---- 9) [F17] 成本口径扩展 ----
    results["cost_ext"] = run_cost_extended(z, FEATURES)

    # ---- 汇总 comparison.json ----
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    _write_comparison(results)

    # 控制台摘要
    print("\n=== 方案对照（v2 退化 -> v3）===")
    for a, b in zip(rows_v2, rows_v3):
        print(f"{a['scheme']}: P {a['p_acc']*100:.1f}->{b['p_acc']*100:.1f}%  "
              f"E[C] {a['exp_cost']:.1f}->{b['exp_cost']:.1f}  "
              f"P95 {a['p95_cost']:.1f}->{b['p95_cost']:.1f}  "
              f"CVaR {a['cvar95']:.1f}->{b['cvar95']:.1f}")
    print("\n全部结果已写入 output/results.json 与 output/comparison.json")


def _write_comparison(R):
    """把改进前(v2)/改进后(v3)关键数值抽出来，写成结构化对照，供文档引用。"""
    comp = {"schemes": [], "grid": R["grid"], "two_level": R["two_level"],
            "sobol": R["sobol"], "schedule": R["schedule"], "cost_ext": R["cost_ext"],
            "sensitivity": R["sensitivity"]}
    for a, b in zip(R["schemes_v2"], R["schemes"]):
        comp["schemes"].append({
            "scheme": a["scheme"], "name": a["scheme_name"],
            "v2": {k: a[k] for k in ("p_acc", "exp_cost", "p95_cost", "p99_cost", "cvar95", "h_end")},
            "v3": {k: b[k] for k in ("p_acc", "exp_cost", "p95_cost", "p99_cost", "cvar95", "h_end")},
        })
    with open(os.path.join(OUT, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comp, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
