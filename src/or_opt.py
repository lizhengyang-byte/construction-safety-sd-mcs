# -*- coding: utf-8 -*-
"""运筹学决策层：在 v3 蒙特卡洛引擎之上做三套规范优化。

给定决策变量 (I, f)，每套优化都用蒙特卡洛样本做 scenario-based 求解：

  1) 期望最优 (risk-neutral, EV)          minimize  E[C(I,f)]
  2) CVaR 最优 (风险厌恶随机规划)          minimize  CVaR_95[C(I,f)]
  3) 鲁棒最优 (min-max, 不确定集)          minimize  max_{θ∈U} CVaR_95[C(I,f;θ)]

同时给出：有效前沿（E[C] vs CVaR95 的 Pareto）、机会约束（P(C>预算)≤α）。

运行：python src/or_opt.py  → output/or_opt.json + fig11_orfrontier.svg
"""
import json
import os
import sys
import itertools

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")

# 决策网格（万元 / 次/天）
I_GRID = np.array([6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 50, 60], float)
F_GRID = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4], float)
BUDGET = 180.0  # 成本预算上限（万元），用于机会约束 P(C>BUDGET)<=alpha


def _simulate_grid(z, n_sample=None):
    """对 (I,f) 网格上每点做蒙特卡洛，返回每格的成本样本矩阵（用于 EV/CVaR/机会约束）。

    返回 costs[i,j] = 该 (I,f) 点的一组成本样本（长度 = len(z)）。
    """
    if n_sample is not None:
        z = {k: v[:n_sample] for k, v in z.items()}
    n = len(z["g0"])
    costs = np.empty((len(I_GRID), len(F_GRID), n))
    pacc = np.empty((len(I_GRID), len(F_GRID), n))
    params0 = dict(M.DEFAULT)
    for i, I in enumerate(I_GRID):
        for j, f in enumerate(F_GRID):
            scheme = {"I": float(I), "f": float(f)}
            r = M.simulate(params0, scheme, z, features=M.FEATURES)
            costs[i, j] = r["cost"]
            pacc[i, j] = (r["n_acc"] > 0).astype(float)
    return costs, pacc


def cvar(x, q=0.95):
    """CVaR_q：尾部 (1-q) 条件期望（越大越坏方向）。"""
    var = np.quantile(x, q)
    tail = x[x >= var]
    return float(tail.mean()) if tail.size else float(var)


def optimize(costs, pacc):
    """在三套目标下各自选最优 (I,f)，并给出有效前沿与机会约束。"""
    nI, nF, _ = costs.shape
    EV = costs.mean(axis=2)
    CV = np.empty((nI, nF))
    PACC = pacc.mean(axis=2)
    POVER = np.empty((nI, nF))
    for i in range(nI):
        for j in range(nF):
            CV[i, j] = cvar(costs[i, j])
            POVER[i, j] = (costs[i, j] > BUDGET).mean()

    def argmin(A):
        k = np.unravel_index(np.argmin(A), A.shape)
        return k, float(A[k])

    # 1) 期望最优
    (ei, ej), ev = argmin(EV)
    # 2) CVaR 最优
    (ci, cj), cv = argmin(CV)
    # 3) 机会约束下的期望最优：P(C>BUDGET) <= alpha 中 EV 最小
    res = {}
    for alpha in (0.10, 0.05):
        mask = POVER <= alpha
        if mask.any():
            Em = np.where(mask, EV, np.inf)
            (ai, aj), av = argmin(Em)
            res[f"alpha_{alpha}"] = {
                "I": float(I_GRID[ai]), "f": float(F_GRID[aj]),
                "exp_cost": float(EV[ai, aj]), "cvar95": float(CV[ai, aj]),
                "p_over_budget": float(POVER[ai, aj]),
            }
        else:
            res[f"alpha_{alpha}"] = None

    # Pareto 前沿（EV vs CV）
    pts = [(float(EV[i, j]), float(CV[i, j]), float(I_GRID[i]), float(F_GRID[j]))
           for i in range(nI) for j in range(nF)]
    pareto = _pareto_min2(pts)

    return {
        "grid": {"I": I_GRID.tolist(), "f": F_GRID.tolist()},
        "EV_best": {"I": float(I_GRID[ei]), "f": float(F_GRID[ej]),
                     "exp_cost": float(EV[ei, ej]), "cvar95": float(CV[ei, ej]),
                     "p_acc": float(PACC[ei, ej]), "p_over_budget": float(POVER[ei, ej])},
        "CVaR_best": {"I": float(I_GRID[ci]), "f": float(F_GRID[cj]),
                      "exp_cost": float(EV[ci, cj]), "cvar95": float(CV[ci, cj]),
                      "p_acc": float(PACC[ci, cj]), "p_over_budget": float(POVER[ci, cj])},
        "chance_constrained": res,
        "pareto_EV_CVaR": [{"exp_cost": a, "cvar95": b, "I": c, "f": d} for a, b, c, d in pareto],
        "frontier_full": {"EV": EV.tolist(), "CVaR": CV.tolist(),
                           "p_over": POVER.tolist()},
    }


def _pareto_min2(pts):
    """两目标（都最小化）Pareto 前沿。"""
    pts = sorted(pts)
    front = []
    best_b = np.inf
    for a, b, I, f in pts:
        if b < best_b - 1e-9:
            front.append((a, b, I, f))
            best_b = b
    return front


def robust_optimize(z_base, theta_lo=-0.25, theta_hi=0.25, n_sample=1500):
    """min-max 鲁棒：对 {g0, theta} 不确定集顶点取最坏 CVaR。"""
    # 不确定集：g0、theta 各 ±25% 的 4 个顶点（盒式）+ 中心
    from itertools import product
    deltas = list(product([1 + theta_lo, 1 + theta_hi], repeat=2))
    # 预采样一次参数，各顶点按乘子缩放 g0/theta（CRN：同一 z，保证可比）
    z = {k: v[:n_sample] for k, v in z_base.items()}
    worst_cv = np.zeros((len(I_GRID), len(F_GRID)))
    per_vertex = {}
    params0 = dict(M.DEFAULT)
    for dg, dt in deltas:
        zz = dict(z)
        zz["g0"] = z["g0"] * dg
        zz["theta"] = z["theta"] * dt
        CV = np.empty((len(I_GRID), len(F_GRID)))
        for i, I in enumerate(I_GRID):
            for j, f in enumerate(F_GRID):
                r = M.simulate(params0, {"I": float(I), "f": float(f)}, zz, features=M.FEATURES)
                CV[i, j] = cvar(r["cost"])
        worst_cv = np.maximum(worst_cv, CV)
        per_vertex[f"g0x{dg:.2f}_thetax{dt:.2f}"] = float(CV.min())
    k = np.unravel_index(np.argmin(worst_cv), worst_cv.shape)
    return {
        "I": float(I_GRID[k[0]]), "f": float(F_GRID[k[1]]),
        "worst_cvar95": float(worst_cv[k]),
        "per_vertex_min_cvar": per_vertex,
        "note": "盒式不确定集 4 顶点 min-max，最坏情形 CVaR 最小",
    }


def chance_conservative_vs_naive(or_res):
    """对比：期望最优 vs CVaR 最优 在「超预算概率」上的差异（风险厌恶价值）。"""
    ev = or_res["EV_best"]
    cv = or_res["CVaR_best"]
    return {
        "EV_best_p_over_budget": ev["p_over_budget"],
        "CVaR_best_p_over_budget": cv["p_over_budget"],
        "p_over_budget_reduction": ev["p_over_budget"] - cv["p_over_budget"],
        "EV_best_cvar": ev["cvar95"], "CVaR_best_cvar": cv["cvar95"],
        "cvar_reduction": ev["cvar95"] - cv["cvar95"],
        "extra_exp_cost_paid": cv["exp_cost"] - ev["exp_cost"],
        "note": "CVaR 最优用少量期望成本换取尾部大幅下降——风险厌恶的定价",
    }


def main():
    z = M.sample_params_lhs(3000, seed=2026, corr=M.CORR)
    print("== 运筹学决策层：网格蒙特卡洛 (I×f) ==")
    costs, pacc = _simulate_grid(z)
    or_res = optimize(costs, pacc)
    print(f"  期望最优: I={or_res['EV_best']['I']:.0f} f={or_res['EV_best']['f']:.2f} "
          f"E[C]={or_res['EV_best']['exp_cost']:.1f} CVaR={or_res['EV_best']['cvar95']:.1f}")
    print(f"  CVaR最优: I={or_res['CVaR_best']['I']:.0f} f={or_res['CVaR_best']['f']:.2f} "
          f"E[C]={or_res['CVaR_best']['exp_cost']:.1f} CVaR={or_res['CVaR_best']['cvar95']:.1f}")
    rp = chance_conservative_vs_naive(or_res)
    print(f"  风险厌恶价值: 超预算概率 {rp['EV_best_p_over_budget']:.3f}->"
          f"{rp['CVaR_best_p_over_budget']:.3f}, CVaR 降 {rp['cvar_reduction']:.1f}")

    print("== 鲁棒 min-max 优化 ==")
    rob = robust_optimize(M.sample_params_lhs(1500, seed=7, corr=M.CORR))
    print(f"  鲁棒最优: I={rob['I']:.0f} f={rob['f']:.2f} worst-CVaR={rob['worst_cvar95']:.1f}")

    payload = {"optimize": or_res, "risk_aversion_value": rp, "robust": rob,
               "budget": BUDGET}
    json.dump(payload, open(os.path.join(OUT, "or_opt.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    _plot(or_res)
    print("已输出 output/or_opt.json + fig11_orfrontier.svg")


def _plot(or_res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"
    EV = np.array(or_res["frontier_full"]["EV"])
    CV = np.array(or_res["frontier_full"]["CVaR"])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(EV.ravel(), CV.ravel(), s=12, c="#C3CAD6", label="可行 (I,f) 网格")
    pf = or_res["pareto_EV_CVaR"]
    ax.plot([p["exp_cost"] for p in pf], [p["cvar95"] for p in pf],
            "-o", color="#1A2B4A", ms=4, lw=1.8, label="Pareto 前沿")
    ev = or_res["EV_best"]; cv = or_res["CVaR_best"]
    ax.scatter([ev["exp_cost"]], [ev["cvar95"]], marker="*", s=260,
               color="#B4762A", zorder=5, label="期望最优 (risk-neutral)")
    ax.scatter([cv["exp_cost"]], [cv["cvar95"]], marker="D", s=110,
               color="#8B1A1A", zorder=5, label="CVaR 最优 (风险厌恶)")
    ax.set_xlabel("期望总成本 E[C]（万元）")
    ax.set_ylabel("CVaR95（万元）")
    ax.set_title("图 11  运筹学决策：期望–尾部有效前沿与两类最优")
    ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "docs", "figures", "fig11_orfrontier.svg"),
                format="svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
