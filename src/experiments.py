# -*- coding: utf-8 -*-
"""实验脚本（v2）：三方案对比、时序、收敛、投入扫描、OAT 敏感性。

相对 v1 的三处统计改进：
1. 全部方案、扫描档位、敏感性基准共用同一组拉丁超立方样本与同一随机种子
   （共同随机数 CRN），消除"同配置不同种子"造成的口径不一致；
2. 每个指标给出 bootstrap 95% 置信区间，不再只报点估计；
3. k_d 纳入抽样维度（v1 中它在敏感性分析里最敏感，却曾是固定值）。

运行：python src/experiments.py
输出：output/results.json、output/*.csv
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import qmc

from model import DEFAULT, SCHEMES, UNCERTAIN, simulate

N = 6000
REPS = 800          # bootstrap 次数
Z_SEED = 2026       # 共用 LHS 样本种子
R_SEED = 7          # 共用随机数流种子
OUT = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUT, exist_ok=True)


def lhs(n, seed=Z_SEED):
    """五维拉丁超立方抽样，映射到各随机参数区间。"""
    sampler = qmc.LatinHypercube(d=len(UNCERTAIN), seed=seed)
    u = sampler.random(n)
    lo = np.array([v[0] for v in UNCERTAIN.values()])
    hi = np.array([v[1] for v in UNCERTAIN.values()])
    X = qmc.scale(u, lo, hi)
    return {k: X[:, i] for i, k in enumerate(UNCERTAIN)}


def boot_ci(x, reps=REPS, seed=11):
    """返回均值、P95、P99 及其 95% bootstrap 置信区间。"""
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(reps, n))
    s = x[idx]
    return {
        "mean": float(x.mean()),
        "mean_ci": [float(np.quantile(s.mean(axis=1), 0.025)),
                    float(np.quantile(s.mean(axis=1), 0.975))],
        "p95": float(np.quantile(x, 0.95)),
        "p95_ci": [float(np.quantile(np.quantile(s, 0.95, axis=1), 0.025)),
                   float(np.quantile(np.quantile(s, 0.95, axis=1), 0.975))],
        "p99": float(np.quantile(x, 0.99)),
        "p99_ci": [float(np.quantile(np.quantile(s, 0.99, axis=1), 0.025)),
                   float(np.quantile(np.quantile(s, 0.99, axis=1), 0.975))],
    }


def run(params, scheme, z):
    return simulate(dict(params, seed=R_SEED), scheme, z)


def describe(res, name=None):
    cost = boot_ci(res["cost"])
    n = len(res["cost"])
    pa = float((res["n_acc"] > 0).mean())
    se = float(np.sqrt(pa * (1 - pa) / n))
    s = {
        "p_acc": pa,
        "p_acc_ci": [pa - 1.96 * se, pa + 1.96 * se],
        "p_acc_se": se,
        "acc_rate": float(res["n_acc"].mean()),
        "exp_cost": cost["mean"],
        "exp_cost_ci": cost["mean_ci"],
        "p95_cost": cost["p95"],
        "p95_ci": cost["p95_ci"],
        "p99_cost": cost["p99"],
        "p99_ci": cost["p99_ci"],
        "exp_loss": float(res["loss"].mean()),
        "exp_delay": float(res["delay"].mean()),
        "h_end": float(res["H_end"].mean()),
        "n": n,
    }
    if name:
        s.update({"scheme": name, "scheme_name": SCHEMES[name]["name"],
                  "I": SCHEMES[name]["I"], "f": SCHEMES[name]["f"]})
    return s


def main():
    z = lhs(N)
    results = {"meta": {
        "N": N, "T": DEFAULT["T"], "REPS": REPS, "z_seed": Z_SEED, "r_seed": R_SEED,
        "uncertain": {k: list(v) for k, v in UNCERTAIN.items()},
        "base": {k: DEFAULT[k] for k in
                 ["g0", "beta", "lambda0", "k_d", "k_t", "theta", "w_rate", "w_load", "delay_unit_cost"]}}}

    # 1. 三方案对比（共用样本与随机流）
    rows, dists, by_scheme = [], {}, {}
    for name in SCHEMES:
        r = run(DEFAULT, SCHEMES[name], z)
        by_scheme[name] = r
        rows.append(describe(r, name))
        dists[name] = r["cost"]
    pd.DataFrame([{k: v for k, v in row.items() if not k.endswith("_ci") and k != "p_acc_se"}
                  for row in rows]).to_csv(os.path.join(OUT, "scheme_summary.csv"),
                                           index=False, encoding="utf-8-sig")
    results["schemes"] = rows

    qs = np.linspace(0.01, 0.99, 41)
    results["cost_dist"] = {name: {"quantile": qs.round(3).tolist(),
                                   "value": [round(float(np.quantile(d, q)), 2) for q in qs]}
                            for name, d in dists.items()}

    # 2. 隐患存量时序
    zs = lhs(500)
    ser = {}
    for name in SCHEMES:
        S = simulate(dict(DEFAULT, seed=R_SEED), SCHEMES[name], zs, collect_series=True)["series"]
        ser[name] = {"t": list(range(DEFAULT["T"] + 1)),
                     "mean": [round(float(v), 3) for v in S.mean(axis=1)],
                     "p5": [round(float(v), 3) for v in np.quantile(S, 0.05, axis=1)],
                     "p95": [round(float(v), 3) for v in np.quantile(S, 0.95, axis=1)],
                     "d30": round(float(S[30].mean()), 2),
                     "d120": round(float(S[120].mean()), 2)}
    results["series"] = ser

    # 3. 收敛性（均衡方案）
    res = by_scheme["S2"]
    pacc = np.cumsum(res["n_acc"] > 0) / np.arange(1, N + 1)
    idx = list(range(50, N + 1, 50))
    results["convergence"] = {"n": idx,
                              "p_acc": [round(float(pacc[i - 1]), 4) for i in idx],
                              "exp_cost": [round(float(np.mean(res["cost"][:i])), 2) for i in idx]}

    # 4. 投入扫描（同一组样本、同一随机流）
    sweep = []
    for I in [5, 10, 15, 20, 25, 30, 35, 40, 45]:
        f = round(0.15 + 0.0215 * I, 3)
        s = describe(run(DEFAULT, {"I": float(I), "f": f}, z))
        s.update({"I": I, "f": f})
        sweep.append(s)
    results["sweep"] = sweep
    pd.DataFrame([{k: v for k, v in s.items() if not k.endswith("_ci") and k != "p_acc_se"}
                  for s in sweep]).to_csv(os.path.join(OUT, "investment_sweep.csv"),
                                          index=False, encoding="utf-8-sig")

    # 5. OAT 敏感性（基准与 S2 完全一致）
    factors = [("g0", "隐患生成率 g0"), ("beta", "正反馈系数 β"), ("lambda0", "基础检出率 λ0"),
               ("k_d", "巡检边际效应 k_d"), ("theta", "事故触发系数 θ"), ("w_rate", "扰动频次 w")]
    base = float(by_scheme["S2"]["cost"].mean())
    tor = []
    for key, label in factors:
        for pct in (-0.3, 0.3):
            p = dict(DEFAULT)
            p[key] = DEFAULT[key] * (1 + pct)
            r = run(p, SCHEMES["S2"], z)
            tor.append({"factor": key, "label": label, "pct": pct,
                        "exp_cost": round(float(r["cost"].mean()), 2),
                        "p_acc": round(float((r["n_acc"] > 0).mean()), 4),
                        "d_cost": round(float(r["cost"].mean()) - base, 2)})
    results["sensitivity"] = {"base_cost": round(base, 2),
                              "base_p_acc": round(float((by_scheme["S2"]["n_acc"] > 0).mean()), 4),
                              "runs": tor}

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    for s in rows:
        print(f"{s['scheme']}: P={s['p_acc']:.3f} [{s['p_acc_ci'][0]:.3f},{s['p_acc_ci'][1]:.3f}]  "
              f"E[C]={s['exp_cost']:.2f} [{s['exp_cost_ci'][0]:.2f},{s['exp_cost_ci'][1]:.2f}]  "
              f"P95={s['p95_cost']:.1f} [{s['p95_ci'][0]:.1f},{s['p95_ci'][1]:.1f}]  "
              f"P99={s['p99_cost']:.1f}  H={s['h_end']:.2f}")
    print("扫描:", [(s["I"], round(s["exp_cost"], 1), round(s["p_acc"] * 100, 1)) for s in sweep])
    print("敏感性基准:", results["sensitivity"]["base_cost"])


if __name__ == "__main__":
    main()
