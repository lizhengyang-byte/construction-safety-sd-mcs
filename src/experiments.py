# -*- coding: utf-8 -*-
"""实验脚本：三方案对比、时序演化带、收敛性、投入扫描、OAT 敏感性。

运行：python src/experiments.py
输出：output/results.json、output/*.csv
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import qmc

from model import DEFAULT, SCHEMES, UNCERTAIN, simulate, summarize

N = 6000
OUT = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUT, exist_ok=True)


def lhs(n, seed=2026):
    sampler = qmc.LatinHypercube(d=len(UNCERTAIN), seed=seed)
    X = qmc.scale(sampler.random(n),
                  np.array([v[0] for v in UNCERTAIN.values()]),
                  np.array([v[1] for v in UNCERTAIN.values()]))
    return {k: X[:, i] for i, k in enumerate(UNCERTAIN)}


def run_scheme(name, params, n=N, seed=11):
    z = lhs(n, seed)
    res = simulate(dict(params, seed=seed + 1), SCHEMES[name], z)
    s = summarize(res)
    s.update({"scheme": name, "scheme_name": SCHEMES[name]["name"],
              "I": SCHEMES[name]["I"], "f": SCHEMES[name]["f"]})
    return s, res


def main():
    results = {"meta": {"N": N, "T": DEFAULT["T"],
                        "uncertain": {k: list(v) for k, v in UNCERTAIN.items()}}}

    # 1. 三方案对比
    rows, dists = [], {}
    for name in SCHEMES:
        s, res = run_scheme(name, DEFAULT)
        rows.append(s)
        dists[name] = res["cost"]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "scheme_summary.csv"), index=False, encoding="utf-8-sig")
    results["schemes"] = df.round(4).to_dict(orient="records")

    qs = np.linspace(0.01, 0.99, 41)
    results["cost_dist"] = {name: {
        "quantile": qs.round(3).tolist(),
        "value": [round(float(np.quantile(d, q)), 2) for q in qs],
    } for name, d in dists.items()}

    # 2. 隐患存量时序（均值与 5%-95% 带）
    z = lhs(500, 33)
    p = dict(DEFAULT, seed=34)
    ser = {}
    for name in SCHEMES:
        S = simulate(p, SCHEMES[name], z, collect_series=True)["series"]
        ser[name] = {
            "t": list(range(DEFAULT["T"] + 1)),
            "mean": [round(float(v), 3) for v in S.mean(axis=1)],
            "p5": [round(float(v), 3) for v in np.quantile(S, 0.05, axis=1)],
            "p95": [round(float(v), 3) for v in np.quantile(S, 0.95, axis=1)],
        }
    results["series"] = ser

    # 3. 收敛性（均衡方案）
    res = simulate(dict(DEFAULT, seed=56), SCHEMES["S2"], lhs(N, 55))
    pacc = np.cumsum(res["n_acc"] > 0) / np.arange(1, N + 1)
    cost = np.cumsum(res["cost"]) / np.arange(1, N + 1)
    idx = list(range(50, N + 1, 50))
    results["convergence"] = {
        "n": idx,
        "p_acc": [round(float(pacc[i - 1]), 4) for i in idx],
        "exp_cost": [round(float(cost[i - 1]), 2) for i in idx],
    }

    # 4. 安全投入扫描
    z = lhs(3000, 77)
    sweep = []
    for I in [5, 10, 15, 20, 25, 30, 35, 40, 45]:
        f = 0.15 + 0.0215 * I
        s = summarize(simulate(dict(DEFAULT, seed=78), {"I": float(I), "f": float(f)}, z))
        sweep.append({**s, "I": I, "f": round(f, 3)})
    results["sweep"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                        for r in sweep]
    pd.DataFrame(sweep).to_csv(os.path.join(OUT, "investment_sweep.csv"), index=False, encoding="utf-8-sig")

    # 5. OAT 单因素敏感性（均衡方案，±30%）
    factors = [("g0", "隐患生成率 g0"), ("beta", "正反馈系数 beta"), ("lambda0", "基础检出率 lambda0"),
               ("k_d", "巡检边际效应 k_d"), ("theta", "事故触发系数 theta"), ("w_rate", "扰动频次 w")]
    z = lhs(3000, 99)
    base = simulate(dict(DEFAULT, seed=100), SCHEMES["S2"], z)
    base_cost = float(base["cost"].mean())
    base_pa = float((base["n_acc"] > 0).mean())
    tor = []
    for key, label in factors:
        for pct in (-0.3, 0.3):
            p = dict(DEFAULT, seed=100)
            p[key] = DEFAULT[key] * (1 + pct)
            r = simulate(p, SCHEMES["S2"], z)
            tor.append({"factor": key, "label": label, "pct": pct,
                        "exp_cost": round(float(r["cost"].mean()), 2),
                        "p_acc": round(float((r["n_acc"] > 0).mean()), 4),
                        "d_cost": round(float(r["cost"].mean()) - base_cost, 2)})
    results["sensitivity"] = {"base_cost": round(base_cost, 2), "base_p_acc": round(base_pa, 4),
                              "runs": tor}

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    print(df[["scheme", "p_acc", "exp_cost", "p95_cost", "exp_delay", "h_end"]].to_string(index=False))


if __name__ == "__main__":
    main()
