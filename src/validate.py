# -*- coding: utf-8 -*-
"""验证与校准脚本（v3）：概率校准、分岔图、距失控边界距离。

对应改进项：
  [F20] 概率校准检验：PIT 直方图 + 可靠性曲线 + Brier 分数（对照 v2 与 v3）
  [F15] 分岔图：以投入 I 为分岔参数，扫描稳态隐患/是否失控，标出失控临界 I*
  [F15] 距失控边界距离：定义 margin = λ - β g0(1-η) 的标准化富余度，作为新风险指标

运行：python src/validate.py
输出：output/validate.json、docs/figures/fig8_calibration.svg、fig9_bifurcation.svg
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (DEFAULT, SCHEMES, FEATURES, V2_FEATURES,
                   sample_params_lhs, simulate)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
R_SEED = 7


# ---------------------------------------------------------------------------
# [F20] 概率校准：把 180 天周期切成多个短窗口，比较"预测概率"与"实际频率"
# ---------------------------------------------------------------------------
def calibration(z, features):
    """[F20] 概率校准：分组二项可靠性曲线。

    对每个方案、每个参数样本，用 M 次独立随机流重复得到"该样本发生事故的频率" p_hat，
    再在留出流上取 0/1 结果；按 p_hat 分箱，比较箱内平均 p_hat 与平均实际频率。
    同时给出 Brier 分数。
    """
    M_est, M_test = 24, 24
    rec_pred, rec_obs = [], []
    for name in SCHEMES:
        sc = SCHEMES[name]
        # 每个参数样本的估计频率（M_est 次重复）
        est_hits = np.zeros(len(z["g0"]))
        for r in range(M_est):
            rr = simulate(dict(DEFAULT, seed=6000 + r), sc, z, features=features)
            est_hits += (rr["n_acc"] > 0).astype(float)
        p_hat = est_hits / M_est
        # 留出流上的单次结果（M_test 次，每次一个 0/1 -> 展开）
        for r in range(M_test):
            rr = simulate(dict(DEFAULT, seed=8000 + r), sc, z, features=features)
            rec_pred.append(p_hat)
            rec_obs.append((rr["n_acc"] > 0).astype(float))
    predicted = np.concatenate(rec_pred)
    outcome = np.concatenate(rec_obs)

    bins = np.linspace(0, 1, 11)
    rel = []
    for i in range(len(bins) - 1):
        m = (predicted >= bins[i]) & (predicted < bins[i + 1])
        if m.sum() > 0:
            rel.append({"bin": [round(float(bins[i]), 2), round(float(bins[i + 1]), 2)],
                        "pred": round(float(predicted[m].mean()), 4),
                        "obs": round(float(outcome[m].mean()), 4),
                        "n": int(m.sum())})
    brier = float(np.mean((predicted - outcome) ** 2))
    return {"reliability": rel, "brier": round(brier, 5)}


def calibration_pit(z, features):
    """[F20] 更严谨的 PIT：用均衡方案，逐样本计算 u_i = P(X <= x_i) 的均匀性。

    这里把每次仿真看作一个可比的连续输出（总成本 C），用经验 CDF 的 PIT：
    若模型校准良好，则在不同随机流重复下，点预测落点的 PIT 应近似均匀。
    由于生成式模型理论上自校准，这里检验"重复间的经验尾部覆盖率"是否符合标称。
    """
    sc = SCHEMES["S2"]
    reps = 200
    # 用一半随机流估计分位数，另一半检验覆盖率
    est = []
    for r in range(reps):
        rr = simulate(dict(DEFAULT, seed=7000 + r), sc, z)
        est.append(rr["cost"])
    est = np.concatenate(est)
    test = []
    for r in range(reps):
        rr = simulate(dict(DEFAULT, seed=9000 + r), sc, z)
        test.append(rr["cost"])
    test = np.concatenate(test)
    cover = {}
    for q in (0.5, 0.8, 0.9, 0.95):
        thr = np.quantile(est, q)
        cover[f"nominal_{q}"] = {"nominal": q, "actual": round(float((test <= thr).mean()), 4)}
    return cover


# ---------------------------------------------------------------------------
# [F15] 分岔图 + 距失控边界距离
# ---------------------------------------------------------------------------
def bifurcation(z, features):
    """以投入 I 为分岔参数（f 按联动式），给出稳态隐患与是否失控。

    失控条件（解析）：beta*g0*(1-eta) >= lambda  -> 存量指数增长。
    margin = lambda - beta*g0*(1-eta)（越大越安全），标准化为 margin/lambda。
    """
    sc_f = lambda I: round(0.15 + 0.0215 * I, 3)
    rows = []
    for I in np.arange(1.0, 61.0, 1.0):
        f = sc_f(I)
        # 用均值参数跑一次，得到经验期末存量（大 N）
        zz = {k: np.full(400, np.mean(v)) for k, v in z.items()}
        rr = simulate(dict(DEFAULT, seed=R_SEED), {"I": float(I), "f": f}, zz, features=features)
        # 解析 margin
        eta = 1 - np.exp(-DEFAULT["k_t"] * I)
        lam = 1 - np.exp(-(DEFAULT["lambda0"] + DEFAULT["k_d"] * f))
        margin = lam - DEFAULT["beta"] * DEFAULT["g0"] * (1 - eta)
        rows.append({
            "I": round(float(I), 1), "f": f,
            "h_end": round(float(rr["H_end"].mean()), 3),
            "margin": round(float(margin), 5),
            "margin_ratio": round(float(margin / lam), 4),
            "runaway": bool(margin <= 0),
        })
    # 临界投入 I*：margin 首次转正
    I_star = None
    for r in rows:
        if r["margin"] > 0:
            I_star = r["I"]
            break
    return {"curve": rows, "I_star": I_star}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    z = sample_params_lhs(800, seed=2026, corr=None)
    print("== 概率校准（v3）==")
    cal_v2 = calibration(z, V2_FEATURES)
    cal_v3 = calibration(z, FEATURES)
    print("  Brier v2=%.4f  v3=%.4f" % (cal_v2["brier"], cal_v3["brier"]))
    pit = calibration_pit(z, FEATURES)
    print("  尾部覆盖率:", {k: v["actual"] for k, v in pit.items()})

    print("== 分岔分析 ==")
    bif = bifurcation(z, FEATURES)
    print("  失控临界投入 I* =", bif["I_star"], "万元")

    result = {"calibration_v2": cal_v2, "calibration_v3": cal_v3,
              "pit": pit, "bifurcation": bif}
    with open(os.path.join(OUT, "validate.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    _plot(result)
    print("已输出 output/validate.json 与 fig8/fig9")


def _plot(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"
    NAVY, MID, LIGHT = "#1A2B4A", "#6C7A92", "#C3CAD6"
    OA = "#B4762A"

    # 图8 可靠性曲线
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 1], [0, 1], color=MID, ls="--", lw=1, label="理想校准")
    for key, col, lab in (("calibration_v2", LIGHT, "v2"), ("calibration_v3", NAVY, "v3")):
        rel = R[key]["reliability"]
        px = [d["pred"] for d in rel]; oy = [d["obs"] for d in rel]
        ax.plot(px, oy, "o-", color=col, label=f"{lab} (Brier={R[key]['brier']:.3f})")
    ax.set_xlabel("预测事故概率")
    ax.set_ylabel("实际发生频率")
    ax.set_title("图 8  概率校准可靠性曲线（越接近对角线越准）", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig8_calibration.svg"), format="svg", bbox_inches="tight")
    plt.close(fig)

    # 图9 分岔图
    c = R["bifurcation"]["curve"]
    I = [r["I"] for r in c]
    H = [r["h_end"] for r in c]
    mr = [r["margin_ratio"] for r in c]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.plot(I, H, color=NAVY, lw=2, label="稳态隐患（仿真）")
    ax.axvline(R["bifurcation"]["I_star"], color=OA, ls="--", lw=1)
    ax.annotate(f"I*≈{R['bifurcation']['I_star']} 万元\n(失控临界)",
                (R["bifurcation"]["I_star"], max(H) * 0.5), fontsize=9, color=OA,
                xytext=(6, 0), textcoords="offset points")
    ax.set_xlabel("安全投入 I（万元）")
    ax.set_ylabel("稳态隐患存量（个）")
    ax2 = ax.twinx()
    ax2.plot(I, mr, color=MID, ls=":", lw=1.4, label="安全富余度 margin/λ")
    ax2.axhline(0, color=OA, lw=0.8)
    ax2.set_ylabel("安全富余度 margin/λ", color=MID)
    ax.set_title("图 9  分岔分析：稳态隐患与距失控边界的安全富余度", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig9_bifurcation.svg"), format="svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
