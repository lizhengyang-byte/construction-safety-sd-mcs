# -*- coding: utf-8 -*-
"""博弈论 A：监管者–承包商 Stackelberg 主从博弈 + 威慑门槛分析（道德风险）。

模型洞察（由蒙特卡洛弹性实测得出，非人为设定）：
  本模型低投入区 p_acc(I) 近似平坦（I=4→12 时 0.36→0.35，噪声主导），
  安全投入对事故概率的边际效力很弱。因此承包商是否值得「诚实投入」高度依赖
  一次事故的综合损失/处罚基数 L_acc 的大小——L_acc 太小时，偷工省下的确定资本
  远超概率性事故损失，偷工是理性的。

据此把 L_acc 当作监管可调的政策杠杆，做两类分析：

  (A) 威慑门槛扫描：给定检查强度 e，扫 L_acc，找承包商会自愿诚实投入(m→0)的门槛
      L_acc*。给出「处罚/赔付基数需要多大才有效」的政策结论。
  (B) Stackelberg 均衡：对给定 L_acc，上层监管者选最优检查强度 e*，
      承包商做最优响应；对比社会最优，量化效率损失。

运行：python src/game.py → output/game.json + fig12_game.svg
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")

I_GRID = np.array([6, 8, 10, 12, 16, 20, 25, 30], float)
M_GRID = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], float)
E_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], float)
L_GRID = np.array([50, 80, 100, 150, 200, 300, 400, 600], float)  # 事故综合损失基数
F_FIX = 0.5
N_SAMPLE = 30000

# 缓存 p_acc(I,m)——与 e / L_acc 无关，只依赖实际投入，预计算加速
_PACC_CACHE = {}


def _pacc(I_actual, params0, z):
    key = round(float(I_actual), 3)
    if key not in _PACC_CACHE:
        r = M.simulate(params0, {"I": float(I_actual), "f": F_FIX}, z, features=M.FEATURES)
        _PACC_CACHE[key] = float((r["n_acc"] > 0).mean())
    return _PACC_CACHE[key]


def contractor_obj(I, m, p_acc, e, L_acc, inv_cost, social=False):
    """承包商自身成本 / 社会成本。

    期望事故损失 = p_acc × L_acc。偷工省 (1-m) 比例的资本 I，但抬高 p_acc 与查处处罚。
    """
    pen_base, pen_exp = 10.0, 1.5
    pre_base = 3.0
    I_actual = I * (1.0 - m)
    capital = I_actual + inv_cost
    exp_loss = p_acc * L_acc
    detect = min(0.9, 0.30 * e) * (0.5 + 0.5 * m)
    exp_penalty = pen_base * (e ** pen_exp) * detect
    share = 1.6 if social else 1.0
    own = capital + exp_loss * share + exp_penalty + pre_base * m - pre_base * (1 - m)
    return float(own)


def contractor_best_response(params0, z, e, L_acc, social=False, allow_cheat=True):
    """给定 (e, L_acc)，承包商最优 (I, m)。"""
    inv_cost = M.DEFAULT["c_f"] * F_FIX * M.DEFAULT["T"]
    m_set = M_GRID if allow_cheat else np.array([0.0])
    best = None
    for I in I_GRID:
        for m in m_set:
            p = _pacc(I * (1 - m), params0, z)
            obj = contractor_obj(I, m, p, e, L_acc, inv_cost, social)
            if best is None or obj < best["obj"]:
                best = {"obj": obj, "I": float(I), "m": float(m),
                        "I_actual": float(I * (1 - m)), "p_acc": p,
                        "social_obj": contractor_obj(I, m, p, e, L_acc, inv_cost, True),
                        "e": float(e), "L_acc": float(L_acc)}
    return best


def main():
    params0 = dict(M.DEFAULT)
    z0 = M.sample_params_lhs(N_SAMPLE, seed=7, corr=M.CORR)
    _PACC_CACHE.clear()
    print("== Stackelberg 主从博弈 + 威慑门槛分析 ==")

    # ---------- Part A: 威慑门槛扫描（固定 e=0 无监管，看 L_acc 多高才能让 m→0）----------
    print("\n[A] 威慑门槛：无监管 e=0 下，L_acc 对承包商偷工行为的影响")
    deter = []
    for L in L_GRID:
        br = contractor_best_response(params0, z0, e=0.0, L_acc=float(L))
        deter.append({"L_acc": float(L), "I": br["I"], "m": br["m"],
                      "p_acc": br["p_acc"], "obj": br["obj"]})
        print(f"    L_acc={L:>4.0f}: 承包 I={br['I']:.0f} 真实投入={br['I_actual']:.0f} "
              f"m={br['m']:.1f} p_acc={br['p_acc']:.3f}")
    # 门槛：m 从此档起始终≈0 的最小 L_acc（稳健判定，避免噪声抖动）
    L_star = None
    for i, d in enumerate(deter):
        if all(x["m"] <= 1e-9 for x in deter[i:]):
            L_star = d["L_acc"]; break
    print(f"  → 威慑门槛 L_acc* ≈ {L_star} 万元（低于此值承包商理性偷工）")

    # ---------- Part B: Stackelberg 均衡（取一个代表性 L_acc）----------
    L_ref = 300.0
    print(f"\n[B] Stackelberg 均衡（L_acc={L_ref:.0f}）")
    resp_free = contractor_best_response(params0, z0, e=0.0, L_acc=L_ref)
    # 社会最优：社会计划者按社会成本(含外部性)全局最优，允许其权衡投入与偷工
    resp_social = contractor_best_response(params0, z0, e=0.0, L_acc=L_ref,
                                           social=True, allow_cheat=True)
    upper = []
    c_e = 2.0
    for e in E_GRID:
        br = contractor_best_response(params0, z0, e=float(e), L_acc=L_ref)
        robj = br["social_obj"] + c_e * e
        upper.append({"e": float(e), "reg_obj": robj, "I": br["I"], "m": br["m"],
                      "contractor_obj": br["obj"], "p_acc": br["p_acc"]})
        print(f"    e={e:.1f}: 承包 I={br['I']:.0f} m={br['m']:.1f} 自身={br['obj']:.1f} "
              f"| 社会成本={robj:.1f}")
    best_e = min(upper, key=lambda d: d["reg_obj"])
    print(f"  均衡 e*={best_e['e']:.1f}, I*={best_e['I']:.0f}, m*={best_e['m']:.1f}")
    eff_loss = best_e["reg_obj"] - resp_social["social_obj"]
    print(f"  效率损失(均衡 - 社会最优) = {eff_loss:.1f} 万元")

    payload = {
        "deterrence_threshold": {
            "scan": deter, "L_star": L_star,
            "note": "L_acc 低于门槛时，承包商理性选择偷工；本模型低投入区 p_acc 弹性弱是门槛偏高的根源",
        },
        "stackelberg": {
            "L_acc_ref": L_ref,
            "free_market": resp_free,
            "social_optimum": resp_social,
            "equilibrium": best_e,
            "upper_level": upper,
            "efficiency_loss": float(eff_loss),
        },
    }
    json.dump(payload, open(os.path.join(OUT, "game.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)
    _plot(deter, upper, best_e, L_ref)
    print("\n已输出 output/game.json + fig12_game.svg")


def _plot(deter, upper, eq, L_ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    # 左：威慑门槛
    Ls = [d["L_acc"] for d in deter]
    a1.plot(Ls, [d["m"] for d in deter], "-o", color="#8B1A1A", label="偷工比例 m*")
    a1.set_xlabel("事故综合损失/处罚基数 L_acc（万元）")
    a1.set_ylabel("承包商最优偷工比例 m*", color="#8B1A1A")
    a1.set_title("威慑门槛：L_acc 需多高才让承包商自愿诚实", fontsize=10)
    a1.grid(alpha=0.3)
    ax1b = a1.twinx()
    ax1b.plot(Ls, [d["p_acc"] for d in deter], "--s", color="#1A2B4A", ms=4, label="事故概率")
    ax1b.set_ylabel("事故概率 p_acc", color="#1A2B4A")
    lines1, lab1 = a1.get_legend_handles_labels(); lines2, lab2 = ax1b.get_legend_handles_labels()
    a1.legend(lines1 + lines2, lab1 + lab2, frameon=False, fontsize=8, loc="upper right")
    # 右：上层均衡
    es = [u["e"] for u in upper]
    a2.plot(es, [u["reg_obj"] for u in upper], "-o", color="#1A2B4A", label="社会总成本")
    a2.plot(es, [u["contractor_obj"] for u in upper], "-s", color="#8B1A1A", ms=4, label="承包商成本")
    a2.axvline(eq["e"], ls="--", color="#B4762A", lw=1.2)
    a2.scatter([eq["e"]], [eq["reg_obj"]], marker="*", s=220, color="#B4762A", zorder=5,
               label=f"均衡 e*={eq['e']:.1f}")
    a2.set_xlabel("监管检查强度 e"); a2.set_ylabel("成本（万元）")
    a2.set_title(f"Stackelberg 均衡（L_acc={L_ref:.0f}）", fontsize=10)
    a2.grid(alpha=0.3); a2.legend(frameon=False, fontsize=8)
    fig.suptitle("图 12  监管者–承包商博弈：威慑门槛与主从均衡", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "docs", "figures", "fig12_game.svg"),
                format="svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
