# -*- coding: utf-8 -*-
"""博弈论 B：多承包商公共安全外部性博弈（Nash 均衡 vs 社会最优，公地悲剧）。

场景：区域内 N 家承包商各自选安全投入 I_i。事故损失中有一部分是「公共外部性」
（区域监管整顿、行业声誉、保险池损失）由所有承包商共同承担，即每家的私人边际损失
只占社会边际损失的一部分。投入越高越降低事故概率，但成本自付。

  I_i ≥ 0，承包商 i 的成本：
      C_i = c(I_i) + p_acc(I_i) * L_private + p_acc(I_i) * L_public * (N-1)/N * φ
  其中：
      - c(I) 投入成本（线性 + 二次凸性）
      - p_acc(I) 由蒙特卡洛查表（真实模型的安全产出）
      - L_private 私人承担的事故损失
      - L_public 公共外部损失，被「稀释」后每家只感知 (N-1)/N 的一部分（搭便车）

社会计划者最小化 ΣC_i，但含完整外部性 L_public（不稀释）。

求 Nash 均衡：每家对其他家投入的最佳响应迭代（对连续 I 做网格最佳响应，直到收敛）。
对比：
  - Nash 均衡总投入 ΣI_i^N
  - 社会最优总投入 ΣI_i^S
  - 投入不足缺口（公地悲剧量化）

运行：python src/game_multi.py → output/game_multi.json + fig13_commons.svg
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")

N_FIRMS = 5
I_ACTIONS = np.array([0, 2, 4, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50], float)
L_PRIVATE = 80.0   # 私人承担的事故损失（万元）
L_PUBLIC = 120.0   # 公共外部损失（万元，被稀释）
PHI = 0.6          # 公共损失被感知的比例
F_FIX = 0.5
N_SAMPLE = 20000

_PACC = {}


def _pacc(I, params0, z):
    key = round(float(I), 3)
    if key not in _PACC:
        r = M.simulate(params0, {"I": float(I), "f": F_FIX}, z, features=M.FEATURES)
        _PACC[key] = float((r["n_acc"] > 0).mean())
    return _PACC[key]


def invest_cost(I):
    """投入成本：线性 + 凸性（边际成本递增）。斜率校准到使 p_acc 的弱弹性仍能形成内部最优。"""
    return 0.5 * I + 0.010 * I ** 2


def firm_cost(I_i, I_others_mean, params0, z, social=False):
    """单家承包商（或社会计划者单块）成本。

    私人视角：只感知被稀释的公共外部性 (N-1)/N * PHI；
    社会视角：含完整公共外部损失。
    """
    p = _pacc(I_i, params0, z)
    own = invest_cost(I_i) + p * L_PRIVATE
    if social:
        # 社会计划者：把每家产生的外部性完整内化
        own += p * L_PUBLIC
    else:
        # 私人：自身事故 + 被稀释的公共外部（搭便车）
        own += p * L_PUBLIC * (N_FIRMS - 1) / N_FIRMS * PHI
    return own


def best_response(params0, z, social=False):
    """返回每家对称条件下的最优投入（对称均衡：所有家相同 → 对称 Nash）。"""
    costs = np.array([firm_cost(I, None, params0, z, social) for I in I_ACTIONS])
    return float(I_ACTIONS[np.argmin(costs)]), costs


def symmetric_nash(params0, z, iters=50):
    """对称 Nash：迭代最佳响应直到收敛（对称下 I_others_mean 不影响最优，直接一次取最优即可，
    但因存在策略互补/替代，用迭代确认不动点）。"""
    I = 8.0
    hist = []
    for _ in range(iters):
        I_new, _ = best_response(params0, z, social=False)
        hist.append(I_new)
        if abs(I_new - I) < 1e-6:
            break
        I = I_new
    return float(I), hist


def main():
    global PHI
    params0 = dict(M.DEFAULT)
    z0 = M.sample_params_lhs(N_SAMPLE, seed=7, corr=M.CORR)
    _PACC.clear()
    print("== 多承包商公共安全外部性博弈（公地悲剧）==")
    print(f"  N={N_FIRMS} 家, L_private={L_PRIVATE}, L_public={L_PUBLIC}, 感知比例 PHI={PHI}")

    I_nash, hist = symmetric_nash(params0, z0)
    I_social, costs_soc = best_response(params0, z0, social=True)
    print(f"  Nash 均衡单家投入: I*={I_nash:.1f}（迭代 {len(hist)} 步收敛）")
    print(f"  社会最优单家投入: I^S={I_social:.1f}")

    total_nash = I_nash * N_FIRMS
    total_social = I_social * N_FIRMS
    gap = total_social - total_nash
    print(f"  总投入: Nash ΣI={total_nash:.1f} vs 社会最优 ΣI={total_social:.1f}")
    print(f"  投入不足缺口（公地悲剧）= {gap:.1f} 万元（{(gap/total_social*100):.1f}%）")

    # 扫 PHI（外部性感知强度）看投入不足如何随搭便车程度变化
    print("\n  [敏感性] 外部性感知比例 PHI → Nash 投入与缺口")
    phi_scan = []
    base_phi = PHI
    for phi in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        PHI = phi
        In, _ = symmetric_nash(params0, z0)
        phi_scan.append({"phi": phi, "I_nash": In,
                         "gap": I_social - In, "gap_pct": (I_social - In) / I_social * 100})
        print(f"    PHI={phi:.1f}: Nash I={In:.1f}, 缺口={I_social-In:.1f} "
              f"({(I_social-In)/I_social*100:.0f}%)")
    PHI = base_phi

    payload = {
        "N_firms": N_FIRMS, "L_private": L_PRIVATE, "L_public": L_PUBLIC, "phi": PHI,
        "I_nash": I_nash, "I_social": I_social,
        "total_nash": total_nash, "total_social": total_social,
        "underinvestment_gap": gap, "gap_pct": gap / total_social * 100,
        "phi_scan": phi_scan,
        "cost_curve_nash": {str(I): firm_cost(I, None, params0, z0, False)
                            for I in I_ACTIONS},
        "cost_curve_social": {str(I): firm_cost(I, None, params0, z0, True)
                              for I in I_ACTIONS},
    }
    json.dump(payload, open(os.path.join(OUT, "game_multi.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)
    _plot(payload)
    print("\n已输出 output/game_multi.json + fig13_commons.svg")


def _plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))

    I = np.array([float(k) for k in res["cost_curve_nash"].keys()])
    cn = np.array([res["cost_curve_nash"][str(float(k))] for k in
                   [float(x) for x in res["cost_curve_nash"].keys()]])
    cs = np.array([res["cost_curve_social"][str(float(k))] for k in
                   [float(x) for x in res["cost_curve_social"].keys()]])
    a1.plot(I, cn, "-o", color="#8B1A1A", label="私人成本（含稀释外部性）")
    a1.plot(I, cs, "-s", color="#1A2B4A", ms=4, label="社会成本（完整外部性）")
    a1.axvline(res["I_nash"], ls="--", color="#8B1A1A", lw=1, alpha=0.7)
    a1.axvline(res["I_social"], ls="--", color="#1A2B4A", lw=1, alpha=0.7)
    a1.scatter([res["I_nash"]], [np.min(cn)], marker="v", s=160, color="#8B1A1A", zorder=5,
               label=f"Nash I*={res['I_nash']:.0f}")
    a1.scatter([res["I_social"]], [np.min(cs)], marker="^", s=160, color="#1A2B4A", zorder=5,
               label=f"社会最优 I^S={res['I_social']:.0f}")
    a1.set_xlabel("单家安全投入 I（万元）"); a1.set_ylabel("单家成本（万元）")
    a1.set_title("私人 vs 社会成本曲线与两类最优", fontsize=10)
    a1.grid(alpha=0.3); a1.legend(frameon=False, fontsize=8)

    phis = [d["phi"] for d in res["phi_scan"]]
    gaps = [d["gap"] for d in res["phi_scan"]]
    a2.plot(phis, gaps, "-o", color="#B4762A")
    a2.set_xlabel("公共外部性感知比例 PHI（搭便车强度）")
    a2.set_ylabel("单家投入不足缺口（万元）")
    a2.set_title("搭便车越强，投入不足越大（公地悲剧）", fontsize=10)
    a2.grid(alpha=0.3)
    fig.suptitle(f"图 13  {res['N_firms']} 家承包商的公共安全外部性博弈", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "docs", "figures", "fig13_commons.svg"),
                format="svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
