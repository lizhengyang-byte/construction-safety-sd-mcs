# -*- coding: utf-8 -*-
"""生成 v2 -> v3 改进前后的对比矢量图（读取 output/comparison.json 与 validate.json）。

输出：
  docs/figures/cmp1_schemes.svg    三方案 事故概率/期望成本/P95/CVaR 的 v2 vs v3 分组柱状
  docs/figures/cmp2_sobol.svg      OAT 单因素 vs Sobol 总效应 的敏感性排序对比
  docs/figures/cmp3_uncertainty.svg 偶然 vs 认知 不确定性分解
运行：python src/compare_figs.py
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIG, exist_ok=True)
R = json.load(open(os.path.join(ROOT, "output", "comparison.json"), encoding="utf-8"))

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"
NAVY, MID, LIGHT, OA = "#1A2B4A", "#6C7A92", "#C3CAD6", "#B4762A"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)


def fig_schemes():
    schemes = R["schemes"]
    labels = [f"{s['scheme']}" for s in schemes]
    metrics = [("p_acc", "事故概率（%）", 100.0), ("exp_cost", "期望总成本（万元）", 1.0),
               ("p95_cost", "P95 成本（万元）", 1.0), ("cvar95", "CVaR95（万元）", 1.0)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    x = np.arange(len(labels))
    w = 0.36
    for ax, (key, title, scale) in zip(axes, metrics):
        v2 = [s["v2"][key] * scale for s in schemes]
        v3 = [s["v3"][key] * scale for s in schemes]
        ax.bar(x - w/2, v2, w, color=LIGHT, label="改进前 v2")
        ax.bar(x + w/2, v3, w, color=NAVY, label="改进后 v3")
        for xi, val in zip(x - w/2, v2):
            ax.text(xi, val, f"{val:.0f}" if scale == 1 else f"{val:.1f}", ha="center", va="bottom", fontsize=7)
        for xi, val in zip(x + w/2, v3):
            ax.text(xi, val, f"{val:.0f}" if scale == 1 else f"{val:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("图 A  三方案核心指标：改进前（v2）vs 改进后（v3）", fontsize=12)
    save(fig, "cmp1_schemes")


def fig_sobol():
    # OAT 最大 |ΔC| 排序 vs Sobol 总效应 ST 排序
    runs = R["sensitivity"]["runs"]
    oat = {}
    for r in runs:
        oat[r["factor"]] = max(oat.get(r["factor"], 0), abs(r["d_cost"]))
    sobol = R.get("sobol", {})
    names = sobol.get("names", [])
    ST = sobol.get("ST", [])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    # OAT
    items = sorted(oat.items(), key=lambda kv: kv[1])
    a1.barh([k for k, _ in items], [v for _, v in items], color=MID)
    a1.set_title("改进前：OAT 单因素 |ΔC|（万元）", fontsize=9)
    a1.grid(axis="x", alpha=0.3)
    # Sobol
    if names:
        order = np.argsort(ST)
        a2.barh([names[i] for i in order], [ST[i] for i in order], color=NAVY)
        a2.set_title("改进后：Sobol 总效应 ST（含交互）", fontsize=9)
        a2.grid(axis="x", alpha=0.3)
    fig.suptitle("图 B  敏感性分析：单因素 OAT → 全局 Sobol", fontsize=12)
    save(fig, "cmp2_sobol")


def fig_uncertainty():
    tl = R["two_level"]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    parts = [tl["aleatory_var"], tl["epistemic_var"]]
    a_pct = tl["aleatory_share"] * 100
    labels = [f"偶然不确定性  {a_pct:.1f}%", f"认知不确定性  {100-a_pct:.1f}%"]
    ax.pie(parts, colors=[LIGHT, NAVY], explode=[0.0, 0.06],
           startangle=90, textprops={"fontsize": 9})
    ax.legend(labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
              frameon=False, fontsize=9)
    ax.set_title("图 C  成本方差分解：偶然 vs 认知不确定性（S2）", fontsize=11)
    save(fig, "cmp3_uncertainty")


if __name__ == "__main__":
    fig_schemes()
    fig_sobol()
    fig_uncertainty()
    print("已生成对比图：cmp1_schemes / cmp2_sobol / cmp3_uncertainty")
