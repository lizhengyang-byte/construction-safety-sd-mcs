# -*- coding: utf-8 -*-
"""生成论文七张矢量图（SVG）：读取 output/results.json，输出 docs/figures/*.svg。

与飞书文档中的图表同源（同一份 results.json、同一套参数与种子），仅渲染后端不同：
飞书版为内联 SVG / HTML5 block，本脚本用 matplotlib 的 SVG 后端输出独立文件，便于入库与引用。

运行：
    python src/experiments.py   # 先生成 output/results.json
    python src/render_svg.py    # 再生成 docs/figures/*.svg
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.environ.get("FIGDIR", os.path.join(ROOT, "docs", "figures"))
os.makedirs(FIG, exist_ok=True)
RES = json.load(open(os.path.join(ROOT, "output", "results.json"), encoding="utf-8"))

NAVY, MID, LIGHT = "#1A2B4A", "#6C7A92", "#C3CAD6"
COL = {"S1": LIGHT, "S2": NAVY, "S3": MID}

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"      # SVG 内保留文本，便于检索与编辑


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)


def fig_structure():
    """图 1：存量流量结构与反馈回路。"""
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#FFFFFF", ec=MID, fs=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(p1, p2, text="", rad=0.0, ls="-"):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12,
                                     color=MID if ls == "-" else NAVY, lw=1.3, ls=ls,
                                     connectionstyle=f"arc3,rad={rad}"))

    box(4.2, 2.0, 2.0, 1.0, "未整改隐患\n存量 H", fc="#F2F4F8", ec=NAVY)
    box(4.2, 3.9, 2.0, 0.7, "隐患生成 G")
    box(7.4, 2.0, 2.0, 0.9, "巡检整改 R")
    box(4.2, 0.4, 2.0, 0.7, "事故 A（只计数）", ec=LIGHT)

    arrow((5.2, 3.88), (5.2, 3.05), "")
    ax.text(5.32, 3.42, "G = g0(1+βH)(1−η)", fontsize=9, color=NAVY, ha="left", va="center")
    arrow((6.2, 2.5), (7.4, 2.5), "")
    ax.text(6.3, 2.62, "R = λH", fontsize=9, color=NAVY, ha="left", va="bottom")
    arrow((5.2, 2.0), (5.2, 1.15), "")
    ax.text(5.32, 1.55, "A ~ Poisson(θH)", fontsize=9, color=MID, ha="left", va="center")
    arrow((6.8, 2.95), (6.35, 3.8), "", rad=-0.3, ls="--")
    ax.text(6.9, 3.45, "β 正反馈", fontsize=9, color=NAVY, ha="left", va="center")

    box(0.4, 3.95, 1.9, 0.65, "安全投入 I", ec=LIGHT)
    box(0.4, 2.55, 1.9, 0.65, "巡检频次 f", ec=LIGHT)
    arrow((2.3, 4.28), (4.2, 4.22), "")
    ax.text(2.38, 4.42, "η = 1−exp(−k_t I)", fontsize=8.5, color=NAVY, ha="left", va="bottom")
    arrow((2.3, 2.55), (4.2, 2.55), "")
    ax.text(2.38, 2.42, "λ = 1−exp(−(λ0+k_d f))", fontsize=8.5, color=NAVY, ha="left", va="top")

    ax.set_title("图 1  存量流量结构与反馈回路（β 正反馈、η 与 λ 两条负反馈）", fontsize=11)
    save(fig, "fig1_structure")


def fig_series():
    s = RES["series"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for k in ("S1", "S2", "S3"):
        t = s[k]["t"]
        ax.plot(t, s[k]["mean"], color=COL[k], lw=2 if k == "S2" else 1.4, label=f"{k} 均值")
        ax.fill_between(t, s[k]["p5"], s[k]["p95"], color=COL[k], alpha=0.15)
    ax.set_xlabel("施工天数 t（d）")
    ax.set_ylabel("隐患存量（个）")
    ax.set_title("图 3  隐患存量 H(t) 时序演化（阴影为 5%—95% 分位带）", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    save(fig, "fig3_series")


def fig_convergence():
    c = RES["convergence"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(c["n"], c["p_acc"], color=NAVY)
    ax.axhline(c["p_acc"][-1], color=MID, ls="--", lw=1)
    ax.text(c["n"][-1], c["p_acc"][-1], f" 收敛值 {c['p_acc'][-1]:.3f}",
            va="bottom", ha="right", fontsize=9, color=MID)
    ax.set_xlabel("蒙特卡洛样本量 N")
    ax.set_ylabel("事故概率运行均值")
    ax.set_title("图 2  事故概率估计的收敛过程（均衡方案，N = 6000）", fontsize=11)
    ax.grid(alpha=0.3)
    save(fig, "fig2_convergence")


def fig_compare():
    rows = sorted(RES["schemes"], key=lambda x: x["scheme"])
    labels = [f"{r['scheme']} {r['scheme_name']}" for r in rows]
    panels = [("事故概率（%）", [r["p_acc"] * 100 for r in rows], "{:.1f}"),
              ("期望总成本（万元）", [r["exp_cost"] for r in rows], "{:.1f}"),
              ("P95 总成本（万元）", [r["p95_cost"] for r in rows], "{:.0f}")]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3))
    for ax, (title, vals, fmt) in zip(axes, panels):
        ax.barh(labels, vals, color=[COL[r["scheme"]] for r in rows], height=0.55)
        for y, v in enumerate(vals):
            ax.text(v, y, " " + fmt.format(v), va="center", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.18)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("图 4  三方案在均值与尾部的表现分叉", fontsize=11)
    save(fig, "fig4_compare")


def fig_cdf():
    d = RES["cost_dist"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for k in ("S1", "S2", "S3"):
        ax.step(d[k]["value"], d[k]["quantile"], where="post", color=COL[k],
                lw=2 if k == "S2" else 1.4, label=k)
    ax.set_xlabel("项目总成本 C（万元）")
    ax.set_ylabel("累积概率 F(C)")
    ax.set_title("图 5  总成本累积分布：期望最优不等于尾部安全", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    save(fig, "fig5_cdf")


def fig_tradeoff():
    sw = RES["sweep"]
    I = [r["I"] for r in sw]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(I, [r["exp_cost"] for r in sw], color=NAVY, marker="o", ms=3, label="期望总成本")
    ax.plot(I, [r["p95_cost"] for r in sw], color=LIGHT, ls="--", marker="s", ms=3, label="P95 成本")
    ax2 = ax.twinx()
    ax2.plot(I, [r["p_acc"] * 100 for r in sw], color=MID, ls="--", label="事故概率（右轴）")
    ax2.set_ylabel("事故概率（%）", color=MID)
    imin = min(range(len(I)), key=lambda i: sw[i]["exp_cost"])
    ax.axvline(I[imin], color="#B4762A", lw=1)
    ax.annotate(f"最优 I≈{I[imin]} 万元：{sw[imin]['exp_cost']:.1f} 万元",
                (I[imin], sw[imin]["exp_cost"]), textcoords="offset points",
                xytext=(12, 26), fontsize=9, color="#8A5A20")
    ax.set_xlabel("安全投入 I（万元/180 d）")
    ax.set_ylabel("成本（万元）")
    ax.set_title("图 6  安全投入的边际效益：期望成本在 20 万元处见底", fontsize=11)
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="upper left")
    save(fig, "fig6_tradeoff")


def fig_tornado():
    runs = RES["sensitivity"]["runs"]
    labels, data = [], {}
    for r in runs:
        if r["label"] not in labels:
            labels.append(r["label"])
        data.setdefault(r["label"], {})["pos" if r["pct"] > 0 else "neg"] = r["d_cost"]
    order = sorted(labels, key=lambda k: -max(abs(data[k]["pos"]), abs(data[k]["neg"])))
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.barh(order, [data[k]["pos"] for k in order], color=NAVY, height=0.5, label="参数 +30%")
    ax.barh(order, [data[k]["neg"] for k in order], color=MID, height=0.5, label="参数 −30%")
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("期望总成本变动 ΔC（万元）")
    ax.set_title("图 7  参数单因素敏感性（基准为均衡方案）", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    save(fig, "fig7_tornado")


if __name__ == "__main__":
    fig_structure()
    fig_convergence()
    fig_series()
    fig_compare()
    fig_cdf()
    fig_tradeoff()
    fig_tornado()
    files = sorted(os.listdir(FIG))
    print("已生成", len(files), "张 SVG：", files)
