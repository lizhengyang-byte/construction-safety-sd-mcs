# -*- coding: utf-8 -*-
"""委托–代理（Principal–Agent）最优安全激励契约设计 —— 博弈论机制设计 + 蒙特卡洛。

背景：既有博弈模块（game.py）只分析「给定处罚，承包商如何反应」。本模块更进一步，
站在监管者（委托人 principal）角度，求解「面对一个安全努力不可观测、只能观测事故结果
的承包商（代理人 agent），如何设计一份最优激励契约，让承包商自愿选择高安全努力」。

信息不对称（moral hazard）：
  - 承包商可选择安全努力 e ∈ {低, 高}，努力成本 c(e) 为其私有信息（随机）
  - 努力降低事故概率，但努力不可观测；委托人只能观测「是否出事」这一随机结果
  - 契约 = 固定支付 w0 + 事故罚金 P（出事时扣），即「结果依赖」的激励

用蒙特卡洛对承包商的随机努力成本和事故结果抽样，求委托人最优契约 (P*)，并量化：
  1) 最优事故罚金强度 P*
  2) 信息租金（承包商因信息优势获得的高于保留效用的部分）
  3) 一阶 vs 二阶最优（可观测努力 vs 不可观测）的效率损失

运行：python src/principal_agent.py → output/principal_agent.json + fig15_contract.svg
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")

# 努力水平 → 事故概率（用 MC 在给定投入下估 p_acc，努力对应不同投入 I）
EFFORTS = {"low": {"I": 8.0}, "high": {"I": 30.0}}
# 努力成本（承包商私有信息，随机；均值随努力上升）
_EFFORT_COST = {"low": (0.0, 1.0), "high": (18.0, 3.0)}  # (均值, 标准差) 万元
# 事故社会损失（委托人承担的外部成本）
SOCIAL_LOSS = 300.0  # 万元
# 事故罚金搜索网格（万元）
P_GRID = np.linspace(0, 300, 61)
# 保留效用（承包商不签约的替代收益）
RESERVE = 5.0  # 万元
N_SAMPLE = 20000


def _pacc(I, z):
    """在给定投入 I 下估事故概率（用共享 MC 样本）。"""
    r = M.simulate(dict(M.DEFAULT), {"I": I, "f": 0.58}, z, features=M.FEATURES)
    return (r["n_acc"] > 0).astype(float)


def main():
    z = M.sample_params_lhs(N_SAMPLE, seed=2026, corr=M.CORR)
    pacc = {k: _pacc(v["I"], z) for k, v in EFFORTS.items()}
    p_low = float(pacc["low"].mean())
    p_high = float(pacc["high"].mean())
    print(f"== 委托–代理最优安全激励契约设计 ==")
    print(f"  事故概率: 低努力={p_low:.3f}  高努力={p_high:.3f}（努力降低事故 {p_low-p_high:+.3f}）")

    # 承包商努力成本抽样（均值, std）
    rng = np.random.default_rng(11)
    c_low_s = rng.normal(*_EFFORT_COST["low"], size=N_SAMPLE)
    c_high_s = rng.normal(*_EFFORT_COST["high"], size=N_SAMPLE)
    c_low = float(c_low_s.mean())
    c_high = float(c_high_s.mean())

    # 用「概率口径」的期望激励。契约 = 固定支付 w0（委托人付给承包商）+ 事故罚金 P。
    #   代理人净收益(effort) = w0 - 努力成本 - P * p_acc(effort)
    #   委托人社会成本(effort) = SOCIAL_LOSS * p_acc(effort) + w0 - P * p_acc(effort)
    # 委托人可自由设定 w0（转移支付），最优时令参与约束 IR 对目标努力恰好绑定：
    #   w0 = 努力成本 + P*p_acc(effort) - RESERVE   （承包商净收益 = -RESERVE 恰好等于保留）
    def agent_net(w0, P, effort):
        cost = c_low if effort == "low" else c_high
        p = p_low if effort == "low" else p_high
        return w0 - cost - P * p

    def principal_cost(w0, P, effort):
        # 真实社会成本：事故社会损失 + 真实努力成本 + 保留效用；
        # 罚金 P 与固定支付 w0 是委托人与代理人之间的转移支付，对社会总成本零和抵消。
        cost = c_low if effort == "low" else c_high
        p = p_low if effort == "low" else p_high
        return SOCIAL_LOSS * p + cost + RESERVE

    contract_curve = []
    for P in P_GRID:
        # 情形1：诱导低努力（IR 绑定低努力）
        w0_low = c_low + P * p_low - RESERVE
        # 低努力下须满足 IC（低努力确实优于高努力）：agent_net(low) >= agent_net(high)
        ic_low = (w0_low - c_low - P * p_low) >= (w0_low - c_high - P * p_high)
        # 情形2：诱导高努力（IR 绑定高努力）
        w0_high = c_high + P * p_high - RESERVE
        ic_high = (w0_high - c_high - P * p_high) >= (w0_high - c_low - P * p_low)
        # 委托人分别算两种目标努力下（满足 IC 时）的社会成本
        opts = []
        if ic_low:
            opts.append(("low", principal_cost(w0_low, P, "low"), w0_low))
        if ic_high:
            opts.append(("high", principal_cost(w0_high, P, "high"), w0_high))
        if not opts:
            continue
        effort, pc, w0 = min(opts, key=lambda t: t[1])
        contract_curve.append({
            "P": float(P), "effort": effort, "w0": float(w0),
            "agent_net": float(-RESERVE),  # IR 绑定 ⇒ 净收益恒为 -RESERVE
            "principal_cost": float(pc),
            "participate": True,
        })

    # 委托人在「承包商愿意接受(IR)且努力为高或低」下最小化社会成本
    feasible = [c for c in contract_curve if c["participate"]]
    # 真实社会成本只取决于目标努力（与 P、w0 无关）：
    #   高努力: SOCIAL_LOSS*p_high + c_high + RESERVE
    #   低努力: SOCIAL_LOSS*p_low  + c_low  + RESERVE
    # 委托人先选社会成本更低的目标努力，再取实现该努力所需的最小罚金 P*（IC 下界）。
    cost_high = SOCIAL_LOSS * p_high + c_high + RESERVE
    cost_low = SOCIAL_LOSS * p_low + c_low + RESERVE
    target = "high" if cost_high <= cost_low else "low"
    # 在契约曲线中找诱导该目标努力的最小 P（IC 恰好满足处）
    induce = [c for c in feasible if c["effort"] == target]
    best = min(induce, key=lambda c: c["P"]) if induce else min(feasible, key=lambda c: c["principal_cost"])
    regime = "诱导高努力" if target == "high" else "诱导低努力"
    second_best = cost_high if target == "high" else cost_low

    print(f"  目标努力 = {target}（高努力社会成本 {cost_high:.1f} vs 低努力 {cost_low:.1f}）")
    print(f"  最优契约: 最小事故罚金 P* = {best['P']:.1f} 万元、配套固定支付 w0={best['w0']:.1f} 万元（{regime}）")

    # 一阶最优（努力可观测）：委托人直接要求高努力，只付真实努力成本，承包商无租金
    first_best_cost = SOCIAL_LOSS * p_high + c_high + RESERVE
    # 二阶最优 = 不可观测下的最优社会成本（可能退化为低努力，若高努力 IC 成本过高）
    info_rent = max(0.0, second_best - first_best_cost)
    efficiency_loss = second_best - first_best_cost
    # 契约设计的价值：不设罚金(P=0)时承包商必选低努力 → 社会成本 = cost_low
    no_contract_cost = cost_low
    contract_value = no_contract_cost - second_best
    print(f"  一阶最优(努力可观测)社会成本 = {first_best_cost:.1f} 万元")
    print(f"  二阶最优(努力不可观测)社会成本 = {second_best:.1f} 万元")
    print(f"  机制设计效率损失(信息租金下限) = {efficiency_loss:.1f} 万元")
    print(f"  不设罚金(P=0)社会成本 = {no_contract_cost:.1f} 万元 → 契约设计价值 = {contract_value:.1f} 万元")

    payload = {
        "p_low": p_low, "p_high": p_high,
        "c_low": c_low, "c_high": c_high,
        "social_loss": SOCIAL_LOSS, "reserve": RESERVE,
        "best_contract": best, "regime": regime, "target_effort": target,
        "P_star": best["P"], "w0_star": best["w0"],
        "cost_high": cost_high, "cost_low": cost_low,
        "info_rent": info_rent,
        "first_best_cost": first_best_cost,
        "second_best_cost": second_best,
        "efficiency_loss": efficiency_loss,
        "no_contract_cost": no_contract_cost,
        "contract_value": contract_value,
        "contract_curve": contract_curve,
    }
    json.dump(payload, open(os.path.join(OUT, "principal_agent.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)
    _plot(payload)
    print("\n已输出 output/principal_agent.json + fig15_contract.svg")


def _plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"
    curve = res["contract_curve"]
    P = [c["P"] for c in curve]
    # 代理商净收益差（高 − 低）随 P 变化：跨零点是 IC 切换阈值
    c_low, c_high = res["c_low"], res["c_high"]
    p_low, p_high = res["p_low"], res["p_high"]
    ic_gap = [(res["p_low"] - res["p_high"]) * p - (c_high - c_low) for p in P]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(P, [res["cost_low"]] * len(P), "-", color="#8B1A1A", label=f"诱导低努力社会成本={res['cost_low']:.0f}")
    a1.plot(P, [res["cost_high"]] * len(P), "-", color="#1A2B4A", label=f"诱导高努力社会成本={res['cost_high']:.0f}")
    a1.axvline(res["P_star"], ls="--", color="#B4762A", lw=1.2, label=f"最优罚金 P*={res['P_star']:.0f}")
    a1.set_xlabel("事故罚金 P（万元）"); a1.set_ylabel("真实社会成本（万元）")
    a1.set_title("委托人选择目标努力", fontsize=10)
    a1.legend(frameon=False, fontsize=8); a1.grid(alpha=0.3)
    a2.plot(P, ic_gap, "-", color="#B4762A", label="高努力相对净收益差")
    a2.axhline(0, ls=":", color="#555", lw=1, label="IC 切换阈值")
    a2.axvline(res["P_star"], ls="--", color="#8B1A1A", lw=1)
    a2.set_xlabel("事故罚金 P（万元）"); a2.set_ylabel("高−低努力净收益差（万元）")
    a2.set_title("激励相容(IC)：罚金跨过阈值即诱导高努力", fontsize=10)
    a2.legend(frameon=False, fontsize=8); a2.grid(alpha=0.3)
    fig.suptitle("图 15  委托–代理最优安全激励契约", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "docs", "figures", "fig15_contract.svg"),
                format="svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
