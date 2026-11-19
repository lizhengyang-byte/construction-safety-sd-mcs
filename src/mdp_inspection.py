# -*- coding: utf-8 -*-
"""基于马尔可夫决策过程（MDP）的自适应巡检策略 —— 运筹学动态规划 + 蒙特卡洛。

背景：既有模块把巡检频次 f 当作常数决策变量。但现实中隐患存量 H 是实时可观测的：
  H 低时可放宽巡检省钱，H 高时应加密巡检防事故。这是一个「状态依赖」的序贯决策，
  静态最优 f 无法刻画——本模块把它做成有限期 MDP，用值迭代求最优策略，
  并用蒙特卡洛评估「自适应策略 vs 静态最优策略」的成本增益。

MDP 定义（每个时间步 = 1 天，共 T=180 天）：
  状态 s = 隐患存量 H 的分档（离散化到 0..H_max）
  动作 a = 当日巡检强度 {低, 中, 高}，对应 f ∈ {0.2, 0.6, 1.2}，巡检成本 c_f·f
  转移 = 隐患生成（含正反馈）、巡检整改、事故注入（事故额外抬升存量）
  奖励 = -(巡检成本 + 事故期望损失)

用蒙特卡洛估计转移概率 P(s'|s,a) 与期望即时成本，再做后向值迭代（有限期 Bellman）:
  V_t(s) = min_a { cost(s,a) + γ Σ_{s'} P(s'|s,a) V_{t+1}(s') }

运行：python src/mdp_inspection.py → output/mdp_inspection.json + fig16_mdp.svg
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")

T = 180
H_MAX = 20          # 隐患存量上限（离散档位 0..20）
ACTIONS = {"低": 0.2, "中": 0.6, "高": 1.2}   # 巡检频次 f
ACT_LIST = list(ACTIONS.items())
CF = M.DEFAULT["c_f"]          # 单位巡检成本 万元/次
DELAY_UNIT = M.DEFAULT["delay_unit_cost"]
LOSS_MED = 300.0               # 事故综合损失（万元，含外部性/社会视角，与博弈模块一致）
GAMMA = 1.0                    # 有限期，不折扣
I_FIXED = 20.0                 # 固定安全投入（MDP 决策只在巡检频次上）
N_MC = 4000                    # 估计转移概率的蒙特卡洛样本


def _step_distribution(H, f, params, z):
    """给定当前存量 H、巡检频次 f，用 MC 估计下一日存量的分布与即时期望成本。

    复用 model 的机制：隐患生成 G = g0(1+βH)，整改 R=λH，事故泊松注入。
    这里做「一步」MC：对每个抽样参数并行推进一日。
    """
    g0 = z["g0"]; beta = params["beta"]; theta = z["theta"]
    lambda0 = params["lambda0"]; k_d = z["k_d"]
    lam = 1.0 - np.exp(-(lambda0 + k_d * f))
    # 隐患生成（向量化 over 参数样本）
    G = g0 * (1.0 + beta * H)
    R = lam * H
    # 事故注入：事故额外抬升存量（事故后整改强化，简化为一日的额外隐患流入）
    n_acc = np.random.default_rng(hash((int(H*100), int(f*100))) & 0xffffffff).poisson(theta * H, size=N_MC)
    dH = G + 0.5 * n_acc - R          # 净值；事故带来额外隐患
    H_next = np.clip(np.round(H + dH).astype(int), 0, H_MAX)
    # 即时成本：巡检成本 + 事故期望损失
    step_cost = CF * f + n_acc.mean() * LOSS_MED + 0.0
    return H_next.astype(int), float(step_cost)


def build_transition_and_cost(z):
    """预计算转移矩阵 P[s,a,s'] 与即时期望成本 C[s,a]。"""
    P = np.zeros((H_MAX + 1, len(ACT_LIST), H_MAX + 1))
    C = np.zeros((H_MAX + 1, len(ACT_LIST)))
    for s in range(H_MAX + 1):
        for ai, (aname, f) in enumerate(ACT_LIST):
            H_next, cost = _step_distribution(s, f, M.DEFAULT, z)
            for s2 in H_next:
                P[s, ai, s2] += 1.0
            P[s, ai, :] /= P[s, ai, :].sum()
            C[s, ai] = cost
    return P, C


def value_iteration(P, C):
    """有限期后向值迭代，返回各期的价值函数与最优策略。"""
    V = np.zeros(H_MAX + 1)            # 终端价值 0（周期结束）
    policy = np.zeros((T, H_MAX + 1), dtype=int)
    for t in range(T - 1, -1, -1):
        Q = C + GAMMA * (P @ V)        # Q[s,a]
        policy[t] = np.argmin(Q, axis=1)
        V = Q[np.arange(H_MAX + 1), policy[t]]
    return V, policy


def evaluate_policy(policy, P, C, s0=5, rng=None, n_episodes=2000):
    """蒙特卡洛前向模拟策略：从 s0 按转移分布采样 T 步，多回合取平均总成本。
    这是真正意义上的 MC 策略评估，能体现存量波动到高危区时自适应策略的临时加密价值。"""
    if rng is None:
        rng = np.random.default_rng(2027)
    # 预构造每个 (s,a) 的状态累积分布，供快速采样
    cumP = np.cumsum(P, axis=2)
    totals = np.empty(n_episodes)
    for ep in range(n_episodes):
        s = s0
        total = 0.0
        for t in range(T):
            a = policy[t, s]
            total += C[s, a]
            u = rng.random()
            s = int(np.searchsorted(cumP[s, a], u))
            if s > H_MAX:
                s = H_MAX
        totals[ep] = total
    return float(totals.mean()), float(totals.std() / np.sqrt(n_episodes))


def static_best(P, C, s0=5):
    """静态最优：整周期固定同一动作，用同样 MC 前向评估，取总成本最小者。"""
    best = None
    for ai, (aname, f) in enumerate(ACT_LIST):
        fixed_policy = np.full((T, H_MAX + 1), ai, dtype=int)
        cost, _ = evaluate_policy(fixed_policy, P, C, s0=s0, n_episodes=2000)
        if best is None or cost < best[2]:
            best = (aname, f, cost)
    return best


def main():
    print("== 基于 MDP 的自适应巡检策略（运筹学动态规划 + MC）==")
    z = M.sample_params_lhs(N_MC, seed=2026, corr=M.CORR)
    P, C = build_transition_and_cost(z)
    V0, policy = value_iteration(P, C)

    dyn_cost, dyn_se = evaluate_policy(policy, P, C, s0=5)
    aname, f, stat_cost = static_best(P, C, s0=5)
    gain = stat_cost - dyn_cost
    print(f"  静态最优巡检: f={f:.1f}（{aname}）→ 期望总成本 {stat_cost:.1f} 万元")
    print(f"  MDP 自适应策略 → 期望总成本 {dyn_cost:.1f} 万元 (±{dyn_se:.2f})")
    print(f"  自适应增益 = {gain:.1f} 万元（{gain/stat_cost*100:.1f}%）")

    # 展示策略形态：起始日、中段、末段各存量档的最优动作
    snapshots = {}
    for day in [0, 45, 90, 135]:
        row = []
        for s in range(0, H_MAX + 1, 2):
            row.append({"H": s, "action": ACT_LIST[policy[day, s]][0], "f": ACT_LIST[policy[day, s]][1]})
        snapshots[f"day_{day}"] = row

    # 平均巡检频次 & 平均成本分解
    avg_f = float(np.mean([ACTIONS[ACT_LIST[policy[t, s]][0]] for t in range(T) for s in range(H_MAX+1)]))

    payload = {
        "T": T, "H_MAX": H_MAX, "actions": ACTIONS, "c_f": CF,
        "static_best": {"action": aname, "f": f, "cost": stat_cost},
        "mdp_adaptive_cost": dyn_cost,
        "gain": gain, "gain_pct": gain / stat_cost * 100,
        "avg_f_adaptive": avg_f,
        "policy_snapshots": snapshots,
        "final_V0": float(V0[5]),
    }
    json.dump(payload, open(os.path.join(OUT, "mdp_inspection.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)
    _plot(policy, payload)
    print("\n已输出 output/mdp_inspection.json + fig16_mdp.svg")


def _plot(policy, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    # 左：策略热图（横=存量档，纵=天）
    cmap = ListedColormap(["#C3CAD6", "#4A6FA5", "#8B1A1A"])
    a1.imshow(policy[:, :H_MAX + 1], aspect="auto", cmap=cmap, origin="upper",
              extent=[0, H_MAX, T, 0], vmin=-0.5, vmax=2.5)
    a1.set_xlabel("隐患存量档 H"); a1.set_ylabel("施工周期天数")
    a1.set_title("MDP 最优巡检策略（颜色=巡检强度）", fontsize=10)
    from matplotlib.patches import Patch
    a1.legend(handles=[Patch(color="#C3CAD6", label="低 f=0.2"),
                       Patch(color="#4A6FA5", label="中 f=0.6"),
                       Patch(color="#8B1A1A", label="高 f=1.2")],
              frameon=False, fontsize=8, loc="upper right")
    # 右：成本对比柱
    labels = ["静态最优", "MDP 自适应"]
    costs = [res["static_best"]["cost"], res["mdp_adaptive_cost"]]
    bars = a2.bar(labels, costs, color=["#C3CAD6", "#1A2B4A"], width=0.5)
    for b, c in zip(bars, costs):
        a2.text(b.get_x() + b.get_width()/2, c, f"{c:.1f}", ha="center", va="bottom", fontsize=9)
    a2.set_ylabel("期望总成本（万元）")
    a2.set_title(f"自适应增益 {res['gain']:.1f} 万元（{res['gain_pct']:.1f}%）", fontsize=10)
    a2.grid(axis="y", alpha=0.3)
    fig.suptitle("图 16  MDP 自适应巡检策略与静态策略对比", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "docs", "figures", "fig16_mdp.svg"),
                format="svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
