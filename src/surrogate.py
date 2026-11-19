# -*- coding: utf-8 -*-
"""代理模型脚本（v3，[F19]）：用 Kriging / 二次响应面替代逐次仿真，加速大规模扫描与优化。

用途：贝叶斯优化（找最优 (I,f)）、大规模 Sobol、实时决策界面都需要"秒出结果"。
做法：
  1. 在 (I,f) 网格上跑少量真仿真得到训练点（期望成本/事故概率）；
  2. 拟合 Kriging（高斯过程）与二次响应面（RBF/sklearn 二选一，纯 numpy 兜底）；
  3. 在密集网格上预测，与真仿真对比误差（R²、MAPE）；
  4. 报告加速比。

运行：python src/surrogate.py
输出：output/surrogate.json、docs/figures/fig10_surrogate.svg
"""
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DEFAULT, FEATURES, sample_params_lhs, simulate, summarize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
R_SEED = 7


def _true_eval(z, I, f):
    r = simulate(dict(DEFAULT, seed=R_SEED), {"I": float(I), "f": float(f)}, z, features=FEATURES)
    s = summarize(r)
    return s["exp_cost"], s["p95_cost"], s["p_acc"]


class RBF:
    """纯 numpy 的径向基（高斯核）插值代理，无需 scikit-learn。"""

    def __init__(self, X, y, gamma=None):
        self.X = np.asarray(X, float)
        self.y = np.asarray(y, float)
        if gamma is None:
            d = np.linalg.norm(self.X[:, None, :] - self.X[None, :, :], axis=2)
            pos = d[d > 0]
            gamma = 1.0 / (np.median(pos) ** 2 + 1e-9)
        self.gamma = gamma
        K = self._kernel(self.X, self.X) + 1e-6 * np.eye(len(self.X))
        self.w = np.linalg.solve(K, self.y)

    def _kernel(self, A, B):
        d2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
        return np.exp(-self.gamma * d2)

    def predict(self, Xq):
        Kq = self._kernel(np.asarray(Xq, float), self.X)
        return Kq @ self.w


def main():
    z = sample_params_lhs(2000, seed=2026, corr=None)   # 代理训练/评估用真仿真底样本

    # 训练网格（稀疏）
    It = [8, 15, 20, 30, 40, 50]
    ft = [0.3, 0.5, 0.7, 0.9, 1.2]
    tt = time.time()
    X, yc, y95, yp = [], [], [], []
    for I in It:
        for f in ft:
            c, p95, p = _true_eval(z, I, f)
            X.append([I, f]); yc.append(c); y95.append(p95); yp.append(p)
    t_train_sims = time.time() - tt

    # 拟合三个代理
    sur_c = RBF(X, yc); sur_95 = RBF(X, y95); sur_p = RBF(X, yp)

    # 密集评估网格
    Ie = np.arange(8, 51, 3.0)
    fe = np.arange(0.3, 1.31, 0.1)
    Xe = np.array([[i, f] for i in Ie for f in fe])

    tt = time.time()
    pc = sur_c.predict(Xe); p95 = sur_95.predict(Xe); pp = sur_p.predict(Xe)
    t_surrogate = time.time() - tt

    # 真值对比（抽样 60 个密集点）
    rng = np.random.default_rng(0)
    sel = rng.choice(len(Xe), size=min(60, len(Xe)), replace=False)
    tt = time.time()
    tc, t95, tp = [], [], []
    for s in sel:
        c, q, p = _true_eval(z, Xe[s, 0], Xe[s, 1])
        tc.append(c); t95.append(q); tp.append(p)
    t_true = time.time() - tt

    def r2(y, yhat):
        y = np.asarray(y); yhat = np.asarray(yhat)
        ss_res = np.sum((y - yhat) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    metrics = {
        "exp_cost_R2": r2(tc, pc[sel]), "exp_cost_MAPE":
            float(np.mean(np.abs((pc[sel]-tc)/np.array(tc)))),
        "p95_R2": r2(t95, p95[sel]),
        "p_acc_R2": r2(tp, pp[sel]),
        "n_train": len(X), "n_dense": len(Xe), "n_check": len(sel),
        "t_train_sims_s": round(t_train_sims, 2),
        "t_surrogate_s": round(t_surrogate, 5),
        "t_true_check_s": round(t_true, 2),
        "speedup_eval": round((t_true / len(sel)) * len(Xe) / max(t_surrogate, 1e-9), 1),
    }

    # 用代理找最优 (I,f)（期望成本最低）
    best = int(np.argmin(pc))
    opt = {"I": float(Xe[best, 0]), "f": float(Xe[best, 1]),
           "exp_cost_pred": float(pc[best]), "p_acc_pred": float(pp[best])}

    result = {"metrics": metrics, "optimum": opt,
              "grid": {"I": Ie.tolist(), "f": fe.tolist(),
                       "exp_cost_pred": pc.reshape(len(Ie), len(fe)).round(2).tolist()}}
    with open(os.path.join(OUT, "surrogate.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    print("== 代理模型 ==")
    print(f"  训练仿真 {metrics['n_train']} 次({metrics['t_train_sims_s']}s), "
          f"密集预测 {metrics['n_dense']} 点仅 {metrics['t_surrogate_s']}s")
    print(f"  期望成本 R²={metrics['exp_cost_R2']:.4f} MAPE={metrics['exp_cost_MAPE']:.3%}, "
          f"P95 R²={metrics['p95_R2']:.4f}, p_acc R²={metrics['p_acc_R2']:.4f}")
    print(f"  代理最优: I={opt['I']:.0f} f={opt['f']:.2f} exp≈{opt['exp_cost_pred']:.1f}")

    _plot(result)


def _plot(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"
    I = np.array(R["grid"]["I"]); f = np.array(R["grid"]["f"])
    Z = np.array(R["grid"]["exp_cost_pred"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cs = ax.contourf(I, f, Z.T, levels=20, cmap="viridis_r")
    ax.contour(I, f, Z.T, levels=8, colors="white", linewidths=0.5, alpha=0.6)
    fig.colorbar(cs, ax=ax, label="期望总成本（万元，代理预测）")
    o = R["optimum"]
    ax.plot(o["I"], o["f"], "r*", ms=16, label=f"代理最优 ({o['I']:.0f},{o['f']:.2f})")
    ax.set_xlabel("安全投入 I（万元）"); ax.set_ylabel("巡检频次 f（次/天）")
    ax.set_title("图 10  (I,f) 期望成本与代理模型最优点", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig10_surrogate.svg"), format="svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
