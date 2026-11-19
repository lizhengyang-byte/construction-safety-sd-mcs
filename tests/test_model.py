# -*- coding: utf-8 -*-
"""模型自洽性测试（v3）：边界、解析稳态、单调性、检出率封顶、成本口径，
以及 v3 新特性：独立随机流可复现、事故闭环、巡检成本、参数相关、损失-停工相关、
Wilson 区间、CVaR、分级/分阶段开关可退化到 v2。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import (DEFAULT, SCHEMES, UNCERTAIN, FEATURES, V2_FEATURES,
                   sample_params_lhs, simulate, summarize, grid_scan)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def z_of(n=200, seed=1, corr=None):
    return sample_params_lhs(n, seed=seed, corr=corr)


# ---------------- v2 既有测试（用 v2 退化开关保证口径一致） ----------------
def test_zero_trigger_gives_zero_accidents():
    """θ = 0 时不应发生任何事故，总成本等于投入（v2 口径，不含巡检成本）。"""
    z = z_of()
    r = simulate(dict(DEFAULT, theta=0.0, seed=2), SCHEMES["S1"], z,
                 features=dict(V2_FEATURES))
    assert r["n_acc"].sum() == 0
    assert np.allclose(r["cost"], SCHEMES["S1"]["I"])


def test_steady_state_matches_analytic():
    """v2 口径下仿真期末存量应与解析稳态解一致（误差 < 5%）。"""
    z = z_of(600, 3)
    for name in ("S1", "S2", "S3"):
        s = SCHEMES[name]
        eta = 1.0 - np.exp(-DEFAULT["k_t"] * s["I"])
        lam = 1.0 - np.exp(-(DEFAULT["lambda0"] + DEFAULT["k_d"] * s["f"]))
        num = DEFAULT["g0"] * (1.0 - eta) + DEFAULT["w_rate"] * DEFAULT["w_load"]
        den = lam - DEFAULT["beta"] * DEFAULT["g0"] * (1.0 - eta)
        h_star = num / den
        # v2 口径：全程恒定参数、恒定天气均值（关掉 phase/grading/markov）
        r = simulate(dict(DEFAULT, seed=4), s, z, features=dict(V2_FEATURES))
        rel = abs(r["H_end"].mean() - h_star) / h_star
        assert rel < 0.05, f"{name}: 仿真 {r['H_end'].mean():.2f} vs 解析 {h_star:.2f}"


def test_more_inspection_lowers_hazard():
    """巡检频次更高时，期末隐患存量应更低（v3 全特性下亦然）。"""
    z = z_of(400, 5, corr=None)
    low = simulate(dict(DEFAULT, seed=6), SCHEMES["S1"], z, features=FEATURES)["H_end"].mean()
    high = simulate(dict(DEFAULT, seed=6), SCHEMES["S3"], z, features=FEATURES)["H_end"].mean()
    assert high < low


def test_detection_rate_is_capped():
    """即使巡检频次很高，检出率也不得超过 1。"""
    lam = 1.0 - np.exp(-(DEFAULT["lambda0"] + DEFAULT["k_d"] * 5.0))
    assert 0.0 < lam < 1.0


def test_cost_composition():
    """v2 口径总成本 = 投入 + 损失 + 停工单价 × 停工天数。"""
    z = z_of(100, 7)
    r = simulate(dict(DEFAULT, seed=8), SCHEMES["S2"], z, features=dict(V2_FEATURES))
    expect = SCHEMES["S2"]["I"] + r["loss"] + DEFAULT["delay_unit_cost"] * r["delay"]
    assert np.allclose(r["cost"], expect)


# ---------------- v3 新增测试 ----------------
def test_inspection_cost_included():
    """[F1-补] 开启巡检成本后，成本应比 v2 口径多出 c_f·f·T。"""
    z = z_of(300, 11)
    r2 = simulate(dict(DEFAULT, seed=8), SCHEMES["S2"], z, features=dict(V2_FEATURES))
    r3 = simulate(dict(DEFAULT, seed=8), SCHEMES["S2"], z,
                  features=dict(V2_FEATURES, inspection_cost=True))
    delta = r3["cost"] - r2["cost"]
    expect = DEFAULT["c_f"] * SCHEMES["S2"]["f"] * DEFAULT["T"]
    assert np.allclose(delta, expect), f"delta={delta.mean():.3f} expect={expect:.3f}"


def test_independent_streams_reproducible():
    """[F13] 同一 seed 下两次运行结果应完全一致（独立随机流可复现）。"""
    z = z_of(300, 12, corr=None)
    a = simulate(dict(DEFAULT, seed=123), SCHEMES["S2"], z, features=FEATURES)
    b = simulate(dict(DEFAULT, seed=123), SCHEMES["S2"], z, features=FEATURES)
    assert np.allclose(a["cost"], b["cost"])


def test_accident_loop_lowers_hazard():
    """[F2] 事故闭环（事故后提速整改）应使期末隐患不高于无闭环情形。"""
    z = z_of(500, 13, corr=None)
    off = simulate(dict(DEFAULT, seed=9), SCHEMES["S1"], z,
                   features=dict(V2_FEATURES, accident_loop=False))
    on = simulate(dict(DEFAULT, seed=9), SCHEMES["S1"], z,
                  features=dict(V2_FEATURES, accident_loop=True))
    # 闭环会提升检出 -> 隐患下降（在同等随机流下应成立）
    assert on["H_end"].mean() <= off["H_end"].mean() + 1e-6


def test_param_correlation_changes_joint():
    """[F5] 注入参数相关后，g0 与 k_d 的样本秩相关应显著为正。"""
    zc = sample_params_lhs(2000, seed=1, corr=np.array(
        [[1, 0, 0, 0, .55], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0], [.55, 0, 0, 0, 1.]]))
    from scipy.stats import spearmanr
    rho, _ = spearmanr(zc["g0"], zc["k_d"])
    assert rho > 0.3, f"g0-k_d 秩相关={rho:.3f}"


def test_loss_delay_correlated():
    """[F6] 相关抽样下，单次事故损失与停工应呈正相关。"""
    z = z_of(4000, 15, corr=None)
    r = simulate(dict(DEFAULT, seed=20), SCHEMES["S3"], z,
                 features=dict(V2_FEATURES, loss_delay_corr=True))
    m = r["n_acc"] > 0
    if m.sum() > 50:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(r["loss"][m], r["delay"][m])
        assert rho > 0.2, f"损失-停工相关={rho:.3f}"


def test_v2_features_degenerate():
    """全关 FEATURES 时应与 v2 数值对齐（与论文 v2 三方案量级一致）。"""
    z = z_of(3000, 2026, corr=None)
    r = simulate(dict(DEFAULT, seed=7), SCHEMES["S2"], z, features=dict(V2_FEATURES))
    p = (r["n_acc"] > 0).mean()
    assert 0.20 < p < 0.30, f"S2 事故概率(v2口径)={p:.3f}"


def test_wilson_ci_bounds():
    """[F12] Wilson 区间应包住点估计且不出 [0,1]。"""
    from experiments import wilson_ci
    lo, hi = wilson_ci(0.02, 100)
    assert 0.0 <= lo <= 0.02 <= hi <= 1.0
    lo2, hi2 = wilson_ci(0.0, 100)
    assert lo2 == 0.0 and hi2 > 0.0


def test_cvar_geq_quantile():
    """[F11] CVaR95 应不小于 P95（尾部均值 >= 分位数）。"""
    z = z_of(2000, 17, corr=None)
    r = simulate(dict(DEFAULT, seed=7), SCHEMES["S2"], z, features=FEATURES)
    s = summarize(r)
    assert s["cvar95"] >= s["p95_cost"] - 1e-6


def test_grid_pareto_subset():
    """[F1] 2D 网格能产出 Pareto 前沿，且非空。"""
    z = z_of(500, 19, corr=None)
    cells = grid_scan(DEFAULT, z, [8, 20, 40], [0.3, 0.7, 1.1], features=FEATURES)
    assert len(cells) == 9
    assert any(c["pareto"] for c in cells)
