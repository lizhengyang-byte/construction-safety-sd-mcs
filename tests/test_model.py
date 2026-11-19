# -*- coding: utf-8 -*-
"""模型自洽性测试：边界条件、解析稳态、单调性与检出率封顶。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import DEFAULT, SCHEMES, UNCERTAIN, simulate  # noqa: E402


def z_of(n=200, seed=1):
    """在抽样区间内生成均匀样本，用于自洽性检查（不追求统计精度）。"""
    rng = np.random.default_rng(seed)
    lo = np.array([v[0] for v in UNCERTAIN.values()])
    hi = np.array([v[1] for v in UNCERTAIN.values()])
    X = rng.uniform(lo, hi, (n, len(UNCERTAIN)))
    return {k: X[:, i] for i, k in enumerate(UNCERTAIN)}


def test_zero_trigger_gives_zero_accidents():
    """θ = 0 时不应发生任何事故，总成本等于投入本身。"""
    z = z_of()
    r = simulate(dict(DEFAULT, theta=0.0, seed=2), SCHEMES["S1"], z)
    assert r["n_acc"].sum() == 0
    assert np.allclose(r["cost"], SCHEMES["S1"]["I"])


def test_steady_state_matches_analytic():
    """仿真期末存量应与解析稳态解一致（误差 < 5%）。"""
    z = z_of(400, 3)
    for name in ("S1", "S2", "S3"):
        s = SCHEMES[name]
        eta = 1.0 - np.exp(-DEFAULT["k_t"] * s["I"])
        lam = 1.0 - np.exp(-(DEFAULT["lambda0"] + DEFAULT["k_d"] * s["f"]))
        num = DEFAULT["g0"] * (1.0 - eta) + DEFAULT["w_rate"] * DEFAULT["w_load"]
        den = lam - DEFAULT["beta"] * DEFAULT["g0"] * (1.0 - eta)
        h_star = num / den
        r = simulate(dict(DEFAULT, seed=4), s, z)
        rel = abs(r["H_end"].mean() - h_star) / h_star
        assert rel < 0.05, f"{name}: 仿真 {r['H_end'].mean():.2f} vs 解析 {h_star:.2f}"


def test_more_inspection_lowers_hazard():
    """巡检频次更高时，期末隐患存量应更低。"""
    z = z_of(200, 5)
    low = simulate(dict(DEFAULT, seed=6), SCHEMES["S1"], z)["H_end"].mean()
    high = simulate(dict(DEFAULT, seed=6), SCHEMES["S3"], z)["H_end"].mean()
    assert high < low


def test_detection_rate_is_capped():
    """即使巡检频次很高，检出率也不得超过 1。"""
    lam = 1.0 - np.exp(-(DEFAULT["lambda0"] + DEFAULT["k_d"] * 5.0))
    assert 0.0 < lam < 1.0


def test_cost_composition():
    """总成本应等于投入 + 损失 + 停工单价 × 停工天数。"""
    z = z_of(100, 7)
    r = simulate(dict(DEFAULT, seed=8), SCHEMES["S2"], z)
    expect = SCHEMES["S2"]["I"] + r["loss"] + DEFAULT["delay_unit_cost"] * r["delay"]
    assert np.allclose(r["cost"], expect)
