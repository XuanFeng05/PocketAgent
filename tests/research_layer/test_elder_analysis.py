from io import BytesIO
import zipfile

import numpy as np
import pandas as pd
import pytest

from research_layer.elder_analysis import assess_family, holm_adjust, metrics, nav_returns, parse_french_daily


def test_initial_capital_counts_first_day_loss_and_drawdown():
    m = metrics([.9, .99])
    assert m["maximum_drawdown"] == pytest.approx(.1)
    assert m["total_return"] == pytest.approx(-.01)
    assert m["longest_underwater_sessions"] == 2
    np.testing.assert_allclose(nav_returns([.9, .99]), [-.1, .1])


def test_no_rf_is_not_silently_a_zero_rf_sharpe():
    nav = [1.01, 1.02, 1.015]
    assert metrics(nav)["sharpe_excess_rf"] is None
    rf = nav_returns(nav)
    assert metrics(nav, rf)["sharpe_excess_rf"] is None
    with pytest.raises(ValueError):
        metrics(nav, [0, 0])


def test_french_percent_units_header_and_filtering():
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr("ff.csv", "Provider vintage\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n20211231,1,2,3,4,5,0.01\n20220103,-1,2,3,4,5,0.02\n20220104,1,2,3,4,5,0.03\n\nAnnual Factors\n")
    frame, meta = parse_french_daily(stream.getvalue(), "2022-01-01", "2022-12-31")
    assert list(frame.index) == ["2022-01-03", "2022-01-04"]
    assert frame.loc["2022-01-03", "RF"] == pytest.approx(.0002)
    assert frame.loc["2022-01-03", "Mkt-RF"] == pytest.approx(-.01)
    assert "Provider vintage" in meta["provider_header"]


def test_holm_monotone_stepdown():
    np.testing.assert_allclose(holm_adjust([.04, .01, .03]), [.06, .03, .06])


def test_synchronized_bootstrap_preserves_duplicate_candidates():
    r = np.array([.01, -.02, .005, .015, -.01, .002] * 4)
    nav = np.cumprod(1 + r)
    dates = pd.bdate_range("2022-01-03", periods=len(nav))
    rf = pd.Series(.0001, index=dates)
    result = assess_family(np.column_stack([nav, nav]), dates, rf, ["a"], ["a", "b"], B=100, block=3, seed=42)
    a, b = result["candidate_results"]
    assert a["positive_excess_one_sided_p"] == b["positive_excess_one_sided_p"]
    assert a["mean_annualized_excess_rf_individual_ci95"] == b["mean_annualized_excess_rf_individual_ci95"]
    assert result["family_centered_max_mean_test"]["p_one_sided"] == a["positive_excess_one_sided_p"]


def test_missing_rf_dates_rejected():
    dates = pd.bdate_range("2022-01-03", periods=10)
    nav = np.cumprod(1 + np.linspace(-.01, .01, 10))[:, None]
    with pytest.raises(ValueError, match="Missing RF dates"):
        assess_family(nav, dates, pd.Series(0., index=dates[:-1]), [], ["a"], B=100, block=2)
