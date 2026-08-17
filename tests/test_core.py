import numpy as np
import pandas as pd
import pytest

from trading_agent.engine import (
    FEATURES,
    _max_drawdown,
    _positions,
    build_dataset,
    run_experiment,
    walk_forward_predictions,
)


def synthetic_prices(rows: int = 650) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0005, 0.015, rows)
    close = 100 * np.exp(np.cumsum(returns))
    spread = rng.uniform(0.002, 0.02, rows)
    index = pd.date_range("2022-01-01", periods=rows, freq="D")
    open_ = close * (1 + rng.normal(0, 0.002, rows))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) * (1 + spread),
            "Low": np.minimum(open_, close) * (1 - spread),
            "Close": close,
            "Volume": rng.integers(1_000, 100_000, rows),
        },
        index=index,
    )


def test_dataset_has_features_and_purges_unfinished_horizon():
    dataset = build_dataset(synthetic_prices(), horizon=10)
    assert set(FEATURES).issubset(dataset.columns)
    assert pd.isna(dataset.iloc[-1]["target"])
    assert dataset["target"].tail(11).isna().all()
    assert dataset[FEATURES].notna().all().all()


def test_experiment_only_scores_out_of_sample_rows():
    result = run_experiment(synthetic_prices(800))
    assert not result.frame.empty
    assert result.frame["probability"].between(0, 1).all()
    assert result.latest_signal in {
        "COMPRAR", "SALIR", "MANTENER POSICIÓN", "ESPERAR", "MODELO NO VALIDADO"
    }
    assert np.isfinite(result.metrics["total_return"])
    assert np.isfinite(result.metrics["roc_auc"])


def test_signal_respects_real_portfolio_state():
    prices = synthetic_prices(800)
    without_position = run_experiment(
        prices, enter_threshold=1.0, exit_threshold=0.0, currently_in_position=False
    )
    with_position = run_experiment(
        prices, enter_threshold=1.0, exit_threshold=0.0, currently_in_position=True
    )
    assert without_position.latest_signal in {"ESPERAR", "MODELO NO VALIDADO"}
    assert with_position.latest_signal in {"MANTENER POSICIÓN", "MODELO NO VALIDADO"}


def test_decision_executes_from_next_open_not_same_close():
    prices = synthetic_prices()
    dataset = build_dataset(prices, horizon=5)
    timestamp = dataset.index[100]
    original = dataset.loc[timestamp, "execution_return"]
    expected_position = prices.index.get_loc(timestamp)
    expected = prices["Open"].iloc[expected_position + 2] / prices["Open"].iloc[expected_position + 1] - 1
    assert original == expected


def test_walk_forward_never_uses_an_unfinished_label():
    dataset = build_dataset(synthetic_prices(800), horizon=20)
    predictions = walk_forward_predictions(dataset).dropna(subset=["probability"])
    assert not predictions.empty
    trained = pd.to_datetime(predictions["trained_through"])
    assert (trained <= predictions.index).all()


def test_drawdown_includes_initial_capital():
    equity = pd.Series([90.0, 81.0])
    assert _max_drawdown(equity, 100.0) == pytest.approx(-0.19)


def test_position_cannot_exceed_maximum_duration():
    probabilities = pd.Series([0.9] * 30)
    position, age = _positions(probabilities, 0.58, 0.45, max_holding_bars=20)
    assert age.max() == 20
    assert position.iloc[20] == 0


def test_prices_are_sorted_and_deduplicated():
    prices = synthetic_prices()
    duplicated = pd.concat([prices.iloc[::-1], prices.iloc[[100]]])
    dataset = build_dataset(duplicated, horizon=5)
    assert dataset.index.is_monotonic_increasing
    assert dataset.index.is_unique
