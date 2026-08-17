from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "return_1", "return_5", "return_10", "return_20", "return_60",
    "ema_gap_10", "ema_gap_30", "ema_gap_60", "rsi_14", "atr_pct_14",
    "volatility_10", "volatility_20", "volatility_60", "volume_ratio_20",
    "drawdown_20", "drawdown_60",
]


@dataclass(frozen=True)
class BacktestResult:
    frame: pd.DataFrame
    latest_signal: str
    latest_probability: float
    metrics: dict[str, Any]
    reasons: list[str]
    horizon: int
    validation_passed: bool
    warnings: list[str]


@dataclass(frozen=True)
class MultiHorizonResult:
    results: dict[int, BacktestResult]
    latest_signal: str
    mean_score: float
    score_range: tuple[float, float]
    validated_horizons: int
    agreement: str


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Faltan columnas OHLCV: {', '.join(sorted(missing))}")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("El índice de precios debe contener fechas.")

    data = prices.loc[:, ["Open", "High", "Low", "Close", "Volume"]].copy()
    data = data[~data.index.duplicated(keep="last")].sort_index()
    data = data.apply(pd.to_numeric, errors="coerce")
    if data.empty or data.isna().any().any():
        raise ValueError("Los datos contienen valores faltantes o no numéricos.")
    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Los precios deben ser positivos.")
    upper = data[["Open", "Low", "Close"]].max(axis=1)
    lower = data[["Open", "High", "Close"]].min(axis=1)
    if (data["High"] < upper).any() or (data["Low"] > lower).any():
        raise ValueError("Se encontraron velas OHLC inconsistentes.")
    return data


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / window, adjust=False).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100)
    return rsi.mask((loss == 0) & (gain == 0), 50)


def build_dataset(
    prices: pd.DataFrame, fee_rate: float = 0.001, horizon: int = 10
) -> pd.DataFrame:
    """Crea variables causales y una etiqueta ejecutable desde la próxima apertura."""
    if not 0 <= fee_rate < 1:
        raise ValueError("El costo por orden debe estar entre 0 y 1.")
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("El horizonte debe ser un entero positivo.")

    data = _prepare_prices(prices)
    close = data["Close"].astype(float)
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    open_ = data["Open"].astype(float)

    for window in (1, 5, 10, 20, 60):
        data[f"return_{window}"] = close.pct_change(window)
    for window in (10, 30, 60):
        data[f"ema_gap_{window}"] = close / close.ewm(span=window, adjust=False).mean() - 1
    data["rsi_14"] = _rsi(close)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    data["atr_pct_14"] = true_range.rolling(14).mean() / close
    for window in (10, 20, 60):
        data[f"volatility_{window}"] = data["return_1"].rolling(window).std()
    data["volume_ratio_20"] = data["Volume"] / data["Volume"].rolling(20).mean().replace(0, np.nan)
    data["drawdown_20"] = close / close.rolling(20).max() - 1
    data["drawdown_60"] = close / close.rolling(60).max() - 1

    # Señal al cierre t; entrada en Open[t+1]. Nunca captura el gap anterior al fill.
    data["execution_return"] = open_.shift(-2) / open_.shift(-1) - 1
    data["forward_return"] = open_.shift(-(horizon + 1)) / open_.shift(-1) - 1
    index_as_series = pd.Series(data.index, index=data.index)
    data["label_end"] = index_as_series.shift(-(horizon + 1))
    round_trip_hurdle = 1 / ((1 - fee_rate) ** 2) - 1
    data["target"] = np.where(
        data["forward_return"].notna(),
        (data["forward_return"] > round_trip_hurdle).astype(float),
        np.nan,
    )
    return data.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)


def _new_model():
    # Regularización fuerte para reducir varianza en muestras financieras pequeñas.
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=0.2, max_iter=1_000, random_state=42)
    )


def walk_forward_predictions(
    dataset: pd.DataFrame,
    min_train_size: int = 180,
    max_train_size: int = 1_260,
    prediction_bars: int | None = None,
) -> pd.DataFrame:
    """Predice semanalmente usando solo etiquetas cuyo horizonte ya finalizó."""
    if len(dataset) < min_train_size + 40:
        raise ValueError(
            f"Se necesitan al menos {min_train_size + 40} velas útiles; se recibieron {len(dataset)}."
        )
    output = pd.DataFrame(
        index=dataset.index,
        columns=["probability", "baseline_probability", "trained_through"],
    )
    model = None
    constant_probability: float | None = None
    baseline = np.nan
    trained_through = pd.NaT
    fitted_week: tuple[int, int] | None = None

    prediction_rows = dataset if prediction_bars is None else dataset.tail(prediction_bars)
    for as_of, row in prediction_rows.iterrows():
        eligible = dataset.loc[
            dataset["target"].notna()
            & dataset["label_end"].notna()
            & (dataset["label_end"] <= as_of)
        ].tail(max_train_size)
        if len(eligible) < min_train_size:
            continue
        iso = as_of.isocalendar()
        week_key = (int(iso.year), int(iso.week))
        if fitted_week != week_key:
            y = eligible["target"].astype(int)
            baseline = float(y.mean())
            trained_through = eligible["label_end"].max()
            if y.nunique() < 2:
                model = None
                constant_probability = baseline
            else:
                model = _new_model().fit(eligible[FEATURES], y)
                constant_probability = None
            fitted_week = week_key

        if constant_probability is not None:
            probability = constant_probability
        elif model is not None:
            probability = float(model.predict_proba(row[FEATURES].to_frame().T)[0, 1])
        else:
            continue
        output.loc[as_of] = [probability, baseline, trained_through]

    output["probability"] = pd.to_numeric(output["probability"], errors="coerce")
    output["baseline_probability"] = pd.to_numeric(
        output["baseline_probability"], errors="coerce"
    )
    return output


def _positions(
    probabilities: pd.Series, enter: float, exit_: float, max_holding_bars: int
) -> tuple[pd.Series, pd.Series]:
    state, held = 0.0, 0
    values: list[float] = []
    ages: list[float] = []
    for probability in probabilities:
        if pd.isna(probability):
            values.append(np.nan)
            ages.append(np.nan)
            continue
        if state == 0 and probability >= enter:
            state, held = 1.0, 1
        elif state == 1 and (probability <= exit_ or held >= max_holding_bars):
            state, held = 0.0, 0
        elif state == 1:
            held += 1
        values.append(state)
        ages.append(float(held))
    return pd.Series(values, index=probabilities.index), pd.Series(ages, index=probabilities.index)


def _infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 252.0
    years = (index[-1] - index[0]).total_seconds() / (365.25 * 24 * 60 * 60)
    return float((len(index) - 1) / years) if years > 0 else 252.0


def _max_drawdown(equity: pd.Series, initial_capital: float) -> float:
    values = np.concatenate(([float(initial_capital)], equity.astype(float).to_numpy()))
    running_max = np.maximum.accumulate(values)
    return float(np.min(values / running_max - 1))


def _trade_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    active, entry_time, growth, previous = False, None, 1.0, 0.0
    for timestamp, row in frame.iterrows():
        position, strategy_return = row["position"], row["strategy_return"]
        if pd.isna(position) or pd.isna(strategy_return):
            continue
        if not active and previous == 0 and position > 0:
            active, entry_time, growth = True, timestamp, 1.0
        if active:
            growth *= 1 + float(strategy_return)
        if active and previous > 0 and position == 0:
            records.append({"entry": entry_time, "exit": timestamp, "return": growth - 1, "closed": True})
            active, entry_time, growth = False, None, 1.0
        previous = float(position)
    if active:
        records.append({"entry": entry_time, "exit": frame.index[-1], "return": growth - 1, "closed": False})
    return pd.DataFrame(records, columns=["entry", "exit", "return", "closed"])


def _reasons(row: pd.Series) -> list[str]:
    reasons = [
        "precio sobre la EMA de 60" if row["ema_gap_60"] > 0 else "precio bajo la EMA de 60",
        "momentum de 20 ruedas positivo" if row["return_20"] > 0 else "momentum de 20 ruedas negativo",
    ]
    if row["rsi_14"] >= 70:
        reasons.append("RSI en zona alta")
    elif row["rsi_14"] <= 30:
        reasons.append("RSI en zona baja")
    else:
        reasons.append(f"RSI neutral ({row['rsi_14']:.0f})")
    return reasons


def _direction(score: float, enter: float, exit_: float) -> str:
    return "ENTRADA" if score >= enter else "SALIDA" if score <= exit_ else "NEUTRAL"


def run_experiment(
    prices: pd.DataFrame,
    interval: str = "1d",
    fee_rate: float = 0.001,
    enter_threshold: float = 0.58,
    exit_threshold: float = 0.45,
    initial_capital: float = 10_000,
    currently_in_position: bool = False,
    horizon: int = 10,
    max_holding_bars: int = 20,
    current_holding_bars: int = 0,
    allocation: float = 0.25,
    evaluation_bars: int = 252,
) -> BacktestResult:
    if interval != "1d":
        raise ValueError("Esta versión está validada únicamente para velas diarias.")
    if not 0 <= exit_threshold < enter_threshold <= 1:
        raise ValueError("Los umbrales deben cumplir 0 <= salida < entrada <= 1.")
    if initial_capital <= 0 or not 0 < allocation <= 1:
        raise ValueError("El capital y la asignación deben ser positivos.")
    if max_holding_bars < 1 or current_holding_bars < 0:
        raise ValueError("La duración de la posición no es válida.")

    prepared = _prepare_prices(prices)
    data = build_dataset(prepared, fee_rate=fee_rate, horizon=horizon)
    prediction_bars = evaluation_bars + max_holding_bars + horizon + 20
    data = data.join(walk_forward_predictions(data, prediction_bars=prediction_bars))
    data["position"], data["holding_bars"] = _positions(
        data["probability"], enter_threshold, exit_threshold, max_holding_bars
    )
    data["exposure"] = data["position"] * allocation
    data["turnover"] = data["exposure"].diff().abs().fillna(data["exposure"])
    gross_factor = 1 + data["exposure"] * data["execution_return"]
    data["strategy_return"] = gross_factor * (1 - fee_rate * data["turnover"]) - 1

    return_frame = data.dropna(subset=["probability", "execution_return"]).copy()
    if return_frame.empty:
        raise ValueError("No quedaron observaciones fuera de muestra para evaluar.")
    evaluation = return_frame.tail(evaluation_bars).copy()
    evaluation["equity"] = initial_capital * (1 + evaluation["strategy_return"]).cumprod()
    benchmark_factor = 1 + allocation * evaluation["execution_return"]
    benchmark_factor.iloc[0] *= 1 - fee_rate * allocation
    evaluation["buy_hold"] = initial_capital * benchmark_factor.cumprod()

    predictive = data.dropna(subset=["probability", "baseline_probability", "target"]).tail(evaluation_bars)
    y_true = predictive["target"].astype(int)
    score = predictive["probability"].astype(float)
    baseline_score = predictive["baseline_probability"].astype(float)
    auc = float(roc_auc_score(y_true, score)) if y_true.nunique() == 2 else 0.5
    brier = float(brier_score_loss(y_true, score))
    baseline_brier = float(brier_score_loss(y_true, baseline_score))

    periods = _infer_periods_per_year(prepared.index)
    returns = evaluation["strategy_return"]
    volatility = float(returns.std(ddof=0))
    sharpe = float(returns.mean() / volatility * sqrt(periods)) if volatility > 0 else 0.0
    ledger = _trade_ledger(return_frame)
    closed = ledger[
        ledger["closed"]
        & (ledger["entry"] >= evaluation.index[0])
        & (ledger["exit"] <= evaluation.index[-1])
    ]
    trade_count = int(len(closed))
    win_rate = float((closed["return"] > 0).mean()) if trade_count else 0.0
    wins = float(closed.loc[closed["return"] > 0, "return"].sum()) if trade_count else 0.0
    losses = float(-closed.loc[closed["return"] < 0, "return"].sum()) if trade_count else 0.0
    profit_factor = wins / losses if losses > 0 else float("inf") if wins > 0 else 0.0

    total_return = float(evaluation["equity"].iloc[-1] / initial_capital - 1)
    checks = {
        "muestras": len(predictive) >= 100,
        "AUC": auc >= 0.52,
        "Brier": brier < baseline_brier,
        "retorno": total_return > 0,
        "operaciones": trade_count >= 3,
    }
    validation_passed = all(checks.values())
    latest = data.iloc[-1]
    probability = float(latest["probability"])
    if currently_in_position and current_holding_bars >= max_holding_bars:
        signal = "SALIR"
    elif not validation_passed:
        signal = "MODELO NO VALIDADO"
    elif not currently_in_position and probability >= enter_threshold:
        signal = "COMPRAR"
    elif currently_in_position and probability <= exit_threshold:
        signal = "SALIR"
    elif currently_in_position:
        signal = "MANTENER POSICIÓN"
    else:
        signal = "ESPERAR"

    metrics: dict[str, Any] = {
        "total_return": total_return,
        "buy_hold_return": float(evaluation["buy_hold"].iloc[-1] / initial_capital - 1),
        "max_drawdown": _max_drawdown(evaluation["equity"], initial_capital),
        "sharpe": sharpe,
        "trades": trade_count,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "exposure": float(evaluation["exposure"].mean()),
        "roc_auc": auc,
        "brier": brier,
        "baseline_brier": baseline_brier,
        "test_samples": int(len(predictive)),
        "direction": _direction(probability, enter_threshold, exit_threshold),
        "evaluation_start": evaluation.index[0],
        "evaluation_end": evaluation.index[-1],
        "periods_per_year": periods,
    }
    return BacktestResult(
        evaluation, signal, probability, metrics, _reasons(latest), horizon,
        validation_passed, [name for name, passed in checks.items() if not passed]
    )


def run_multi_horizon_experiment(
    prices: pd.DataFrame, horizons: Iterable[int] = (5, 10, 20), **kwargs
) -> MultiHorizonResult:
    horizon_list = [int(h) for h in horizons]
    if not horizon_list:
        raise ValueError("Debe indicarse al menos un horizonte.")
    with ThreadPoolExecutor(max_workers=min(3, len(horizon_list))) as executor:
        futures = {
            horizon: executor.submit(run_experiment, prices, horizon=horizon, **kwargs)
            for horizon in horizon_list
        }
        results = {horizon: future.result() for horizon, future in futures.items()}
    scores = [result.latest_probability for result in results.values()]
    enter, exit_ = float(kwargs.get("enter_threshold", 0.58)), float(kwargs.get("exit_threshold", 0.45))
    in_position = bool(kwargs.get("currently_in_position", False))
    held, max_holding = int(kwargs.get("current_holding_bars", 0)), int(kwargs.get("max_holding_bars", 20))
    directions = [_direction(score, enter, exit_) for score in scores]
    validated = sum(result.validation_passed for result in results.values())
    dispersion = max(scores) - min(scores)

    if in_position and held >= max_holding:
        signal, agreement = "SALIR", "Límite de duración alcanzado"
    elif validated < 2:
        signal = "MODELO NO VALIDADO"
        agreement = f"Solo {validated} de {len(results)} horizontes superan la validación"
    elif directions.count("ENTRADA") >= 2 and dispersion <= 0.15:
        signal, agreement = ("MANTENER POSICIÓN" if in_position else "COMPRAR"), "Consenso alcista"
    elif directions.count("SALIDA") >= 2 and dispersion <= 0.15:
        signal, agreement = ("SALIR" if in_position else "ESPERAR"), "Consenso defensivo"
    elif len(set(directions)) > 1 or dispersion > 0.15:
        signal, agreement = "SEÑAL INCIERTA", "Los horizontes no coinciden"
    else:
        signal = "MANTENER POSICIÓN" if in_position else "ESPERAR"
        agreement = "Sin ventaja suficiente para una entrada nueva"
    return MultiHorizonResult(
        results, signal, float(np.mean(scores)), (float(min(scores)), float(max(scores))),
        validated, agreement
    )
