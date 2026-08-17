"""Nucleo del copiloto experimental de trading."""

from .engine import (
    BacktestResult,
    MultiHorizonResult,
    build_dataset,
    run_experiment,
    run_multi_horizon_experiment,
)

__all__ = [
    "BacktestResult",
    "MultiHorizonResult",
    "build_dataset",
    "run_experiment",
    "run_multi_horizon_experiment",
]
