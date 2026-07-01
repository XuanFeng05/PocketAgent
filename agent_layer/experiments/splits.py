from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class DateRange:
    start: pd.Timestamp
    end: pd.Timestamp
    trading_days: int

    def payload(self) -> dict[str, object]:
        return {
            "start": str(self.start.date()),
            "end": str(self.end.date()),
            "trading_days": self.trading_days,
        }


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train: DateRange
    validation: DateRange
    embargo_days: int


@dataclass(frozen=True)
class ExperimentSplits:
    folds: tuple[WalkForwardFold, ...]
    final_development: DateRange
    test: DateRange
    test_embargo_days: int

    def payload(self) -> dict[str, object]:
        return {
            "folds": [
                {
                    "fold": fold.fold,
                    "train": fold.train.payload(),
                    "validation": fold.validation.payload(),
                    "embargo_days": fold.embargo_days,
                }
                for fold in self.folds
            ],
            "final_development": self.final_development.payload(),
            "test": self.test.payload(),
            "test_embargo_days": self.test_embargo_days,
        }


def build_walk_forward_splits(
    trading_dates: Iterable[str | pd.Timestamp],
    *,
    validation_days: int = 126,
    test_days: int = 252,
    embargo_days: int = 20,
    folds: int = 3,
) -> ExperimentSplits:
    dates = sorted(
        {
            pd.Timestamp(value).normalize()
            for value in trading_dates
            if not pd.isna(pd.Timestamp(value))
        }
    )
    if validation_days <= 0 or test_days <= 0 or embargo_days < 0 or folds <= 0:
        raise ValueError("Split lengths and fold count must be positive; embargo may be zero.")
    required = test_days + embargo_days + folds * validation_days + folds * embargo_days + 1
    if len(dates) < required:
        raise ValueError(
            f"At least {required} trading days are required for the configured walk-forward split; got {len(dates)}."
        )

    test_start_index = len(dates) - test_days
    test = _range(dates, test_start_index, len(dates) - 1)
    development_end_index = test_start_index - embargo_days - 1
    final_development = _range(dates, 0, development_end_index)

    result: list[WalkForwardFold] = []
    for fold_index in range(folds):
        blocks_from_end = folds - fold_index - 1
        validation_end = development_end_index - blocks_from_end * (
            validation_days + embargo_days
        )
        validation_start = validation_end - validation_days + 1
        train_end = validation_start - embargo_days - 1
        if train_end < 0:
            raise ValueError("Not enough training history before the earliest validation fold.")
        result.append(
            WalkForwardFold(
                fold=fold_index + 1,
                train=_range(dates, 0, train_end),
                validation=_range(dates, validation_start, validation_end),
                embargo_days=embargo_days,
            )
        )
    return ExperimentSplits(
        folds=tuple(result),
        final_development=final_development,
        test=test,
        test_embargo_days=embargo_days,
    )


def _range(dates: list[pd.Timestamp], start: int, end: int) -> DateRange:
    return DateRange(dates[start], dates[end], end - start + 1)
