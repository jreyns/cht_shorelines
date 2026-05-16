from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd

from .io import (
    NumericScalar,
    NumericTableLike,
    PathLike,
    ShorelinesModelProtocol,
    coerce_numeric_table,
    maybe_path,
    normalize_optional_file_name,
    parse_compact_datetime,
    read_numeric_table,
)

ChangeInput: TypeAlias = NumericScalar | NumericTableLike | pd.DataFrame


class ShorelinesClimateChange:
    def __init__(self, model: ShorelinesModelProtocol | None = None) -> None:
        self.model = model
        self.sea_level_rise = 0.0
        self.wave_height_change = 0.0
        self.wave_direction_change = 0.0
        self.sea_level_rise_file = None
        self.wave_height_change_file = None
        self.wave_direction_change_file = None

    @property
    def root(self) -> Path:
        """
        Return the case root directory.

        Returns
        -------
        pathlib.Path
            Directory used to resolve and write climate-change files.
        """
        if self.model is not None:
            return Path(self.model.path)
        return Path.cwd()

    def set_sea_level_rise(
        self,
        value: ChangeInput,
        file_name: PathLike | None = None,
    ) -> None:
        """
        Set sea-level-rise forcing.

        Parameters
        ----------
        value : ChangeInput
            Scalar offset or two-column time series containing time and value.
        file_name : str or pathlib.Path, optional
            Output file name used when ``value`` is a time series.
        """
        self.sea_level_rise = _normalize_change_input(value)
        self.sea_level_rise_file = normalize_optional_file_name(file_name)
        if self.model is not None:
            self.model.input.variables.ccslr = (
                self.sea_level_rise_file or self.sea_level_rise
            )

    def set_wave_height_change(
        self,
        value: ChangeInput,
        file_name: PathLike | None = None,
    ) -> None:
        """
        Set wave-height change forcing.

        Parameters
        ----------
        value : ChangeInput
            Scalar offset or two-column time series containing time and value.
        file_name : str or pathlib.Path, optional
            Output file name used when ``value`` is a time series.
        """
        self.wave_height_change = _normalize_change_input(value)
        self.wave_height_change_file = normalize_optional_file_name(file_name)
        if self.model is not None:
            self.model.input.variables.cchs = (
                self.wave_height_change_file or self.wave_height_change
            )

    def set_wave_direction_change(
        self,
        value: ChangeInput,
        file_name: PathLike | None = None,
    ) -> None:
        """
        Set wave-direction change forcing.

        Parameters
        ----------
        value : ChangeInput
            Scalar offset or two-column time series containing time and value.
        file_name : str or pathlib.Path, optional
            Output file name used when ``value`` is a time series.
        """
        self.wave_direction_change = _normalize_change_input(value)
        self.wave_direction_change_file = normalize_optional_file_name(file_name)
        if self.model is not None:
            self.model.input.variables.ccdir = (
                self.wave_direction_change_file or self.wave_direction_change
            )

    def read(self) -> None:
        """
        Read climate-change settings from the attached model input.

        Notes
        -----
        File-backed series are loaded into pandas data frames. Scalar values are
        preserved as-is.
        """
        variables = getattr(self.model.input, "variables", None) if self.model else None
        if variables is None:
            return
        self.sea_level_rise, self.sea_level_rise_file = self._read_source(
            variables.ccslr
        )
        self.wave_height_change, self.wave_height_change_file = self._read_source(
            variables.cchs
        )
        self.wave_direction_change, self.wave_direction_change_file = self._read_source(
            variables.ccdir
        )

    def write(self) -> None:
        """
        Write climate-change settings to files and model variables.

        Notes
        -----
        Time-series inputs are written to disk and the corresponding runfile
        variables are updated with the generated file names.
        """
        if self.model is None:
            return
        self.model.input.variables.ccslr = self._write_source(
            self.sea_level_rise, self.sea_level_rise_file, "ccslr.txt"
        )
        self.model.input.variables.cchs = self._write_source(
            self.wave_height_change, self.wave_height_change_file, "cchs.txt"
        )
        self.model.input.variables.ccdir = self._write_source(
            self.wave_direction_change, self.wave_direction_change_file, "ccdir.txt"
        )

    def _read_source(self, value: object) -> tuple[object, str | None]:
        if isinstance(value, str) and value:
            path = maybe_path(self.root, value)
            if path is not None and path.exists():
                return _read_change_file(path), value
            return value, value
        if _is_series_like(value):
            return _change_dataframe(np.asarray(value, dtype=float)), None
        return value, None

    def _write_source(
        self,
        value: object,
        file_name: str | None,
        default_name: str,
    ) -> object:
        if isinstance(value, pd.DataFrame):
            target = self.root / (file_name or default_name)
            _write_change_file(target, value)
            return target.name
        return value


def _normalize_change_input(value: ChangeInput) -> NumericScalar | pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if _is_series_like(value):
        return _change_dataframe(
            coerce_numeric_table(value, argument="value", min_columns=2)
        )
    return value


def _is_series_like(value: object) -> bool:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return arr.ndim == 2 and arr.shape[1] >= 2


def _read_change_file(path: Path) -> pd.DataFrame:
    return _change_dataframe(read_numeric_table(path))


def _change_dataframe(data: NumericTableLike) -> pd.DataFrame:
    arr = coerce_numeric_table(data, argument="data", min_columns=2)
    return pd.DataFrame(
        {
            "time": [parse_compact_datetime(value) for value in arr[:, 0]],
            "value": arr[:, 1],
        }
    )


def _write_change_file(path: Path, data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fid:
        for row in data.itertuples(index=False):
            timestamp = pd.Timestamp(row.time)
            if timestamp.second:
                time_value = timestamp.strftime("%Y%m%d%H%M%S")
            elif timestamp.hour or timestamp.minute:
                time_value = timestamp.strftime("%Y%m%d%H%M")
            else:
                time_value = timestamp.strftime("%Y%m%d")
            fid.write(f"{time_value} {float(row.value):15.6f}\n")
