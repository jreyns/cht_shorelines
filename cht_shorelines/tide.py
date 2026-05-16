from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import (
    NumericTableLike,
    PathLike,
    ShorelinesModelProtocol,
    coerce_numeric_table,
    maybe_path,
    normalize_file_name,
    read_numeric_table,
    write_numeric_table,
)
from .wave_boundary_conditions import (
    _is_numeric_table,
    _is_time_series,
    _read_point_list,
    _water_level_climate_dataframe,
    _water_levels_dataframe,
)


class ShorelinesTide:
    def __init__(self, model: ShorelinesModelProtocol | None = None) -> None:
        self.model = model
        self.tide_type = None
        self.tide_data = None
        self.tide_file = None
        self.water_levels = None
        self.water_level_series = {}
        self.water_level_points = []
        self.tide_profile = None
        self.tide_profile_file = None

    @property
    def root(self) -> Path:
        """
        Return the case root directory.

        Returns
        -------
        pathlib.Path
            Directory used to resolve and write tide-related files.
        """
        if self.model is not None:
            return Path(self.model.path)
        return Path.cwd()

    def set_tide_data(
        self,
        data: NumericTableLike,
        file_name: PathLike = "tide.txt",
    ) -> None:
        """
        Set tide forcing data.

        Parameters
        ----------
        data : array-like
            Numeric tide table. Eleven columns indicate harmonic input; other
            compatible tables are interpreted as water-level forcing.
        file_name : str or pathlib.Path, default "tide.txt"
            Output file name.
        """
        self.tide_data = coerce_numeric_table(data, argument="data")
        self.tide_file = normalize_file_name(file_name)
        self.tide_type = 1 if self.tide_data.ndim == 2 and self.tide_data.shape[1] == 11 else 2
        if self.model is not None:
            self.model.input.variables.tidefile = self.tide_file

    def set_tide_profile(
        self,
        data: NumericTableLike,
        file_name: PathLike = "tideprofile.txt",
    ) -> None:
        """
        Set the tide profile table.

        Parameters
        ----------
        data : array-like
            Numeric tide profile table.
        file_name : str or pathlib.Path, default "tideprofile.txt"
            Output file name.
        """
        self.tide_profile = coerce_numeric_table(data, argument="data")
        self.tide_profile_file = normalize_file_name(file_name)
        if self.model is not None:
            self.model.input.variables.tideprofile = self.tide_profile_file

    def read(self) -> None:
        """
        Read tide forcing and tide profile inputs from the model.
        """
        variables = getattr(self.model.input, "variables", None) if self.model else None
        if variables is None:
            return

        self.tide_type = None
        self.tide_data = None
        self.water_levels = None
        self.water_level_series = {}
        self.water_level_points = []
        self.tide_file = None

        tide_value = getattr(variables, "tidefile", "")
        if isinstance(tide_value, str) and tide_value:
            self._read_tide_file(tide_value)
        elif _is_matrix_like(tide_value):
            self._read_tide_array(tide_value)

        profile_value = getattr(variables, "tideprofile", "")
        if isinstance(profile_value, str) and profile_value:
            path = maybe_path(self.root, profile_value)
            if path is not None and path.exists():
                self.tide_profile = read_numeric_table(path)
                self.tide_profile_file = profile_value
        elif _is_matrix_like(profile_value):
            self.tide_profile = np.asarray(profile_value, dtype=float)

    def write(self) -> None:
        """
        Write tide forcing and tide profile inputs to disk.

        Notes
        -----
        When a model is attached, the corresponding runfile variables are updated
        with the written file names.
        """
        if self.model is None:
            return
        if self.tide_data is not None:
            file_name = self.tide_file or "tide.txt"
            write_numeric_table(self.root / file_name, self.tide_data)
            self.model.input.variables.tidefile = file_name
        elif self.water_levels is not None:
            file_name = self.tide_file or "tide.wat"
            _write_water_levels(self.root / file_name, self.water_levels)
            self.model.input.variables.tidefile = file_name
        if self.tide_profile is not None:
            file_name = self.tide_profile_file or "tideprofile.txt"
            write_numeric_table(self.root / file_name, self.tide_profile)
            self.model.input.variables.tideprofile = file_name

    def _read_tide_file(self, file_name: PathLike) -> None:
        path = maybe_path(self.root, file_name)
        if path is None or not path.exists():
            return
        self.tide_file = file_name
        if _is_numeric_table(path):
            data = read_numeric_table(path)
            if data.size == 0:
                return
            if data.shape[1] == 11:
                self.tide_type = 1
                self.tide_data = data
                return
            self.tide_type = 2
            if _is_time_series(data, path):
                self.water_levels = _water_levels_dataframe(data)
            else:
                self.water_levels = _water_level_climate_dataframe(data)
            return
        self.tide_type = 2
        for point in _read_point_list(path):
            point_path = maybe_path(path.parent, point["file_name"])
            if point_path is None or not point_path.exists():
                continue
            data = read_numeric_table(point_path)
            self.water_level_series[point["file_name"]] = (
                _water_levels_dataframe(data)
                if _is_time_series(data, point_path)
                else _water_level_climate_dataframe(data)
            )
            self.water_level_points.append(point)

    def _read_tide_array(self, value: NumericTableLike) -> None:
        data = np.asarray(value, dtype=float)
        if data.ndim != 2:
            return
        if data.shape[1] == 11:
            self.tide_type = 1
            self.tide_data = data
            return
        self.tide_type = 2
        if data.shape[1] >= 2:
            if data[0, 0] > 1000:
                self.water_levels = _water_levels_dataframe(data)
            else:
                self.water_levels = _water_level_climate_dataframe(data)


def _is_matrix_like(value: object) -> bool:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return arr.ndim == 2 and arr.size > 0


def _write_water_levels(path: Path, data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fid:
        for row in data.itertuples(index=False):
            values = []
            if "time" in data:
                values.append(int(pd.Timestamp(row.time).strftime("%Y%m%d%H%M")))
            values.append(float(row.swl))
            for name in ("htide", "vtide", "refdep", "prob"):
                if name in data:
                    values.append(float(getattr(row, name)))
            if "time" in data:
                first = f"{int(values[0]):12d}"
                rest = "".join(f"{float(v):15.6f}" for v in values[1:])
                fid.write(first + rest + "\n")
            else:
                fid.write("".join(f"{float(v):15.6f}" for v in values) + "\n")
