from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from .io import (
    NumericTableLike,
    PathLike,
    RecordDataLike,
    ShorelinesModelProtocol,
    coerce_numeric_table,
    dataframe_from_records,
    maybe_path,
    normalize_file_name,
    normalize_probabilities,
    parse_compact_datetime,
    read_numeric_table,
    yyyymmddhhmm,
)


class BoundaryPoint(TypedDict):
    file_name: str
    x: float
    y: float
    hs_factor: float
    dir_offset: float


class ShorelinesWaveBoundaryConditions:
    def __init__(self, model: ShorelinesModelProtocol | None = None) -> None:
        self.model = model
        self.wave_climate = None
        self.wave_climate_file = None
        self.wave_timeseries = {}
        self.spatial_wave_file = None
        self.spatial_wave_points = []
        self.water_levels = None
        self.water_level_file = None
        self.water_level_series = {}
        self.water_level_list_file = None
        self.water_level_points = []
        self.wind = None
        self.wind_file = None
        self.wind_series = {}
        self.wind_list_file = None
        self.wind_points = []

    @property
    def root(self) -> Path:
        """
        Return the case root directory.

        Returns
        -------
        pathlib.Path
            Directory used to resolve and write boundary-condition files.
        """
        if self.model is not None:
            return Path(self.model.path)
        return Path.cwd()

    def set_wave_climate(
        self,
        data: RecordDataLike,
        file_name: PathLike = "waves.wvc",
    ) -> None:
        """Set a wave climate table with columns hs, tp, dir, and optional prob."""
        self.wave_climate = dataframe_from_records(data, ["hs", "tp", "dir"])
        self.wave_climate_file = normalize_file_name(file_name)
        self.wave_timeseries = {}
        self.spatial_wave_points = []
        self.spatial_wave_file = None
        if self.model is not None:
            self.model.input.variables.wvcfile = self.wave_climate_file

    def set_timeseries(
        self,
        data: RecordDataLike,
        file_name: PathLike = "waves.wvt",
    ) -> None:
        """Set a single wave time series with columns time, hs, tp, dir."""
        normalized_file_name = normalize_file_name(file_name)
        self.wave_timeseries = {
            normalized_file_name: dataframe_from_records(data, ["time", "hs", "tp", "dir"])
        }
        self.wave_climate = None
        self.wave_climate_file = normalized_file_name
        self.spatial_wave_points = []
        self.spatial_wave_file = None
        if self.model is not None:
            self.model.input.variables.wvcfile = normalized_file_name

    def add_timeseries_point(
        self,
        data: RecordDataLike,
        x: float,
        y: float,
        file_name: PathLike,
        hs_factor: float = 1.0,
        dir_offset: float = 0.0,
    ) -> None:
        """Add one spatially varying wave time-series point."""
        normalized_file_name = normalize_file_name(file_name)
        self.wave_timeseries[normalized_file_name] = dataframe_from_records(
            data, ["time", "hs", "tp", "dir"]
        )
        self.spatial_wave_points.append(
            {
                "file_name": normalized_file_name,
                "x": float(x),
                "y": float(y),
                "hs_factor": float(hs_factor),
                "dir_offset": float(dir_offset),
            }
        )

    def set_spatial_timeseries(
        self,
        points: Sequence[Mapping[str, object]],
        list_file: PathLike = "waves.wvt",
    ) -> None:
        """Set multiple wave time-series points.

        Each point is a mapping with ``data``, ``x``, ``y``, and ``file_name``.
        Optional keys are ``hs_factor`` and ``dir_offset``.
        """
        self.wave_climate = None
        self.wave_timeseries = {}
        self.spatial_wave_points = []
        for point in points:
            self.add_timeseries_point(
                point["data"],
                point["x"],
                point["y"],
                point["file_name"],
                point.get("hs_factor", 1.0),
                point.get("dir_offset", 0.0),
            )
        self.spatial_wave_file = normalize_file_name(list_file)
        if self.model is not None:
            self.model.input.variables.wvcfile = self.spatial_wave_file

    def set_water_levels(
        self,
        data: RecordDataLike,
        file_name: PathLike = "waterlevels.wat",
    ) -> None:
        """Set water-level data with columns time and swl."""
        self.water_levels = dataframe_from_records(data, ["time", "swl"])
        self.water_level_file = normalize_file_name(file_name)
        self.water_level_series = {}
        self.water_level_points = []
        self.water_level_list_file = None
        if self.model is not None:
            self.model.input.variables.watfile = self.water_level_file

    def set_wind(
        self,
        data: RecordDataLike,
        file_name: PathLike = "wind.wnd",
    ) -> None:
        """Set wind data with columns time, uz, and dir."""
        self.wind = dataframe_from_records(data, ["time", "uz", "dir"])
        self.wind_file = normalize_file_name(file_name)
        self.wind_series = {}
        self.wind_points = []
        self.wind_list_file = None
        if self.model is not None:
            self.model.input.variables.wndfile = self.wind_file

    def read(self) -> None:
        """
        Read wave, water-level, and wind boundary inputs from the model.
        """
        variables = getattr(self.model.input, "variables", None) if self.model else None
        if variables is None:
            return

        self._read_wave_source(getattr(variables, "wvcfile", ""))
        self._read_water_level_source(getattr(variables, "watfile", ""))
        self._read_wind_source(getattr(variables, "wndfile", ""))

    def write(self) -> None:
        """
        Write configured boundary-condition inputs to disk.

        Notes
        -----
        When a model is attached, the corresponding runfile variables are updated
        with the written file names.
        """
        if self.wave_climate is not None:
            self._write_wave_climate(
                self.root / (self.wave_climate_file or "waves.wvc")
            )

        for file_name, data in self.wave_timeseries.items():
            self._write_wave_timeseries(self.root / file_name, data)

        if self.spatial_wave_points:
            list_file = self.spatial_wave_file or self.wave_climate_file or "waves.wvt"
            self._write_spatial_point_list(self.root / list_file, self.spatial_wave_points)
            if self.model is not None:
                self.model.input.variables.wvcfile = list_file

        if self.water_levels is not None:
            self._write_water_levels(
                self.root / (self.water_level_file or "waterlevels.wat")
            )

        for file_name, data in self.water_level_series.items():
            self._write_water_levels(self.root / file_name, data)

        if self.water_level_points:
            list_file = self.water_level_list_file or self.water_level_file or "waterlevels.wat"
            self._write_spatial_point_list(self.root / list_file, self.water_level_points)
            if self.model is not None:
                self.model.input.variables.watfile = list_file

        if self.wind is not None:
            self._write_wind(self.root / (self.wind_file or "wind.wnd"))

        for file_name, data in self.wind_series.items():
            self._write_wind(self.root / file_name, data)

        if self.wind_points:
            list_file = self.wind_list_file or self.wind_file or "wind.wnd"
            self._write_spatial_point_list(self.root / list_file, self.wind_points)
            if self.model is not None:
                self.model.input.variables.wndfile = list_file

    def check_times(self) -> tuple[bool, list[str]]:
        """
        Check whether time-series forcing covers the simulation period.

        Returns
        -------
        tuple of bool and list of str
            Boolean success flag and coverage messages for incomplete inputs.
        """
        messages = []
        variables = self.model.input.variables if self.model is not None else None
        if variables is None:
            return True, messages
        start = pd.Timestamp(variables.reftime)
        end = pd.Timestamp(variables.endofsimulation)

        for label, data in self.wave_timeseries.items():
            ok, msg = _covers(data, start, end, label)
            if not ok:
                messages.append(msg)
        for label, data in self.water_level_series.items():
            ok, msg = _covers(data, start, end, label)
            if not ok:
                messages.append(msg)
        for label, data in self.wind_series.items():
            ok, msg = _covers(data, start, end, label)
            if not ok:
                messages.append(msg)
        if self.water_levels is not None:
            ok, msg = _covers(self.water_levels, start, end, self.water_level_file)
            if not ok:
                messages.append(msg)
        if self.wind is not None:
            ok, msg = _covers(self.wind, start, end, self.wind_file)
            if not ok:
                messages.append(msg)
        return len(messages) == 0, messages

    def _read_wave_source(self, value: object) -> None:
        self.wave_climate = None
        self.wave_climate_file = None
        self.wave_timeseries = {}
        self.spatial_wave_points = []
        self.spatial_wave_file = None
        if not value:
            return
        path = maybe_path(self.root, value)
        if path is None or not path.exists():
            return
        if _is_numeric_table(path):
            data = read_numeric_table(path)
            if data.size == 0:
                return
            if _is_time_series(data, path):
                self.wave_timeseries[path.name] = _wave_timeseries_dataframe(data)
                self.wave_climate_file = path.name
            else:
                self.wave_climate = _wave_climate_dataframe(data)
                self.wave_climate_file = path.name
            return
        self.spatial_wave_file = path.name
        for point in _read_point_list(path):
            point_path = maybe_path(path.parent, point["file_name"])
            if point_path is None or not point_path.exists():
                continue
            data = read_numeric_table(point_path)
            self.wave_timeseries[point["file_name"]] = (
                _wave_timeseries_dataframe(data)
                if _is_time_series(data, point_path)
                else _wave_climate_dataframe(data)
            )
            self.spatial_wave_points.append(point)

    def _read_water_level_source(self, value: object) -> None:
        self.water_levels = None
        self.water_level_file = None
        self.water_level_series = {}
        self.water_level_points = []
        self.water_level_list_file = None
        if not value:
            return
        path = maybe_path(self.root, value)
        if path is None or not path.exists():
            return
        if _is_numeric_table(path):
            data = read_numeric_table(path)
            if data.size == 0:
                return
            if _is_time_series(data, path):
                self.water_levels = _water_levels_dataframe(data)
            else:
                self.water_levels = _water_level_climate_dataframe(data)
            self.water_level_file = path.name
            return
        self.water_level_list_file = path.name
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

    def _read_wind_source(self, value: object) -> None:
        self.wind = None
        self.wind_file = None
        self.wind_series = {}
        self.wind_points = []
        self.wind_list_file = None
        if not value:
            return
        path = maybe_path(self.root, value)
        if path is None or not path.exists():
            return
        if _is_numeric_table(path):
            data = read_numeric_table(path)
            if data.size == 0:
                return
            if _is_time_series(data, path):
                self.wind = _wind_timeseries_dataframe(data)
            else:
                self.wind = _wind_climate_dataframe(data)
            self.wind_file = path.name
            return
        self.wind_list_file = path.name
        for point in _read_point_list(path):
            point_path = maybe_path(path.parent, point["file_name"])
            if point_path is None or not point_path.exists():
                continue
            data = read_numeric_table(point_path)
            self.wind_series[point["file_name"]] = (
                _wind_timeseries_dataframe(data)
                if _is_time_series(data, point_path)
                else _wind_climate_dataframe(data)
            )
            self.wind_points.append(point)

    def _write_wave_climate(self, path: Path) -> None:
        self._write_wave_climate_data(path, self.wave_climate)

    def _write_wave_timeseries(self, path: Path, data: pd.DataFrame) -> None:
        if "time" not in data:
            self._write_wave_climate_data(path, data)
            return
        rows = (
            (yyyymmddhhmm(row.time), row.hs, row.tp, row.dir)
            for row in data.itertuples(index=False)
        )
        _write_rows(path, rows, first_col_int=True)

    def _write_wave_climate_data(self, path: Path, data: pd.DataFrame) -> None:
        cols = ["hs", "tp", "dir"] + (["prob"] if "prob" in data else [])
        _write_rows(path, data[cols].itertuples(index=False, name=None))

    def _write_spatial_point_list(
        self,
        path: Path,
        points: Sequence[BoundaryPoint],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as fid:
            for point in points:
                fid.write(
                    f"{point['file_name']} {point['x']:15.6f} {point['y']:15.6f}"
                    f" {point.get('hs_factor', 1.0):10.6f}"
                    f" {point.get('dir_offset', 0.0):10.6f}\n"
                )

    def _write_water_levels(self, path: Path, data: pd.DataFrame | None = None) -> None:
        data = self.water_levels if data is None else data
        optional = [col for col in ["htide", "vtide", "refdep", "prob"] if col in data]
        rows = []
        for row in data.itertuples(index=False):
            values = [yyyymmddhhmm(row.time), row.swl] if "time" in data else [row.swl]
            for col in optional:
                values.append(getattr(row, col))
            rows.append(values)
        _write_rows(path, rows, first_col_int="time" in data)

    def _write_wind(self, path: Path, data: pd.DataFrame | None = None) -> None:
        data = self.wind if data is None else data
        if "time" in data:
            rows = (
                (yyyymmddhhmm(row.time), row.uz, row.dir)
                for row in data.itertuples(index=False)
            )
            _write_rows(path, rows, first_col_int=True)
            return
        cols = ["uz", "dir"] + (["prob"] if "prob" in data else [])
        _write_rows(path, data[cols].itertuples(index=False, name=None))


def _is_numeric_table(path: Path) -> bool:
    try:
        read_numeric_table(path)
    except ValueError:
        return False
    return True


def _is_time_series(data: np.ndarray, path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in {".wvt", ".wvd", ".wlt", ".wdt"}:
        return True
    return bool(data.size) and data[0, 0] > 1000


def _read_point_list(path: Path) -> list[BoundaryPoint]:
    points: list[BoundaryPoint] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("%", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            x = float(parts[1])
            y = float(parts[2])
            file_name = parts[0]
            hs_factor = float(parts[3]) if len(parts) > 3 else 1.0
            dir_offset = float(parts[4]) if len(parts) > 4 else 0.0
        except ValueError:
            x = float(parts[0])
            y = float(parts[1])
            file_name = parts[2]
            hs_factor = float(parts[3]) if len(parts) > 3 else 1.0
            dir_offset = float(parts[4]) if len(parts) > 4 else 0.0
        points.append(
            {
                "file_name": file_name,
                "x": x,
                "y": y,
                "hs_factor": hs_factor,
                "dir_offset": dir_offset,
            }
        )
    return points


def _wave_climate_dataframe(data: NumericTableLike) -> pd.DataFrame:
    data = coerce_numeric_table(data, argument="data", min_columns=3)
    columns = ["hs", "tp", "dir"] + (["prob"] if data.shape[1] > 3 else [])
    df = pd.DataFrame(data[:, : len(columns)], columns=columns)
    if "prob" in df:
        df["prob"] = normalize_probabilities(df["prob"])
    return df


def _wave_timeseries_dataframe(data: NumericTableLike) -> pd.DataFrame:
    data = coerce_numeric_table(data, argument="data", min_columns=4)
    df = pd.DataFrame(
        {
            "time": [parse_compact_datetime(value) for value in data[:, 0]],
            "hs": data[:, 1],
            "tp": data[:, 2],
            "dir": data[:, 3],
        }
    )
    if data.shape[1] > 4:
        df["prob"] = normalize_probabilities(data[:, 4])
    return df


def _water_levels_dataframe(data: NumericTableLike) -> pd.DataFrame:
    data = coerce_numeric_table(data, argument="data", min_columns=2)
    payload = {
        "time": [parse_compact_datetime(value) for value in data[:, 0]],
        "swl": data[:, 1],
    }
    if data.shape[1] > 2:
        payload["htide"] = data[:, 2]
    if data.shape[1] > 3:
        payload["vtide"] = data[:, 3]
    if data.shape[1] > 4:
        payload["refdep"] = data[:, 4]
    if data.shape[1] > 5:
        payload["prob"] = normalize_probabilities(data[:, 5])
    return pd.DataFrame(payload)


def _water_level_climate_dataframe(data: NumericTableLike) -> pd.DataFrame:
    data = coerce_numeric_table(data, argument="data", min_columns=1)
    payload = {"swl": data[:, 0]}
    if data.shape[1] == 2:
        payload["prob"] = normalize_probabilities(data[:, 1])
        return pd.DataFrame(payload)
    if data.shape[1] > 1:
        payload["htide"] = data[:, 1]
    if data.shape[1] > 2:
        payload["vtide"] = data[:, 2]
    if data.shape[1] > 3:
        payload["refdep"] = data[:, 3]
    if data.shape[1] == 5:
        payload["prob"] = normalize_probabilities(data[:, 4])
    return pd.DataFrame(payload)


def _wind_timeseries_dataframe(data: NumericTableLike) -> pd.DataFrame:
    data = coerce_numeric_table(data, argument="data", min_columns=3)
    payload = {
        "time": [parse_compact_datetime(value) for value in data[:, 0]],
        "uz": data[:, 1],
        "dir": data[:, 2],
    }
    if data.shape[1] > 3:
        payload["prob"] = normalize_probabilities(data[:, 3])
    return pd.DataFrame(payload)


def _wind_climate_dataframe(data: NumericTableLike) -> pd.DataFrame:
    data = coerce_numeric_table(data, argument="data", min_columns=2)
    payload = {"uz": data[:, 0], "dir": data[:, 1]}
    if data.shape[1] > 2:
        payload["prob"] = normalize_probabilities(data[:, 2])
    return pd.DataFrame(payload)


def _write_rows(
    path: Path,
    rows: Iterable[Sequence[object]],
    first_col_int: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fid:
        for row in rows:
            values = list(row)
            if first_col_int:
                first = f"{int(values[0]):12d}"
                rest = "".join(f"{float(v):15.6f}" for v in values[1:])
                fid.write(first + rest + "\n")
            else:
                fid.write("".join(f"{float(v):15.6f}" for v in values) + "\n")


def _covers(
    data: pd.DataFrame | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    label: str | None,
) -> tuple[bool, str]:
    if data is None or "time" not in data:
        return True, ""
    times = pd.to_datetime(data["time"])
    if times.empty:
        return False, f"{label} has no time records"
    if times.min() > start or times.max() < end:
        return False, f"{label} does not cover {start.date()} through {end.date()}"
    return True, ""
