from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .io import dataframe_from_records, yyyymmddhhmm

class ShorelinesWaveBoundaryConditions:
    def __init__(self, model=None):
        self.model = model
        self.wave_climate = None
        self.wave_climate_file = None
        self.wave_timeseries = {}
        self.spatial_wave_file = None
        self.spatial_wave_points = []
        self.water_levels = None
        self.water_level_file = None
        self.wind = None
        self.wind_file = None

    @property
    def root(self) -> Path:
        if self.model is not None:
            return Path(self.model.path)
        return Path.cwd()

    def set_wave_climate(self, data, file_name="waves.wvc"):
        """Set a wave climate table with columns hs, tp, dir, and optional prob."""
        self.wave_climate = dataframe_from_records(data, ["hs", "tp", "dir"])
        self.wave_climate_file = file_name
        if self.model is not None:
            self.model.input.variables.wvcfile = file_name

    def set_timeseries(self, data, file_name="waves.wvt"):
        """Set a single wave time series with columns time, hs, tp, dir."""
        self.wave_timeseries = {file_name: dataframe_from_records(data, ["time", "hs", "tp", "dir"])}
        self.wave_climate = None
        self.wave_climate_file = file_name
        if self.model is not None:
            self.model.input.variables.wvcfile = file_name

    def add_timeseries_point(
        self,
        data,
        x: float,
        y: float,
        file_name: str,
        hs_factor: float = 1.0,
        dir_offset: float = 0.0,
    ):
        """Add one spatially varying wave time-series point."""
        self.wave_timeseries[file_name] = dataframe_from_records(
            data, ["time", "hs", "tp", "dir"]
        )
        self.spatial_wave_points.append(
            {
                "file_name": file_name,
                "x": x,
                "y": y,
                "hs_factor": hs_factor,
                "dir_offset": dir_offset,
            }
        )

    def set_spatial_timeseries(self, points, list_file="waves.wvt"):
        """Set multiple wave time-series points.

        Each point is a mapping with ``data``, ``x``, ``y``, and ``file_name``.
        Optional keys are ``hs_factor`` and ``dir_offset``.
        """
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
        self.spatial_wave_file = list_file
        if self.model is not None:
            self.model.input.variables.wvcfile = list_file

    def set_water_levels(self, data, file_name="waterlevels.wat"):
        """Set water-level data with columns time and swl."""
        self.water_levels = dataframe_from_records(data, ["time", "swl"])
        self.water_level_file = file_name
        if self.model is not None:
            self.model.input.variables.watfile = file_name

    def set_wind(self, data, file_name="wind.wnd"):
        """Set wind data with columns time, uz, and dir."""
        self.wind = dataframe_from_records(data, ["time", "uz", "dir"])
        self.wind_file = file_name
        if self.model is not None:
            self.model.input.variables.wndfile = file_name

    def read(self):
        pass

    def write(self):
        if self.wave_climate is not None:
            self._write_wave_climate(self.root / (self.wave_climate_file or "waves.wvc"))

        for file_name, data in self.wave_timeseries.items():
            self._write_wave_timeseries(self.root / file_name, data)

        if self.spatial_wave_points:
            list_file = self.spatial_wave_file or self.wave_climate_file or "waves.wvt"
            self._write_spatial_wave_list(self.root / list_file)
            if self.model is not None:
                self.model.input.variables.wvcfile = list_file

        if self.water_levels is not None:
            self._write_water_levels(self.root / (self.water_level_file or "waterlevels.wat"))

        if self.wind is not None:
            self._write_wind(self.root / (self.wind_file or "wind.wnd"))

    def check_times(self) -> Tuple[bool, List[str]]:
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
        if self.water_levels is not None:
            ok, msg = _covers(self.water_levels, start, end, self.water_level_file)
            if not ok:
                messages.append(msg)
        if self.wind is not None:
            ok, msg = _covers(self.wind, start, end, self.wind_file)
            if not ok:
                messages.append(msg)
        return len(messages) == 0, messages

    def _write_wave_climate(self, path: Path):
        cols = ["hs", "tp", "dir"] + (["prob"] if "prob" in self.wave_climate else [])
        _write_rows(path, self.wave_climate[cols].itertuples(index=False, name=None))

    def _write_wave_timeseries(self, path: Path, data):
        rows = (
            (yyyymmddhhmm(row.time), row.hs, row.tp, row.dir)
            for row in data.itertuples(index=False)
        )
        _write_rows(path, rows, first_col_int=True)

    def _write_spatial_wave_list(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as fid:
            for point in self.spatial_wave_points:
                fid.write(
                    f"{point['file_name']} {point['x']:15.6f} {point['y']:15.6f}"
                    f" {point['hs_factor']:10.6f} {point['dir_offset']:10.6f}\n"
                )

    def _write_water_levels(self, path: Path):
        optional = [col for col in ["htide", "vtide", "refdep"] if col in self.water_levels]
        rows = []
        for row in self.water_levels.itertuples(index=False):
            values = [yyyymmddhhmm(row.time), row.swl]
            for col in optional:
                values.append(getattr(row, col))
            rows.append(values)
        _write_rows(path, rows, first_col_int=True)

    def _write_wind(self, path: Path):
        rows = (
            (yyyymmddhhmm(row.time), row.uz, row.dir)
            for row in self.wind.itertuples(index=False)
        )
        _write_rows(path, rows, first_col_int=True)


def _write_rows(path: Path, rows, first_col_int=False):
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


def _covers(data, start, end, label):
    if data is None or "time" not in data:
        return True, ""
    times = pd.to_datetime(data["time"])
    if times.empty:
        return False, f"{label} has no time records"
    if times.min() > start or times.max() < end:
        return False, f"{label} does not cover {start.date()} through {end.date()}"
    return True, ""
