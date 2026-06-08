from __future__ import annotations

from pathlib import Path

from .io import PathLike, ShorelinesModelProtocol, maybe_path, read_numeric_table, write_numeric_table
from .wave_boundary_conditions import (
    _is_numeric_table,
    _is_time_series,
    _read_point_list,
    _wave_climate_dataframe,
    _wave_timeseries_dataframe,
    _water_level_climate_dataframe,
    _water_levels_dataframe,
)


class ShorelinesRunup:
    def __init__(self, model: ShorelinesModelProtocol | None = None) -> None:
        self.model = model
        self.water_levels = None
        self.water_level_file = None
        self.water_level_series = {}
        self.water_level_points = []
        self.water_locations = None
        self.water_location_file = None
        self.wave_conditions = None
        self.wave_file = None
        self.wave_series = {}
        self.wave_points = []
        self.wave_locations = None
        self.wave_location_file = None

    @property
    def root(self) -> Path:
        """
        Return the case root directory.

        Returns
        -------
        pathlib.Path
            Directory used to resolve and write runup-related files.
        """
        if self.model is not None:
            return Path(self.model.path)
        return Path.cwd()

    def read(self) -> None:
        """
        Read runup water levels, wave conditions, and location files.

        Notes
        -----
        Sources are read from file names referenced by the attached model input.
        """
        variables = getattr(self.model.input, "variables", None) if self.model else None
        if variables is None:
            return
        self._read_water_source(getattr(variables, "watfile", "") or getattr(variables, "watclimfile", ""))
        self._read_wave_source(
            getattr(variables, "wvdfile", "")
            or getattr(variables, "wvcfile", "")
            or getattr(variables, "waveclimfile", "")
        )
        self._read_locations(getattr(variables, "watlocfile", ""), water=True)
        self._read_locations(getattr(variables, "WaveLocfile", ""), water=False)

    def write(self) -> None:
        """
        Write runup location tables to disk.

        Notes
        -----
        This component currently writes only location files. Runup forcing files
        are read-only through this interface.
        """
        if self.model is None:
            return
        if self.water_locations is not None:
            file_name = self.water_location_file or "water_locations.txt"
            write_numeric_table(self.root / file_name, self.water_locations)
            self.model.input.variables.watlocfile = file_name
        if self.wave_locations is not None:
            file_name = self.wave_location_file or "wave_locations.txt"
            write_numeric_table(self.root / file_name, self.wave_locations)
            self.model.input.variables.WaveLocfile = file_name

    def _read_water_source(self, value: object) -> None:
        self.water_levels = None
        self.water_level_file = None
        self.water_level_series = {}
        self.water_level_points = []
        if not value:
            return
        path = maybe_path(self.root, value)
        if path is None or not path.exists():
            return
        self.water_level_file = value
        if _is_numeric_table(path):
            data = read_numeric_table(path)
            if data.size == 0:
                return
            if _is_time_series(data, path):
                self.water_levels = _water_levels_dataframe(data)
            else:
                self.water_levels = _water_level_climate_dataframe(data)
            return
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

    def _read_wave_source(self, value: object) -> None:
        self.wave_conditions = None
        self.wave_file = None
        self.wave_series = {}
        self.wave_points = []
        if not value:
            return
        path = maybe_path(self.root, value)
        if path is None or not path.exists():
            return
        self.wave_file = value
        if _is_numeric_table(path):
            data = read_numeric_table(path)
            if data.size == 0:
                return
            self.wave_conditions = (
                _wave_timeseries_dataframe(data)
                if _is_time_series(data, path)
                else _wave_climate_dataframe(data)
            )
            return
        for point in _read_point_list(path):
            point_path = maybe_path(path.parent, point["file_name"])
            if point_path is None or not point_path.exists():
                continue
            data = read_numeric_table(point_path)
            self.wave_series[point["file_name"]] = (
                _wave_timeseries_dataframe(data)
                if _is_time_series(data, point_path)
                else _wave_climate_dataframe(data)
            )
            self.wave_points.append(point)

    def _read_locations(self, file_name: PathLike | str, water: bool) -> None:
        if not file_name:
            return
        path = maybe_path(self.root, file_name)
        if path is None or not path.exists():
            return
        data = read_numeric_table(path)
        if data.size == 0:
            return
        if water:
            self.water_locations = data[:, :2]
            self.water_location_file = file_name
        else:
            self.wave_locations = data[:, :2]
            self.wave_location_file = file_name
