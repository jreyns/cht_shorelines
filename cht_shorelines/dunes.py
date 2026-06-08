from __future__ import annotations

from pathlib import Path

import pandas as pd

from .initial_conditions import ShorelinesInitialConditions
from .input import Variables
from .io import (
    PathLike,
    RecordDataLike,
    ShorelinesModelProtocol,
    maybe_path,
    normalize_file_name,
    read_numeric_table,
)
from .wave_boundary_conditions import (
    _covers,
    _is_numeric_table,
    _is_time_series,
    _read_point_list,
    _water_level_climate_dataframe,
    _water_levels_dataframe,
    _wind_climate_dataframe,
    _wind_timeseries_dataframe,
)

_UNSET = object()
_PARAMETER_NAMES = (
    "cs",
    "cstill",
    "xtill",
    "perctill",
    "aoverwash",
    "dtdune",
    "duneaw",
    "rhoa",
    "d50r",
    "kw",
    "k",
    "segmaw",
    "maxslope",
    "csmodel",
    "runupform",
    "runupfactor",
    "uz",
    "z",
    "phiwnd0",
    "swl0",
)


class ShorelinesDunes:
    """
    Manage dune geometry, parameters, and dune-specific forcing inputs.

    Parameters
    ----------
    model : ShorelinesModelProtocol, optional
        Attached ShorelineS model instance used for runfile synchronization.
    """

    def __init__(self, model: ShorelinesModelProtocol | None = None) -> None:
        self.model = model
        defaults = Variables()

        self.enabled = bool(defaults.dune)
        for name in _PARAMETER_NAMES:
            setattr(self, name, getattr(defaults, name))

        self.wind = None
        self.wind_file = None
        self.wind_series = {}
        self.wind_points = []
        self.wind_list_file = None

        self.water_levels = None
        self.water_level_file = None
        self.water_level_series = {}
        self.water_level_points = []
        self.water_level_list_file = None

        self._initial_conditions = None if model is None else model.initial_conditions

    @property
    def root(self) -> Path:
        """
        Return the case root directory.

        Returns
        -------
        pathlib.Path
            Directory used to resolve and write dune-related files.
        """
        if self.model is not None:
            return Path(self.model.path)
        return Path.cwd()

    @property
    def initial_conditions(self) -> ShorelinesInitialConditions | None:
        """
        Return the attached initial-conditions component.

        Returns
        -------
        ShorelinesInitialConditions or None
            Component that stores dune geometry and related setup files.
        """
        if self.model is None:
            return self._initial_conditions
        return self.model.initial_conditions

    @property
    def geometry(self):
        """
        Return the dune geometry table.

        Returns
        -------
        numpy.ndarray or None
            Dune rows with columns matching the ``.dun`` file layout.
        """
        initial = self.initial_conditions
        return None if initial is None else initial.dunes

    @property
    def geometry_file(self) -> str | None:
        """
        Return the dune geometry file name.

        Returns
        -------
        str or None
            File name used for dune geometry, when configured.
        """
        initial = self.initial_conditions
        return None if initial is None else initial.dune_file

    def set_enabled(self, enabled: bool = True) -> None:
        """
        Enable or disable dune computations.

        Parameters
        ----------
        enabled : bool, default True
            Whether dune computations should be active in the runfile.
        """
        self.enabled = bool(enabled)
        if self.model is not None:
            self.model.input.variables.dune = int(self.enabled)

    def set_dunes(
        self,
        data,
        file_name: PathLike = "dunes.dun",
    ) -> None:
        """
        Set dune geometry rows.

        Parameters
        ----------
        data : array-like
            Dune table with columns ``x``, ``y``, ``wberm``, ``dfelev``,
            ``dcelev``, and optional dune-material parameters.
        file_name : str or pathlib.Path, default "dunes.dun"
            Output file name for the dune geometry table.
        """
        initial = self.initial_conditions
        if initial is None:
            raise RuntimeError("Dune geometry requires an attached model")
        initial.set_dunes(data, file_name=file_name)
        self.set_enabled(True)

    def configure(
        self,
        *,
        enabled: bool | None = None,
        cs=_UNSET,
        cstill=_UNSET,
        xtill=_UNSET,
        perctill=_UNSET,
        aoverwash=_UNSET,
        dtdune=_UNSET,
        duneaw=_UNSET,
        rhoa=_UNSET,
        d50r=_UNSET,
        kw=_UNSET,
        k=_UNSET,
        segmaw=_UNSET,
        maxslope=_UNSET,
        csmodel=_UNSET,
        runupform=_UNSET,
        runupfactor=_UNSET,
        uz=_UNSET,
        z=_UNSET,
        phiwnd0=_UNSET,
        swl0=_UNSET,
    ) -> None:
        """
        Update dune parameters stored in the ShorelineS runfile.

        Parameters
        ----------
        enabled : bool, optional
            Whether dune computations are enabled.
        cs, cstill, xtill, perctill, aoverwash, dtdune, duneaw, rhoa, d50r, kw, k, segmaw, maxslope, csmodel, runupform, runupfactor, uz, z, phiwnd0, swl0 : object, optional
            Dune, wind, and runup parameters matching ShorelineS runfile keys.
            Parameters omitted from the call are left unchanged.
        """
        if enabled is not None:
            self.set_enabled(enabled)
        updates = {
            "cs": cs,
            "cstill": cstill,
            "xtill": xtill,
            "perctill": perctill,
            "aoverwash": aoverwash,
            "dtdune": dtdune,
            "duneaw": duneaw,
            "rhoa": rhoa,
            "d50r": d50r,
            "kw": kw,
            "k": k,
            "segmaw": segmaw,
            "maxslope": maxslope,
            "csmodel": csmodel,
            "runupform": runupform,
            "runupfactor": runupfactor,
            "uz": uz,
            "z": z,
            "phiwnd0": phiwnd0,
            "swl0": swl0,
        }
        variables = self.model.input.variables if self.model is not None else None
        for name, value in updates.items():
            if value is _UNSET:
                continue
            normalized = normalize_file_name(value) if name == "csmodel" and value else value
            setattr(self, name, normalized)
            if variables is not None:
                setattr(variables, name, normalized)

    def set_wind_static(
        self,
        velocity: float,
        direction: float,
        *,
        measurement_height: float | None = None,
    ) -> None:
        """
        Set static wind conditions for dune evolution.

        Parameters
        ----------
        velocity : float
            Wind velocity in meters per second.
        direction : float
            Wind direction in degrees north.
        measurement_height : float, optional
            Height above the surface where the wind velocity is defined.
        """
        self.wind = None
        self.wind_file = None
        self.wind_series = {}
        self.wind_points = []
        self.wind_list_file = None
        self.configure(uz=float(velocity), phiwnd0=float(direction))
        if measurement_height is not None:
            self.configure(z=float(measurement_height))
        if self.model is not None:
            self.model.input.variables.wndfile = ""

    def set_water_level_static(self, level: float) -> None:
        """
        Set a static still-water level for dune evolution.

        Parameters
        ----------
        level : float
            Still-water level relative to mean sea level.
        """
        self.water_levels = None
        self.water_level_file = None
        self.water_level_series = {}
        self.water_level_points = []
        self.water_level_list_file = None
        self.configure(swl0=float(level))
        if self.model is not None:
            self.model.input.variables.watfile = ""

    def set_wind(
        self,
        data: RecordDataLike,
        file_name: PathLike = "wind.wnd",
    ) -> None:
        """
        Set dune wind forcing from a single file-backed dataset.

        Parameters
        ----------
        data : RecordDataLike
            Wind climate or time series with columns ``uz`` and ``dir`` and
            optional ``time`` and ``prob``.
        file_name : str or pathlib.Path, default "wind.wnd"
            Output file name for the wind forcing.
        """
        self.wind = _normalize_wind_input(data)
        self.wind_file = normalize_file_name(file_name)
        self.wind_series = {}
        self.wind_points = []
        self.wind_list_file = None
        if self.model is not None:
            self.model.input.variables.wndfile = self.wind_file

    def add_wind_point(
        self,
        data: RecordDataLike,
        x: float,
        y: float,
        file_name: PathLike,
    ) -> None:
        """
        Add one spatially varying dune wind point.

        Parameters
        ----------
        data : RecordDataLike
            Wind climate or time series for this point.
        x, y : float
            Point coordinates.
        file_name : str or pathlib.Path
            File name written for this point.
        """
        normalized = normalize_file_name(file_name)
        self.wind_series[normalized] = _normalize_wind_input(data)
        self.wind_points.append({"file_name": normalized, "x": float(x), "y": float(y)})

    def set_spatial_wind(
        self,
        points,
        list_file: PathLike = "wind.wnd",
    ) -> None:
        """
        Set multiple spatial dune wind points.

        Parameters
        ----------
        points : sequence of mapping
            Point definitions containing ``data``, ``x``, ``y``, and
            ``file_name`` entries.
        list_file : str or pathlib.Path, default "wind.wnd"
            File that lists the point-specific wind files and coordinates.
        """
        self.wind = None
        self.wind_file = None
        self.wind_series = {}
        self.wind_points = []
        for point in points:
            self.add_wind_point(point["data"], point["x"], point["y"], point["file_name"])
        self.wind_list_file = normalize_file_name(list_file)
        if self.model is not None:
            self.model.input.variables.wndfile = self.wind_list_file

    def set_water_levels(
        self,
        data: RecordDataLike,
        file_name: PathLike = "waterlevels.wat",
    ) -> None:
        """
        Set dune still-water forcing from a single file-backed dataset.

        Parameters
        ----------
        data : RecordDataLike
            Water-level climate or time series with column ``swl`` and optional
            ``time``, ``htide``, ``vtide``, ``refdep``, and ``prob``.
        file_name : str or pathlib.Path, default "waterlevels.wat"
            Output file name for the water-level forcing.
        """
        self.water_levels = _normalize_water_level_input(data)
        self.water_level_file = normalize_file_name(file_name)
        self.water_level_series = {}
        self.water_level_points = []
        self.water_level_list_file = None
        if self.model is not None:
            self.model.input.variables.watfile = self.water_level_file

    def add_water_level_point(
        self,
        data: RecordDataLike,
        x: float,
        y: float,
        file_name: PathLike,
    ) -> None:
        """
        Add one spatially varying dune water-level point.

        Parameters
        ----------
        data : RecordDataLike
            Water-level climate or time series for this point.
        x, y : float
            Point coordinates.
        file_name : str or pathlib.Path
            File name written for this point.
        """
        normalized = normalize_file_name(file_name)
        self.water_level_series[normalized] = _normalize_water_level_input(data)
        self.water_level_points.append({"file_name": normalized, "x": float(x), "y": float(y)})

    def set_spatial_water_levels(
        self,
        points,
        list_file: PathLike = "waterlevels.wat",
    ) -> None:
        """
        Set multiple spatial dune water-level points.

        Parameters
        ----------
        points : sequence of mapping
            Point definitions containing ``data``, ``x``, ``y``, and
            ``file_name`` entries.
        list_file : str or pathlib.Path, default "waterlevels.wat"
            File that lists the point-specific water-level files and
            coordinates.
        """
        self.water_levels = None
        self.water_level_file = None
        self.water_level_series = {}
        self.water_level_points = []
        for point in points:
            self.add_water_level_point(point["data"], point["x"], point["y"], point["file_name"])
        self.water_level_list_file = normalize_file_name(list_file)
        if self.model is not None:
            self.model.input.variables.watfile = self.water_level_list_file

    def read(self) -> None:
        """
        Read dune parameters and forcing referenced by the attached model.
        """
        variables = getattr(self.model.input, "variables", None) if self.model else None
        if variables is None:
            return

        self.enabled = bool(getattr(variables, "dune", False))
        for name in _PARAMETER_NAMES:
            setattr(self, name, getattr(variables, name))

        self._read_wind_source(getattr(variables, "wndfile", ""))
        self._read_water_level_source(getattr(variables, "watfile", ""))

    def write(self) -> None:
        """
        Write dune parameters and forcing files to disk.

        Notes
        -----
        When a model is attached, matching runfile variables are updated with
        generated file names.
        """
        if self.model is None:
            return

        variables = self.model.input.variables
        variables.dune = int(self.enabled)
        for name in _PARAMETER_NAMES:
            setattr(variables, name, getattr(self, name))

        if self.wind is not None:
            target = self.root / (self.wind_file or "wind.wnd")
            _write_wind(target, self.wind)
            variables.wndfile = target.name
        for file_name, data in self.wind_series.items():
            _write_wind(self.root / file_name, data)
        if self.wind_points:
            list_file = self.wind_list_file or self.wind_file or "wind.wnd"
            _write_point_list(self.root / list_file, self.wind_points)
            variables.wndfile = list_file

        if self.water_levels is not None:
            target = self.root / (self.water_level_file or "waterlevels.wat")
            _write_water_levels(target, self.water_levels)
            variables.watfile = target.name
        for file_name, data in self.water_level_series.items():
            _write_water_levels(self.root / file_name, data)
        if self.water_level_points:
            list_file = self.water_level_list_file or self.water_level_file or "waterlevels.wat"
            _write_point_list(self.root / list_file, self.water_level_points)
            variables.watfile = list_file

    def check_times(self) -> tuple[bool, list[str]]:
        """
        Check whether dune forcing time series cover the simulation period.

        Returns
        -------
        tuple of bool and list of str
            Boolean success flag and coverage messages for incomplete inputs.
        """
        variables = self.model.input.variables if self.model is not None else None
        if variables is None:
            return True, []
        start = pd.Timestamp(variables.reftime)
        end = pd.Timestamp(variables.endofsimulation)
        messages = []
        for label, data in self.wind_series.items():
            ok, message = _covers(data, start, end, label)
            if not ok:
                messages.append(message)
        for label, data in self.water_level_series.items():
            ok, message = _covers(data, start, end, label)
            if not ok:
                messages.append(message)
        if self.wind is not None:
            ok, message = _covers(self.wind, start, end, self.wind_file)
            if not ok:
                messages.append(message)
        if self.water_levels is not None:
            ok, message = _covers(self.water_levels, start, end, self.water_level_file)
            if not ok:
                messages.append(message)
        return len(messages) == 0, messages

    def _read_wind_source(self, value: object) -> None:
        """Read wind forcing from a file or point-list reference."""
        self.wind = None
        self.wind_file = value if isinstance(value, str) and value else None
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
            self.wind = (
                _wind_timeseries_dataframe(data)
                if _is_time_series(data, path)
                else _wind_climate_dataframe(data)
            )
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

    def _read_water_level_source(self, value: object) -> None:
        """Read water-level forcing from a file or point-list reference."""
        self.water_levels = None
        self.water_level_file = value if isinstance(value, str) and value else None
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
            self.water_levels = (
                _water_levels_dataframe(data)
                if _is_time_series(data, path)
                else _water_level_climate_dataframe(data)
            )
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


def _normalize_wind_input(data: RecordDataLike) -> pd.DataFrame:
    """
    Normalize wind input records to a dune-forcing data frame.

    Parameters
    ----------
    data : RecordDataLike
        Wind climate or time-series input.

    Returns
    -------
    pandas.DataFrame
        Normalized data frame with lowercase column names.
    """
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    df = df.rename(columns={col: str(col).lower() for col in df.columns})
    missing = [name for name in ("uz", "dir") if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        columns = ["time", "uz", "dir"] + (["prob"] if "prob" in df.columns else [])
    else:
        columns = ["uz", "dir"] + (["prob"] if "prob" in df.columns else [])
    return df[columns].copy()


def _normalize_water_level_input(data: RecordDataLike) -> pd.DataFrame:
    """
    Normalize water-level input records to a dune-forcing data frame.

    Parameters
    ----------
    data : RecordDataLike
        Water-level climate or time-series input.

    Returns
    -------
    pandas.DataFrame
        Normalized data frame with lowercase column names.
    """
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    df = df.rename(columns={col: str(col).lower() for col in df.columns})
    if "swl" not in df.columns:
        raise ValueError("Missing required columns: swl")
    optional = [name for name in ("htide", "vtide", "refdep", "prob") if name in df.columns]
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        columns = ["time", "swl"] + optional
    else:
        columns = ["swl"] + optional
    return df[columns].copy()


def _write_point_list(path: Path, points) -> None:
    """
    Write a spatial point-list file for dune forcing.

    Parameters
    ----------
    path : pathlib.Path
        Destination path.
    points : sequence of mapping
        Point records containing ``file_name``, ``x``, and ``y``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fid:
        for point in points:
            fid.write(f"{point['file_name']} {point['x']:15.6f} {point['y']:15.6f}\n")


def _write_wind(path: Path, data: pd.DataFrame) -> None:
    """
    Write dune wind forcing to disk.

    Parameters
    ----------
    path : pathlib.Path
        Destination path.
    data : pandas.DataFrame
        Wind climate or time-series data frame.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fid:
        if "time" in data.columns:
            for row in data.itertuples(index=False):
                timestamp = pd.Timestamp(row.time).strftime("%Y%m%d%H%M")
                values = [timestamp, float(row.uz), float(row.dir)]
                if "prob" in data.columns:
                    values.append(float(row.prob))
                fid.write(f"{values[0]:>12}{''.join(f'{value:15.6f}' for value in values[1:])}\n")
            return
        columns = ["uz", "dir"] + (["prob"] if "prob" in data.columns else [])
        for row in data[columns].itertuples(index=False, name=None):
            fid.write("".join(f"{float(value):15.6f}" for value in row) + "\n")


def _write_water_levels(path: Path, data: pd.DataFrame) -> None:
    """
    Write dune water-level forcing to disk.

    Parameters
    ----------
    path : pathlib.Path
        Destination path.
    data : pandas.DataFrame
        Water-level climate or time-series data frame.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    optional = [name for name in ("htide", "vtide", "refdep", "prob") if name in data.columns]
    with path.open("w", encoding="utf-8", newline="\n") as fid:
        if "time" in data.columns:
            for row in data.itertuples(index=False):
                timestamp = pd.Timestamp(row.time).strftime("%Y%m%d%H%M")
                values = [timestamp, float(row.swl)]
                for name in optional:
                    values.append(float(getattr(row, name)))
                fid.write(f"{values[0]:>12}{''.join(f'{value:15.6f}' for value in values[1:])}\n")
            return
        columns = ["swl"] + optional
        for row in data[columns].itertuples(index=False, name=None):
            fid.write("".join(f"{float(value):15.6f}" for value in row) + "\n")
