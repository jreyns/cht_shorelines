from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import (
    CoordinateArray,
    NumericTableLike,
    PathLike,
    ShorelinesModelProtocol,
    coerce_numeric_table,
    normalize_file_name,
    read_numeric_table,
    read_xy,
    validate_xy_sections,
    write_numeric_table,
    write_xy,
    xy_columns_to_sections,
)


class ShorelinesInitialConditions:
    def __init__(self, model: ShorelinesModelProtocol | None = None) -> None:
        self.model = model
        self.dunes = None
        self.dune_file = None
        self.sediment_limiter = None
        self.sediment_limiter_file = None
        self.channel = None
        self.channel_file = None
        self.spit_polygon = None
        self.spit_file = None
        self.flood_delta = None
        self.flood_delta_file = None
        self.river_discharges = None
        self.river_file = None
        self.mangroves = None
        self.mangrove_file = None

    @property
    def root(self) -> Path:
        """
        Return the case root directory.

        Returns
        -------
        pathlib.Path
            Directory used to resolve and write initial-condition files.
        """
        if self.model is not None:
            return Path(self.model.path)
        return Path.cwd()

    def set_dunes(
        self,
        data: NumericTableLike,
        file_name: PathLike = "dunes.dun",
    ) -> None:
        """Set dune rows: x, y, wberm, dfelev, dcelev, optional cs/cstill/xtill/perctill."""
        self.dunes = coerce_numeric_table(data, argument="data", min_columns=5)
        self.dune_file = normalize_file_name(file_name)
        if self.model is not None:
            variables = self.model.input.variables
            variables.dune = 1
            variables.ldbdune = self.dune_file

    def set_sediment_limiter(
        self,
        coordinates: CoordinateArray,
        width: NumericTableLike | None = None,
        file_name: PathLike = "sediment_limiter.ldb",
    ) -> None:
        """Set sediment-limiter coordinates, optionally with per-point width."""
        arr = coerce_numeric_table(
            coordinates,
            argument="coordinates",
            exact_columns=2,
        )
        if width is not None:
            width_arr = np.asarray(width, dtype=float).reshape(-1, 1)
            if width_arr.shape[0] not in {1, arr.shape[0]}:
                raise ValueError("width must contain one value or one value per coordinate")
            arr = np.column_stack([arr, width_arr])
        self.sediment_limiter = arr
        self.sediment_limiter_file = normalize_file_name(file_name)
        if self.model is not None:
            variables = self.model.input.variables
            variables.sedlim = 1
            variables.ldbsedlim = self.sediment_limiter_file

    def set_channel_axis(
        self,
        coordinates: CoordinateArray,
        file_name: PathLike = "channel.ldb",
    ) -> None:
        """
        Set the channel axis polyline.

        Parameters
        ----------
        coordinates : numpy.ndarray
            NaN-separated ``Nx2`` coordinate array.
        file_name : str or pathlib.Path, default "channel.ldb"
            Output file name.
        """
        self.channel = validate_xy_sections(coordinates)
        self.channel_file = normalize_file_name(file_name)
        if self.model is not None:
            variables = self.model.input.variables
            variables.channel = 1
            variables.ldbchannel = self.channel_file

    def set_spit_polygon(
        self,
        coordinates: CoordinateArray,
        file_name: PathLike = "spit.ldb",
    ) -> None:
        """
        Set the spit polygon coordinates.

        Parameters
        ----------
        coordinates : numpy.ndarray
            NaN-separated ``Nx2`` coordinate array.
        file_name : str or pathlib.Path, default "spit.ldb"
            Output file name.
        """
        self.spit_polygon = validate_xy_sections(coordinates)
        self.spit_file = normalize_file_name(file_name)
        if self.model is not None:
            self.model.input.variables.ldbspit = self.spit_file

    def set_flood_delta(
        self,
        coordinates: CoordinateArray,
        file_name: PathLike = "flood_delta.ldb",
    ) -> None:
        """
        Set the flood-delta polygon coordinates.

        Parameters
        ----------
        coordinates : numpy.ndarray
            NaN-separated ``Nx2`` coordinate array.
        file_name : str or pathlib.Path, default "flood_delta.ldb"
            Output file name.
        """
        self.flood_delta = validate_xy_sections(coordinates)
        self.flood_delta_file = normalize_file_name(file_name)
        if self.model is not None:
            variables = self.model.input.variables
            variables.flooddelta = 1
            variables.ldbflood = self.flood_delta_file

    def set_river_discharges(
        self,
        data: NumericTableLike,
        file_name: PathLike = "river_discharge.riv",
    ) -> None:
        """Set mud river rows: xriv1, yriv1, xriv2, yriv2, tstart, tend, rate."""
        self.river_discharges = coerce_numeric_table(data, argument="data", min_columns=7)
        self.river_file = normalize_file_name(file_name)
        if self.model is not None:
            variables = self.model.input.variables
            variables.mud = 1
            variables.ldbriverdisch = self.river_file

    def set_mangroves(
        self,
        data: NumericTableLike,
        file_name: PathLike = "mangroves.mgv",
    ) -> None:
        """Set mangrove rows: xmgv, ymgv, Bf, Bm, Bfm."""
        self.mangroves = coerce_numeric_table(data, argument="data", min_columns=5)
        self.mangrove_file = normalize_file_name(file_name)
        if self.model is not None:
            variables = self.model.input.variables
            variables.mud = 1
            variables.ldbmangrove = self.mangrove_file

    def read(self) -> None:
        """
        Read supported initial-condition inputs from the model.

        Notes
        -----
        File-based inputs are preferred. When files are absent, compatible values
        stored directly in the runfile are converted into arrays.
        """
        variables = getattr(self.model.input, "variables", None) if self.model else None
        if variables is None:
            return

        self._read_numeric("ldbdune", "dunes", "dune_file")
        self._read_numeric("ldbsedlim", "sediment_limiter", "sediment_limiter_file")
        self._read_xy("ldbchannel", "channel", "channel_file")
        self._read_xy("ldbspit", "spit_polygon", "spit_file")
        self._read_xy("ldbflood", "flood_delta", "flood_delta_file")
        self._read_numeric("ldbriverdisch", "river_discharges", "river_file")
        self._read_numeric("ldbmangrove", "mangroves", "mangrove_file")

        if self.dunes is None and _has_data(getattr(variables, "xdune", "")):
            self.dunes = _stack_optional_columns(
                variables.xdune,
                variables.ydune,
                variables.wberm,
                variables.dfelev,
                variables.dcelev,
                variables.cs,
                variables.cstill,
                variables.xtill,
                variables.perctill,
            )

        if self.sediment_limiter is None and _has_data(getattr(variables, "xsedlim", "")):
            data = xy_columns_to_sections(variables.xsedlim, variables.ysedlim)
            self.sediment_limiter = _append_width_column(data, getattr(variables, "widthsedlim", []))

        if self.channel is None and _has_data(getattr(variables, "xrmc", "")):
            self.channel = xy_columns_to_sections(variables.xrmc, variables.yrmc)

        if self.spit_polygon is None and _has_data(getattr(variables, "xspitpol", "")):
            self.spit_polygon = xy_columns_to_sections(variables.xspitpol, variables.yspitpol)

        if self.flood_delta is None and _has_data(getattr(variables, "xfloodpol", "")):
            self.flood_delta = xy_columns_to_sections(variables.xfloodpol, variables.yfloodpol)

    def write(self) -> None:
        """
        Write configured initial-condition inputs to disk.

        Notes
        -----
        When a model is attached, the corresponding runfile variables are updated
        with the written file names.
        """
        if self.dunes is not None:
            file_name = self.dune_file or "dunes.dun"
            write_numeric_table(self.root / file_name, self.dunes)
            if self.model is not None:
                self.model.input.variables.ldbdune = file_name

        if self.sediment_limiter is not None:
            file_name = self.sediment_limiter_file or "sediment_limiter.ldb"
            write_numeric_table(self.root / file_name, self.sediment_limiter)
            if self.model is not None:
                self.model.input.variables.ldbsedlim = file_name

        if self.channel is not None:
            file_name = self.channel_file or "channel.ldb"
            write_xy(self.root / file_name, self.channel)
            if self.model is not None:
                self.model.input.variables.ldbchannel = file_name

        if self.spit_polygon is not None:
            file_name = self.spit_file or "spit.ldb"
            write_xy(self.root / file_name, self.spit_polygon)
            if self.model is not None:
                self.model.input.variables.ldbspit = file_name

        if self.flood_delta is not None:
            file_name = self.flood_delta_file or "flood_delta.ldb"
            write_xy(self.root / file_name, self.flood_delta)
            if self.model is not None:
                self.model.input.variables.ldbflood = file_name

        if self.river_discharges is not None:
            file_name = self.river_file or "river_discharge.riv"
            write_numeric_table(self.root / file_name, self.river_discharges)
            if self.model is not None:
                self.model.input.variables.ldbriverdisch = file_name

        if self.mangroves is not None:
            file_name = self.mangrove_file or "mangroves.mgv"
            write_numeric_table(self.root / file_name, self.mangroves)
            if self.model is not None:
                self.model.input.variables.ldbmangrove = file_name

    def _read_numeric(
        self,
        variable_name: str,
        attribute_name: str,
        file_attribute_name: str,
    ) -> None:
        file_name = getattr(self.model.input.variables, variable_name, "")
        if not file_name:
            return
        path = self.root / file_name
        if not path.exists():
            return
        setattr(self, attribute_name, read_numeric_table(path))
        setattr(self, file_attribute_name, file_name)

    def _read_xy(
        self,
        variable_name: str,
        attribute_name: str,
        file_attribute_name: str,
    ) -> None:
        file_name = getattr(self.model.input.variables, variable_name, "")
        if not file_name:
            return
        path = self.root / file_name
        if not path.exists():
            return
        setattr(self, attribute_name, read_xy(path))
        setattr(self, file_attribute_name, file_name)


def _stack_optional_columns(*columns: object) -> np.ndarray:
    arrays = [np.asarray(column, dtype=float).reshape(-1) for column in columns]
    length = max(arr.size for arr in arrays if arr.size)
    normalized = []
    for arr in arrays:
        if arr.size == 0:
            normalized.append(np.full(length, np.nan))
        elif arr.size == 1 and length > 1:
            normalized.append(np.full(length, float(arr[0])))
        elif arr.size == length:
            normalized.append(arr)
        else:
            raise ValueError("Incompatible column length in initial condition vectors")
    return np.column_stack(normalized)


def _append_width_column(
    data: np.ndarray | list[np.ndarray],
    width: NumericTableLike | None,
) -> np.ndarray:
    if width is None:
        width_arr = np.asarray([], dtype=float)
    else:
        width_arr = np.asarray(width, dtype=float).reshape(-1)
    if isinstance(data, list):
        sections = data
        if width_arr.size:
            offset = 0
            combined = []
            for section in sections:
                local = section
                count = len(section)
                if width_arr.size == 1:
                    local_width = np.full(count, width_arr[0])
                else:
                    local_width = width_arr[offset : offset + count]
                combined.append(np.column_stack([local, local_width]))
                offset += count
            rows = []
            for index, section in enumerate(combined):
                if index:
                    rows.append([np.nan, np.nan, np.nan])
                rows.extend(section.tolist())
            return np.asarray(rows, dtype=float)
        rows = []
        for index, section in enumerate(sections):
            if index:
                rows.append([np.nan, np.nan])
            rows.extend(section.tolist())
        return np.asarray(rows, dtype=float)
    if width_arr.size:
        if width_arr.size == 1 and len(data) > 1:
            width_arr = np.full(len(data), width_arr[0])
        return np.column_stack([np.asarray(data, dtype=float), width_arr])
    return np.asarray(data, dtype=float)


def _has_data(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    try:
        return len(value) > 0
    except TypeError:
        return True
