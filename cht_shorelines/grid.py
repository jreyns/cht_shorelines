from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import (
    NumericTableLike,
    PathLike,
    ShorelinesModelProtocol,
    normalize_file_name,
    read_xy,
    validate_xy_sections,
    write_numeric_table,
    write_xy,
    xy_columns_to_sections,
)


class ShorelinesDomain:
    def __init__(self, model: ShorelinesModelProtocol | None = None) -> None:
        self.model = model
        self.coastline = None
        self.coastline_file = None
        self.extra_xy_files = {}

    @property
    def root(self) -> Path:
        if self.model is not None:
            return Path(self.model.path)
        return Path.cwd()

    def set_coastline(
        self,
        coordinates: object,
        file_name: PathLike = "coastline.ldb",
    ) -> None:
        """Set the initial coastline as one or more Nx2 coordinate sections."""
        self.coastline = validate_xy_sections(coordinates)
        self.coastline_file = normalize_file_name(file_name)
        if self.model is not None:
            self.model.input.variables.ldbcoastline = self.coastline_file

    def set_xy_file(
        self,
        variable_name: str,
        coordinates: object,
        file_name: PathLike,
    ) -> None:
        """Set a generic ShorelineS x/y input file variable."""
        normalized_file_name = normalize_file_name(file_name)
        validated_coordinates = validate_xy_sections(coordinates)
        self.extra_xy_files[variable_name] = (normalized_file_name, validated_coordinates)
        if self.model is not None:
            setattr(self.model.input.variables, variable_name, normalized_file_name)

    def read(self) -> None:
        variables = getattr(self.model.input, "variables", None) if self.model else None
        if variables is None:
            return
        if getattr(variables, "ldbcoastline", ""):
            path = self.root / variables.ldbcoastline
            if path.exists():
                self.coastline = read_xy(path)
                self.coastline_file = variables.ldbcoastline
                return
        if _has_data(getattr(variables, "xmc", "")) and _has_data(getattr(variables, "ymc", "")):
            self.coastline = xy_columns_to_sections(variables.xmc, variables.ymc)

    def write(self) -> None:
        if self.coastline is not None:
            file_name = self.coastline_file or "coastline.ldb"
            write_xy(self.root / file_name, self.coastline)
            if self.model is not None:
                self.model.input.variables.ldbcoastline = file_name

        for variable_name, (file_name, coordinates) in self.extra_xy_files.items():
            write_xy(self.root / file_name, coordinates)
            if self.model is not None:
                setattr(self.model.input.variables, variable_name, file_name)

    def write_table(
        self,
        variable_name: str,
        data: NumericTableLike,
        file_name: PathLike,
        header: str | None = None,
    ) -> None:
        normalized_file_name = normalize_file_name(file_name)
        write_numeric_table(self.root / normalized_file_name, data, header=header)
        if self.model is not None:
            setattr(self.model.input.variables, variable_name, normalized_file_name)

    def clear_spatial_attributes(self) -> None:
        self.coastline = None
        self.coastline_file = None
        self.extra_xy_files.clear()


def _has_data(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    try:
        return len(value) > 0
    except TypeError:
        return True
