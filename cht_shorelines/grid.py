from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import write_numeric_table, write_xy


class ShorelinesDomain:
    def __init__(self, model=None):
        self.model = model
        self.coastline = None
        self.coastline_file = None
        self.extra_xy_files = {}

    @property
    def root(self) -> Path:
        if self.model is not None:
            return Path(self.model.path)
        return Path.cwd()

    def set_coastline(self, coordinates, file_name="coastline.ldb"):
        """Set the initial coastline as one or more Nx2 coordinate sections."""
        self.coastline = coordinates
        self.coastline_file = file_name
        if self.model is not None:
            self.model.input.variables.ldbcoastline = file_name

    def set_xy_file(self, variable_name: str, coordinates, file_name: str):
        """Set a generic ShorelineS x/y input file variable."""
        self.extra_xy_files[variable_name] = (file_name, coordinates)
        if self.model is not None:
            setattr(self.model.input.variables, variable_name, file_name)

    def read(self):
        variables = getattr(self.model.input, "variables", None) if self.model else None
        if variables is None or not getattr(variables, "ldbcoastline", ""):
            return
        path = self.root / variables.ldbcoastline
        if path.exists():
            self.coastline = np.loadtxt(path)
            self.coastline_file = variables.ldbcoastline

    def write(self):
        if self.coastline is not None:
            file_name = self.coastline_file or "coastline.ldb"
            write_xy(self.root / file_name, self.coastline)
            if self.model is not None:
                self.model.input.variables.ldbcoastline = file_name

        for variable_name, (file_name, coordinates) in self.extra_xy_files.items():
            write_xy(self.root / file_name, coordinates)
            if self.model is not None:
                setattr(self.model.input.variables, variable_name, file_name)

    def write_table(self, variable_name: str, data, file_name: str, header=None):
        write_numeric_table(self.root / file_name, data, header=header)
        if self.model is not None:
            setattr(self.model.input.variables, variable_name, file_name)

    def clear_spatial_attributes(self):
        self.coastline = None
        self.coastline_file = None
        self.extra_xy_files.clear()
