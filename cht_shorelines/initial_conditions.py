from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import write_numeric_table, write_xy


class ShorelinesInitialConditions:
    def __init__(self, model=None):
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
        if self.model is not None:
            return Path(self.model.path)
        return Path.cwd()

    def set_dunes(self, data, file_name="dunes.dun"):
        """Set dune rows: x, y, wberm, dfelev, dcelev, optional cs/cstill/xtill/perctill."""
        self.dunes = data
        self.dune_file = file_name
        if self.model is not None:
            variables = self.model.input.variables
            variables.dune = 1
            variables.ldbdune = file_name

    def set_sediment_limiter(self, coordinates, width=None, file_name="sediment_limiter.ldb"):
        """Set sediment-limiter coordinates, optionally with per-point width."""
        arr = np.asarray(coordinates, dtype=float)
        if width is not None:
            width_arr = np.asarray(width, dtype=float).reshape(-1, 1)
            arr = np.column_stack([arr, width_arr])
        self.sediment_limiter = arr
        self.sediment_limiter_file = file_name
        if self.model is not None:
            variables = self.model.input.variables
            variables.sedlim = 1
            variables.ldbsedlim = file_name

    def set_channel_axis(self, coordinates, file_name="channel.ldb"):
        self.channel = coordinates
        self.channel_file = file_name
        if self.model is not None:
            variables = self.model.input.variables
            variables.channel = 1
            variables.ldbchannel = file_name

    def set_spit_polygon(self, coordinates, file_name="spit.ldb"):
        self.spit_polygon = coordinates
        self.spit_file = file_name
        if self.model is not None:
            self.model.input.variables.ldbspit = file_name

    def set_flood_delta(self, coordinates, file_name="flood_delta.ldb"):
        self.flood_delta = coordinates
        self.flood_delta_file = file_name
        if self.model is not None:
            variables = self.model.input.variables
            variables.flooddelta = 1
            variables.ldbflood = file_name

    def set_river_discharges(self, data, file_name="river_discharge.riv"):
        """Set mud river rows: xriv1, yriv1, xriv2, yriv2, tstart, tend, rate."""
        self.river_discharges = data
        self.river_file = file_name
        if self.model is not None:
            variables = self.model.input.variables
            variables.mud = 1
            variables.ldbriverdisch = file_name

    def set_mangroves(self, data, file_name="mangroves.mgv"):
        """Set mangrove rows: xmgv, ymgv, Bf, Bm, Bfm."""
        self.mangroves = data
        self.mangrove_file = file_name
        if self.model is not None:
            variables = self.model.input.variables
            variables.mud = 1
            variables.ldbmangrove = file_name

    def read(self):
        pass

    def write(self):
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
