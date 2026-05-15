from __future__ import annotations

from pathlib import Path

from .io import write_numeric_table, write_xy


class ShorelinesStructures:
    def __init__(self, model=None):
        self.model = model
        self.structures = None
        self.structures_file = None
        self.permeable = None
        self.permeable_file = None
        self.revetments = None
        self.revetments_file = None
        self.transmission_characteristics = None
        self.transmission_file = None

    @property
    def root(self) -> Path:
        if self.model is not None:
            return Path(self.model.path)
        return Path.cwd()

    def set_structures(self, coordinates, file_name="structures.ldb", structure_type=None):
        """Set hard structures as one or more x/y coordinate sections."""
        self.structures = coordinates
        self.structures_file = file_name
        if self.model is not None:
            variables = self.model.input.variables
            variables.struct = 1
            variables.ldbstructures = file_name
            if structure_type is not None:
                variables.structtype = structure_type

    def set_permeable(self, coordinates, file_name="permeable.ldb", wavetransm=1.0, qstransm=1.0):
        self.permeable = coordinates
        self.permeable_file = file_name
        if self.model is not None:
            variables = self.model.input.variables
            variables.perm = 1
            variables.ldbpermeable = file_name
            variables.wavetransm = wavetransm
            variables.qstransm = qstransm

    def set_revetments(self, coordinates, file_name="revetments.ldb"):
        self.revetments = coordinates
        self.revetments_file = file_name
        if self.model is not None:
            variables = self.model.input.variables
            variables.revet = 1
            variables.ldbrevetments = file_name

    def set_transmission_characteristics(
        self,
        data,
        file_name="transmission.txt",
        form="angr",
    ):
        """Set transmission rows: depth, crest height, slope, width, optional d50."""
        self.transmission_characteristics = data
        self.transmission_file = file_name
        if self.model is not None:
            variables = self.model.input.variables
            variables.transmission = 1
            variables.diffraction = 1
            variables.transmfile = file_name
            variables.transmform = form

    def read(self):
        pass

    def write(self):
        if self.structures is not None:
            file_name = self.structures_file or "structures.ldb"
            write_xy(self.root / file_name, self.structures)
            if self.model is not None:
                self.model.input.variables.ldbstructures = file_name

        if self.permeable is not None:
            file_name = self.permeable_file or "permeable.ldb"
            write_xy(self.root / file_name, self.permeable)
            if self.model is not None:
                self.model.input.variables.ldbpermeable = file_name

        if self.revetments is not None:
            file_name = self.revetments_file or "revetments.ldb"
            write_xy(self.root / file_name, self.revetments)
            if self.model is not None:
                self.model.input.variables.ldbrevetments = file_name

        if self.transmission_characteristics is not None:
            file_name = self.transmission_file or "transmission.txt"
            write_numeric_table(self.root / file_name, self.transmission_characteristics)
            if self.model is not None:
                self.model.input.variables.transmfile = file_name
