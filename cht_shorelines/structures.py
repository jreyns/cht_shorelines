from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import (
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


class ShorelinesStructures:
    def __init__(self, model: ShorelinesModelProtocol | None = None) -> None:
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

    def set_structures(
        self,
        coordinates: object,
        file_name: PathLike = "structures.ldb",
        structure_type: str | list[str] | None = None,
    ) -> None:
        """Set hard structures as one or more x/y coordinate sections."""
        self.structures = validate_xy_sections(coordinates)
        self.structures_file = normalize_file_name(file_name)
        if self.model is not None:
            variables = self.model.input.variables
            variables.struct = 1
            variables.ldbstructures = self.structures_file
            if structure_type is not None:
                variables.structtype = structure_type

    def set_permeable(
        self,
        coordinates: object,
        file_name: PathLike = "permeable.ldb",
        wavetransm: float = 1.0,
        qstransm: float = 1.0,
    ) -> None:
        self.permeable = validate_xy_sections(coordinates)
        self.permeable_file = normalize_file_name(file_name)
        if self.model is not None:
            variables = self.model.input.variables
            variables.perm = 1
            variables.ldbpermeable = self.permeable_file
            variables.wavetransm = wavetransm
            variables.qstransm = qstransm

    def set_revetments(
        self,
        coordinates: object,
        file_name: PathLike = "revetments.ldb",
    ) -> None:
        self.revetments = validate_xy_sections(coordinates)
        self.revetments_file = normalize_file_name(file_name)
        if self.model is not None:
            variables = self.model.input.variables
            variables.revet = 1
            variables.ldbrevetments = self.revetments_file

    def set_transmission_characteristics(
        self,
        data: NumericTableLike,
        file_name: PathLike = "transmission.txt",
        form: str = "angr",
    ) -> None:
        """Set transmission rows: depth, crest height, slope, width, optional d50."""
        self.transmission_characteristics = coerce_numeric_table(
            data,
            argument="data",
            min_columns=4,
        )
        self.transmission_file = normalize_file_name(file_name)
        if self.model is not None:
            variables = self.model.input.variables
            variables.transmission = 1
            variables.diffraction = 1
            variables.transmfile = self.transmission_file
            variables.transmform = form

    def read(self) -> None:
        variables = getattr(self.model.input, "variables", None) if self.model else None
        if variables is None:
            return

        self._read_xy("ldbstructures", "structures", "structures_file")
        self._read_xy("ldbpermeable", "permeable", "permeable_file")
        self._read_xy("ldbrevetments", "revetments", "revetments_file")
        self._read_numeric("transmfile", "transmission_characteristics", "transmission_file")

        if self.structures is None and _has_data(getattr(variables, "xhard", "")):
            self.structures = xy_columns_to_sections(variables.xhard, variables.yhard)
        if self.permeable is None and _has_data(getattr(variables, "xperm", "")):
            self.permeable = xy_columns_to_sections(variables.xperm, variables.yperm)
        if self.revetments is None and _has_data(getattr(variables, "xrevet", "")):
            self.revetments = xy_columns_to_sections(variables.xrevet, variables.yrevet)

        if self.transmission_characteristics is None and any(
            _has_data(getattr(variables, name, []))
            for name in (
                "transmbwdepth",
                "transmcrestheight",
                "transmslope",
                "transmcrestwidth",
                "transmd50",
            )
        ):
            self.transmission_characteristics = _stack_optional_columns(
                variables.transmbwdepth,
                variables.transmcrestheight,
                variables.transmslope,
                variables.transmcrestwidth,
                variables.transmd50,
            )

    def write(self) -> None:
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


def _stack_optional_columns(*columns: object) -> np.ndarray | None:
    arrays = []
    length = 0
    for column in columns:
        arr = column if isinstance(column, list) else [column]
        arr = [float(value) for value in arr]
        arrays.append(arr)
        if arr:
            length = max(length, len(arr))
    if length == 0:
        return None
    normalized = []
    for arr in arrays:
        if not arr:
            normalized.append([float("nan")] * length)
        elif len(arr) == 1 and length > 1:
            normalized.append(arr * length)
        elif len(arr) == length:
            normalized.append(arr)
        else:
            raise ValueError("Incompatible transmission parameter lengths")
    return np.asarray(list(map(list, zip(*normalized))), dtype=float)


def _has_data(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    try:
        return len(value) > 0
    except TypeError:
        return True
