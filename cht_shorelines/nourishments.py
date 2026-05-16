from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from .io import (
    DatetimeLike,
    NumericTableLike,
    PathLike,
    RecordDataLike,
    ShorelinesModelProtocol,
    coerce_numeric_table,
    dataframe_from_records,
    normalize_file_name,
    read_numeric_table,
    write_numeric_table,
    yyyymmdd,
)


class ShorelinesNourishments:
    def __init__(self, model: ShorelinesModelProtocol | None = None) -> None:
        self.model = model
        self.nourishments = None
        self.nourishment_file = None
        self.shoreface_nourishments = None
        self.shoreface_file = None

    @property
    def root(self) -> Path:
        if self.model is not None:
            return Path(self.model.path)
        return Path.cwd()

    def set_nourishments(
        self,
        data: RecordDataLike,
        file_name: PathLike = "nourishments.nor",
    ) -> None:
        """Set nourishment rows.

        Required columns: xstart, ystart, xend, yend, tstart, tend, totalvolume.
        Dates may be strings, datetime-like values, or yyyymmdd integers.
        """
        self.nourishments = dataframe_from_records(
            data, ["xstart", "ystart", "xend", "yend", "tstart", "tend", "totalvolume"]
        )
        self.nourishment_file = normalize_file_name(file_name)
        if self.model is not None:
            variables = self.model.input.variables
            variables.nourish = 1
            variables.norfile = self.nourishment_file
            variables.ldbnourish = self.nourishment_file

    def set_shoreface_nourishments(
        self,
        data: NumericTableLike,
        file_name: PathLike = "shoreface.fnor",
    ) -> None:
        self.shoreface_nourishments = coerce_numeric_table(data, argument="data")
        self.shoreface_file = normalize_file_name(file_name)
        if self.model is not None:
            variables = self.model.input.variables
            variables.fnourish = 1
            variables.fnorfile = self.shoreface_file

    def read(self) -> None:
        variables = getattr(self.model.input, "variables", None) if self.model else None
        if variables is None:
            return
        nourishment_file = ""
        if getattr(variables, "norfile", ""):
            nourishment_file = variables.norfile
        elif str(getattr(variables, "ldbnourish", "")).lower().endswith(".nor"):
            nourishment_file = variables.ldbnourish
        if nourishment_file:
            path = self.root / nourishment_file
            if path.exists():
                data = read_numeric_table(path)
                if data.size:
                    self.nourishments = pd.DataFrame(
                        {
                            "xstart": data[:, 0],
                            "ystart": data[:, 1],
                            "xend": data[:, 2],
                            "yend": data[:, 3],
                            "tstart": [str(int(value)) for value in data[:, 4]],
                            "tend": [str(int(value)) for value in data[:, 5]],
                            "totalvolume": data[:, 6],
                        }
                    )
                    self.nourishment_file = nourishment_file

        shoreface_file = getattr(variables, "fnorfile", "")
        if shoreface_file:
            path = self.root / shoreface_file
            if path.exists():
                self.shoreface_nourishments = read_numeric_table(path)
                self.shoreface_file = shoreface_file

    def write(self) -> None:
        if self.nourishments is not None:
            file_name = self.nourishment_file or "nourishments.nor"
            rows = []
            for row in self.nourishments.itertuples(index=False):
                rows.append(
                    [
                        row.xstart,
                        row.ystart,
                        row.xend,
                        row.yend,
                        _date_or_int(row.tstart),
                        _date_or_int(row.tend),
                        row.totalvolume,
                    ]
                )
            _write_nor(
                self.root / file_name,
                rows,
                "%     xstart     ystart       xend       yend     tstart       tend  totalvolume",
            )
            if self.model is not None:
                self.model.input.variables.norfile = file_name
                self.model.input.variables.ldbnourish = file_name

        if self.shoreface_nourishments is not None:
            file_name = self.shoreface_file or "shoreface.fnor"
            write_numeric_table(self.root / file_name, self.shoreface_nourishments)
            if self.model is not None:
                self.model.input.variables.fnorfile = file_name


def _date_or_int(value: DatetimeLike | int) -> int:
    if isinstance(value, int):
        return value
    text = str(value)
    if text.isdigit() and len(text) == 8:
        return int(text)
    return yyyymmdd(value)


def _write_nor(
    path: Path,
    rows: Iterable[Sequence[float | int]],
    header: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fid:
        fid.write(header + "\n")
        for row in rows:
            fid.write(
                f"{row[0]:12.1f} {row[1]:12.1f} {row[2]:12.1f} {row[3]:12.1f}"
                f" {int(row[4]):10d} {int(row[5]):10d} {row[6]:12.1f}\n"
            )
