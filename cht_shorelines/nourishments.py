from __future__ import annotations

from pathlib import Path

from .io import dataframe_from_records, write_numeric_table, yyyymmdd


class ShorelinesNourishments:
    def __init__(self, model=None):
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

    def set_nourishments(self, data, file_name="nourishments.nor"):
        """Set nourishment rows.

        Required columns: xstart, ystart, xend, yend, tstart, tend, totalvolume.
        Dates may be strings, datetime-like values, or yyyymmdd integers.
        """
        self.nourishments = dataframe_from_records(
            data, ["xstart", "ystart", "xend", "yend", "tstart", "tend", "totalvolume"]
        )
        self.nourishment_file = file_name
        if self.model is not None:
            variables = self.model.input.variables
            variables.nourish = 1
            variables.norfile = file_name
            variables.ldbnourish = file_name

    def set_shoreface_nourishments(self, data, file_name="shoreface.fnor"):
        self.shoreface_nourishments = data
        self.shoreface_file = file_name
        if self.model is not None:
            variables = self.model.input.variables
            variables.fnourish = 1
            variables.fnorfile = file_name

    def write(self):
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


def _date_or_int(value):
    if isinstance(value, int):
        return value
    text = str(value)
    if text.isdigit() and len(text) == 8:
        return int(text)
    return yyyymmdd(value)


def _write_nor(path: Path, rows, header: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fid:
        fid.write(header + "\n")
        for row in rows:
            fid.write(
                f"{row[0]:12.1f} {row[1]:12.1f} {row[2]:12.1f} {row[3]:12.1f}"
                f" {int(row[4]):10d} {int(row[5]):10d} {row[6]:12.1f}\n"
            )
