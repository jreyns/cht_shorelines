"""File and MATLAB literal helpers for ShorelineS input generation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def matlab_repr(value) -> str:
    """Return a MATLAB expression that ``readkeys.m`` can evaluate."""
    if value is None:
        return "[]"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if math.isnan(value):
                return "NaN"
            if math.isinf(value):
                return "Inf" if value > 0 else "-Inf"
        return f"{value:.15g}"
    if isinstance(value, Path):
        return matlab_repr(str(value))
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, pd.Series):
        return matlab_repr(value.to_numpy())
    if isinstance(value, pd.DataFrame):
        return matlab_repr(value.to_numpy())
    if isinstance(value, np.ndarray):
        return _array_repr(value)
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return _list_repr(value)
    if isinstance(value, Mapping) and not value:
        return "{}"
    if isinstance(value, Mapping):
        raise TypeError("MATLAB struct serialization is not supported in runfiles")
    return matlab_repr(str(value))


def _array_repr(value: np.ndarray) -> str:
    if value.size == 0:
        return "[]"
    if value.ndim == 0:
        return matlab_repr(value.item())
    if value.ndim == 1:
        return "[" + " ".join(matlab_repr(v) for v in value.tolist()) + "]"
    if value.ndim == 2:
        rows = [" ".join(matlab_repr(v) for v in row) for row in value.tolist()]
        return "[" + "; ".join(rows) + "]"
    raise ValueError("Only scalar, vector, and matrix arrays can be serialized")


def _list_repr(value: list) -> str:
    if not value:
        return "[]"
    if any(_is_cell_like(v) for v in value):
        rows = []
        for item in value:
            if isinstance(item, (list, tuple)):
                rows.append(" ".join(matlab_repr(v) for v in item))
            else:
                rows.append(matlab_repr(item))
        return "{" + "; ".join(rows) + "}"
    return "[" + " ".join(matlab_repr(v) for v in value) + "]"


def _is_cell_like(value) -> bool:
    if isinstance(value, str):
        return True
    if isinstance(value, (list, tuple)):
        return any(isinstance(v, str) for v in value)
    return False


def write_runfile(path: Path, variables: Mapping[str, object]) -> None:
    """Write ShorelineS key/value input consumed by ``readkeys.m``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fid:
        for key, value in variables.items():
            if key.startswith("_"):
                continue
            fid.write(f"{key} = {matlab_repr(value)}\n")


def write_numeric_table(
    path: Path,
    data,
    header: str | None = None,
    fmt: str = "{:15.6f}",
) -> None:
    """Write a whitespace-separated numeric table."""
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fid:
        if header:
            fid.write(header.rstrip() + "\n")
        for row in arr:
            fid.write(" ".join(_format_number(v, fmt) for v in row).rstrip() + "\n")


def write_xy(path: Path, sections: Sequence) -> None:
    """Write one or more x/y polylines with NaN separators."""
    rows = []
    for index, section in enumerate(_as_sections(sections)):
        arr = np.asarray(section, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("Coordinate sections must be Nx2 arrays")
        if index:
            rows.append([np.nan, np.nan])
        rows.extend(arr.tolist())
    write_numeric_table(path, rows)


def _as_sections(sections: Sequence) -> list:
    if isinstance(sections, np.ndarray):
        return [sections]
    if not sections:
        return []
    first = sections[0]
    if isinstance(first, (int, float, np.number)):
        return [sections]
    arr = np.asarray(sections, dtype=object)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return [sections]
    return list(sections)


def dataframe_from_records(data, required: Iterable[str]) -> pd.DataFrame:
    """Normalize records or a DataFrame and check required columns."""
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    rename = {col: str(col).lower() for col in df.columns}
    df = df.rename(columns=rename)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return df


def yyyymmddhhmm(value) -> int:
    timestamp = pd.Timestamp(value)
    return int(timestamp.strftime("%Y%m%d%H%M"))


def yyyymmdd(value) -> int:
    timestamp = pd.Timestamp(value)
    return int(timestamp.strftime("%Y%m%d"))


def _format_number(value: float, fmt: str) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    return fmt.format(value)
