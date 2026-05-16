"""File and MATLAB literal helpers for ShorelineS input generation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
import math
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import numpy as np
import pandas as pd


PathLike: TypeAlias = str | Path
NumericScalar: TypeAlias = int | float | np.number
NumericTableLike: TypeAlias = Sequence[object] | np.ndarray | pd.Series | pd.DataFrame
RecordDataLike: TypeAlias = (
    pd.DataFrame
    | Mapping[str, Sequence[object]]
    | Sequence[Mapping[str, object]]
    | Sequence[Sequence[object]]
)
DatetimeLike: TypeAlias = (
    str | int | float | date | datetime | pd.Timestamp | np.datetime64
)


class ShorelinesModelProtocol(Protocol):
    path: str
    input: Any


def matlab_repr(value: Any) -> str:
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


def _list_repr(value: list[object]) -> str:
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


def _is_cell_like(value: object) -> bool:
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
    data: NumericTableLike,
    header: str | None = None,
    fmt: str = "{:15.6f}",
) -> None:
    """Write a whitespace-separated numeric table."""
    arr = coerce_numeric_table(data, argument="data")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fid:
        if header:
            fid.write(header.rstrip() + "\n")
        for row in arr:
            fid.write(" ".join(_format_number(v, fmt) for v in row).rstrip() + "\n")


def write_xy(path: Path, sections: Sequence[object] | np.ndarray) -> None:
    """Write one or more x/y polylines with NaN separators."""
    rows = []
    for index, section in enumerate(_as_sections(sections)):
        arr = coerce_numeric_table(
            section,
            argument=f"sections[{index}]",
            exact_columns=2,
        )
        if index:
            rows.append([np.nan, np.nan])
        rows.extend(arr.tolist())
    write_numeric_table(path, rows)


def read_numeric_table(path: Path | str) -> np.ndarray:
    """Read a whitespace-separated numeric table, skipping ``%`` comments."""
    path = Path(path)
    rows = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("%", 1)[0].strip()
        if not line:
            continue
        rows.append([float(value) for value in line.split()])
    if not rows:
        return np.empty((0, 0), dtype=float)
    width = max(len(row) for row in rows)
    if any(len(row) != width for row in rows):
        raise ValueError(f"Inconsistent column count in {path}")
    return np.asarray(rows, dtype=float)


def read_xy(path: Path | str) -> np.ndarray | list[np.ndarray]:
    """Read one or more x/y polylines from a numeric table with NaN separators."""
    data = read_numeric_table(path)
    if data.size == 0:
        return np.empty((0, 2), dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in XY file {path}")
    return split_xy_sections(data[:, :2])


def _as_sections(sections: Sequence[object] | np.ndarray) -> list[object]:
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


def split_xy_sections(data: NumericTableLike) -> list[np.ndarray] | np.ndarray:
    """Split an ``Nx2`` array into sections at rows with NaN coordinates."""
    arr = coerce_numeric_table(data, argument="data", exact_columns=2)
    if arr.size == 0:
        return arr
    mask = np.any(np.isnan(arr), axis=1)
    if not mask.any():
        return arr
    sections = []
    start = 0
    for index, is_nan in enumerate(mask):
        if is_nan:
            if index > start:
                sections.append(arr[start:index])
            start = index + 1
    if start < len(arr):
        sections.append(arr[start:])
    return sections


def xy_columns_to_sections(
    x: NumericTableLike,
    y: NumericTableLike,
) -> list[np.ndarray] | np.ndarray:
    """Combine ``x`` and ``y`` vectors into one or more coordinate sections."""
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    return split_xy_sections(np.column_stack([x_arr, y_arr]))


def dataframe_from_records(data: RecordDataLike, required: Iterable[str]) -> pd.DataFrame:
    """Normalize records or a DataFrame and check required columns."""
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    rename = {col: str(col).lower() for col in df.columns}
    df = df.rename(columns=rename)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return df


def yyyymmddhhmm(value: DatetimeLike) -> int:
    timestamp = pd.Timestamp(value)
    return int(timestamp.strftime("%Y%m%d%H%M"))


def yyyymmdd(value: DatetimeLike) -> int:
    timestamp = pd.Timestamp(value)
    return int(timestamp.strftime("%Y%m%d"))


def parse_compact_datetime(value: DatetimeLike) -> pd.Timestamp:
    text = str(int(float(value)))
    if len(text) == 8:
        return pd.to_datetime(text, format="%Y%m%d")
    if len(text) == 12:
        return pd.to_datetime(text, format="%Y%m%d%H%M")
    if len(text) == 14:
        return pd.to_datetime(text, format="%Y%m%d%H%M%S")
    raise ValueError(f"Unsupported compact datetime value: {value}")


def maybe_path(root: Path | str, value: PathLike | None) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str) and value:
        path = Path(value)
    else:
        return None
    if not path.is_absolute():
        path = Path(root) / path
    return path


def normalize_probabilities(values: NumericTableLike) -> np.ndarray:
    prob = np.asarray(values, dtype=float).reshape(-1)
    total = np.nansum(prob)
    if total > 1.1 and total < 100:
        return prob / 100.0
    if total > 200:
        return prob / 365.0
    if total and not np.isclose(total, 1.0):
        return prob / total
    return prob


def _format_number(value: float, fmt: str) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    return fmt.format(value)


def normalize_file_name(file_name: PathLike, *, argument: str = "file_name") -> str:
    if isinstance(file_name, Path):
        text = str(file_name)
    elif isinstance(file_name, str):
        text = file_name
    else:
        raise TypeError(f"{argument} must be a string or Path")
    if not text:
        raise ValueError(f"{argument} must not be empty")
    return text


def normalize_optional_file_name(
    file_name: PathLike | None,
    *,
    argument: str = "file_name",
) -> str | None:
    if file_name is None:
        return None
    return normalize_file_name(file_name, argument=argument)


def coerce_numeric_table(
    data: NumericTableLike,
    *,
    argument: str = "data",
    min_columns: int = 1,
    exact_columns: int | None = None,
) -> np.ndarray:
    try:
        arr = np.asarray(data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{argument} must be numeric") from exc
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{argument} must be a 1D or 2D numeric array")
    if exact_columns is not None and arr.shape[1] != exact_columns:
        raise ValueError(
            f"{argument} must have exactly {exact_columns} columns, got {arr.shape[1]}"
        )
    if arr.shape[1] < min_columns:
        raise ValueError(
            f"{argument} must have at least {min_columns} columns, got {arr.shape[1]}"
        )
    return arr


def validate_xy_sections(
    sections: Sequence[object] | np.ndarray,
    *,
    argument: str = "coordinates",
) -> Sequence[object] | np.ndarray:
    for index, section in enumerate(_as_sections(sections)):
        coerce_numeric_table(
            section,
            argument=f"{argument}[{index}]",
            exact_columns=2,
        )
    return sections
