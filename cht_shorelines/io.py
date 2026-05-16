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
CoordinateArray: TypeAlias = np.ndarray
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


def write_xy(path: Path, coordinates: CoordinateArray) -> None:
    """Write one or more x/y polylines stored as a NaN-separated ``Nx2`` array."""
    write_numeric_table(path, validate_coordinate_array(coordinates))


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


def read_xy(path: Path | str) -> np.ndarray:
    """Read one or more x/y polylines as a NaN-separated ``Nx2`` array."""
    data = read_numeric_table(path)
    if data.size == 0:
        return np.empty((0, 2), dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in XY file {path}")
    return validate_coordinate_array(data[:, :2], argument="data")


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
) -> np.ndarray:
    """Combine ``x`` and ``y`` vectors into a NaN-separated ``Nx2`` array."""
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    return validate_coordinate_array(
        np.column_stack([x_arr, y_arr]),
        argument="coordinates",
    )


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
    """
    Convert a date-like value to ``YYYYMMDDHHMM`` integer format.

    Parameters
    ----------
    value : datetime-like
        Value accepted by :class:`pandas.Timestamp`.

    Returns
    -------
    int
        Compact date-time representation.
    """
    timestamp = pd.Timestamp(value)
    return int(timestamp.strftime("%Y%m%d%H%M"))


def yyyymmdd(value: DatetimeLike) -> int:
    """
    Convert a date-like value to ``YYYYMMDD`` integer format.

    Parameters
    ----------
    value : datetime-like
        Value accepted by :class:`pandas.Timestamp`.

    Returns
    -------
    int
        Compact date representation.
    """
    timestamp = pd.Timestamp(value)
    return int(timestamp.strftime("%Y%m%d"))


def parse_compact_datetime(value: DatetimeLike) -> pd.Timestamp:
    """
    Parse a compact numeric date or date-time value.

    Parameters
    ----------
    value : datetime-like
        Integer-like value in ``YYYYMMDD``, ``YYYYMMDDHHMM``, or
        ``YYYYMMDDHHMMSS`` format.

    Returns
    -------
    pandas.Timestamp
        Parsed timestamp.
    """
    text = str(int(float(value)))
    if len(text) == 8:
        return pd.to_datetime(text, format="%Y%m%d")
    if len(text) == 12:
        return pd.to_datetime(text, format="%Y%m%d%H%M")
    if len(text) == 14:
        return pd.to_datetime(text, format="%Y%m%d%H%M%S")
    raise ValueError(f"Unsupported compact datetime value: {value}")


def maybe_path(root: Path | str, value: PathLike | None) -> Path | None:
    """
    Resolve an optional path relative to a root directory.

    Parameters
    ----------
    root : str or pathlib.Path
        Base directory for relative paths.
    value : str or pathlib.Path, optional
        Candidate path value.

    Returns
    -------
    pathlib.Path or None
        Resolved path when ``value`` is non-empty, otherwise ``None``.
    """
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
    """
    Normalize probability-like values to fractions.

    Parameters
    ----------
    values : array-like
        Probability values expressed as fractions, percentages, or day counts.

    Returns
    -------
    numpy.ndarray
        Normalized one-dimensional probability vector.
    """
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
    """
    Validate and normalize a required file name argument.

    Parameters
    ----------
    file_name : str or pathlib.Path
        File name to normalize.
    argument : str, default "file_name"
        Argument name used in error messages.

    Returns
    -------
    str
        Normalized file name.
    """
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
    """
    Validate and normalize an optional file name argument.

    Parameters
    ----------
    file_name : str or pathlib.Path, optional
        File name to normalize.
    argument : str, default "file_name"
        Argument name used in error messages.

    Returns
    -------
    str or None
        Normalized file name, or ``None`` when not provided.
    """
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
    """
    Convert numeric table input to a validated two-dimensional array.

    Parameters
    ----------
    data : array-like
        Numeric input to coerce.
    argument : str, default "data"
        Argument name used in error messages.
    min_columns : int, default 1
        Minimum allowed number of columns.
    exact_columns : int, optional
        Exact required number of columns.

    Returns
    -------
    numpy.ndarray
        Two-dimensional floating-point array.
    """
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


def validate_coordinate_array(
    coordinates: object,
    *,
    argument: str = "coordinates",
) -> np.ndarray:
    """
    Validate a public coordinate array input.

    Parameters
    ----------
    coordinates : object
        Candidate coordinate input. Public interfaces require a
        :class:`numpy.ndarray`.
    argument : str, default "coordinates"
        Argument name used in error messages.

    Returns
    -------
    numpy.ndarray
        Validated ``Nx2`` floating-point coordinate array.
    """
    if not isinstance(coordinates, np.ndarray):
        raise TypeError(f"{argument} must be provided as a numpy.ndarray")
    arr = np.asarray(coordinates, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{argument} must be an Nx2 array")
    nan_mask = np.any(np.isnan(arr), axis=1)
    if np.any(np.isnan(arr[~nan_mask])) or np.any(np.isinf(arr)):
        raise ValueError(
            f"{argument} must contain finite values except full NaN separator rows"
        )
    if nan_mask.any():
        invalid_rows = nan_mask & ~np.all(np.isnan(arr), axis=1)
        if invalid_rows.any():
            raise ValueError(f"{argument} NaN separator rows must be [NaN, NaN]")
        if nan_mask[0] or nan_mask[-1]:
            raise ValueError(f"{argument} must not start or end with a NaN separator row")
        if np.any(nan_mask[:-1] & nan_mask[1:]):
            raise ValueError(f"{argument} must not contain consecutive NaN separator rows")
    return arr


def validate_xy_sections(
    coordinates: object,
    *,
    argument: str = "coordinates",
) -> np.ndarray:
    """
    Validate coordinate input for XY-based public APIs.

    Parameters
    ----------
    coordinates : object
        Candidate coordinate input.
    argument : str, default "coordinates"
        Argument name used in error messages.

    Returns
    -------
    numpy.ndarray
        Validated ``Nx2`` coordinate array.
    """
    return validate_coordinate_array(coordinates, argument=argument)
