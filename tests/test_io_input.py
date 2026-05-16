from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from cht_shorelines.input import ShorelinesInput
from cht_shorelines.io import (
    coerce_numeric_table,
    dataframe_from_records,
    matlab_repr,
    maybe_path,
    normalize_file_name,
    normalize_probabilities,
    parse_compact_datetime,
    read_numeric_table,
    read_xy,
    split_xy_sections,
    write_numeric_table,
    write_xy,
    xy_columns_to_sections,
    yyyymmdd,
    yyyymmddhhmm,
)


def test_numeric_and_xy_roundtrip_helpers(tmp_path):
    table_path = tmp_path / "table.txt"
    xy_path = tmp_path / "coast.ldb"
    sections = [
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([[2.0, 2.0], [3.0, 3.0]]),
    ]

    write_numeric_table(table_path, [[1.0, 2.0], [3.0, 4.0]], header="% header")
    write_xy(xy_path, sections)

    table = read_numeric_table(table_path)
    coast = read_xy(xy_path)

    np.testing.assert_allclose(table, np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert isinstance(coast, list)
    assert len(coast) == 2
    np.testing.assert_allclose(coast[0], sections[0])
    np.testing.assert_allclose(coast[1], sections[1])


def test_io_helpers_validate_and_normalize_inputs(tmp_path):
    data = dataframe_from_records([{"HS": 1.0, "TP": 5.0, "DIR": 270.0}], ["hs", "tp", "dir"])
    assert list(data.columns) == ["hs", "tp", "dir"]

    dt = parse_compact_datetime(200001020304)
    assert yyyymmdd(dt) == 20000102
    assert yyyymmddhhmm(dt) == 200001020304

    probs = normalize_probabilities([20.0, 30.0, 50.0])
    np.testing.assert_allclose(probs, np.array([0.2, 0.3, 0.5]))

    path = maybe_path(tmp_path, "inner/file.txt")
    assert path == tmp_path / "inner" / "file.txt"
    assert normalize_file_name(Path("case.txt")) == "case.txt"

    matrix = coerce_numeric_table([1.0, 2.0, 3.0])
    np.testing.assert_allclose(matrix, np.array([[1.0], [2.0], [3.0]]))

    sections = xy_columns_to_sections([0.0, 1.0, np.nan, 2.0], [0.0, 1.0, np.nan, 2.0])
    assert isinstance(sections, list)
    assert len(sections) == 2

    split = split_xy_sections(np.array([[0.0, 0.0], [np.nan, np.nan], [1.0, 1.0]]))
    assert isinstance(split, list)
    assert len(split) == 2


def test_matlab_repr_rejects_structs():
    with pytest.raises(TypeError):
        matlab_repr({"a": 1})


def test_shorelines_input_write_and_read_roundtrip(tmp_path):
    inp = ShorelinesInput(root=tmp_path, runfile="case.txt")
    inp.variables.trform = "CERC"
    inp.variables.randomseed = 42
    inp.variables.xlimits = [0.0, 100.0]
    inp.variables.structtype = [["Closed", -6000], ["Angleconstant", 300]]

    inp.write()

    reloaded = ShorelinesInput(root=tmp_path, runfile="case.txt")
    reloaded.read()

    assert reloaded.runfile_path == tmp_path / "case.txt"
    assert reloaded.variables.trform == "CERC"
    assert reloaded.variables.randomseed == 42
    assert reloaded.variables.xlimits == [0.0, 100.0]
    assert reloaded.variables.structtype == [["Closed", -6000], ["Angleconstant", 300]]
