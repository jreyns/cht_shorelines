from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pydantic")
pytest.importorskip("pyproj")

import cht_shorelines
from cht_shorelines import Shorelines, ShorelinesClimateChange, ShorelinesRunup, ShorelinesTide
from cht_shorelines.output import ShorelinesOutput
from cht_shorelines.validation import GridSpec


def test_package_exports_and_output_class():
    assert "Shorelines" in cht_shorelines.__all__
    assert Shorelines is cht_shorelines.Shorelines
    assert ShorelinesClimateChange is cht_shorelines.ShorelinesClimateChange
    assert ShorelinesRunup is cht_shorelines.ShorelinesRunup
    assert ShorelinesTide is cht_shorelines.ShorelinesTide
    assert isinstance(ShorelinesOutput(), ShorelinesOutput)


def test_validation_gridspec_accepts_supported_formats_and_rejects_invalid():
    linear = GridSpec(grids=[[0.0, 0.0, 1.0, 1.0]])
    assert linear.grids == [[0.0, 0.0, 1.0, 1.0]]

    curvilinear = GridSpec(grids=[np.array([[0.0, 0.0], [1.0, 1.0]])])
    assert curvilinear.grids[0].shape == (2, 2)

    with pytest.raises(ValueError):
        GridSpec(grids=[[0.0, 1.0, 2.0]])


def test_shorelines_clear_spatial_attributes_and_runner(tmp_path):
    model = Shorelines(root=tmp_path, runfile="case.txt")
    model.grid.set_coastline([[0.0, 0.0], [1.0, 0.0]])
    model.initial_conditions.set_channel_axis([[0.0, 0.0], [1.0, 1.0]])
    model.structures.set_structures([[0.0, 0.0], [2.0, 2.0]])
    model.write()

    runner = model.write_matlab_runner(name="run_case_custom.m", shoreline_functions_path="C:/shorelines/functions")
    model.clear_spatial_attributes()

    assert runner.name == "run_case_custom.m"
    assert "addpath(genpath('C:/shorelines/functions'));" in runner.read_text()
    assert model.grid.coastline is None
    assert model.initial_conditions.channel is None
    assert model.structures.structures is None
