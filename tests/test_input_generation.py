import pytest


def test_matlab_repr_scalars_vectors_matrices_and_cells():
    np = pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    from cht_shorelines.io import matlab_repr

    assert matlab_repr("CERC") == "'CERC'"
    assert matlab_repr(True) == "1"
    assert matlab_repr([]) == "[]"
    assert matlab_repr({}) == "{}"
    assert matlab_repr([1, 2.5, np.nan]) == "[1 2.5 NaN]"
    assert matlab_repr(np.array([[1, 2], [3, 4]])) == "[1 2; 3 4]"
    assert matlab_repr([["Closed", -6000], ["Angleconstant", 300]]) == (
        "{'Closed' -6000; 'Angleconstant' 300}"
    )


def test_minimal_case_writes_runfile_and_attribute_files(tmp_path):
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyproj")
    from cht_shorelines import Shorelines

    model = Shorelines(root=tmp_path, runfile="case.txt")
    model.input.variables.reftime = "2000-01-01"
    model.input.variables.endofsimulation = "2000-01-03"
    model.input.variables.tc = 0
    model.input.variables.dt = 0.01
    model.input.variables.trform = "CERC"
    model.grid.set_coastline(
        [np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 10.0]])]
    )
    model.wave_boundary_conditions.set_timeseries(
        pd.DataFrame(
            {
                "time": ["2000-01-01 00:00", "2000-01-02 00:00", "2000-01-03 00:00"],
                "hs": [1.0, 1.2, 1.1],
                "tp": [6.0, 6.5, 6.2],
                "dir": [300.0, 305.0, 310.0],
            }
        )
    )

    model.write()
    runner = model.write_matlab_runner()

    runfile = (tmp_path / "case.txt").read_text()
    assert "ldbcoastline = 'coastline.ldb'" in runfile
    assert "wvcfile = 'waves.wvt'" in runfile
    assert "trform = 'CERC'" in runfile
    assert (tmp_path / "coastline.ldb").exists()
    assert "200001010000" in (tmp_path / "waves.wvt").read_text()
    assert runner.name == "run_case.m"
    assert "runShorelineS('case.txt')" in runner.read_text()


def test_spatial_wave_list_and_nourishments(tmp_path):
    pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyproj")
    from cht_shorelines import Shorelines

    model = Shorelines(root=tmp_path, runfile="spatial.txt")
    wave_data = pd.DataFrame(
        {
            "time": ["2000-01-01", "2000-01-02"],
            "hs": [0.8, 0.9],
            "tp": [5.0, 5.2],
            "dir": [280.0, 285.0],
        }
    )
    model.wave_boundary_conditions.set_spatial_timeseries(
        [
            {"data": wave_data, "x": 0.0, "y": 0.0, "file_name": "wvt/a.wvt"},
            {
                "data": wave_data,
                "x": 100.0,
                "y": 50.0,
                "file_name": "wvt/b.wvt",
                "hs_factor": 0.9,
                "dir_offset": 2.0,
            },
        ],
        list_file="waves_list.wvt",
    )
    model.nourishments.set_nourishments(
        [
            {
                "xstart": 0.0,
                "ystart": 0.0,
                "xend": 100.0,
                "yend": 0.0,
                "tstart": "2000-01-01",
                "tend": "2000-12-01",
                "totalvolume": 120000.0,
            }
        ],
        file_name="nourish.nor",
    )

    model.write()

    assert "wvt/a.wvt" in (tmp_path / "waves_list.wvt").read_text()
    assert (tmp_path / "wvt" / "a.wvt").exists()
    assert "20000101" in (tmp_path / "nourish.nor").read_text()
    runfile = (tmp_path / "spatial.txt").read_text()
    assert "wvcfile = 'waves_list.wvt'" in runfile
    assert "norfile = 'nourish.nor'" in runfile
