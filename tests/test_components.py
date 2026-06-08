from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from cht_shorelines.climate_change import ShorelinesClimateChange
from cht_shorelines.dunes import ShorelinesDunes
from cht_shorelines.grid import ShorelinesDomain
from cht_shorelines.initial_conditions import ShorelinesInitialConditions
from cht_shorelines.input import ShorelinesInput
from cht_shorelines.nourishments import ShorelinesNourishments
from cht_shorelines.runup import ShorelinesRunup
from cht_shorelines.structures import ShorelinesStructures
from cht_shorelines.tide import ShorelinesTide
from cht_shorelines.wave_boundary_conditions import ShorelinesWaveBoundaryConditions


def make_model(tmp_path):
    return SimpleNamespace(path=str(tmp_path), input=ShorelinesInput(root=tmp_path, runfile="case.txt"))


def test_grid_domain_write_table_and_read_fallback(tmp_path):
    model = make_model(tmp_path)
    domain = ShorelinesDomain(model)

    domain.set_coastline(
        [np.array([[0.0, 0.0], [1.0, 0.0]])],
        file_name="coast.ldb",
    )
    domain.set_xy_file(
        "ldbplot",
        [np.array([[2.0, 2.0], [3.0, 3.0]])],
        "plot.ldb",
    )
    domain.write_table("xyout", [[1.0, 2.0], [3.0, 4.0]], "table.txt", header="% data")
    domain.write()

    assert (tmp_path / "coast.ldb").exists()
    assert (tmp_path / "plot.ldb").exists()
    assert "1.000000" in (tmp_path / "table.txt").read_text()
    assert model.input.variables.ldbcoastline == "coast.ldb"

    model.input.variables.ldbcoastline = ""
    model.input.variables.xmc = [0.0, 1.0]
    model.input.variables.ymc = [0.0, 0.0]
    fallback = ShorelinesDomain(model)
    fallback.read()
    np.testing.assert_allclose(fallback.coastline, np.array([[0.0, 0.0], [1.0, 0.0]]))


def test_initial_conditions_write_and_read(tmp_path):
    model = make_model(tmp_path)
    initial = ShorelinesInitialConditions(model)

    initial.set_dunes([[0.0, 0.0, 10.0, 3.0, 8.0]])
    initial.set_sediment_limiter(np.array([[0.0, 0.0], [1.0, 1.0]]), width=[5.0, 6.0])
    initial.set_channel_axis([np.array([[0.0, 0.0], [10.0, 0.0]])])
    initial.set_spit_polygon([np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])])
    initial.set_flood_delta([np.array([[0.0, 0.0], [0.0, 1.0]])])
    initial.set_river_discharges([[0.0, 0.0, 1.0, 1.0, 20000101, 20000102, 10.0]])
    initial.set_mangroves([[0.0, 0.0, 1.0, 2.0, 3.0]])
    initial.write()

    reread = ShorelinesInitialConditions(model)
    reread.read()

    assert reread.dunes.shape == (1, 5)
    assert reread.sediment_limiter.shape == (2, 3)
    np.testing.assert_allclose(reread.channel, np.array([[0.0, 0.0], [10.0, 0.0]]))
    np.testing.assert_allclose(reread.river_discharges[0, -1], 10.0)
    np.testing.assert_allclose(reread.mangroves[0, -1], 3.0)


def test_dunes_roundtrip_with_forcing(tmp_path):
    from cht_shorelines import Shorelines

    model = Shorelines(root=tmp_path, runfile="case.txt")
    model.input.variables.reftime = "2000-01-01"
    model.input.variables.endofsimulation = "2000-01-03"

    dunes = model.dunes
    assert isinstance(dunes, ShorelinesDunes)
    dunes.set_dunes([[0.0, 0.0, 12.0, 3.5, 8.5, 0.001, 0.00001, 4.0, 70.0]])
    dunes.configure(
        enabled=True,
        cs=0.001,
        cstill=0.00001,
        xtill=[4.0],
        perctill=[70.0],
        aoverwash=4.0,
        dtdune=0.25,
        duneaw=0.15,
        rhoa=1.3,
        d50r=0.0003,
        kw=4.5,
        segmaw=0.2,
        maxslope=1 / 20,
        runupform="Larson",
        runupfactor=1.2,
        z=12.0,
        swl0=0.4,
    )
    dunes.set_wind(
        [
            {"time": "2000-01-01", "uz": 8.0, "dir": 270.0},
            {"time": "2000-01-02", "uz": 9.0, "dir": 275.0},
            {"time": "2000-01-03", "uz": 10.0, "dir": 280.0},
        ],
        file_name="dune_wind.wnd",
    )
    dunes.set_water_levels(
        [
            {"time": "2000-01-01", "swl": 0.1},
            {"time": "2000-01-02", "swl": 0.2},
            {"time": "2000-01-03", "swl": 0.3},
        ],
        file_name="dune_water.wat",
    )

    model.write()
    reread = Shorelines(root=tmp_path, runfile="case.txt", mode="r")

    assert reread.dunes.enabled is True
    assert reread.dunes.geometry.shape == (1, 9)
    assert reread.dunes.wind_file == "dune_wind.wnd"
    assert reread.dunes.water_level_file == "dune_water.wat"
    assert list(reread.dunes.wind.columns) == ["time", "uz", "dir"]
    assert list(reread.dunes.water_levels.columns) == ["time", "swl"]
    assert reread.input.variables.runupform == "Larson"
    assert reread.input.variables.runupfactor == 1.2
    assert reread.input.variables.cs == 0.001
    np.testing.assert_allclose(reread.dunes.geometry[0, 2:5], [12.0, 3.5, 8.5])


def test_structures_write_and_read(tmp_path):
    model = make_model(tmp_path)
    structures = ShorelinesStructures(model)

    structures.set_structures([np.array([[0.0, 0.0], [1.0, 1.0]])], structure_type="groin")
    structures.set_permeable([np.array([[2.0, 2.0], [3.0, 3.0]])], wavetransm=0.7, qstransm=0.5)
    structures.set_revetments([np.array([[4.0, 4.0], [5.0, 5.0]])])
    structures.set_transmission_characteristics([[1.0, 2.0, 3.0, 4.0, 5.0]])
    structures.write()

    reread = ShorelinesStructures(model)
    reread.read()

    np.testing.assert_allclose(reread.transmission_characteristics, np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]))
    assert model.input.variables.structtype == "groin"
    assert model.input.variables.wavetransm == 0.7
    assert model.input.variables.qstransm == 0.5


def test_climate_change_roundtrip(tmp_path):
    model = make_model(tmp_path)
    climate = ShorelinesClimateChange(model)
    series = pd.DataFrame({"time": pd.to_datetime(["2000-01-01", "2001-01-01"]), "value": [0.1, 0.2]})

    climate.set_sea_level_rise(series, file_name="ccslr.txt")
    climate.set_wave_height_change(0.3)
    climate.set_wave_direction_change([[20000101, 5.0], [20010101, 6.0]], file_name="ccdir.txt")
    climate.write()

    reread = ShorelinesClimateChange(model)
    reread.read()

    assert isinstance(reread.sea_level_rise, pd.DataFrame)
    np.testing.assert_allclose(reread.sea_level_rise["value"], [0.1, 0.2])
    assert reread.wave_height_change == 0.3
    np.testing.assert_allclose(reread.wave_direction_change["value"], [5.0, 6.0])


def test_nourishments_write_and_read(tmp_path):
    model = make_model(tmp_path)
    nourishments = ShorelinesNourishments(model)

    nourishments.set_nourishments(
        [
            {
                "xstart": 0.0,
                "ystart": 1.0,
                "xend": 2.0,
                "yend": 3.0,
                "tstart": "2000-01-01",
                "tend": "2000-02-01",
                "totalvolume": 1000.0,
            }
        ]
    )
    nourishments.set_shoreface_nourishments([[1.0, 2.0], [3.0, 4.0]])
    nourishments.write()

    reread = ShorelinesNourishments(model)
    reread.read()

    assert list(reread.nourishments.columns) == [
        "xstart",
        "ystart",
        "xend",
        "yend",
        "tstart",
        "tend",
        "totalvolume",
    ]
    np.testing.assert_allclose(reread.shoreface_nourishments, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_tide_write_and_read(tmp_path):
    model = make_model(tmp_path)
    tide = ShorelinesTide(model)

    tide.set_tide_data([[200001010000, 0.1], [200001020000, 0.2]], file_name="tide.wat")
    tide.set_tide_profile([[0.0, 1.0], [2.0, 3.0]], file_name="profile.txt")
    tide.write()

    reread = ShorelinesTide(model)
    reread.read()

    assert reread.tide_type == 2
    assert list(reread.water_levels.columns)[:2] == ["time", "swl"]
    np.testing.assert_allclose(reread.tide_profile, np.array([[0.0, 1.0], [2.0, 3.0]]))


def test_runup_reads_sources_and_locations(tmp_path):
    model = make_model(tmp_path)
    model.input.variables.watfile = "runup_water.wat"
    model.input.variables.wvdfile = "runup_wave.wvd"
    model.input.variables.watlocfile = "water_locations.txt"
    model.input.variables.WaveLocfile = "wave_locations.txt"

    (tmp_path / "runup_water.wat").write_text("200001010000 0.1\n200001020000 0.2\n")
    (tmp_path / "runup_wave.wvd").write_text("200001010000 1.0 5.0 270.0\n200001020000 1.1 5.5 275.0\n")
    (tmp_path / "water_locations.txt").write_text("0.0 0.0\n1.0 1.0\n")
    (tmp_path / "wave_locations.txt").write_text("2.0 2.0\n3.0 3.0\n")

    runup = ShorelinesRunup(model)
    runup.read()

    assert list(runup.water_levels.columns)[:2] == ["time", "swl"]
    assert list(runup.wave_conditions.columns)[:4] == ["time", "hs", "tp", "dir"]
    np.testing.assert_allclose(runup.water_locations, np.array([[0.0, 0.0], [1.0, 1.0]]))
    np.testing.assert_allclose(runup.wave_locations, np.array([[2.0, 2.0], [3.0, 3.0]]))


def test_wave_boundary_conditions_write_read_and_check_times(tmp_path):
    model = make_model(tmp_path)
    model.input.variables.reftime = "2000-01-01"
    model.input.variables.endofsimulation = "2000-01-03"
    waves = ShorelinesWaveBoundaryConditions(model)

    waves.set_spatial_timeseries(
        [
            {
                "data": pd.DataFrame(
                    {
                        "time": ["2000-01-01", "2000-01-02", "2000-01-03"],
                        "hs": [1.0, 1.1, 1.2],
                        "tp": [5.0, 5.1, 5.2],
                        "dir": [270.0, 271.0, 272.0],
                    }
                ),
                "x": 0.0,
                "y": 0.0,
                "file_name": "waves/a.wvt",
            }
        ],
        list_file="waves_list.wvt",
    )
    waves.set_water_levels(
        pd.DataFrame(
            {
                "time": ["2000-01-01", "2000-01-02", "2000-01-03"],
                "swl": [0.1, 0.2, 0.3],
            }
        ),
        file_name="waterlevels.wat",
    )
    waves.set_wind(
        pd.DataFrame(
            {
                "time": ["2000-01-01", "2000-01-02", "2000-01-03"],
                "uz": [5.0, 6.0, 7.0],
                "dir": [180.0, 190.0, 200.0],
            }
        ),
        file_name="wind.wnd",
    )
    waves.write()

    reread = ShorelinesWaveBoundaryConditions(model)
    reread.read()
    ok, messages = reread.check_times()

    assert ok is True
    assert messages == []
    assert "waves/a.wvt" in (tmp_path / "waves_list.wvt").read_text()
    assert list(reread.water_levels.columns) == ["time", "swl"]
    assert list(reread.wind.columns) == ["time", "uz", "dir"]


def test_coordinate_setters_accept_list_of_ndarrays_and_reject_plain_lists(tmp_path):
    model = make_model(tmp_path)

    ShorelinesDomain(model).set_coastline([np.array([[0.0, 0.0], [1.0, 0.0]])])
    ShorelinesInitialConditions(model).set_channel_axis(
        [np.array([[0.0, 0.0], [1.0, 0.0]])]
    )
    ShorelinesStructures(model).set_structures(
        [np.array([[0.0, 0.0], [1.0, 0.0]])]
    )

    with pytest.raises(TypeError):
        ShorelinesDomain(model).set_coastline([[0.0, 0.0], [1.0, 0.0]])

    with pytest.raises(TypeError):
        ShorelinesInitialConditions(model).set_channel_axis([[0.0, 0.0], [1.0, 0.0]])

    with pytest.raises(TypeError):
        ShorelinesStructures(model).set_structures([[0.0, 0.0], [1.0, 0.0]])
