from pathlib import Path

import pytest
import requests_mock

from ebfloeseg.load import load, ImageType, Satellite, DataSet


ExampleDataSetBeaufortSea = DataSet(
    datetime="2016-07-01T00:00:00Z",
    wrap="day",
    satellite=Satellite.terra,
    kind=ImageType.truecolor,
    scale=10000,
    bbox=(
        -2330000,
        -420000,
        -1130000,
        750000,
    ),
    crs="EPSG:3413",
    ts=1683675557694,
)


@pytest.mark.slow
@pytest.mark.parametrize("satellite", Satellite)
@pytest.mark.parametrize("kind", ImageType)
def test_load_runs_without_crashing_for_different_parameters(kind, satellite):
    load(kind=kind, satellite=satellite, scale=100000)


def test_error_on_empty_file():
    with requests_mock.Mocker() as m:
        m.get(
            "https://wvs.earthdata.nasa.gov/api/v1/snapshot",
            content=Path("tests/load/empty.tiff").read_bytes(),
        )
        with pytest.raises(AssertionError):
            load()
