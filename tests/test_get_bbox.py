from io import StringIO

import pytest
from ebfloeseg.app import get_bbox


@pytest.fixture()
def csv():
    return StringIO(
        """location,center_lat,center_lon,top_left_lat,top_left_lon,lower_right_lat,lower_right_lon,left_x,right_x,lower_y,top_y,startdate,enddate
beaufort_sea,75,-135,67.22298,-152.46426,79.32881,-94.68433,-2383879.497,-883879.4975,-750000,750000,2020-09-05,2020-09-08
hudson_bay,60,-83,59.65687,-101.24295,57.54266,-66.04186,-2795941.755,-1295941.755,-3368686.029,-1868686.029,2020-09-06,2020-09-09
"""
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        ("hudson_bay", "-2795941.755,-3368686.029,-1295941.755,-1868686.029"),
        ("beaufort_sea", "-2383879.497,-750000.0,-883879.4975,750000.0"),
    ],
)
def test_get_bbox(csv, name, expected, capsys):
    get_bbox(csv, name)
    captured = capsys.readouterr()
    assert captured.out == expected + "\n"
