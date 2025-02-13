import datetime
import pytest
from pathlib import Path
import logging
import numpy as np

import rasterio
from ebfloeseg.preprocess import (
    preprocess,
    preprocess_b,
    count_blobs_per_label,
    clean_labels_with_multiple_blobs,
)

logger = logging.getLogger(__name__)


def test_process_exception(tmpdir):
    fcloud = "cloud.tif"
    ftci = "tci.tif"
    fcloud_direc = Path(tmpdir, "clouds")
    ftci_direc = Path(tmpdir, "tcis")
    save_figs = True
    save_direc = tmpdir
    land_mask = "land_mask.tif"
    erode_itmax = 8
    erode_itmin = 3
    step = -1
    erosion_kernel_type = "diamond"
    erosion_kernel_size = 1

    with pytest.raises(Exception):
        preprocess(
            fcloud,
            ftci,
            fcloud_direc,
            ftci_direc,
            save_figs,
            save_direc,
            land_mask,
            erode_itmax,
            erode_itmin,
            step,
            erosion_kernel_type,
            erosion_kernel_size,
        )


test_dir = Path(__file__).parent


@pytest.mark.parametrize(
    "truecolorimg,cloudimg,landmask",
    [
        (
            test_dir / "process/truecolor.tiff",
            test_dir / "process/cloud.tiff",
            test_dir / "process/landmask.tiff",
        ),
        (
            test_dir / "input/tci/tci_2012-08-01_214_terra.tiff",
            test_dir / "input/cloud/cloud_2012-08-01_214_terra.tiff",
            test_dir / "input/reproj_land.tiff",
        ),
        (
            test_dir / "input/tci/tci_2012-08-02_215_terra.tiff",
            test_dir / "input/cloud/cloud_2012-08-02_215_terra.tiff",
            test_dir / "input/reproj_land.tiff",
        ),
    ],
)
def test_process_no_duplicated_labels(
    truecolorimg: Path, cloudimg: Path, landmask: Path, tmp_path
):
    preprocess_b(
        ftci=truecolorimg,
        fcloud=cloudimg,
        fland=landmask,
        save_figs=True,
        save_direc=tmp_path,
        fname_prefix="",
        itmax=8,
        itmin=3,
        step=-1,
        erosion_kernel_type="diamond",
        erosion_kernel_size=1,
        date=datetime.date.fromisoformat("2001-01-01"),
    )

    with rasterio.open(tmp_path / "final.tif") as dataset:
        image_array = dataset.read()
        blobs_per_label = count_blobs_per_label(image_array)
        background_label = 0
        for row in blobs_per_label.itertuples():
            if row.label == background_label:
                continue
            logger.debug(row)
            assert (
                row.count == 1
            ), f"{row.count} disconnected components detected for {row.label=}"


@pytest.mark.parametrize(
    "original",
    [
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
        np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]]),
        np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]]),
        np.array([[1, 1, 0], [1, 0, 0], [0, 0, 2]]),
        np.array([[1, 1, 2], [1, 2, 2], [2, 2, 2]]),
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        np.array([[1, 1, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]),
        np.array([[1, 1, 1, 1], [1, 0, 0, 0], [0, 3, 0, 0], [1, 2, 2, 2]]),
    ],
)
def test_clean_labels_with_multiple_blobs_is_idempotent(original):
    cleaned_once = clean_labels_with_multiple_blobs(original)
    cleaned_twice = clean_labels_with_multiple_blobs(cleaned_once)
    np.testing.assert_array_equal(cleaned_once, cleaned_twice)


@pytest.mark.parametrize(
    "original,expected_cleaned",
    [
        (  # No change, all zeros
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ),
        (  # No change, all ones
            np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        ),
        (  # No change, diamond
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
        ),
        (  # Small cleanup, opposite corners
            np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]]),
            np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]]),
        ),
        (  # 8-connected opposite corner, no change
            np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]),
            np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]),
        ),
        (  # No change
            np.array([[1, 1, 0], [1, 0, 0], [0, 0, 2]]),
            np.array([[1, 1, 0], [1, 0, 0], [0, 0, 2]]),
        ),
        (  # No change
            np.array([[1, 1, 2], [1, 2, 2], [2, 2, 2]]),
            np.array([[1, 1, 2], [1, 2, 2], [2, 2, 2]]),
        ),
        (  # No change
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        ),
        (  # 4x4, small cleanup
            np.array([[1, 1, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]),
            np.array([[1, 1, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        ),
        (  # 4x4, small cleanup
            np.array([[1, 1, 1, 1], [1, 0, 0, 0], [0, 3, 0, 0], [1, 2, 2, 2]]),
            np.array([[1, 1, 1, 1], [1, 0, 0, 0], [0, 3, 0, 0], [0, 2, 2, 2]]),
        ),
    ],
)
def test_clean_labels_with_multiple_blobs_works_for_simple_cases(
    original, expected_cleaned
):
    np.testing.assert_array_equal(
        clean_labels_with_multiple_blobs(original), expected_cleaned
    )


@pytest.mark.parametrize(
    "original",
    [
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]),
        np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]]),
        np.array([[1, 1, 1], [2, 2, 2], [1, 1, 1]]),
    ],
)
def test_clean_labels_with_multiple_blobs_throws_errors_when_two_blobs_have_the_same_size(
    original,
):
    with pytest.raises(AssertionError):
        clean_labels_with_multiple_blobs(original)


if __name__ == "__main__":
    pytest.main()
