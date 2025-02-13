import datetime
from logging import getLogger
from typing import Optional

import numpy as np
import pandas as pd
import cv2
from scipy import ndimage
import skimage
from skimage.filters import threshold_local
from skimage.morphology import diamond, opening
import rasterio
from rasterio.enums import ColorInterp

from ebfloeseg.masking import create_land_mask, maskrgb, mask_image, create_cloud_mask
from ebfloeseg.savefigs import imsave, save_ice_mask_hist
from ebfloeseg.utils import (
    write_mask_values,
    get_wcuts,
    getmeta,
    getres,
    get_region_properties,
    smallest_dtype,
)


def extract_features(
    output,
    red_c,
    target_dir,
    fname,
):
    props = get_region_properties(output, red_c)
    df = pd.DataFrame.from_dict(props)
    df.to_csv(target_dir / fname)


def get_remove_small_mask(watershed, it):
    area_lim = (it) ** 4
    props = skimage.measure.regionprops_table(watershed, properties=["label", "area"])
    df = pd.DataFrame.from_dict(props)
    mask = np.isin(watershed, df[df.area < area_lim].label.values)
    return mask


def get_erosion_kernel(erosion_kernel_type="diamond", erosion_kernel_size=1):
    if erosion_kernel_type == "diamond":
        erosion_kernel = diamond(erosion_kernel_size)
    elif erosion_kernel_type == "ellipse":
        erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple([erosion_kernel_size] * 2)
        )
    return erosion_kernel


logger = getLogger(__name__)


def _preprocess(
    ftci,
    fcloud,
    land_mask,
    itmax,
    itmin,
    step,
    erosion_kernel_type,
    erosion_kernel_size,
    save_figs,
    save_direc,
    doy="",
    year="",
    sat="",
    res="",
    fname_prefix="",
):
    tci = rasterio.open(ftci)

    save_direc.mkdir(exist_ok=True, parents=True)

    cloud_mask = create_cloud_mask(fcloud)

    match tci.colorinterp:
        case (ColorInterp.red, ColorInterp.green, ColorInterp.blue):
            red_c, green_c, blue_c = tci.read()
            assert tci.colorinterp[0] is ColorInterp.red
        case (ColorInterp.red, ColorInterp.green, ColorInterp.blue, _):
            red_c, green_c, blue_c, _ = tci.read()
        case _:
            msg = "unknown number of dimensions %s" % tci.colorinterp
            raise ValueError(msg)

    rgb_masked = np.dstack([red_c, green_c, blue_c])  # masked below

    maskrgb(rgb_masked, cloud_mask)
    if save_figs:
        fname = f"{fname_prefix}cloud_mask_on_rgb.tif"
        imsave(tci, rgb_masked, save_direc, fname)

    maskrgb(rgb_masked, land_mask)
    if save_figs:
        fname = f"{fname_prefix}land_cloud_mask_on_rgb.tif"
        imsave(tci, rgb_masked, save_direc, fname)

    ## adaptive threshold for ice mask
    red_masked = rgb_masked[:, :, 0]
    thresh_adaptive = threshold_local(red_c, block_size=399)

    # here just determining the min and max values for the adaptive threshold
    ow_cut_min, ow_cut_max, bins = get_wcuts(red_masked)

    if save_figs:
        save_ice_mask_hist(
            red_masked=red_masked,
            bins=bins,
            mincut=ow_cut_min,
            maxcut=ow_cut_max,
            target_dir=save_direc,
            fname=f"{fname_prefix}ice_mask_hist.png",
        )

    thresh_adaptive = np.clip(thresh_adaptive, ow_cut_min, ow_cut_max)

    ice_mask = red_masked > thresh_adaptive

    # a simple text file with columns: 'doy','ice_area','unmasked','sic'
    write_mask_values(
        lmd=land_mask + cloud_mask,
        ice_mask=ice_mask,
        doy=doy,
        save_direc=save_direc,
        fname=f"{fname_prefix}mask_values.txt",
    )

    # saving ice mask
    fname = f"{fname_prefix}ice_mask_bw.tif"
    if save_figs:
        imsave(
            tci=tci,
            img=ice_mask,
            save_direc=save_direc,
            fname=fname,
            count=1,
            rollaxis=False,
            dtype=np.bool_,
            res=res,
        )

    # here dilating the land and cloud mask so any floes that are adjacent to the mask can be removed later
    land_cloud_mask = (land_mask + cloud_mask).astype(int)
    land_cloud_mask_dilated = skimage.morphology.binary_dilation(
        land_cloud_mask, diamond(10)
    )

    # setting up different kernel for erosion-expansion algo
    erosion_kernel = get_erosion_kernel(erosion_kernel_type, erosion_kernel_size)

    # TODO: clarify this block
    inp = ice_mask
    input_no = ice_mask
    output = np.zeros((np.shape(ice_mask)), dtype=np.int16)
    itmax = itmax
    itmin = itmin
    step = step
    highest_label_so_far = 0

    for r, it in enumerate(range(itmax, itmin - 1, step)):
        # erode a lot at first, decrease number of iterations each time
        eroded_ice_mask = cv2.erode(inp.astype(np.uint8), erosion_kernel, iterations=it)
        eroded_ice_mask = ndimage.binary_fill_holes(eroded_ice_mask.astype(np.uint8))

        dilated_ice_mask = cv2.dilate(
            inp.astype(np.uint8), erosion_kernel, iterations=it
        )

        # label floes remaining after erosion
        n, markers, _, _ = cv2.connectedComponentsWithStats(
            eroded_ice_mask.astype(np.uint8)
        )

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        unknown = cv2.subtract(
            dilated_ice_mask.astype(np.uint8), eroded_ice_mask.astype(np.uint8)
        )

        # Now, mark the region of unknown with zero
        # markers[unknown == 255] = 0
        mask_image(markers, unknown == 255, 0)

        # dilate each marker
        for a in np.arange(0, it + 1, 1):
            markers = skimage.morphology.dilation(markers, erosion_kernel)

        # rewatershed
        watershed = cv2.watershed(rgb_masked, markers)

        # get rid of floes that intersect the dilated land mask
        watershed[
            np.isin(
                watershed,
                np.unique(watershed[land_cloud_mask_dilated & (watershed > 1)]),
            )
        ] = 1

        # set the open water and already identified floes to no
        # watershed[~input_no] = 1
        mask_image(watershed, ~input_no, 1)

        # get rid of ones that are too small
        area_lim = (it) ** 4
        props = skimage.measure.regionprops_table(
            watershed, properties=["label", "area"]
        )
        df = pd.DataFrame.from_dict(props)
        watershed[np.isin(watershed, df[df.area < area_lim].label.values)] = 1

        if save_figs:
            fname = f"{fname_prefix}identification_round_{r}.tif"
            imsave(
                tci=tci,
                img=watershed,
                save_direc=save_direc,
                fname=fname,
                count=1,
                rollaxis=False,
                dtype=np.uint8,
                res=res,
            )

        input_no = ice_mask + inp
        inp = (watershed == 1) & (inp == 1) & ice_mask
        watershed[watershed < 2] = 0
        new_label_mask = watershed > 0
        output[new_label_mask] = watershed[new_label_mask] + highest_label_so_far
        highest_label_so_far = np.max(output)

    # Clean the final props
    output = opening(output)
    output = clean_labels_with_multiple_blobs(output)

    # saving the props table
    fname_infix = ""
    if sat:
        fname_infix = f"{sat}_{fname_infix}"
    if res:
        fname_infix = f"{res}_{fname_infix}"

    extract_features(
        output, red_c, save_direc, fname=f"{fname_prefix}{fname_infix}props.csv"
    )

    # saving the label floes tif
    fname = "final.tif"
    assert (
        output.min() >= 0
    ), "negative values found, but values should never be smaller than zero"
    if sat:
        fname = f"{sat}_{fname}"
    if fname_prefix:
        fname = f"{fname_prefix}{fname}"

    imsave(
        tci=tci,
        img=output,
        save_direc=save_direc,
        fname=fname,
        count=1,
        rollaxis=False,
        dtype=smallest_dtype(output),
        res=res,
    )


def count_blobs_per_label(label_array):
    df = pd.DataFrame({"label": [], "count": []})
    for label in np.unique(label_array):
        mask = label == label_array
        _, count = skimage.measure.label(mask, return_num=True)
        df = pd.concat([df, pd.DataFrame({"label": [label], "count": [count]})])
    return df


def clean_labels_with_multiple_blobs(label_array, factor_threshold=5):
    label_array_ = np.copy(label_array)
    blobs_per_label = count_blobs_per_label(label_array_)
    for row in blobs_per_label.query("label > 0 & count > 1").itertuples():
        mask = label_array_ == row.label
        relabeled, count = skimage.measure.label(mask, return_num=True)
        assert count > 1
        relabeled_props = pd.DataFrame(
            skimage.measure.regionprops_table(relabeled, properties=["label", "area"])
        )
        ordered_relabeled_props = relabeled_props.sort_values(
            by="area", ascending=False
        )
        blob_generator = ordered_relabeled_props.itertuples()
        largest_blob = next(blob_generator)
        for blob in blob_generator:
            # Exception if this blob isn't strictly smaller than the largest blob
            assert (
                blob.area < largest_blob.area
            ), "blob %s area %s is not smaller than 'largest_blob' %s with area %s" % (
                blob.label,
                blob.area,
                largest_blob.label,
                largest_blob.area,
            )
            # Check that the blob is at least factor_threshold times smaller than the "main" blob
            # and throw a warning if it isn't
            if blob.area * factor_threshold > largest_blob.area:
                logger.warning(
                    "Blob %s has area %s, larger than 1/%s of largest blob area %s"
                    % (blob.label, blob.area, factor_threshold, largest_blob.area)
                )
            blob_mask = relabeled == blob.label
            label_array_[blob_mask] = 0
    return label_array_


def preprocess(
    ftci,
    fcloud,
    land_mask,
    itmax,
    itmin,
    step,
    erosion_kernel_type,
    erosion_kernel_size,
    save_figs,
    save_direc,
):
    try:
        doy, year, sat = getmeta(fcloud)
        res = getres(doy, year)
        save_direc = save_direc / doy
        fname_prefix = ""

        _preprocess(
            ftci=ftci,
            fcloud=fcloud,
            land_mask=land_mask,
            itmax=itmax,
            itmin=itmin,
            step=step,
            erosion_kernel_type=erosion_kernel_type,
            erosion_kernel_size=erosion_kernel_size,
            save_figs=save_figs,
            save_direc=save_direc,
            doy=doy,
            year=year,
            sat=sat,
            res=res,
            fname_prefix=fname_prefix,
        )
    except Exception as e:
        logger.exception(f"Error processing {fcloud} and {ftci}: {e}")
        raise


def preprocess_b(
    ftci,
    fcloud,
    fland,
    itmax,
    itmin,
    step,
    erosion_kernel_type,
    erosion_kernel_size,
    save_figs,
    save_direc,
    fname_prefix,
    date: Optional[datetime.datetime],
):
    try:
        if date is not None:
            doy = date.timetuple().tm_yday
            year = date.year
        else:
            doy = None
            year = None
        _preprocess(
            ftci=ftci,
            fcloud=fcloud,
            land_mask=create_land_mask(fland),
            itmax=itmax,
            itmin=itmin,
            step=step,
            erosion_kernel_type=erosion_kernel_type,
            erosion_kernel_size=erosion_kernel_size,
            save_figs=save_figs,
            save_direc=save_direc,
            doy=doy,
            year=year,
            sat=None,
            res=None,
            fname_prefix=fname_prefix,
        )
    except Exception as e:
        logger.exception(f"Error processing {fcloud} and {ftci}: {e}")
        raise
