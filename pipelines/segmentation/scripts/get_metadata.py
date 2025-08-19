#!/usr/bin/env python3
"""
Extract OME-TIFF metadata and write it to a JSON file.

Usage:
    python get_ome_metadata.py <input_image.ome.tiff> <output_metadata.json>
"""

import sys
import os
import json
import xml.etree.ElementTree as ET
from tifffile import TiffFile

# OME namespace
_OME_NS = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}


def load_ome_physical_info(fpath):
    """Return OME physical metadata as dict."""
    with TiffFile(fpath) as tif:
        ome_xml = tif.ome_metadata
        dtype = tif.pages[0].dtype if tif.pages else None

    root = ET.fromstring(ome_xml)

    # Pixels block
    px = root.find(".//ome:Image/ome:Pixels", _OME_NS)
    sizeX = int(px.get("SizeX"))
    sizeY = int(px.get("SizeY"))
    sizeZ = int(px.get("SizeZ", 1))
    sizeC = int(px.get("SizeC", 1))
    sizeT = int(px.get("SizeT", 1))
    dim_order = px.get("DimensionOrder")
    dtype_xml = px.get("Type")

    sx = float(px.get("PhysicalSizeX"))
    sy = float(px.get("PhysicalSizeY"))
    sx_unit = px.get("PhysicalSizeXUnit") or "µm"
    sy_unit = px.get("PhysicalSizeYUnit") or "µm"

    # Origin info
    plate = root.find(".//ome:Plate", _OME_NS)
    well_sample = root.find(".//ome:Well/ome:WellSample", _OME_NS)
    well_origin_x = float(plate.get("WellOriginX")) if plate is not None and plate.get("WellOriginX") else 0.0
    well_origin_y = float(plate.get("WellOriginY")) if plate is not None and plate.get("WellOriginY") else 0.0
    pos_x = float(well_sample.get("PositionX")) if well_sample is not None and well_sample.get("PositionX") else 0.0
    pos_y = float(well_sample.get("PositionY")) if well_sample is not None and well_sample.get("PositionY") else 0.0

    origin_x = well_origin_x + pos_x
    origin_y = well_origin_y + pos_y
    origin_unit = "µm"

    # Channel names
    ch_names = [
        ch.get("Name") or f"Channel-{i}"
        for i, ch in enumerate(root.findall(".//ome:Image/ome:Pixels/ome:Channel", _OME_NS))
    ]

    # Acquisition date
    acq = root.find(".//ome:Image/ome:AcquisitionDate", _OME_NS)
    acq_time = acq.text if acq is not None else None

    info = {
        "size": {"X": sizeX, "Y": sizeY, "Z": sizeZ, "C": sizeC, "T": sizeT},
        "dimension_order": dim_order,
        "dtype_xml": dtype_xml,
        "dtype_tiff": str(dtype) if dtype is not None else None,
        "physical_pixel_size": {"X": sx, "X_unit": sx_unit, "Y": sy, "Y_unit": sy_unit},
        "origin": {
            "X": origin_x,
            "Y": origin_y,
            "unit": origin_unit,
            "components": {
                "WellOriginX": well_origin_x,
                "WellOriginY": well_origin_y,
                "PositionX": pos_x,
                "PositionY": pos_y,
            },
        },
        "channels": ch_names,
        "acquisition_time": acq_time,
    }
    return info


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python get_ome_metadata.py <input_image.ome.tiff> <output_metadata.json>")

    image_path = sys.argv[1]
    out_json = sys.argv[2]

    if not os.path.exists(image_path):
        sys.exit(f"Error: Input file not found: {image_path}")

    info = load_ome_physical_info(image_path)

    # Write metadata
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(info, f, indent=2)

    print(f"Metadata written to {out_json}")
