#!/usr/bin/env python3
"""
Stub segmentation script.

CLI (matches Snakefile rule):
  --image PATH               input image
  --meta PATH                metadata JSON
  --out-seg PATH             output labeled segmentation (TIFF)
  --out-props PATH           output regionprops CSV
  --model-info STR           e.g., "2D_versatile_fluo"
  --channel INT              channel index to use (if applicable)
  --prob-thresh FLOAT        placeholder (unused in stub)
  --nms-thresh FLOAT         placeholder (unused in stub)
  --summarize-channels STR   e.g., "all"

Behavior:
  - Loads the image, picks a 2D plane (best-effort) using --channel if present.
  - Writes a zero-labeled uint16 mask with the same Y×X shape.
  - Writes an empty CSV with regionprops headers.
  - Validates that metadata JSON is readable.
"""
import argparse
import json
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm

from tifffile import imread
from tifffile import TiffFile

from skimage.color import rgb2gray
from skimage.color import label2rgb
from skimage.transform import rescale
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops
from skimage.measure import regionprops_table
from skimage.segmentation import mark_boundaries

# stardist
from csbdeep.utils import normalize
from stardist.models import StarDist2D


def resolve_channel_index(segment_channel, meta):
    """
    Return integer channel index.
    """
    chs = (meta.get("channels") or [])
    C = meta.get("size", {}).get("C", len(chs) or 1)

    # default / auto
    if not segment_channel or str(segment_channel).lower() in {"auto", "none"}:
        return 0

    s = str(segment_channel).strip()

    # numeric index (supports negatives), clamped to [0, C-1]
    try:
        idx = int(s)
        if idx < 0:
            idx += C
        return max(0, min(idx, C - 1))
    except ValueError:
        pass

    # name match (case-insensitive) against reversed list
    try:
        return [c.lower() for c in chs].index(s.lower())
    except ValueError:
        return 0

# minimal image IO
def imread_any(path):
    try:
        import tifffile as tiff
        return tiff.imread(path)
    except Exception:
        pass
    try:
        import imageio.v3 as iio
        return iio.imread(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read image '{path}': {e}")


def print_params(args, title="[PARAMS]"):
    """Pretty-print parsed argparse parameters in an aligned block."""
    d = vars(args)
    width = max(len(k) for k in d)
    lines = [title]
    for k in sorted(d):
        lines.append(f"  {k:<{width}} : {d[k]}")
    print("\n".join(lines) + "\n")


def _probe_supported_props(labels, intensity_image=None):
    """Return list of regionprops_table properties supported by current skimage."""
    # Comprehensive candidate list of scalar/table-friendly props
    candidates = [
        "label", "area", "bbox", "bbox_area", "centroid", "eccentricity",
        "equivalent_diameter", "euler_number", "extent", "feret_diameter_max",
        "filled_area", "inertia_tensor_eigvals", "local_centroid",
        "major_axis_length", "minor_axis_length", "orientation",
        "perimeter", "perimeter_crofton", "solidity"
    ]
    supported = []
    for prop in candidates:
        try:
            regionprops_table(labels, intensity_image=intensity_image, properties=[prop])
            supported.append(prop)
        except Exception:
            # not supported in this skimage build
            pass
    return supported

def segment_single_channel_allprops(
    img,
    info,
    stardist_model,
    channel,                     # index or name
    prob_thresh=0.4,
    nms_thresh=0.2,
    summarize_channels="all"     # "all" or list of indices/names
):
    """
    Segment a single channel from (C, T, Y, X, 3), return labels and a table with:
      - ALL supported scalar region properties (auto-detected)
      - Intensity stats for every requested channel: mean/max/min/sum
      - Pixel + physical centroids
    """
    # ---- resolve channels ----
    ch_names = (info.get("channels", []))
    def _to_idx(ch):
        return ch_names.index(ch) if isinstance(ch, str) else int(ch)

    seg_idx = _to_idx(channel)
    seg_name = ch_names[seg_idx] if seg_idx < len(ch_names) else f"Channel-{seg_idx}"

    if summarize_channels == "all":
        sum_idx = list(range(img.shape[0]))
    else:
        sum_idx = [_to_idx(c) for c in summarize_channels]
    if seg_idx not in sum_idx:
        sum_idx = [seg_idx] + sum_idx

    # ---- metadata (µm) ----
    sx = float(info["physical_pixel_size"]["X"])
    sy = float(info["physical_pixel_size"]["Y"])
    ox = float(info["origin"]["X"])
    oy = float(info["origin"]["Y"])
    unit_px = info["physical_pixel_size"].get("X_unit", "µm")
    unit_org = info["origin"].get("unit", "µm")
    if unit_px != unit_org:
        raise ValueError("Pixel-size and origin units differ; convert first.")

    # ---- shapes ----
    C, T, Y, X, _ = img.shape
    if not (0 <= seg_idx < C):
        raise IndexError(f"Channel index {seg_idx} out of range [0, {C-1}]")

    # ---- outputs ----
    segments = np.zeros((T, Y, X), dtype=np.int32)
    rows = []

    # ---- model ----
    model = StarDist2D.from_pretrained(stardist_model)

    # Probe supported properties once (using a dummy tiny label if needed later)
    # We'll probe on the fly after first successful frame to match your skimage build.
    supported_props = None

    with tqdm(total=T, desc=f"Segmenting [{seg_name}]", unit="frame") as pbar:
        for t in range(T):
            # segment on selected channel
            raw_seg = rgb2gray(img[seg_idx, t])
            try:
                labels, _ = model.predict_instances(
                    normalize(raw_seg),
                    prob_thresh=prob_thresh,
                    nms_thresh=nms_thresh
                )
            except Exception:
                pbar.update(1)
                continue

            segments[t] = labels.astype(np.int32, copy=False)

            # discover supported props (once)
            if supported_props is None:
                supported_props = _probe_supported_props(labels, intensity_image=raw_seg)

            # geometry + intensity from segmentation channel
            main_props = regionprops_table(
                labels,
                intensity_image=raw_seg,
                properties=supported_props + ["mean_intensity", "max_intensity", "min_intensity"]
            )
            df = pd.DataFrame(main_props)

            
            if df.empty:
                pbar.update(1)
                continue

            # physical centroids
            df["centroid_x_px"] = df["centroid-1"]
            df["centroid_y_px"] = df["centroid-0"]
            df["centroid_x_um"] = ox + df["centroid_x_px"] * sx
            df["centroid_y_um"] = oy + df["centroid_y_px"] * sy
            df["centroid_unit"] = unit_org

            # add segmentation-channel explicit names
            df.rename(columns={
                "mean_intensity": f"{seg_name}_mean",
                "max_intensity":  f"{seg_name}_max",
                "min_intensity":  f"{seg_name}_min",
            }, inplace=True)
            df[f"{seg_name}_sum"] = df[f"{seg_name}_mean"] * df["area"]

            # add intensity summaries for other channels
            for c in sum_idx:
                if c == seg_idx:
                    continue
                raw_c = rgb2gray(img[c, t])
                ch_label = ch_names[c] if c < len(ch_names) else f"Channel-{c}"
                add = regionprops_table(
                    labels,
                    intensity_image=raw_c,
                    properties=["label", "mean_intensity", "max_intensity", "min_intensity"]
                )
                add = pd.DataFrame(add).rename(columns={
                    "mean_intensity": f"{ch_label}_mean",
                    "max_intensity":  f"{ch_label}_max",
                    "min_intensity":  f"{ch_label}_min",
                })
                add[f"{ch_label}_sum"] = add[f"{ch_label}_mean"] * df["area"]
                df = df.merge(add, on="label", how="left")

            # annotate indices
            df["time"] = t
            df["seg_channel_idx"] = seg_idx
            df["seg_channel_name"] = seg_name

            rows.append(df)
            pbar.set_postfix({"T": t, "Cells": int(labels.max())})
            pbar.update(1)

    props = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return segments, props


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--out-seg", required=True)
    p.add_argument("--out-props", required=True)
    p.add_argument("--model-info", default="2D_versatile_fluo")
    p.add_argument("--segment_channel", type=str, default="TaGFP")
    p.add_argument("--prob-thresh", type=float, default=0.4)
    p.add_argument("--nms-thresh", type=float, default=0.2)
    p.add_argument("--summarize-channels", default="all")
    args = p.parse_args()

    print_params(args)

    # load image
    img = imread_any(args.image)
    print(f"[IMAGE] path={args.image}")
    print(f"[IMAGE] shape={getattr(img, 'shape', None)} dtype={getattr(img, 'dtype', None)}\n")

    # load metadata JSON
    meta_path = Path(args.meta)
    try:
        with meta_path.open("r") as fh:
            info = json.load(fh)
        print(f"[META] path={meta_path}")
        print("[META] contents:")
        print(json.dumps(info, indent=2, sort_keys=True))
        print()
    except Exception as e:
        print(f"[META][ERROR] failed to read '{meta_path}': {e}\n")

    # get the numeric channel from the metadata
    idx = resolve_channel_index(args.segment_channel, info)
    chs = (info.get("channels") or [])
    name = chs[idx] if 0 <= idx < len(chs) else "unknown"
    print(f"[SEGMENTATION CHANNEL] index={idx} name={name}")

    """ SEGMENT """
    segments, props = segment_single_channel_allprops(
        img=img,
        info=info,
        stardist_model=args.model_info,
        channel=idx,
        prob_thresh=args.prob_thresh,
        nms_thresh=args.nms_thresh,
        summarize_channels=args.summarize_channels,
    )    
    
   # --- save segmentation with NumPy (.npy) ---
    seg_path = Path(args.out_seg)
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    
    seg = np.asarray(segments)
    if seg.dtype == bool:
        seg = seg.astype(np.uint8)
    elif not np.issubdtype(seg.dtype, np.integer):
        seg = seg.astype(np.uint16)
    
    # ensure .npy (np.save will append if missing; better to set explicitly)
    if seg_path.suffix != ".npy":
        seg_path = seg_path.with_suffix(".npy")
    
    np.save(seg_path, seg)
    print(f"[WRITE] seg (npy): {seg_path}  shape={seg.shape} dtype={seg.dtype}")
    
    # --- save region properties CSV (unchanged) ---
    props_path = Path(args.out_props)
    props_path.parent.mkdir(parents=True, exist_ok=True)
    df_props = props if isinstance(props, pd.DataFrame) else pd.DataFrame(props)
    df_props.to_csv(props_path, index=False)
    print(f"[WRITE] props     : {props_path}  shape={df_props.shape}")

