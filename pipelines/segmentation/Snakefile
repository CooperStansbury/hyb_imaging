import os
import sys
import glob
import re
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import yaml
import json
import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

""" PATH CONFIG """
BASE_DIRECTORY = Path(workflow.basedir)

# Config details
CONFIG_PATH = "/config/config.yaml"
CONFIG_BASENAME = os.path.basename(CONFIG_PATH)
CONFIG_ABS_PATH = str(BASE_DIRECTORY) + CONFIG_PATH
configfile: CONFIG_ABS_PATH 

# Canonicalize to an absolute, real path
CONDA_ENV_PATH = os.fspath(Path(config["conda_env"]).expanduser().resolve(strict=True))

# Sanity checks (optional but useful)
assert Path(CONDA_ENV_PATH, "conda-meta").exists(), f"Not a conda env: {CONDA_ENV_PATH}"
assert Path(CONDA_ENV_PATH, "bin/python").exists(), f"Missing python in: {CONDA_ENV_PATH}"

""" EXECUTION DETAILS """
logging.info(f"Base directory : {BASE_DIRECTORY}")
logging.info(f"Config file    : {CONFIG_ABS_PATH}")
logging.info(json.dumps(config, indent=2))

""" HELPER VARIABLES """
OUTPUT_PATH = config["output_path"]

# Load input paths file
input_paths_file = config["input_paths"]
df_inputs = pd.read_csv(input_paths_file)

image_names = df_inputs['image_name'].to_list()

# Add output_path column (preserving full multi-part extension, e.g. ".ome.tiff")
df_inputs["output_path"] = df_inputs.apply(
    lambda row: os.path.join(
        OUTPUT_PATH, "raw_images", row["image_name"] + "".join(Path(row["file_path"]).suffixes)
    ),
    axis=1
)

# Collect all extensions (multi-suffix aware, e.g. ".ome.tiff")
extensions = df_inputs["file_path"].apply(lambda f: "".join(Path(f).suffixes)).unique()

if len(extensions) == 0:
    raise ValueError("No input files found to determine extension.")
elif len(extensions) > 1:
    raise ValueError(f"Multiple extensions detected: {extensions}. Please standardize inputs.")
else:
    EXT = extensions[0]  # the single extension to use
    logging.info(f"Using extension: {EXT}")

# Pretty print table without index
logging.info("Input paths with output paths:")
logging.info("\n" + tabulate.tabulate(df_inputs.values, headers=df_inputs.columns, tablefmt="psql"))

# Lists from df_inputs
input_file_paths  = df_inputs["file_path"].tolist()
output_file_paths = df_inputs["output_path"].tolist()


# --- Aggregate rule ---
rule all:
    """
    Final aggregation rule: raw image copies, config copies, and per-image metadata.
    """
    input:
        output_file_paths,
        os.path.join(OUTPUT_PATH, "config", CONFIG_BASENAME),
        os.path.join(OUTPUT_PATH, "config", os.path.basename(config["input_paths"])),
        expand(
            os.path.join(OUTPUT_PATH, "metadata", "{image_name}.json"),
            image_name=image_names
        ),
        expand(
            os.path.join(OUTPUT_PATH, "segments", "{image_name}.npy"),
            image_name=image_names
        ),
        expand(
            os.path.join(OUTPUT_PATH, "regionprops", "{image_name}_props.csv"),
            image_name=image_names
        ),




rule get_config:
    """
    Copy the main configuration file into the pipeline's output/config/ directory.

    Input:
        CONFIG_ABS_PATH (absolute path to the config.yaml used by the workflow)
    Output:
        {OUTPUT_PATH}/config/{CONFIG_BASENAME}
    """
    input:
        CONFIG_ABS_PATH
    output:
        os.path.join(OUTPUT_PATH, "config", CONFIG_BASENAME)
    run:
        import os, shutil
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        shutil.copy(input[0], output[0])


rule get_input_list:
    """
    Copy the input list file (from config['input_paths'])
    into the pipeline's output/config/ directory.
    """
    input:
        config["input_paths"]
    output:
        os.path.join(OUTPUT_PATH, "config", os.path.basename(config["input_paths"]))
    run:
        import os, shutil
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        shutil.copy(input[0], output[0])


rule copy_raw_images:
    """
    Copy raw image files from their original locations (input_file_paths)
    to their standardized locations (output_file_paths).
    Each input file is paired with its corresponding output file by index.
    """
    input:
        input_file_paths
    output:
        output_file_paths
    run:
        import os, shutil
        for src, dst in zip(input, output):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)


rule get_metadata:
    """
    Create a metadata JSON for each raw image.

    Input : {OUTPUT_PATH}/raw_images/{image_name}{EXT}
    Output: {OUTPUT_PATH}/metadata/{image_name}.json
    Notes : Uses the conda env at CONDA_ENV_PATH; image_name is constrained by wildcard_constraints.
    """
    input:
        os.path.join(OUTPUT_PATH, "raw_images", "{image_name}" + EXT)
    output:
        os.path.join(OUTPUT_PATH, "metadata", "{image_name}" + ".json")
    conda:
        CONDA_ENV_PATH
    wildcard_constraints:
        image_name="|".join(image_names)
    shell:
        "python scripts/get_metadata.py {input} {output}"


_SD = config["stardist_params"]  # fail fast if missing

rule segment:
    """
    Segment an image using StarDist params from `config['stardist_params']`.
    Inputs : raw image + metadata JSON
    Outputs: labeled segmentation (.tif) + region properties (.csv)
    """
    input:
        image = os.path.join(OUTPUT_PATH, "raw_images", "{image_name}" + EXT),
        meta  = os.path.join(OUTPUT_PATH, "metadata", "{image_name}.json")
    output:
        seg   = os.path.join(OUTPUT_PATH, "segments", "{image_name}.npy"),
        props = os.path.join(OUTPUT_PATH, "regionprops", "{image_name}_props.csv")
    params:
        model_info = _SD["info"],
        segment_channel = _SD["segment_channel"],
        prob_thresh = float(_SD["prob_thresh"]),
        nms_thresh = float(_SD["nms_thresh"]),
        summarize = _SD.get("summarize_channels", "all")
    conda:
        CONDA_ENV_PATH
    wildcard_constraints:
        image_name="|".join(image_names)
    shell:
        # scripts/segment.py should accept these flags
        "python scripts/segment.py "
        "--image {input.image} --meta {input.meta} "
        "--out-seg {output.seg} --out-props {output.props} "
        "--model-info '{params.model_info}' "
        "--segment_channel {params.segment_channel} "
        "--prob-thresh {params.prob_thresh} "
        "--nms-thresh {params.nms_thresh} "
        "--summarize-channels '{params.summarize}'"

