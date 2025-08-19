# Segmentation Pipeline

Snakemake workflow for segmenting hybridization imaging data.

## Usage

Run the pipeline from this directory:

```bash
snakemake --cores 1 --use-conda --drop-metadata --verbose
```

Configuration files live in `config/` and custom scripts in `scripts/`.
