# OlmoEarth Embeddings Tutorial

End-to-end examples that compute, download, and analyze OlmoEarth embeddings
via the Studio API. These scripts reproduce the use cases from the
[custom embeddings blog post](https://blog.allenai.org/olmoearth-embeddings):

1. **Similarity search** -- cosine-similarity heatmap and patch mosaic over
   California's Central Valley.
2. **Per-pixel segmentation** -- few-shot land-cover classification from 60
   labeled pixels in Ca Mau (Vietnam).
3. **Change detection** -- wildfire burn scar detection via cosine distance
   between monthly embeddings (Park Fire, California, Sept 2023 vs Sept 2024).
4. **PCA exploration** -- false-color RGB from the first three principal
   components of Flevoland (Netherlands) embeddings.

Examples 1, 2, and 4 use **OlmoEarth-v1-Tiny** (192-dim, 6.2M parameters)
at 40-meter resolution with 12 monthly periods (annual composites) of
Sentinel-2 L2A imagery. Example 3 uses the same encoder at 1 monthly period.

## Prerequisites

Install dependencies from the repo root:

```bash
uv sync
```

You also need an **OlmoEarth Studio API key**. Set it as an environment
variable or pass it via `--api-key`:

```bash
export OLMOEARTH_API_KEY="your-key-here"
```

## Quick start

Run all commands from this directory
(`docs/tutorials/OlmoEarthEmbeddings/`).

### 1. One-time setup

Create a Studio project and two embeddings models (annual and monthly):

```bash
python -m setup_model
```

This writes `config.json` with `project_id`, `model_id` (annual), and
`monthly_model_id` that subsequent scripts read automatically.

### 2. Similarity search

```bash
python -m similarity.compute --config config.json
python -m similarity.analyze \
    --embed data/central_valley/embeddings.tif \
    --rgb data/central_valley/s2_rgb.tif \
    --out figures/
```

Outputs: `figures/similarity_heatmap.png`, `figures/similarity_mosaic.png`.

### 3. Few-shot segmentation

```bash
python -m segmentation.compute --config config.json
python -m segmentation.analyze \
    --data-dir data/segmentation/ca_mau \
    --out figures/
```

Output: `figures/fewshot_60labels.png`.

### 4. Change detection

```bash
python -m change_detection.compute --config config.json
python -m change_detection.analyze \
    --before-dir data/change_detection/sept_2023 \
    --after-dir data/change_detection/sept_2024 \
    --out figures/
```

Output: `figures/change_detection.png`.

### 5. PCA false-color

```bash
python -m pca.compute --config config.json
python -m pca.analyze \
    --embed data/flevoland/embeddings.tif \
    --rgb data/flevoland/s2_rgb.tif \
    --out figures/
```

Output: `figures/pca_false_color.png`.

## Notebooks

The `notebooks/` directory contains Jupyter notebooks for the analyze
steps. They import the same functions used by the CLI scripts, so there is
no duplicated logic. Run the corresponding `compute.py` first to download
data, then open a notebook to explore results interactively:

```bash
jupyter notebook notebooks/similarity.ipynb
```

## Directory layout

```
OlmoEarthEmbeddings/
    config.json                 # (generated) project/model IDs
    setup_model.py              # one-time project + model creation
    change_detection/
        analyze.py              # cosine distance + figure
        compute.py              # submit + download Park Fire (monthly)
    common/
        embedding_utils.py      # load embedding COGs
        imagery_sources.py      # Sentinel-2 RGB + WorldCover download
        studio_client.py        # API client, polling, result download
    data/                       # (generated, git-ignored)
    figures/                    # (generated, git-ignored)
    notebooks/
        change_detection.ipynb  # interactive change detection
        pca.ipynb               # interactive PCA analysis
        segmentation.ipynb      # interactive few-shot classification
        similarity.ipynb        # interactive similarity analysis
    pca/
        analyze.py              # PCA false-color figure
        compute.py              # submit + download Flevoland
    segmentation/
        analyze.py              # few-shot linear probe + figure
        compute.py              # submit + download Ca Mau
    similarity/
        analyze.py              # heatmap + patch mosaic
        compute.py              # submit + download Central Valley
```

## Notes

- The **compute** scripts require network access and a valid API key. They
  submit predictions and download satellite imagery from Planetary Computer.
- The **analyze** scripts are standalone: they take local file paths and can
  run offline once the data is downloaded.
- Sentinel-2 median composites can take several minutes to build depending on
  scene count and area size.
