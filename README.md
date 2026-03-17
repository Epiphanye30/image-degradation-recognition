# ECE176 Project

This repository contains code for training image degradation classifiers from parquet datasets and generating experiment figures.

## Project Structure

- `train.py`: main training entry point, supporting `resnet18`, truncated ResNet variants, and the custom `LMS-CNN`
- `dataloader.py`: loads images and labels from parquet files and builds dataloaders
- `generate_parquet.py`: builds a single parquet dataset from the raw MDC data
- `split_parquet.py`: splits one parquet file into training and validation sets
- `preview_parquet.py`: previews random samples from a parquet dataset
- `plots.py`: unified plotting entry point, replacing the old `plot_B.py` and `plot_C.py`

## Environment Setup

Python 3.10 or 3.11 is recommended.

```bash
conda create -n ece176 python=3.11
conda activate ece176
pip install -r requirements.txt
```

## Data Preparation

### 1. Prepare Data

Download the **ImageNet** dataset and the **FoundIR-V1** test set. For the ImageNet dataset, synthesize the Low-Quality (LQ) images following the instructions in [AgenticIR/dataset](https://github.com/Kaiwen-Zhu/AgenticIR/tree/main/dataset).

Organize the datasets according to the following directory structure:

```text
datasets/
├── mdc/                          # ImageNet-based Multi-Degradation Corpus
│   ├── noise+rain/
│   ├── motion_blur+jpeg/
│   └── ...                       # Other degradation combinations
└── foundir/                      # FoundIR-V1 dataset
    ├── dark+noise/
    ├── noise+haze/
    └── ...
```

2. Generate the parquet dataset:

```bash
python generate_parquet.py
```

3. Split the dataset into training and validation sets:

```bash
python split_parquet.py --input datasets/mdc_dataset.parquet --train_out datasets/train.parquet --val_out datasets/val.parquet
```

4. Preview random samples:

```bash
python preview_parquet.py --parquet datasets/train.parquet --num_samples 6 --save_path figures/data_preview.png
```

## Training

Basic example:

```bash
python train.py --train_parquet datasets/train.parquet --val_parquet datasets/val.parquet --save_dir checkpoints/standard_resnet18
```

Custom backbone example:

```bash
python train.py --backbone customized_cnn --base_channels 8 --train_parquet datasets/train.parquet --val_parquet datasets/val.parquet --save_dir checkpoints/standard_customized_bc8
```

The main outputs are saved under `checkpoints/`, including:

- `metrics.csv`: per-epoch training and validation metrics
- `test_metrics.csv`: final test metrics
- `best_model.pt` / `last_model.pt`: model checkpoints

## Plotting

Use the unified plotting script:

```bash
python plots.py --figure all
```

Available `--figure` options:

- `fig1`: Plan A final performance bar chart
- `fig2`: model capacity vs. performance
- `fig3`: pretrained vs. scratch training curves
- `fig4`: training curves across base channel sizes
- `fig5`: pretrained vs. scratch final performance
- `part-b`: Part B OOD generalization figure
- `part-c`: Part C training strategy figure

If your experiment outputs are stored somewhere else, specify the directories explicitly:

```bash
python plots.py --figure part-b --checkpoints-dir checkpoints/full_redo --output-dir figures
```

## Notes

- `.gitignore` excludes local datasets, checkpoints, generated figures, and cache files so large artifacts are not committed.
- `train.py` currently contains fixed `HF_HOME` and `TORCH_HOME` paths. If you run this on another machine, it is better to replace them with local paths or environment variables.
