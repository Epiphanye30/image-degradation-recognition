import os
import random
import pandas as pd
from tqdm import tqdm

LQ_ROOT = "./datasets/foundir-v1" # or "./dataset/mdc"
OUTPUT_PARQUET = "./datasets/foundir_dataset.parquet" # or "mdc_dataset.parquet"
SAMPLES_PER_DEG = 50


def collect_lq_images():
    samples = []

    for deg_str in os.listdir(LQ_ROOT):
        deg_path = os.path.join(LQ_ROOT, deg_str, "LQ")

        if not os.path.isdir(deg_path):
            continue

        image_list = []

        for img in os.listdir(deg_path):
            lq_path = os.path.join(deg_path, img)
            image_list.append((lq_path, deg_str))

        random.shuffle(image_list)
        samples.extend(image_list[:SAMPLES_PER_DEG])

    return samples


def generate_parquet():
    samples = collect_lq_images()
    rows = []

    for lq_path, degradation in tqdm(samples):
        with open(lq_path, "rb") as f:
            lq_bytes = f.read()

        rows.append(
            {
                "image_bytes": lq_bytes,
                "degradation": degradation,
            }
        )

    df = pd.DataFrame(rows)
    df.to_parquet(
        OUTPUT_PARQUET,
        engine="pyarrow",
        compression="snappy",
    )
    print("Saved:", OUTPUT_PARQUET)
    print("Total samples:", len(df))


if __name__ == "__main__":
    random.seed(42)
    generate_parquet()