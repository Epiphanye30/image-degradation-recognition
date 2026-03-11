import os
import random
import pandas as pd
from tqdm import tqdm

LQ_ROOT = "/root/pubdatasets2/imagenet/MDC/LQ"
HQ_ROOT = "/root/pubdatasets2/imagenet/MDC/HQ"
D_LEVELS = ["d2", "d3"]
OUTPUT_PARQUET = "mdc_dataset.parquet"
SAMPLES_PER_DEG = 500


def collect_lq_images():
    samples = []

    for d in D_LEVELS:
        d_root = os.path.join(LQ_ROOT, d)

        for degradation in os.listdir(d_root):
            deg_path = os.path.join(d_root, degradation)

            if not os.path.isdir(deg_path):
                continue

            image_list = []

            for cls in os.listdir(deg_path):
                cls_path = os.path.join(deg_path, cls)

                if not os.path.isdir(cls_path):
                    continue

                for img in os.listdir(cls_path):
                    lq_path = os.path.join(cls_path, img)
                    image_list.append((lq_path, degradation))

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
