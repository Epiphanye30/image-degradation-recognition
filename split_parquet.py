import argparse
import pandas as pd


def split_parquet(input_path, train_path, val_path, train_ratio=0.8, seed=42):

    df = pd.read_parquet(input_path)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_train = int(len(df) * train_ratio)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:]
    train_df.to_parquet(train_path, engine="pyarrow", compression="snappy")
    val_df.to_parquet(val_path, engine="pyarrow", compression="snappy")

    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Saved train -> {train_path}")
    print(f"Saved val   -> {val_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--train_out", type=str, default="train.parquet")
    parser.add_argument("--val_out", type=str, default="val.parquet")
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    split_parquet(
        args.input,
        args.train_out,
        args.val_out,
        args.ratio,
        args.seed,
    )