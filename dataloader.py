import io
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split

DEGRADATION_DICT = {
    "dark": 0,
    "noise": 1,
    "haze": 2,
    "jpeg compression artifact": 3,
    "defocus blur": 4,
    "motion blur": 5,
    "rain": 6,
    "low resolution": 7,
}


class ParquetDegradationDataset(Dataset):
    def __init__(
        self,
        parquet_path,
        image_size=224,
        degradation_dict=DEGRADATION_DICT,
    ):
        if isinstance(parquet_path, str):
            self.df = pd.read_parquet(parquet_path)

        elif isinstance(parquet_path, (list, tuple)):
            dfs = [pd.read_parquet(p) for p in parquet_path]
            self.df = pd.concat(dfs, ignore_index=True)

        self.image_size = image_size
        self.degradation_dict = degradation_dict
        self.num_classes = len(degradation_dict)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.df)

    def decode_image(self, img_bytes):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = self.transform(img)

        return img

    def parse_degradation(self, degradation_str):
        degra_tensor = torch.zeros(self.num_classes)
        degra_list = degradation_str.split("+")

        for d in degra_list:
            d = d.strip()
            if d in self.degradation_dict:
                degra_tensor[self.degradation_dict[d]] = 1

        return degra_tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        lq_bytes = row["image_bytes"]
        # hq_bytes = row["hq_bytes"]
        degradation = row["degradation"]
        LQ = self.decode_image(lq_bytes)
        # HQ = self.decode_image(hq_bytes)
        degra_tensor = self.parse_degradation(degradation)
        return LQ, degra_tensor


def build_dataloaders(args):
    train_dataset = ParquetDegradationDataset(
        parquet_path=args.train_parquet,
        image_size=224,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    if isinstance(args.val_parquet, str):
        val_dataset = ParquetDegradationDataset(
            parquet_path=args.val_parquet,
            image_size=224,
        )
        val_loaders = [DataLoader(
            val_dataset,
            batch_size=args.bs,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )]

    elif isinstance(args.val_parquet, (list, tuple)):
        val_dataset = []
        val_loaders = []
        for val_parquet in args.val_parquet:
            val_dataset = ParquetDegradationDataset(
                parquet_path=val_parquet,
                image_size=224,
            )

            val_loaders.append(DataLoader(
                val_dataset,
                batch_size=args.bs,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            ))
    
    args.num_classes = len(DEGRADATION_DICT)
    return train_loader, val_loaders
