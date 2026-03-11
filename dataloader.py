import io
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
        self.df = pd.read_parquet(parquet_path)
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
    val_dataset = ParquetDegradationDataset(
        parquet_path=args.val_parquet,
        image_size=224,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader
