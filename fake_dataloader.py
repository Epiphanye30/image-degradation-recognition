import torch
from torch.utils.data import Dataset, DataLoader


class FakeDegradationDataset(Dataset):
    """
    Return:
        LQ:  [3, H, W]
        HQ:  [3, H, W]
        degra_tensor: [num_classes]
    """

    def __init__(
        self,
        length: int = 128,
        num_classes: int = 7,
        image_size: int = 224,
        seed: int = 42,
    ):
        self.length = length
        self.num_classes = num_classes
        self.image_size = image_size
        self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        g = torch.Generator().manual_seed(self.seed + idx)

        # HQ image
        HQ = torch.rand(3, self.image_size, self.image_size, generator=g)

        # create LQ by adding noise
        noise = 0.1 * torch.randn(3, self.image_size, self.image_size, generator=g)
        LQ = torch.clamp(HQ + noise, 0.0, 1.0)

        # multilabel degradation vector
        degra_tensor = torch.randint(
            low=0,
            high=2,
            size=(self.num_classes,),
            generator=g,
        ).float()

        if degra_tensor.sum() == 0:
            degra_tensor[
                torch.randint(0, self.num_classes, (1,), generator=g).item()
            ] = 1.0

        return LQ, HQ, degra_tensor


def build_dataloaders(args):

    train_dataset = FakeDegradationDataset(
        length=128,
        num_classes=args.num_classes,
        image_size=224,
        seed=args.seed,
    )

    val_dataset = FakeDegradationDataset(
        length=32,
        num_classes=args.num_classes,
        image_size=224,
        seed=args.seed + 1000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader