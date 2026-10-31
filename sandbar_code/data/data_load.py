import os
from pathlib import Path
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    image_dir = os.path.join(path, 'train')

    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(256, pad_if_needed=True, padding_mode='reflect'),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=transform, input_dir='input', target_dir='target',
                      names_file=Path(__file__).resolve().parents[2] / 'data/splits/train.txt'),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'valid')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False,
                 input_dir='blur', target_dir='sharp', names_file=None):
        self.image_dir = image_dir
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.image_list = (names_file.read_text().splitlines() if names_file is not None
                           else os.listdir(os.path.join(image_dir, input_dir)))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.input_dir, self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, self.target_dir, self.image_list[idx]))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        # Case-insensitive extension check (dataset contains both .jpg and .JPEG).
        for x in lst:
            splits = x.split('.')
            if splits[-1].lower() not in ['png', 'jpg', 'jpeg', 'tif']:
                raise ValueError
