from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class SequentialMnist(Dataset):
    def __init__(self, root: str, train: bool, task: str, transform: Optional[Callable] = None):
        self.dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
        self.task = task.upper()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]
        return img, label


def make_permutation(size: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    perm_path = Path("result") / f"psmnist_perm_{seed}.json"
    if perm_path.exists():
        with perm_path.open("r") as f:
            return np.array(json.load(f), dtype=np.int64)
    perm = rng.permutation(size)
    perm_path.parent.mkdir(parents=True, exist_ok=True)
    with perm_path.open("w") as f:
        json.dump(perm.tolist(), f)
    return perm


def image_to_sequence(image: torch.Tensor, in_dim: int, task: str, perm: Optional[np.ndarray] = None) -> torch.Tensor:
    flat = image.view(-1)
    if task.upper() == "PSMNIST":
        if perm is None:
            raise ValueError("Permutation must be provided for PSMNIST")
        flat = flat[torch.from_numpy(perm)]
    if flat.numel() % in_dim != 0:
        raise ValueError("Flattened image length must be divisible by in_dim")
    seq = flat.view(-1, in_dim)
    return seq


def collate_sequences(batch, in_dim: int, task: str, perm: Optional[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    sequences = []
    labels = []
    for img, label in batch:
        seq = image_to_sequence(img, in_dim=in_dim, task=task, perm=perm)
        sequences.append(seq)
        labels.append(label)
    x = torch.stack(sequences, dim=0)  # [B, T, in_dim]
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def make_dataloaders(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = SequentialMnist(args.data_root, train=True, task=args.task, transform=transform)
    test_ds = SequentialMnist(args.data_root, train=False, task=args.task, transform=transform)

    perm = None
    if args.task.upper() == "PSMNIST":
        perm = make_permutation(28 * 28, args.seed)

    collate_fn = lambda batch: collate_sequences(batch, in_dim=args.in_dim, task=args.task, perm=perm)

    if getattr(args, "val_split", 0.0) > 0:
        val_size = int(len(train_ds) * args.val_split)
        train_size = len(train_ds) - val_size
        train_ds, val_ds = random_split(train_ds, [train_size, val_size])
    else:
        val_ds = None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=collate_fn,
        )
    return train_loader, val_loader, test_loader
