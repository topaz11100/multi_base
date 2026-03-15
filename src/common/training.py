from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long().view(-1)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += float(loss.item()) * int(y.numel())
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    if total == 0:
        return 0.0, 0.0
    return correct / total, loss_sum / total


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    soft_mask_epochs: Optional[int] = None,
    stabilize_epochs: int = 0,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    weight_decay_dend_soma: Optional[float] = None,
    grad_clip: Optional[float] = 1.0,
    lambda_ortho: float = 0.0,
    lambda_s: float = 0.0,
    log_every: int = 1,
    max_train_batches: Optional[int] = None,
    max_test_batches: Optional[int] = None,
) -> Dict[str, Any]:
    model.to(device)
    from src.common.optim import build_adamw

    optimizer, _ = build_adamw(
        model,
        lr=float(lr),
        weight_decay=float(weight_decay),
        weight_decay_dend_soma=weight_decay_dend_soma,
    )
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    soft_e = int(epochs) if soft_mask_epochs is None else int(soft_mask_epochs)
    stb_e = int(stabilize_epochs)
    total_e = int(soft_e + stb_e)

    pbar = tqdm(range(1, total_e + 1), desc="epoch", total=int(total_e), leave=True)
    for epoch in pbar:
        if int(stb_e) > 0 and int(epoch) == int(soft_e) + 1:
            from src.common.model_utils import harden_variable_branches_

            harden_variable_branches_(model)
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0

        for bi, (x, y) in enumerate(train_loader):
            if max_train_batches is not None and bi >= max_train_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long().view(-1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)

            # Optional regularization from model
            if hasattr(model, "regularization_loss") and callable(getattr(model, "regularization_loss")):
                reg = model.regularization_loss(lambda_ortho=lambda_ortho, lambda_s=lambda_s)  # type: ignore
                loss = loss + reg

            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            loss_sum += float(loss.item()) * int(y.numel())
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())

        train_acc = correct / total if total > 0 else 0.0
        train_loss = loss_sum / total if total > 0 else 0.0

        test_acc, test_loss = evaluate_classifier(model, test_loader, device, max_batches=max_test_batches)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if log_every and (epoch % log_every == 0):
            pbar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                train_acc=f"{train_acc:.4f}",
                test_loss=f"{test_loss:.4f}",
                test_acc=f"{test_acc:.4f}",
            )

    return history
