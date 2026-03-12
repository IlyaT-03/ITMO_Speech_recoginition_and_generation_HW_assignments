from typing import Any
import torch
from torch import nn

from src.model import SpeechCommandModel
from src.train import train_model, evaluate
from src.utils import count_parameters, count_flops


def run_mel_experiments(
    mel_values: list[int],
    train_loader,
    val_loader,
    test_loader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str | None = None,
    early_stopping_patience: int | None = None,
) -> list[dict[str, Any]]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []
    criterion = nn.CrossEntropyLoss()

    for n_mels in mel_values:
        print(f"\nRunning experiment with n_mels={n_mels}")

        model = SpeechCommandModel(n_mels=n_mels, groups=1).to(device)

        history, best_model, best_epoch, best_val_acc = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            early_stopping_patience=early_stopping_patience,
        )

        test_loss, test_acc = evaluate(
            model=best_model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        result = {
            "n_mels": n_mels,
            "history": history,
            "best_model": best_model,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "num_params": count_parameters(best_model),
            "flops": count_flops(best_model, device=device),
        }

        print(
            f"[RESULT] n_mels={n_mels} | "
            f"best_epoch={best_epoch} | "
            f"best_val_acc={best_val_acc:.4f} | "
            f"test_acc={test_acc:.4f} | "
            f"params={result['num_params']} | "
            f"flops={result['flops']}"
        )

        results.append(result)

    return results


def run_group_experiments(
    group_values: list[int],
    train_loader,
    val_loader,
    test_loader,
    n_mels: int = 80,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str | None = None,
    early_stopping_patience: int | None = None,
) -> list[dict[str, Any]]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []
    criterion = nn.CrossEntropyLoss()

    for groups in group_values:
        print(f"\nRunning experiment with groups={groups}")

        model = SpeechCommandModel(n_mels=n_mels, groups=groups).to(device)

        history, best_model, best_epoch, best_val_acc = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            early_stopping_patience=early_stopping_patience,
        )

        test_loss, test_acc = evaluate(
            model=best_model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        result = {
            "groups": groups,
            "n_mels": n_mels,
            "history": history,
            "best_model": best_model,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "num_params": count_parameters(best_model),
            "flops": count_flops(best_model, device=device),
        }

        print(
            f"[RESULT] groups={groups} | "
            f"best_epoch={best_epoch} | "
            f"best_val_acc={best_val_acc:.4f} | "
            f"test_acc={test_acc:.4f} | "
            f"params={result['num_params']} | "
            f"flops={result['flops']}"
        )

        results.append(result)

    return results