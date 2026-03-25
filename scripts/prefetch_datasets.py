#!/usr/bin/env python3
"""Pre-download dataset files and build STL10 grayscale caches used by the app."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> None:
    data_root = REPO / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    root = str(data_root)

    import torch
    import torchvision

    print(f"torch {torch.__version__}, torchvision {torchvision.__version__}")

    print("Fashion-MNIST (train + test)...")
    from torchvision.datasets import FashionMNIST

    FashionMNIST(root=root, train=True, download=True)
    FashionMNIST(root=root, train=False, download=True)
    print("  ok")

    print("STL10 raw archives (train + test)...")
    from torchvision.datasets import STL10

    STL10(root=root, split="train", download=True)
    STL10(root=root, split="test", download=True)
    print("  ok")

    print("STL10 grayscale .npz caches (one-time build, may take a few minutes)...")
    from src.data_loader import load_stl10

    load_stl10(root=root, train=True, max_samples=1)
    load_stl10(root=root, train=False, max_samples=1)
    print("  ok")

    print("Prefetch finished.")


if __name__ == "__main__":
    main()
