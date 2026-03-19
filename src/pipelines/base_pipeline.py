import torch
from abc import ABC, abstractmethod
from typing import Any


def convert_to_tensor(data: Any, dtype: torch.dtype) -> torch.Tensor:
    """
    Converts input data to a PyTorch tensor.
    """
    print(f"Converting to PyTorch Tensors with dtype {dtype}")
    return torch.as_tensor(data, dtype=dtype)


class BasePipeline(ABC):
    @abstractmethod
    def _train_test_split(
        self,
        X,
        y,
        stratify,
        test_split: float = 0.2,
        random_state: int = 42,
        shuffle: bool = True,
    ):
        pass

    @abstractmethod
    def pipeline(self) -> tuple[torch.Tensor, ...]:
        pass
