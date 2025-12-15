"""A FlashTensor is a wrapper on top of torch.nn.Module that enables blazing fast loading"""

import torch


class FlashTensor(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        """
        Args:
            module (torch.nn.Module): The module to wrap
        """
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            *args: Arguments to pass to the module
            **kwargs: Keyword arguments to pass to the module
        """
        return self.module(*args, **kwargs)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        """
        Args:
            state_dict (dict[str, torch.Tensor]): The state dict to load
        """
        self.module.load_state_dict(state_dict)

    def to(self, device: torch.device, non_blocking: bool = False):
        """
        Args:
            device (torch.device): The device to move the module to
            non_blocking (bool): Whether to use non-blocking memory transfer
        """
        self.module.to(device, non_blocking=non_blocking)
        return self
