from flashtensors.tensor_reference import TensorReference
from typing import List
from pydantic import BaseModel, Field


class FlashState(BaseModel):
    """Container describing a full FlashTensors checkpoint index.

    Attributes:
        tensor_references (List[TensorReference]):
            A list containing metadata entries for all tensors.
    """

    layout: List[TensorReference] = Field(
        ..., description="Metadata for all tensors in the checkpoint."
    )


def build_flash_state(state_dict: Dict[str, torch.Tensor]):
    return FlashState(layout=[])
