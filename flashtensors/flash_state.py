from typing import List

from pydantic import BaseModel, Field

from .tensor_reference import TensorReference


class FlashState(BaseModel):
    """Container describing a full FlashTensors checkpoint index.

    Attributes:
        layout (List[TensorReference]):
            A list containing metadata entries for all tensors.
    """

    layout: List[TensorReference] = Field(
        ..., description="Metadata for all tensors in the checkpoint."
    )
