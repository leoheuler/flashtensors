from typing import List
from pydantic import BaseModel, Field
from .dtypes import DTYPE_MAP

class TensorReference(BaseModel):
    """Metadata describing the location and layout of a tensor in a FlashTensors shard.

    Attributes:
        name (str): 
            Symbolic tensor name (e.g., `"linear1.weight"`).
        offset (int): 
            Byte offset in the `.data` file where the tensor's raw storage begins.
        size (int): 
            Total storage size in bytes.
        shape (List[int]): 
            Tensor shape in row-major order.
        stride (List[int]): 
            Tensor stride (in number of elements) per dimension.
        dtype (str): 
            String key identifying the tensor dtype (canonical, torch.*, numpy.*, etc.).
    """

    name: str = Field(..., description="Tensor name (e.g. 'linear.weight').")
    offset: int = Field(..., description="Byte offset within the data file.")
    size: int = Field(..., description="Tensor size in bytes.")
    shape: List[int] = Field(..., description="Tensor shape.")
    stride: List[int] = Field(..., description="Tensor stride (in elements).")
    dtype: str = Field(..., description="Dtype key into DTYPE_MAP.")

    def torch_dtype(self):
        """Return the tensor dtype as a `torch.dtype`.

        Returns:
            torch.dtype: The corresponding PyTorch dtype.
        """
        return DTYPE_MAP[self.dtype].astorch()

    def numpy_dtype(self):
        """Return the tensor dtype as a `numpy.dtype`.

        Returns:
            numpy.dtype: The corresponding NumPy dtype.
        """
        return DTYPE_MAP[self.dtype].asnumpy()


def build_tensor_reference(name, offset, size, shape, stride, dtype):
    return TensorReference(
        name=name,
        offset=offset,
        size=size,
        shape=shape,
        stride=stride,
        dtype=dtype,
    )
    