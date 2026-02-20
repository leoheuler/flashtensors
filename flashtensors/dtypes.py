import torch
import numpy as np

from pydantic import BaseModel, Field


class DType(BaseModel):
    name: str = Field(description="Canonical dtype name, e.g. float32")
    torch_name: str = Field(description="Stored torch dtype name, e.g. float32")
    numpy_name: str = Field(description="Stored numpy dtype name, e.g. float32")
    itemsize: int = Field(description="Size in bytes")

    def astorch(self):
        return getattr(torch, self.torch_name)

    def asnumpy(self):
        if self.numpy_name == "bfloat16":
            raise TypeError(
                "bfloat16 has no NumPy equivalent; use uint16 as the storage dtype"
            )
        return np.dtype(self.numpy_name)


# TODO: Support Quantizations
DTYPES = [
    DType(name="float16", torch_name="float16", numpy_name="float16", itemsize=2),
    DType(name="float32", torch_name="float32", numpy_name="float32", itemsize=4),
    DType(name="float64", torch_name="float64", numpy_name="float64", itemsize=8),
    DType(name="bfloat16", torch_name="bfloat16", numpy_name="bfloat16", itemsize=2),
    DType(name="int8", torch_name="int8", numpy_name="int8", itemsize=1),
    DType(name="uint8", torch_name="uint8", numpy_name="uint8", itemsize=1),
    DType(name="int16", torch_name="int16", numpy_name="int16", itemsize=2),
    DType(name="int32", torch_name="int32", numpy_name="int32", itemsize=4),
    DType(name="int64", torch_name="int64", numpy_name="int64", itemsize=8),
    DType(name="bool", torch_name="bool", numpy_name="bool_", itemsize=1),
    DType(name="complex64", torch_name="complex64", numpy_name="complex64", itemsize=8),
    DType(
        name="complex128", torch_name="complex128", numpy_name="complex128", itemsize=16
    ),
]


DTYPE_MAP = {
    alias: dt
    for dt in DTYPES
    for alias in {
        dt.name,
        dt.torch_name,
        dt.numpy_name,
        f"torch.{dt.torch_name}",
        f"numpy.{dt.numpy_name}",
    }
}
