from pydantic import BaseModel
from enum import Enum

from pydantic import BaseModel, Field
import torch
import numpy as np

from typing import List
from pydantic import BaseModel, Field



def save_tensors(tensor_names, tensor_data_index, model_path):
    os.makedirs(model_path, exist_ok=True)
    tensor_filename = os.path.join(model_path, "tensor.flashtensors")
    offsets = {}
    data_record = {}

    with open(tensor_filename, "wb") as f:
        offset = 0
        total = len(tensor_names)

        for count, name in tqdm(enumerate(tensor_names, 1)):
            data_ptr, size = tensor_data_index[name]
            tensor = state_dict[name]
            buf = tensor.cpu().numpy().tobytes()

            # Deduplication based on data pointer
            if data_ptr in data_record:
                offsets[name] = offsets[data_record[data_ptr]]
                continue

            data_record[data_ptr] = name

            f.write(buf)
            offsets[name] = offset
            offset += len(buf)

            # Optional progress display
            print(f"\rSaving tensors: {count}/{total} ({100*count/total:.1f}%)", end="")

    print("\nDone.")
    return offsets

# --- 3. Provided save_dict implementation ---
def save_dict(state_dict: Dict[str, torch.Tensor], model_path: Union[str, os.PathLike]):
    tensor_names = list(state_dict.keys())
    tensor_data_index = {}
    for name, param in state_dict.items():
        param_storage = param.untyped_storage()
        data_ptr = param_storage.data_ptr()
        size = param_storage.size()
        tensor_data_index[name] = (data_ptr, size)

    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    tensor_offsets = save_tensors(tensor_names, tensor_data_index, model_path)

    layout = []
    for name, param in state_dict.items():
        # name: offset, size, shape, stride, dtype
        layout.append(
            TensorReference(name=name,
            offset= tensor_offsets[name],
            size = tensor_data_index[name][1],
            shape = tuple(param.shape),
            stride=tuple(param.stride()),
            dtype = str(param.dtype))
        )

    flash_state = FlashState(tensor_references = tensor_index)

    with open(os.path.join(model_path, "tensor_index.json"), "w") as f:
        json.dump(flash_state.model_dump(), f, indent=2)

# --- 4. Save model ---
save_dir = "./tiny_model_raw/"
save_dict(state_dict, save_dir)

print(f"✅ Model tensors saved under {save_dir}")
print(os.listdir(save_dir))


del model
del state_dict

