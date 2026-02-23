import abc


class BaseLoader(abc.ABC):
    @abc.abstractmethod
    def load(self, data_path, file_size, layout, device_map, num_workers, chunk_size):
        """Load tensors from binary data file. Returns Dict[str, Tensor]."""
