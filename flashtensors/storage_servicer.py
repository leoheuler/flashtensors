# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #

import asyncio
import grpc
from .proto import storage_pb2_grpc, storage_pb2
from .utils import init_logger
from ._checkpoint_store import (CheckpointStore, MemCopyChunk)

class StorageServicer(storage_pb2_grpc.StorageServicer):

    def __init__(
        self,
        storage_path,
        mem_pool_size,
        num_thread,
        chunk_size,
        registration_required,
    ):
        """Initialize the storage servicer
        Args:
            storage_path: The path to the storage directory
            mem_pool_size: The size of the memory pool
            num_thread: The number of threads to use
            chunk_size: The size of each chunk
            registration_required: Whether registration is required
        """

        self.logger = init_logger(__name__)

        self.storage_path = storage_path
        self.mem_pool_size = mem_pool_size
        self.num_thread = num_thread
        self.chunk_size = chunk_size
        self.registration_required = registration_required

        self.logger.info(
            f"StorageServicer: storage_path={storage_path}, "
            f"mem_pool_size={mem_pool_size}, num_thread={num_thread}, "
            f"chunk_size={chunk_size}, "
            f"registration_required={registration_required}"
        )

        if not CheckpointStore:
            raise ValueError("Unable to initialize CheckpointStore")

        self.storage = CheckpointStore(
            storage_path, mem_pool_size, num_thread, chunk_size
        )

        return

    async def LoadModelAsync(self, request, context):
        model_path = request.model_path
        if not model_path:
            self.logger.error("model_path is empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return storage_pb2.LoadModelResponse()

        if not self.registration_required:
            model_size = self.storage.register_model_info(model_path)
            if model_size < 0:
                self.logger.error("RegisterModel failed")
                context.set_code(grpc.StatusCode.INTERNAL)
                return storage_pb2.LoadModelResponse()

        device_type = request.target_device_type
        if device_type == storage_pb2.DEVICE_TYPE_CPU:
            ret = self.storage.load_model_from_disk_async(model_path)
        elif device_type == storage_pb2.DEVICE_TYPE_GPU:
            replica_uuid = request.replica_uuid
            if not replica_uuid:
                self.logger.error("replica_uuid is empty")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return storage_pb2.LoadModelResponse()

            gpu_memory_handles = {
                device_uuid: [
                    handle.cuda_ipc_handle for handle in handle_list.handles
                ]
                for device_uuid, handle_list in request.handles.items()
            }

            def create_mem_copy_chunk(chunk):
                mem_copy_chunk = MemCopyChunk()
                mem_copy_chunk.src_offset = chunk.src_offset
                mem_copy_chunk.size = chunk.size
                mem_copy_chunk.dst_offset = chunk.dst_offset
                mem_copy_chunk.handle_idx = chunk.handle_idx
                return mem_copy_chunk

            mem_copy_chunks = {
                device_uuid: [
                    create_mem_copy_chunk(chunk) for chunk in chunk_list.chunks
                ]
                for device_uuid, chunk_list in request.chunks.items()
            }
            # self.logger.debug(
            #     f"LoadModelAsync: {model_path}, {replica_uuid}, "
            #     f"{gpu_memory_handles}, {mem_copy_chunks}"
            # )
            ret = self.storage.load_model_from_mem_async(
                model_path, replica_uuid, gpu_memory_handles, mem_copy_chunks
            )
        else:
            self.logger.error(f"Unsupported device type: {device_type}")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return storage_pb2.LoadModelResponse()

        if ret != 0:
            self.logger.error("LoadModel failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            return storage_pb2.LoadModelResponse()

        self.logger.info(
            f"LoadModel: success {model_path} with target {device_type}"
        )
        return storage_pb2.LoadModelResponse(model_path=model_path)

    async def ConfirmModel(self, request, context):
        model_path = request.model_path
        replica_uuid = request.replica_uuid
        device_type = request.target_device_type

        if not model_path:
            self.logger.error("model_path is empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return storage_pb2.ConfirmModelResponse()

        if device_type != storage_pb2.DEVICE_TYPE_GPU:
            self.logger.error(f"Unsupported device type: {device_type}")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return storage_pb2.ConfirmModelResponse()

        for i in range(10):  # Increase retries for large models like PaliGemma
            try:
                ret = self.storage.wait_model_in_gpu(model_path, replica_uuid)
                if ret == 0:
                    self.logger.info(
                        f"Confirm model {model_path} replica {replica_uuid} success"
                    )
                    return storage_pb2.ConfirmModelResponse(model_path=model_path)
                elif ret == 1:
                    self.logger.warning(f"Model {model_path} replica {replica_uuid} is interrupted")
                    context.set_code(grpc.StatusCode.INTERNAL)
                    return storage_pb2.ConfirmModelResponse()
                else:
                    self.logger.warning(f"Confirm model failed with ret={ret}, retry {i + 1}")
            except Exception as e:
                self.logger.error(f"Exception in wait_model_in_gpu: {e}")
                if i == 19:  # Last retry
                    raise

            await asyncio.sleep(0.1)  # Longer sleep for stability

        self.logger.error(
            f"Confirm model {model_path} replica {replica_uuid} failed"
        )
        context.set_code(grpc.StatusCode.INTERNAL)
        return storage_pb2.ConfirmModelResponse()

    async def UnloadModel(self, request, context):
        model_path = request.model_path
        device_type = request.target_device_type

        if not model_path:
            self.logger.error("model_path is empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return storage_pb2.UnloadModelResponse()

        if device_type != storage_pb2.DEVICE_TYPE_CPU:
            self.logger.error(f"Unsupported device type: {device_type}")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return storage_pb2.UnloadModelResponse()

        for i in range(5):
            ret = self.storage.unload_model_from_host(model_path)
            if ret == 0:
                self.logger.info(f"UnloadModel: success {model_path}")
                return storage_pb2.UnloadModelResponse(model_path=model_path)
            self.logger.info(f"UnloadModel failed, retry {i + 1}")

            await asyncio.sleep(0.01)

        self.logger.error(f"UnloadModel failed for model {model_path}")
        context.set_code(grpc.StatusCode.INTERNAL)
        return storage_pb2.UnloadModelResponse()

    async def ClearMem(self, request, context):
        ret = self.storage.clear_mem()
        if ret != 0:
            self.logger.error("ClearMem failed")
            context.set_code(grpc.StatusCode.INTERNAL)
        return storage_pb2.ClearMemResponse()

    async def RegisterModel(self, request, context):
        model_path = request.model_path
        if not model_path:
            self.logger.error("model_path is empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return storage_pb2.RegisterModelResponse()

        model_size = self.storage.register_model_info(model_path)
        if model_size < 0:
            self.logger.error("RegisterModel failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            return storage_pb2.RegisterModelResponse()

        return storage_pb2.RegisterModelResponse(
            model_path=model_path, model_size=model_size
        )

    async def GetServerConfig(self, request, context):
        return storage_pb2.GetServerConfigResponse(
            mem_pool_size=self.storage.get_mem_pool_size(),
            chunk_size=self.storage.get_chunk_size(),
        )
