import asyncio
import logging
import os
from typing import Optional

import grpc
from grpc.aio import Server

from .storage_servicer import StorageServicer
from .proto import storage_pb2_grpc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class StorageServer:
    def __init__(
        self,
        host: str,
        port: int,
        storage_path: str,
        num_thread: int,
        chunk_size: int,
        mem_pool_size: int,
        registration_required: bool,
        max_workers: Optional[int] = None,
    ):
        """
        Args:
            host: The host to bind the server to
            port: The port to bind the server to
            storage_path: The path to the storage directory
            num_thread: The number of threads to use
            chunk_size: The size of each chunk
            mem_pool_size: The size of the memory pool
            registration_required: Whether registration is required
            max_workers: The maximum number of workers
        Raises:
            ValueError: If any of the arguments are invalids
        """

        if not storage_path:
            storage_path = os.getenv("STORAGE_PATH", "/tmp/flashtensors_models")
            
        if not storage_path:
            raise ValueError("storage_path cannot be empty")

        if not os.path.isdir(storage_path):
            raise ValueError(f"storage_path must be a directory: {storage_path}")

        if mem_pool_size <= 0:
            raise ValueError(f"mem_pool_size must be greater than 0: {mem_pool_size}")

        if num_thread <= 0:
            raise ValueError(f"num_thread must be greater than 0: {num_thread}")

        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be greater than 0: {chunk_size}")

        self.host = host
        self.port = port
        self.server: Optional[Server] = None
        self.servicer = StorageServicer(
            storage_path,
            mem_pool_size,
            num_thread,
            chunk_size,
            registration_required,
        )
        self.max_workers = max_workers or num_thread

    async def start(self):
        """Start the storage server.

        Raises:
            Exception: If the server fails to start
        """
        try:
            self.server = grpc.aio.server()
            storage_pb2_grpc.add_StorageServicer_to_server(self.servicer, self.server)

            listen_addr = f"{self.host}:{self.port}"
            port = self.server.add_insecure_port(listen_addr)

            if port != self.port:
                logger.warning(f"Port {self.port} is not available, using {port}")
                self.port = port

            logger.info(f"Starting gRPC server on {listen_addr}")
            await self.server.start()
            logger.info("gRPC server started successfully")

            await self.server.wait_for_termination()

        except Exception as e:
            logger.error(f"Server startup failed: {str(e)}", exc_info=True)
            if self.server:
                await self.server.stop(5)
            raise

    async def stop(self, grace: float = 5):
        """Stop the storage server.

        Args:
            grace: The grace period in seconds

        Raises:
            Exception: If the server fails to stop
        """

        if self.server:
            logger.info(f"Shutting down gRPC server with grace period {grace}s")
            await self.server.stop(grace)
            logger.info("gRPC server stopped successfully")
        else:
            logger.warning("Server not running - cannot stop")


async def serve(
    host: str,
    port: int,
    storage_path: str,
    num_thread: int,
    chunk_size: int,
    mem_pool_size: int,
    registration_required: bool,
    max_workers: Optional[int] = None,
):
    """Start the storage server.

    Args:
        host: The host to bind the server to
        port: The port to bind the server to
        storage_path: The path to the storage directory
        num_thread: The number of threads to use
        chunk_size: The size of each chunk
        mem_pool_size: The size of the memory pool
        registration_required: Whether registration is required
        max_workers: The maximum number of workers

    Returns:
        None

    Raises:
        KeyboardInterrupt: If the server is interrupted
        asyncio.CancelledError: If the server is cancelled
        Exception: If the server fails to start
    """
    server = StorageServer(
        host,
        port,
        storage_path,
        num_thread,
        chunk_size,
        mem_pool_size,
        registration_required,
        max_workers,
    )
    try:
        await server.start()
    except (KeyboardInterrupt, asyncio.CancelledError):
        await server.stop()
    except Exception as e:
        logger.error(f"Server failed to start: {str(e)}", exc_info=True)
        raise


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Storage Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8073)
    parser.add_argument("--storage-path", default="/workspace")
    parser.add_argument("--num-thread", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=1024**2*32)
    parser.add_argument("--mem-pool-size", type=int, default=1024**3*12) # 1024**3*12 = 12GB
    parser.add_argument("--registration-required", action="store_true")
    parser.add_argument("--max-workers", type=int)

    args = parser.parse_args()

    logger.info(
        f"Starting storage server with args: {args.host}, {args.port}, {args.storage_path}, {args.num_thread}, {args.chunk_size}, {args.mem_pool_size}, {args.registration_required}, {args.max_workers}"
    )

    asyncio.run(
        serve(
            args.host,
            args.port,
            args.storage_path,
            args.num_thread,
            args.chunk_size,
            args.mem_pool_size,
            args.registration_required,
            args.max_workers,
        )
    )


if __name__ == "__main__":
    main()
