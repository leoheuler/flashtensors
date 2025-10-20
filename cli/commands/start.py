"""
Start command for the Flash engine in the Flash CLI.
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional

import appdirs
import click
from click import Context

from rich.console import Console
from rich.panel import Panel

import flashtensors as flash
from cli.commands.base import BaseCommand, InteractiveCommandMixin
from cli.styles.colors import ERROR, SUCCESSFULL_UPDATE, WARNING


_DEFAULT_STORAGE_PATH = "/tmp/models"
_DEFAULT_MEMORY_LIMIT = 30
_DEFAULT_PORT = 8073


class StartCommand(BaseCommand, InteractiveCommandMixin):
    """Handle starting the Flash engine in the Flash CLI."""

    def __init__(self):
        super().__init__()

    @property
    def command(self):
        """Return the Click command instance."""

        @click.command(help="Start the Flash engine")
        @click.option(
            "--port",
            "-p",
            type=int,
            default=_DEFAULT_PORT,
            help=f"Port to run the engine on (default: {_DEFAULT_PORT})",
        )
        @click.option(
            "--storage_path",
            "-s",
            type=str,
            default=_DEFAULT_STORAGE_PATH,
            help=f"Path to store engine data (default: {_DEFAULT_STORAGE_PATH})",
        )
        @click.option(
            "--memory-limit",
            "-m",
            type=int,
            default=_DEFAULT_MEMORY_LIMIT,
            help=f"Memory limit for the engine (default: {_DEFAULT_MEMORY_LIMIT})",
        )
        def cmd(port: int, storage_path: Optional[str], memory_limit: int):
            self._start_engine(port, storage_path, memory_limit)

        return cmd

    def _start_engine(
        self,
        port: Optional[int] = _DEFAULT_PORT,
        storage_path: Optional[str] = _DEFAULT_STORAGE_PATH,
        memory_limit: Optional[int] = _DEFAULT_MEMORY_LIMIT,
    ) -> None:
        """Start the Flash engine with the given parameters.

        Args:
            port: Port to run the engine on
            storage_path: Path to store engine data
            memory_limit: Memory limit for the engine
        """
        print(storage_path)
        print(memory_limit)

        flash.configure(
            storage_path=storage_path,  # Where models will be stored
            mem_pool_size=1024**3 * memory_limit,  # 30GB memory pool (GPU Size)
            chunk_size=1024**2 * 32,  # 32MB chunks
            num_threads=8,  # Number of threads
            gpu_memory_utilization=0.8,  # Use 80% of GPU memory
            server_host="0.0.0.0",  # gRPC server host
            server_port=port,  # gRPC server port
        )

        self.console.print(
            Panel.fit(
                f"🚀 Starting Flash Engine\n\n"
                f"• Port: {port}\n"
                f"• Storage: {storage_path}\n"
                f"• Memory Limit: {memory_limit}",
                title="Flash Engine",
                border_style="green",
            )
        )

        self.console.print("Engine is running")

    @classmethod
    def get_interactive_commands(cls):
        """Return interactive command help text."""
        return {
            "start": "Start the Flash engine",
            "start --port <port>": "Start on a specific port",
            "start --storage_path <path>": "Set custom storage path",
            "start --memory-limit <limit>": "Set memory limit",
        }

    @classmethod
    def handle_interactive(cls, command: str, *args, **kwargs):
        """Handle interactive start commands."""
        if command.startswith("start"):
            instance = cls()
            cmd = instance.command
            with Context(cmd) as ctx:
                try:
                    # Parse the arguments using Click's parser
                    args_list = list(args)  # Convert tuple to list
                    parsed_args = cmd.make_parser(ctx).parse_args(args_list)

                    if parsed_args:
                        port = parsed_args[0].get("port", _DEFAULT_PORT)
                        storage_path = parsed_args[0].get(
                            "storage_path", _DEFAULT_STORAGE_PATH
                        )
                        memory_limit = parsed_args[0].get(
                            "memory_limit", _DEFAULT_MEMORY_LIMIT
                        )
                    else:
                        port = _DEFAULT_PORT
                        storage_path = _DEFAULT_STORAGE_PATH
                        memory_limit = _DEFAULT_MEMORY_LIMIT

                except Exception as e:
                    print("Error parsing arguments: ", e)

            instance._start_engine(port, storage_path, memory_limit)
            return True
        return False
