"""
Stop command for the Flash engine in the Flash CLI.
"""

import os
import signal
import subprocess
from typing import Optional

import click
from rich.console import Console

from cli.commands.base import BaseCommand, InteractiveCommandMixin
from cli.styles.colors import ERROR, SUCCESSFULL_UPDATE


class StopCommand(BaseCommand, InteractiveCommandMixin):
    """Handle stopping the Flash engine in the Flash CLI."""

    def __init__(self):
        super().__init__()

    @property
    def command(self):
        """Return the Click command instance."""

        @click.command(help="Stop the Flash engine if it's running")
        @click.pass_context
        def stop(ctx: click.Context):
            """Stop the Flash engine if it's running."""
            self.execute()

        return stop

    def execute(self, **kwargs) -> bool:
        """Execute the stop command.

        Returns:
            bool: True if the engine was stopped, False otherwise
        """
        try:
            # Find the process ID of the running Flash engine
            try:
                result = subprocess.run(
                    ["pkill", "-9", "-f", "flashtensors.storage_server"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                pids = result.stdout.strip().split("\n")
            except subprocess.CalledProcessError:
                self.console.print("No running Flash engine process found.", style=ERROR)
                return False

            self.console.print(
                "Flash engine has been stopped successfully", style=SUCCESSFULL_UPDATE
            )
            return True

        except Exception as e:
            self.console.print(f"Error stopping Flash engine: {str(e)}", style=ERROR)
            return False

    @classmethod
    def get_interactive_commands(cls):
        """Return a dictionary of command names to help text for interactive mode."""
        return {"stop": "Stop the Flash engine"}

    @classmethod
    def handle_interactive(cls, command: str, *args, **kwargs):
        """Handle the stop command in interactive mode.

        Returns:
            bool: True if the command was handled, False otherwise
        """
        if command == "stop":
            instance = cls()
            instance.execute()
            return True
        return False
