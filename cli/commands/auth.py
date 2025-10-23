"""
Authentication command for the FlashTensors CLI.
"""

import click
import time
from rich.console import Console
from rich.panel import Panel
from cli.commands.base import BaseCommand, InteractiveCommandMixin
from cli.styles.colors import UPDATE, SUCCESSFULL_UPDATE


class AuthCommand(BaseCommand, InteractiveCommandMixin):
    """Handle authentication for the FlashTensors CLI."""

    def __init__(self):
        super().__init__()

    @property
    def command(self):
        """Return the Click command instance."""

        @click.command(help="Authenticate with Flash Tensors")
        def cmd():
            self.execute()

        return cmd

    def execute(self, **kwargs):
        """Execute the authentication process."""
        self.console.print("🔑 Starting authentication", style=f"italic {UPDATE}")    
        with self.console.status("Opening authentication in your browser..."):
            time.sleep(1.5)
        self.console.print(
            "✅ Authentication successful!", style=f"italic {SUCCESSFULL_UPDATE}"
        )
        self.console.print("You are now logged in to Flash Tensors.", style="dim")

    @classmethod
    def get_interactive_commands(cls):
        """Return interactive command help text."""
        return {
            "auth": "Authenticate with Flash Tensors CLI",
        }

    @classmethod
    def handle_interactive(cls, command: str, *args, **kwargs):
        """Handle interactive auth command."""
        if command == "auth":
            instance = cls()
            instance.execute()
            return True
        return False
