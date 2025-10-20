"""
Models command for listing available and downloaded models in the Teil CLI.
"""

import click
from rich.console import Console
from rich.table import Table
import flashtensors as flash
from cli.commands.base import BaseCommand, InteractiveCommandMixin


class ModelsCommand(BaseCommand, InteractiveCommandMixin):
    """Handle model listing in the Teil CLI."""

    def __init__(self):
        super().__init__()

    @property
    def command(self):
        """Return the Click command instance."""

        @click.command(help="List available and downloaded models")
        def cmd():
            self.execute()

        return cmd

    def execute(self, **kwargs):
        """Execute the model listing."""
        models = flash.list_models()
        console = Console()
        table = Table(title="Models", show_header=True, header_style="bold magenta")
        table.add_column("Model", style="cyan", width=40)
        table.add_column("Size (GB)", style="magenta")
        for model_key, model_info in models.items():
            model_size = model_info["size"] / (1024**3)
            table.add_row(model_key, f"{model_size:.2f}")
        console.print(table)

    @classmethod
    def get_interactive_commands(cls):
        """Return interactive command help text."""
        return {
            "models": "List all available models",
        }

    @classmethod
    def handle_interactive(cls, command: str, *args, **kwargs):
        """Handle interactive models command."""
        if command == "models":
            instance = cls()
            options = {}

            instance.execute()
            return True
        return False
