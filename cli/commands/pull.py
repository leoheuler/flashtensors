"""
Pull command for downloading models in the Teil CLI.
"""

import os
from pathlib import Path
from typing import Optional

import click
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.text import Text

import flashtensors as flash
from cli.commands.base import BaseCommand, InteractiveCommandMixin
from cli.components import create_simple_command_box
from cli.styles.colors import WARNING
from cli.components.model_status import display_model_status


class PullCommand(BaseCommand, InteractiveCommandMixin):
    """Handle model downloads in the Teil CLI."""

    def __init__(self):
        super().__init__()
        self.default_model_dir = os.path.expanduser("~/.teil/models")

    @property
    def command(self):
        """Return the Click command instance."""

        @click.command(help="Download a model")
        @click.argument("model_name", required=False)
        def cmd(model_name: Optional[str] = None):
            self.execute(model_name=model_name)

        return cmd

    def execute(self, model_name: Optional[str] = None):
        """Execute the model download process."""
        if not model_name:
            self.console.print(
                "Please specify a model name. Available models:", style=WARNING
            )
            self._list_available_models()
            return

        self._download_model(model_name)

    def _list_available_models(self):
        """List available models to download"""
        # List available models
        models = ["llama2-7b", "llama2-13b", "mistral-7b"]
        for model in models:
            self.console.print(f"  • {model}", style=WARNING)

        # Show example command
        create_simple_command_box(
            command="teil pull llama2-7b",
            description="Download a specific model",
            console=self.console,
        )

    def _download_model(self, model_name: str):
        """Simulate model download with progress"""
        self.console.print(f"\n🔄 Transforming model {model_name}...")
        result = teil.register_model(
            model_id=model_name,
            backend="transformers",  # We should have an "auto" backend option
            torch_dtype="float16",
            force=False,  # Don't overwrite if already exists
            hf_token=None,  # Add HuggingFace token if needed for private models
        )

        display_model_status(result)

    @classmethod
    def get_interactive_commands(cls):
        """Return interactive command help text."""
        return {
            "pull": "Download a model",
            "pull <model>": "Download a specific model",
        }

    @classmethod
    def handle_interactive(cls, command: str, *args, **kwargs):
        """Handle interactive pull command."""
        if command == "pull":
            instance = cls()
            model_name = args[0] if args else None
            instance.execute(model_name=model_name, **kwargs)
            return True
        return False
