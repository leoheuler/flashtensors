"""
Base command class for all CLI commands.
"""

from typing import Optional

import click
from rich.console import Console


class BaseCommand:
    """Base class for all CLI commands."""
    
    def __init__(self):
        self.console = Console()
        
    @property
    def command(self):
        """Return the Click command instance.
        
        Subclasses should implement this to return a Click command.
        """
        raise NotImplementedError("Subclasses must implement this method")
        
    def execute(self, **kwargs):
        """Execute the command logic.
        
        Subclasses should implement this with their specific command logic.
        """
        raise NotImplementedError("Subclasses must implement this method")


class InteractiveCommandMixin:
    """Mixin for commands that can be run in interactive mode."""
    
    @classmethod
    def get_interactive_commands(cls) -> dict:
        """Return a dictionary of command names to help text for interactive mode."""
        return {}
        
    @classmethod
    def handle_interactive(cls, command: str, *args, **kwargs) -> bool:
        """Handle an interactive command.
        
        Returns:
            bool: True if the command was handled, False otherwise
        """
        return False
