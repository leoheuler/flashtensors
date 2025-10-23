"""
Flash Tensors CLI Commands Package

This package contains all the CLI commands organized in a modular way.
Each command should be in its own file and registered in __init__.py
"""
from typing import Dict, Type
from .base import BaseCommand, InteractiveCommandMixin
from .auth import AuthCommand
from .pull import PullCommand
from .run import RunCommand
from .remove import RemoveCommand
from .models import ModelsCommand
from .start import StartCommand
from .stop import StopCommand

__all__ = ['BaseCommand', 'InteractiveCommandMixin', 'register_commands']

# Register all commands here
COMMANDS: Dict[str, Type[BaseCommand]] = {
    "auth": AuthCommand,
    "pull": PullCommand,
    "stop": StopCommand,
    "run": RunCommand,
    "remove": RemoveCommand,
    "models": ModelsCommand,
    "start": StartCommand,
    # Add more commands here
}

def register_commands(cli):
    """Register all commands with the Click CLI"""
    for name, cmd_class in COMMANDS.items():
        cmd = cmd_class()
        cli.add_command(cmd.command, name=name)
