"""
CommandBox component for displaying commands in a consistent, styled format.
"""

from typing import List, Optional, Tuple

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from cli.styles.colors import COMMAND, PARAGRAPH


class CommandBox:
    """A component for displaying commands in a consistent, styled format."""

    def __init__(
        self,
        command_parts: List[Tuple[str, str]],
        description: Optional[str] = None,
        width: int = 70,  # Increased default width
        show_copy_hint: bool = True,
    ):
        """Initialize the CommandBox.

        Args:
            command_parts: List of (text, style) tuples for each part of the command
            description: Optional description to show after the command
            width: Width of the command box
            show_copy_hint: Whether to show the copy hint (clipboard emoji and instruction)
        """
        self.command_parts = command_parts
        self.description = description
        self.width = width
        self.show_copy_hint = show_copy_hint

    def render(self) -> Panel:
        """Render the command box as a Rich Panel."""
        # Create the actual command string for easy copying
        command_str = " ".join(
            part[0] for part in self.command_parts if not part[0].startswith("#")
        ).strip()

        # Create command display with better visibility and spacing
        command_text = Text()
        command_text.append(" ", style="dim")  # Left padding
        command_text.append("$", style=f"{COMMAND} dim")
        command_text.append(" ", style="dim")  # Space after $

        # Add each part of the command with proper spacing
        for i, (text, style) in enumerate(self.command_parts):
            if text == "$":
                continue
            command_text.append(text, style=style)
            # Add space between command parts, but not after the last one
            if i < len(self.command_parts) - 1 and not text.endswith(" "):
                command_text.append(" ", style="dim")

        # Add description on the same line if provided
        if self.show_copy_hint and self.description:
            command_text.append("  ", style="dim")  # Add some space after command
            command_text.append(f"# {self.description}", style=f"{PARAGRAPH} dim")

        return Panel(
            command_text,
            border_style=f"{COMMAND} dim",
            box=ROUNDED,
            padding=(1, 2),  # More vertical padding
            width=self.width,
            style="",
        )

    def print(self, console: Optional[Console] = None) -> None:
        """Print the command box to the console.

        Args:
            console: Rich Console instance to print to. Uses the default console if not provided.
        """
        console = console or Console()
        console.print(self.render())


def create_command_box(
    command_parts: List[Tuple[str, str]],
    description: Optional[str] = None,
    width: int = 60,
    show_copy_hint: bool = True,
    console: Optional[Console] = None,
) -> None:
    """Helper function to create and print a command box in one step.

    Args:
        command_parts: List of (text, style) tuples for each part of the command
        description: Optional description to show after the command
        width: Width of the command box
        show_copy_hint: Whether to show the copy hint (clipboard emoji and instruction)
        console: Rich Console instance to print to. Uses the default console if not provided.
    """
    box = CommandBox(
        command_parts=command_parts,
        description=description,
        width=width,
        show_copy_hint=show_copy_hint,
    )
    box.print(console)


def create_simple_command_box(
    command: str,
    description: Optional[str] = None,
    width: int = 60,
    show_copy_hint: bool = True,
    console: Optional[Console] = None,
) -> None:
    """Helper function to create and print a command box from a simple command string.

    Args:
        command: The command string to display
        description: Optional description to show after the command
        width: Width of the command box
        show_copy_hint: Whether to show the copy hint (clipboard emoji and instruction)
        console: Rich Console instance to print to. Uses the default console if not provided.
    """
    # Split command into parts (simple whitespace split for now)
    parts = command.split()
    command_parts = [(part, f"{COMMAND} bold") for part in parts]

    create_command_box(
        command_parts=command_parts,
        description=description,
        width=width,
        show_copy_hint=show_copy_hint,
        console=console,
    )
