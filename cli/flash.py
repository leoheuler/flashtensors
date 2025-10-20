import os
import sys
import time
from typing import Dict, Optional, Type

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completion, WordCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.text import Text

from cli.ascii_art.logo import create_gradient_title

from cli.commands import InteractiveCommandMixin, register_commands
from cli.styles.colors import BORDER, COMMAND, HEADER, PARAGRAPH
import flashtensors as flash
from flashtensors.utils.logger import enable_quiet_mode


console = Console()


def show_startup_screen():
    """Display the beautiful startup screen with command flow"""
    console.clear()

    # First show only the logo
    gradient_title = create_gradient_title()
    console.print(gradient_title)
    console.print()

    hero = Text(
        "Run any AI model on your device or in the cloud.",
        style=f"bold {HEADER}",
    )
    console.print(hero)
    console.print()

    # Create command sections with highlighted commands
    getting_started = [
        Text.assemble(
            ("1. ", PARAGRAPH),
            ("auth", f"bold {COMMAND}"),
            (" - Authenticate with Teil", PARAGRAPH),
        ),
        Text.assemble(
            ("2. ", PARAGRAPH),
            ("engine start", f"bold {COMMAND}"),
            (" - Start Teil engine", PARAGRAPH),
        ),
        Text.assemble(
            ("3. ", PARAGRAPH),
            ("pull ", f"bold {COMMAND}"),
            ("<model>", "bold #ff79c6"),
            (" - Add a model to your device", PARAGRAPH),
        ),
        Text.assemble(
            ("4. ", PARAGRAPH),
            ("run ", f"bold {COMMAND}"),
            ("<model>", "bold #ff79c6"),
            (" - Run a model in ephemeral mode", PARAGRAPH),
        ),
        Text.assemble(
            ("5. ", PARAGRAPH),
            ("remove ", f"bold {COMMAND}"),
            ("<model>", "bold #ff79c6"),
            (" - Remove a model from your device", PARAGRAPH),
        ),
        Text.assemble(
            ("6. ", PARAGRAPH),
            ("models", f"bold {COMMAND}"),
            (" - List all available models", PARAGRAPH),
        ),
    ]

    # Create panels for each section
    def create_section(title, commands, style="dim"):
        # Create a single text object with newlines
        content = Text()
        for i, cmd in enumerate(commands):
            if i > 0:
                content.append("\n")
            content.append(cmd)

        return Panel(
            content,
            title=f"[b]{title}[/]",
            border_style=style,
            padding=(1, 2, 1, 4),
            title_align="left",
        )

    # Print the combined sections
    console.print(create_section("Getting Started", getting_started, BORDER))
    console.print()


@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show version information")
@click.pass_context
def cli(ctx, version):
    """Teil CLI - Your AI-powered development assistant"""
    if version:
        console.print("Teil-cli v2.5-pro (99% context left)")
        return

    if ctx.invoked_subcommand is None:
        # Try to connect to existing server when entering interactive mode
        try:
            enable_quiet_mode()
            
            console.print("🔌 Connecting to flashtensors server...", style="dim")
            flash.connect()
            console.print("✅ Connected to flashtensors server", style="green")
        except ConnectionError as e:
            console.print(f"⚠️  {str(e)}", style="yellow")
            console.print("💡 Run 'flash start' to start the server", style="dim")
        except Exception as e:
            console.print(f"❌ Connection error: {str(e)}", style="red")
        
        show_startup_screen()
        interactive_mode()


# Register all commands
register_commands(cli)


class CommandCompleter(WordCompleter):
    """Custom completer that supports command help in suggestions"""

    def get_completions(self, document: Document, complete_event):
        word_before_cursor = document.get_word_before_cursor(WORD=True)

        for cmd, help_text in self.words.items():
            if cmd.startswith(word_before_cursor):
                yield Completion(
                    cmd, start_position=-len(word_before_cursor), display_meta=help_text
                )


def get_interactive_commands() -> Dict[str, str]:
    """Get all available interactive commands and their help text"""
    commands = {}
    for cmd_class in InteractiveCommandMixin.__subclasses__():
        commands.update(cmd_class.get_interactive_commands())
    return commands


def handle_interactive_command(user_input: str) -> bool:
    """Handle an interactive command"""
    parts = user_input.strip().split()
    if not parts:
        return False

    command = parts[0].lower()
    args = parts[1:]

    for cmd_class in InteractiveCommandMixin.__subclasses__():
        if cmd_class.handle_interactive(command, *args):
            return True

    return False


def interactive_mode():
    """Interactive REPL with command autocompletion"""
    # Get available commands
    commands = get_interactive_commands()

    # Add built-in commands
    commands.update(
        {
            "exit": "Exit the interactive shell",
            "quit": "Exit the interactive shell",
            "clear": "Clear the screen",
            "help": "Show this help message",
        }
    )

    # Create command completer
    completer = CommandCompleter(commands, ignore_case=True, sentence=True)

    # Simple prompt style
    def create_prompt():
        return [
            ("#ff79c6", "❯"),
            ("#bd93f9", "❯ "),
        ]

    # Set up key bindings
    kb = KeyBindings()

    @kb.add(Keys.ControlC)
    @kb.add(Keys.ControlD)
    def _(event):
        raise KeyboardInterrupt()

    # Create prompt session
    session = PromptSession(
        completer=completer,
        complete_while_typing=True,
        key_bindings=kb,
        complete_style="column",
        mouse_support=True,
    )

    while True:
        try:
            console.print(Rule(style="dim"))

            try:
                user_input = session.prompt(create_prompt()).strip()

                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit"):
                    console.print("Goodbye! 👋", style="bold green")
                    break

                if user_input.lower() == "clear":
                    console.clear()
                    continue

                if user_input.lower() == "help":
                    show_help(commands)
                    continue

                # Try to handle with command handlers
                if not handle_interactive_command(user_input):
                    console.print(
                        f"Unknown command: {user_input}",
                        style="bold red",
                    )

            except KeyboardInterrupt:
                console.print("\nGoodbye! 👋", style="bold green")
                break

        except Exception as e:
            console.print(f"Error: {e}", style="bold red")
            continue


def handle_write_command(user_input):
    """Handle write commands like in the screenshot"""
    console.print(
        '• I will start by searching the web for "Teil CLI" to understand its main features and purpose.',
        style="dim",
    )

    # Simulate web search with progress
    with console.status(
        '[bold green]GoogleSearch[/] Searching the web for: "Teil CLI features and purpose"'
    ):
        time.sleep(2)

    console.print()
    console.print("📍 Uncovering Teil's Awesome", style=f"bold {PARAGRAPH}", end="")
    console.print(" (esc to cancel, 21s)", style="dim")
    console.print()

    # Show simulated search results
    search_result = """
Using a Teil.md files

The Teil CLI is a powerful command-line interface that brings AI assistance directly to your terminal. 
Key features include:

- Interactive chat mode with context awareness
- File editing and code generation capabilities  
- Web search integration for real-time information
- Beautiful terminal UI with gradient text and rich formatting
- Progress indicators and status displays
- Multi-modal input support
"""

    with console.status("Processing information..."):
        time.sleep(1)

    console.print(
        Panel(search_result.strip(), title="Search Results", border_style=BORDER)
    )


def show_help(commands: Dict[str, str]):
    """Show help for interactive commands"""
    from rich.table import Table

    table = Table(title="Available Commands", show_header=False, box=None)
    table.add_column("Command", style=COMMAND, no_wrap=True)
    table.add_column("Description", style=PARAGRAPH)

    for cmd, help_text in sorted(commands.items()):
        table.add_row(cmd, help_text)

    console.print(table)


if __name__ == "__main__":
    cli()
