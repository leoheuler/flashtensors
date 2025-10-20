"""
Run command for executing models in the Flash CLI.
"""

import os
from pathlib import Path
from typing import Optional

import click
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from cli.commands.base import BaseCommand, InteractiveCommandMixin
from cli.components import create_simple_command_box
from cli.components.command_box import create_command_box
from cli.styles.colors import ERROR, WARNING, SUCCESSFULL_UPDATE
import flashtensors as flash
import torch
from vllm import SamplingParams


class RunCommand(BaseCommand, InteractiveCommandMixin):
    """Handle model execution in the Flash CLI."""

    def __init__(self):
        super().__init__()
        self.default_model_dir = os.path.expanduser("~/.flash/models")
        self.running_process = None

    @property
    def command(self):
        """Return the Click command instance."""

        @click.command(help="Run a model")
        @click.argument("model_name", required=True)
        @click.argument("prompt", required=True)
        def cmd(model_name: str, prompt: str):
            self.execute(model_name=model_name, prompt=prompt)

        return cmd

    def execute(self, **kwargs):
        """Execute the model."""
        model_name = kwargs.get("model_name")
        prompt = kwargs.get("prompt")

        if not model_name:
            self._list_available_models()
            return

        self._run_model(model_name, prompt)

    def _list_available_models(self):
        """List available models in the model directory."""
        model_dir = Path(self.default_model_dir)

        if not model_dir.exists():
            self.console.print(
                f"\n[bold {WARNING}]No models found. Download a model first:[/]"
            )

            create_simple_command_box(
                command="flash pull <model_name>",
                description="Download a model",
                console=self.console,
            )
            return

        models = [d.name for d in model_dir.iterdir() if d.is_dir()]

        if not models:
            self.console.print(
                f"\n[bold {WARNING}]No models found. Download a model first:[/]"
            )

            create_simple_command_box(
                command="flash pull llama2-7b",
                description="Example model to download",
                console=self.console,
            )
            return

        table = Table(
            title="\nAvailable Models", show_header=True, header_style="bold magenta"
        )
        table.add_column("Model", style="cyan", width=30)
        table.add_column("Path")

        for model in sorted(models):
            table.add_row(model, str(model_dir / model))

        self.console.print(table)

        create_command_box(
            command="flash run <model_name>",
            description="Run a specific model",
            console=self.console,
        )

    def _run_model(self, model_name: str, prompt: Optional[str] = None):
        """Run the specified model."""
        import time
        load_start_time = time.time()
        
        model, tokenizer = flash.load_model(
            model_id=model_name,
            backend="transformers",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        load_time = time.time() - load_start_time
        
        print(prompt)
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Generate text
        start_generate = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 50,
                temperature=0.8,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the original prompt from the output
            generated_only = generated_text[len(prompt):].strip()
            
            self.console.print(f"({model_name}): {generated_only}")
            self.console.print()

            self.console.print(f"Model loaded: {load_time:.2f}s | Response generated in:  {time.time() - start_generate:.2f}s", style=f"bold {SUCCESSFULL_UPDATE}")
            self.console.print(f"")
            flash.cleanup_gpu()

    @classmethod
    def get_interactive_commands(cls):
        """Return interactive command help text."""
        return {
            "run": "Run a model",
            "run <model>": "Run a specific model",
        }

    @classmethod
    def handle_interactive(cls, command: str, *args, **kwargs):
        """Handle interactive run command."""
        if command == "run":
            instance = cls()
            model_name = args[0] if args else None
            prompt = args[1] if len(args) > 1 else None

            instance.execute(model_name=model_name, prompt=prompt)

            return True
        return False
