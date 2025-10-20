"""
Remove command for deleting models in the Flash CLI.
"""
import os
import shutil
from pathlib import Path
from typing import Optional

import click

from cli.commands.base import BaseCommand, InteractiveCommandMixin
from cli.styles.colors import WARNING


class RemoveCommand(BaseCommand, InteractiveCommandMixin):
    """Handle model removal in the Flash CLI."""
    
    def __init__(self):
        super().__init__()
        self.default_model_dir = os.path.expanduser("~/.flash/models")
        
    @property
    def command(self):
        """Return the Click command instance."""
        @click.command(help="Remove a downloaded model")
        @click.argument("model_name", required=False)
        @click.option("--all", "remove_all", is_flag=True, help="Remove all downloaded models")
        @click.option("--force", "-f", is_flag=True, help="Force removal without confirmation")
        def cmd(model_name: Optional[str] = None, remove_all: bool = False, force: bool = False):
            self.execute(
                model_name=model_name,
                remove_all=remove_all,
                force=force
            )
        return cmd
    
    def execute(self, **kwargs):
        """Execute the model removal process."""
        model_name = kwargs.get("model_name")
        remove_all = kwargs.get("remove_all", False)
        force = kwargs.get("force", False)
        
        if not any([model_name, remove_all]):
            self._list_models_for_removal()
            return
            
        if remove_all:
            self._remove_all_models(force)
        elif model_name:
            self._remove_single_model(model_name, force)
    
    def _list_models_for_removal(self):
        """List models that can be removed."""
        model_dir = Path(self.default_model_dir)
        
        if not model_dir.exists() or not any(model_dir.iterdir()):
            self.console.print("\nNo models found to remove.", style=f"bold {WARNING}")
            return
            
        models = [d.name for d in model_dir.iterdir() if d.is_dir()]
        
        if not models:
            self.console.print("\nNo models found to remove.", style=f"bold {WARNING}")
            return
            
        self.console.print("\n[bold]Available models to remove:[/]")
        for i, model in enumerate(sorted(models), 1):
            self.console.print(f"  {i}. {model}")
            
        self.console.print("\nTo remove a model, run:")
        self.console.print("  [bold blue]flash remove <model_name>[/]")
        self.console.print("\nTo remove all models, run:")
        self.console.print("  [bold blue]flash remove --all[/]")
    
    def _remove_single_model(self, model_name: str, force: bool):
        """Remove a single model."""
        model_path = Path(self.default_model_dir) / model_name
        
        if not model_path.exists():
            self.console.print(f"\n❌ Model '{model_name}' not found.", style="bold red")
            self.console.print("Use [bold blue]flash models[/] to list available models.")
            return
            
        if not force and not click.confirm(f"\n⚠️  Are you sure you want to remove '{model_name}'?"):
            self.console.print("\n🚫 Model removal cancelled.", style=f"bold {WARNING}")
            return
            
        try:
            with self.console.status(f"[bold]Removing {model_name}...") as status:
                if model_path.is_symlink():
                    model_path.unlink()
                else:
                    shutil.rmtree(model_path)
                
            self.console.print(f"\n✅ Successfully removed model: {model_name}", style="bold green")
            
        except Exception as e:
            self.console.print(f"\n❌ Failed to remove model: {str(e)}", style="bold red")
    
    def _remove_all_models(self, force: bool):
        """Remove all downloaded models."""
        model_dir = Path(self.default_model_dir)
        
        if not model_dir.exists() or not any(model_dir.iterdir()):
            self.console.print("\nNo models found to remove.", style=f"bold {WARNING}")
            return
            
        models = [d.name for d in model_dir.iterdir() if d.is_dir()]
        
        if not models:
            self.console.print("\nNo models found to remove.", style=f"bold {WARNING}")
            return
            
        self.console.print("\n[bold]The following models will be removed:[/]")
        for model in sorted(models):
            self.console.print(f"  • {model}")
            
        if not force and not click.confirm("\n⚠️  Are you sure you want to remove ALL models? This cannot be undone."):
            self.console.print("\n🚫 Model removal cancelled.", style=f"bold {WARNING}")
            return
            
        try:
            with self.console.status("[bold]Removing all models...") as status:
                for model in models:
                    model_path = model_dir / model
                    if model_path.is_symlink():
                        model_path.unlink()
                    else:
                        shutil.rmtree(model_path, ignore_errors=True)
            
            self.console.print("\n✅ Successfully removed all models.", style="bold green")
            
        except Exception as e:
            self.console.print(f"\n❌ Failed to remove models: {str(e)}", style="bold red")
    
    @classmethod
    def get_interactive_commands(cls):
        """Return interactive command help text."""
        return {
            'remove': 'Remove a downloaded model',
            'remove <model>': 'Remove a specific model',
            'remove --all': 'Remove all downloaded models',
            'remove --force': 'Remove without confirmation',
        }
        
    @classmethod
    def handle_interactive(cls, command: str, *args, **kwargs):
        """Handle interactive remove command."""
        if command == 'remove':
            instance = cls()
            
            # Parse arguments
            options = {}
            
            if '--all' in args:
                options['remove_all'] = True
                args.remove('--all')
                
            if '--force' in args or '-f' in args:
                options['force'] = True
                if '--force' in args:
                    args.remove('--force')
                if '-f' in args:
                    args.remove('-f')
            
            model_name = args[0] if args else None
            if model_name and model_name.startswith('--'):
                model_name = None
                
            instance.execute(model_name=model_name, **options)
            return True
        return False
