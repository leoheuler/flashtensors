"""Rich UI components for displaying model status information."""
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text
from rich.panel import Panel
from typing import Dict, Any, Optional


def display_model_status(result: Dict[str, Any]) -> None:
    """Display model transformation status in a rich formatted way.
    
    Args:
        result: Dictionary containing model transformation results with status and metrics
    """
    console = Console()
    
    # Main status panel
    status_emoji = "✅" if result['status'].lower() == 'success' else "❌"
    status_text = Text.assemble(
        (f"{status_emoji} Model Transformation ", "bold green"),
        (f"{result['status'].upper()}", f"bold {'green' if result['status'].lower() == 'success' else 'red'}")
    )
    
    # Create a table for metrics
    table = Table(show_header=False, box=box.ROUNDED, padding=(0, 2))
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="magenta")
    
    # Add path row
    table.add_row("Path", result.get('path', 'N/A'))
    
    # Add metrics if available
    if "metrics" in result:
        metrics = result["metrics"]
        table.add_row("Download time", f"{metrics['download_time']:.2f} seconds")
        table.add_row("Transform time", f"{metrics['transform_time']:.2f} seconds")
        table.add_row("Total time", f"{metrics['total_time']:.2f} seconds")
        table.add_row("Model size", f"{metrics['model_size'] / (1024**3):.2f} GB")
    
    # Print the output
    console.print("\n")
    console.print(Panel.fit(
        table,
        title=status_text,
        border_style="green",
        padding=(1, 2)
    ))
    console.print("")  # Add an extra newline for better spacing
