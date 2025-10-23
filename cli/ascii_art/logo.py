from rich.text import Text
from cli.styles.colors import ASCII_ART_GRADIENT

FLASH_LOGO = """
████████╗██╗      █████╗ ███████╗██╗  ██╗████████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗ ███████╗
██╔═════╝██║     ██╔══██╗██╔════╝██║  ██║╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗██╔════╝
█████╗   ██║     ███████║███████╗███████║   ██║   █████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝███████╗
██╔══╝   ██║     ██╔══██║╚════██║██╔══██║   ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗╚════██║
██║      ███████╗██║  ██║███████║██║  ██║   ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║███████║
╚═╝      ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝"""


def create_gradient_title(
    ascii_art: str = FLASH_LOGO, colors: list = ASCII_ART_GRADIENT
) -> Text:
    """Create the Flash Tensors gradient title with high-resolution ASCII art and horizontal gradient"""

    lines = ascii_art.split("\n")
    gradient_ascii = Text()

    for line in lines:
        line_text = Text()
        if line.strip():
            line_length = len(line)
            segment_size = line_length // len(colors)

            for i, char in enumerate(line):
                if char.strip():
                    color_index = min(i // segment_size, len(colors) - 1)
                    line_text.append(char, style=f"bold {colors[color_index]}")
                else:
                    line_text.append(char)
        else:
            line_text.append(line)

        gradient_ascii.append(line_text)
        gradient_ascii.append("\n")

    return gradient_ascii
