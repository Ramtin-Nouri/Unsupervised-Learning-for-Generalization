"""All sort of helper functions for the project."""

# PRINT IN TERMINAL WITH COLOR ------------------------------------------------
terminal_colors = {"purple": '\033[95m', "blue": '\033[94m',"cyan": '\033[96m',
                   "green": '\033[92m', "warning": '\033[93m', "fail": '\033[91m',
                   "end": '\033[0m', "bold": '\033[1m', "underline": '\033[4m'}

def print_color(text, color):
    """Prints text in color."""
    print(terminal_colors[color] + text + terminal_colors["end"])

def print_warning(text):
    """Prints text in warning color."""
    print_color(text, "warning")

def print_fail(text):
    """Prints text in fail color."""
    print_color(text, "fail")
# ------------------------------------------------------------------------------

